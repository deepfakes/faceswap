#!/usr/bin python3
""" Main entry point to the training process of FaceSwap """
from __future__ import annotations
import logging
import os
import sys
import typing as T

from time import sleep
from threading import Event

import cv2
import numpy as np

from lib.gui.utils.image import TRAININGPREVIEW
from lib.image import read_image_meta
from lib.keypress import KBHit
from lib.multithreading import MultiThread, FSThread
from lib.training import Preview, PreviewBuffer, TriggerType
from lib.utils import (get_folder, get_image_paths, handle_deprecated_cliopts,
                       FaceswapError, IMAGE_EXTENSIONS)
from plugins.plugin_loader import PluginLoader

if T.TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from plugins.train.model._base import ModelBase
    from plugins.train.trainer._base import TrainerBase


logger = logging.getLogger(__name__)


class Train():
    """ The Faceswap Training Process.

    The training process is responsible for training a model on a set of source faces and a set of
    destination faces.

    The training process is self contained and should not be referenced by any other scripts, so it
    contains no public properties.

    Parameters
    ----------
    arguments: argparse.Namespace
        The arguments to be passed to the training process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments: argparse.Namespace) -> None:
        logger.debug("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self._args = handle_deprecated_cliopts(arguments)

        if self._args.summary:
            # If just outputting summary we don't need to initialize everything
            return

        self._images = self._get_images()
        self._timelapse = self._set_timelapse()
        gui_cache = os.path.join(
            os.path.realpath(os.path.dirname(sys.argv[0])), "lib", "gui", ".cache")
        self._gui_triggers: dict[T.Literal["mask", "refresh"], str] = {
            "mask": os.path.join(gui_cache, ".preview_mask_toggle"),
            "refresh": os.path.join(gui_cache, ".preview_trigger")}
        self._stop: bool = False
        self._save_now: bool = False
        self._preview = PreviewInterface(self._args.preview)

        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_images(self) -> dict[T.Literal["a", "b"], list[str]]:
        """ Check the image folders exist and contains valid extracted faces. Obtain image paths.

        Returns
        -------
        dict
            The image paths for each side. The key is the side, the value is the list of paths
            for that side.
        """
        logger.debug("Getting image paths")
        images = {}
        for side in ("a", "b"):
            side = T.cast(T.Literal["a", "b"], side)
            image_dir = getattr(self._args, f"input_{side}")
            if not os.path.isdir(image_dir):
                logger.error("Error: '%s' does not exist", image_dir)
                sys.exit(1)

            images[side] = get_image_paths(image_dir, ".png")
            if not images[side]:
                logger.error("Error: '%s' contains no images", image_dir)
                sys.exit(1)
            # Validate the first image is a detected face
            test_image = next(img for img in images[side])
            meta = read_image_meta(test_image)
            logger.debug("Test file: (filename: %s, metadata: %s)", test_image, meta)
            if "itxt" not in meta or "alignments" not in meta["itxt"]:
                logger.error("The input folder '%s' contains images that are not extracted faces.",
                             image_dir)
                logger.error("You can only train a model on faces generated from Faceswap's "
                             "extract process. Please check your sources and try again.")
                sys.exit(1)

            logger.info("Model %s Directory: '%s' (%s images)",
                        side.upper(), image_dir, len(images[side]))
        logger.debug("Got image paths: %s", [(key, str(len(val)) + " images")
                                             for key, val in images.items()])
        self._validate_image_counts(images)
        return images

    @classmethod
    def _validate_image_counts(cls, images: dict[T.Literal["a", "b"], list[str]]) -> None:
        """ Validate that there are sufficient images to commence training without raising an
        error.

        Confirms that there are at least 24 images in each folder. Whilst this is not enough images
        to train a Neural Network to any successful degree, it should allow the process to train
        without raising errors when generating previews.

        A warning is raised if there are fewer than 250 images on any side.

        Parameters
        ----------
        images: dict
            The image paths for each side. The key is the side, the value is the list of paths
            for that side.
        """
        counts = {side: len(paths) for side, paths in images.items()}
        msg = ("You need to provide a significant number of images to successfully train a Neural "
               "Network. Aim for between 500 - 5000 images per side.")
        if any(count < 25 for count in counts.values()):
            logger.error("At least one of your input folders contains fewer than 25 images.")
            logger.error(msg)
            sys.exit(1)
        if any(count < 250 for count in counts.values()):
            logger.warning("At least one of your input folders contains fewer than 250 images. "
                           "Results are likely to be poor.")
            logger.warning(msg)

    def _set_timelapse(self) -> dict[T.Literal["input_a", "input_b", "output"], str]:
        """ Set time-lapse paths if requested.

        Returns
        -------
        dict
            The time-lapse keyword arguments for passing to the trainer

        """
        if (not self._args.timelapse_input_a and
                not self._args.timelapse_input_b and
                not self._args.timelapse_output):
            return {}
        if (not self._args.timelapse_input_a or
                not self._args.timelapse_input_b or
                not self._args.timelapse_output):
            raise FaceswapError("To enable the timelapse, you have to supply all the parameters "
                                "(--timelapse-input-A, --timelapse-input-B and "
                                "--timelapse-output).")

        timelapse_output = get_folder(self._args.timelapse_output)

        for side in ("a", "b"):
            side = T.cast(T.Literal["a", "b"], side)
            folder = getattr(self._args, f"timelapse_input_{side}")
            if folder is not None and not os.path.isdir(folder):
                raise FaceswapError(f"The Timelapse path '{folder}' does not exist")

            training_folder = getattr(self._args, f"input_{side}")
            if folder == training_folder:
                continue  # Time-lapse folder is training folder

            filenames = [fname for fname in os.listdir(folder)
                         if os.path.splitext(fname)[-1].lower() in IMAGE_EXTENSIONS]
            if not filenames:
                raise FaceswapError(f"The Timelapse path '{folder}' does not contain any valid "
                                    "images")

            # Time-lapse images must appear in the training set, as we need access to alignment and
            # mask info. Check filenames are there to save failing much later in the process.
            training_images = [os.path.basename(img) for img in self._images[side]]
            if not all(img in training_images for img in filenames):
                raise FaceswapError(f"All images in the Timelapse folder '{folder}' must exist in "
                                    f"the training folder '{training_folder}'")

        TKey = T.Literal["input_a", "input_b", "output"]
        kwargs = {T.cast(TKey, "input_a"): self._args.timelapse_input_a,
                  T.cast(TKey, "input_b"): self._args.timelapse_input_b,
                  T.cast(TKey, "output"): timelapse_output}
        logger.debug("Timelapse enabled: %s", kwargs)
        return kwargs

    def process(self) -> None:
        """ The entry point for triggering the Training Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        if self._args.summary:
            self._load_model()
            return
        logger.debug("Starting Training Process")
        logger.info("Training data directory: %s", self._args.model_dir)
        thread = self._start_thread()
        # from lib.queue_manager import queue_manager; queue_manager.debug_monitor(1)
        err = self._monitor(thread)
        self._end_thread(thread, err)
        logger.debug("Completed Training Process")

    def _start_thread(self) -> MultiThread:
        """ Put the :func:`_training` into a background thread so we can keep control.

        Returns
        -------
        :class:`lib.multithreading.MultiThread`
            The background thread for running training
        """
        logger.debug("Launching Trainer thread")
        thread = MultiThread(target=self._training)
        thread.start()
        logger.debug("Launched Trainer thread")
        return thread

    def _end_thread(self, thread: MultiThread, err: bool) -> None:
        """ Output message and join thread back to main on termination.

        Parameters
        ----------
        thread: :class:`lib.multithreading.MultiThread`
            The background training thread
        err: bool
            Whether an error has been detected in :func:`_monitor`
        """
        logger.debug("Ending Training thread")
        if err:
            msg = "Error caught! Exiting..."
            log = logger.critical
        else:
            msg = ("Exit requested! The trainer will complete its current cycle, "
                   "save the models and quit (This can take a couple of minutes "
                   "depending on your training speed).")
            if not self._args.redirect_gui:
                msg += " If you want to kill it now, press Ctrl + c"
            log = logger.info
        log(msg)
        self._stop = True
        thread.join()
        sys.stdout.flush()
        logger.debug("Ended training thread")

    def _training(self) -> None:
        """ The training process to be run inside a thread. """
        try:
            sleep(0.5)  # Let preview instructions flush out to logger
            logger.debug("Commencing Training")
            logger.info("Loading data, this may take a while...")
            model = self._load_model()
            trainer = self._load_trainer(model)
            if trainer.exit_early:
                self._stop = True
                return
            self._run_training_cycle(model, trainer)
        except KeyboardInterrupt:
            try:
                logger.debug("Keyboard Interrupt Caught. Saving Weights and exiting")
                model.io.save(is_exit=True)
                trainer.clear_tensorboard()
            except KeyboardInterrupt:
                logger.info("Saving model weights has been cancelled!")
            sys.exit(0)
        except Exception as err:
            raise err

    def _load_model(self) -> ModelBase:
        """ Load the model requested for training.

        Returns
        -------
        :file:`plugins.train.model` plugin
            The requested model plugin
        """
        logger.debug("Loading Model")
        model_dir = get_folder(self._args.model_dir)
        model: ModelBase = PluginLoader.get_model(self._args.trainer)(
            model_dir,
            self._args,
            predict=False)
        model.build()
        logger.debug("Loaded Model")
        return model

    def _load_trainer(self, model: ModelBase) -> TrainerBase:
        """ Load the trainer requested for training.

        Parameters
        ----------
        model: :file:`plugins.train.model` plugin
            The requested model plugin

        Returns
        -------
        :file:`plugins.train.trainer` plugin
            The requested model trainer plugin
        """
        logger.debug("Loading Trainer")
        base = PluginLoader.get_trainer(model.trainer)
        trainer: TrainerBase = base(model,
                                    self._images,
                                    self._args.batch_size,
                                    self._args.configfile)
        logger.debug("Loaded Trainer")
        return trainer

    def _run_training_cycle(self, model: ModelBase, trainer: TrainerBase) -> None:
        """ Perform the training cycle.

        Handles the background training, updating previews/time-lapse on each save interval,
        and saving the model.

        Parameters
        ----------
        model: :file:`plugins.train.model` plugin
            The requested model plugin
        trainer: :file:`plugins.train.trainer` plugin
            The requested model trainer plugin
        """
        logger.debug("Running Training Cycle")
        update_preview_images = False
        if self._args.write_image or self._args.redirect_gui or self._args.preview:
            display_func: Callable | None = self._show
        else:
            display_func = None

        for iteration in range(1, self._args.iterations + 1):
            logger.trace("Training iteration: %s", iteration)  # type:ignore
            save_iteration = iteration % self._args.save_interval == 0 or iteration == 1
            gui_triggers = self._process_gui_triggers()

            if self._preview.should_toggle_mask or gui_triggers["mask"]:
                trainer.toggle_mask()
                update_preview_images = True

            if self._preview.should_refresh or gui_triggers["refresh"] or update_preview_images:
                viewer = display_func
                update_preview_images = False
            else:
                viewer = None

            timelapse = self._timelapse if save_iteration else {}
            trainer.train_one_step(viewer, timelapse)

            if viewer is not None and not save_iteration:
                # Spammy but required by GUI to know to update window
                print("")
                logger.info("[Preview Updated]")

            if self._stop:
                logger.debug("Stop received. Terminating")
                break

            if save_iteration or self._save_now:
                logger.debug("Saving (save_iterations: %s, save_now: %s) Iteration: "
                             "(iteration: %s)", save_iteration, self._save_now, iteration)
                model.io.save(is_exit=False)
                self._save_now = False
                update_preview_images = True

        logger.debug("Training cycle complete")
        model.io.save(is_exit=True)
        trainer.clear_tensorboard()
        self._stop = True

    def _output_startup_info(self) -> None:
        """ Print the startup information to the console. """
        logger.debug("Launching Monitor")
        logger.info("===================================================")
        logger.info("  Starting")
        if self._args.preview:
            logger.info("  Using live preview")
        if sys.stdout.isatty():
            logger.info("  Press '%s' to save and quit",
                        "Stop" if self._args.redirect_gui else "ENTER")
        if not self._args.redirect_gui and sys.stdout.isatty():
            logger.info("  Press 'S' to save model weights immediately")
        logger.info("===================================================")

    def _check_keypress(self, keypress: KBHit) -> bool:
        """ Check if a keypress has been detected.

        Parameters
        ----------
        keypress: :class:`lib.keypress.KBHit`
            The keypress monitor

        Returns
        -------
        bool
            ``True`` if an exit keypress has been detected otherwise ``False``
        """
        retval = False
        if keypress.kbhit():
            console_key = keypress.getch()
            if console_key in ("\n", "\r"):
                logger.debug("Exit requested")
                retval = True
            if console_key in ("s", "S"):
                logger.info("Save requested")
                self._save_now = True
        return retval

    def _process_gui_triggers(self) -> dict[T.Literal["mask", "refresh"], bool]:
        """ Check whether a file drop has occurred from the GUI to manually update the preview.

        Returns
        -------
        dict
            The trigger name as key and boolean as value
        """
        retval: dict[T.Literal["mask", "refresh"], bool] = {key: False
                                                            for key in self._gui_triggers}
        if not self._args.redirect_gui:
            return retval

        for trigger, filename in self._gui_triggers.items():
            if os.path.isfile(filename):
                logger.debug("GUI Trigger received for: '%s'", trigger)
                retval[trigger] = True
                logger.debug("Removing gui trigger file: %s", filename)
                os.remove(filename)
                if trigger == "refresh":
                    print("")  # Let log print on different line from loss output
                    logger.info("Refresh preview requested...")
        return retval

    def _monitor(self, thread: MultiThread) -> bool:
        """ Monitor the background :func:`_training` thread for key presses and errors.

        Parameters
        ----------
        thread: :class:~`lib.multithreading.MultiThread`
            The thread containing the training loop

        Returns
        -------
        bool
            ``True`` if there has been an error in the background thread otherwise ``False``
        """
        self._output_startup_info()
        keypress = KBHit(is_gui=self._args.redirect_gui)
        err = False
        while True:
            try:
                if thread.has_error:
                    logger.debug("Thread error detected")
                    err = True
                    break
                if self._stop:
                    logger.debug("Stop received")
                    break

                # Preview Monitor
                if self._preview.should_quit:
                    break
                if self._preview.should_save:
                    self._save_now = True

                # Console Monitor
                if self._check_keypress(keypress):
                    break  # Exit requested

                sleep(1)
            except KeyboardInterrupt:
                logger.debug("Keyboard Interrupt received")
                break
        self._preview.shutdown()
        keypress.set_normal_term()
        logger.debug("Closed Monitor")
        return err

    def _show(self, image: np.ndarray, name: str = "") -> None:
        """ Generate the preview and write preview file output.

        Handles the output and display of preview images.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The preview image to be displayed and/or written out
        name: str, optional
            The name of the image for saving or display purposes. If an empty string is passed
            then it will automatically be named. Default: ""
        """
        logger.debug("Updating preview: (name: %s)", name)
        try:
            scriptpath = os.path.realpath(os.path.dirname(sys.argv[0]))
            if self._args.write_image:
                logger.debug("Saving preview to disk")
                img = "training_preview.png"
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)  # pylint:disable=no-member
                logger.debug("Saved preview to: '%s'", img)
            if self._args.redirect_gui:
                logger.debug("Generating preview for GUI")
                img = TRAININGPREVIEW
                imgfile = os.path.join(scriptpath, "lib", "gui", ".cache", "preview", img)
                cv2.imwrite(imgfile, image)  # pylint:disable=no-member
                logger.debug("Generated preview for GUI: '%s'", imgfile)
            if self._args.preview:
                logger.debug("Generating preview for display: '%s'", name)
                self._preview.buffer.add_image(name, image)
                logger.debug("Generated preview for display: '%s'", name)
        except Exception as err:
            logging.error("could not preview sample")
            raise err
        logger.debug("Updated preview: (name: %s)", name)


class PreviewInterface():
    """ Run the preview window in a thread and interface with it

    Parameters
    ----------
    use_preview: bool
        ``True`` if pop-up preview window has been requested otherwise ``False``
    """
    def __init__(self, use_preview: bool) -> None:
        self._active = use_preview
        self._triggers: TriggerType = {"toggle_mask": Event(),
                                       "refresh": Event(),
                                       "save": Event(),
                                       "quit": Event(),
                                       "shutdown": Event()}
        self._buffer = PreviewBuffer()
        self._thread = self._launch_thread()

    @property
    def buffer(self) -> PreviewBuffer:
        """ :class:`PreviewBuffer`: The thread save preview image object """
        return self._buffer

    @property
    def should_toggle_mask(self) -> bool:
        """ bool: Check whether the mask should be toggled and return the value. If ``True`` is
        returned then resets mask toggle back to ``False`` """
        if not self._active:
            return False
        retval = self._triggers["toggle_mask"].is_set()
        if retval:
            logger.debug("Sending toggle mask")
            self._triggers["toggle_mask"].clear()
        return retval

    @property
    def should_refresh(self) -> bool:
        """ bool: Check whether the preview should be updated and return the value. If ``True`` is
        returned then resets the refresh trigger back to ``False`` """
        if not self._active:
            return False
        retval = self._triggers["refresh"].is_set()
        if retval:
            logger.debug("Sending should refresh")
            self._triggers["refresh"].clear()
        return retval

    @property
    def should_save(self) -> bool:
        """ bool: Check whether a save request has been made. If ``True`` is returned then save
        trigger is set back to ``False`` """
        if not self._active:
            return False
        retval = self._triggers["save"].is_set()
        if retval:
            logger.debug("Sending should save")
            self._triggers["save"].clear()
        return retval

    @property
    def should_quit(self) -> bool:
        """ bool: Check whether an exit request has been made. ``True`` if an exit request has
        been made otherwise ``False``.

        Raises
        ------
        Error
            Re-raises any error within the preview thread
         """
        if self._thread is None:
            return False

        self._thread.check_and_raise_error()

        retval = self._triggers["quit"].is_set()
        if retval:
            logger.debug("Sending should stop")
        return retval

    def _launch_thread(self) -> FSThread | None:
        """ Launch the preview viewer in it's own thread if preview has been selected

        Returns
        -------
        :class:`lib.multithreading.FSThread` or ``None``
            The thread that holds the preview viewer if preview is selected otherwise ``None``
        """
        if not self._active:
            return None
        thread = FSThread(target=Preview,
                          name="preview",
                          args=(self._buffer, ),
                          kwargs={"triggers": self._triggers})
        thread.start()
        return thread

    def shutdown(self) -> None:
        """ Send a signal to shutdown the preview window. """
        if not self._active:
            return
        logger.debug("Sending shutdown to preview viewer")
        self._triggers["shutdown"].set()
