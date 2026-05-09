#!/usr/bin python3
"""Main entry point to the training process of FaceSwap """
from __future__ import annotations
import logging
import os
import sys
import typing as T

from time import sleep
from threading import Event

import cv2
import numpy as np

from lib.gui.utils.image import TRAINING_PREVIEW
from lib.image import read_image_meta
from lib.keypress import KBHit
from lib.logger import parse_class_init
from lib.multithreading import MultiThread, FSThread
from lib.training import Preview, PreviewBuffer, TriggerType
from lib.training.data import get_label
from lib.training.train import Trainer
from lib.utils import (get_folder, get_image_paths, get_module_objects, handle_deprecated_cli_opts,
                       FaceswapError)
from plugins.plugin_loader import PluginLoader
from plugins.train.trainer.base import TrainConfig


if T.TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from plugins.train.model._base import ModelBase


logger = logging.getLogger(__name__)


class Train():
    """The Faceswap Training Process.

    The training process is responsible for training a model on a set of source faces and a set of
    destination faces.

    The training process is self contained and should not be referenced by any other scripts, so it
    contains no public properties.

    Parameters
    ----------
    arguments
        The arguments to be passed to the training process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments: argparse.Namespace) -> None:
        logger.debug(parse_class_init(locals()))
        self._args = handle_deprecated_cli_opts(arguments)

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

    @classmethod
    def _validate_image_counts(cls, side: str, num_images: int) -> None:
        """Validate that there are sufficient images to commence training without raising an
        error.

        Confirms that there are at least 24 images in each folder. Whilst this is not enough images
        to train a Neural Network to any successful degree, it should allow the process to train
        without raising errors when generating previews.

        A warning is raised if there are fewer than 250 images on any side.

        Parameters
        ----------
        side
            The side of the model that we are validating counts for
        num_images
            The number of images for the side
        """
        msg = ("You need to provide a significant number of images to successfully train a Neural "
               "Network. Aim for between 500 - 5000 images per side.")
        if num_images < 25:
            logger.error("Side %s contains fewer than 25 images.", side)
            logger.error(msg)
            sys.exit(1)
        if num_images < 250:
            logger.warning("Side %s contains fewer than 250 images. "
                           "Results are likely to be poor.", side)
            logger.warning(msg)

    @classmethod
    def _validate_faceswap_image(cls, image_path: str) -> None:
        """Validate that the given image path is to a faceswap training image. Exits with error
        if a non-faceswap image is found

        Parameters
        ----------
        image_path
            Full path to a faceswap .png to validate
        """
        meta = read_image_meta(image_path)
        logger.debug("[Train] Test file: (filename: %s, metadata: %s)", image_path, meta)
        if "itxt" not in meta or "alignments" not in meta["itxt"]:
            logger.error("The input folder '%s' contains images that are not extracted faces.",
                         os.path.dirname(image_path))
            logger.error("You can only train a model on faces generated from Faceswap's "
                         "extract process. Please check your sources and try again.")
            sys.exit(1)

    def _get_images(self) -> list[str]:
        """Check the image folders exist and contains valid extracted faces.

        Returns
        -------
        The folder path for each side of the model to be trained
        """
        logger.debug("[Train] Getting image paths")
        retval: list[str] = []
        input_folders = [self._args.input_a, self._args.input_b]
        for idx, image_dir in enumerate(input_folders):
            key = get_label(idx, len(input_folders))
            if not os.path.isdir(image_dir):
                logger.error("Error: '%s' does not exist", image_dir)
                sys.exit(1)

            test = get_image_paths(image_dir, ".png")
            if not test:
                logger.error("Error: '%s' contains no images", image_dir)
                sys.exit(1)
            # Validate the first image is a detected face
            self._validate_faceswap_image(next(img for img in test))
            self._validate_image_counts(key, len(test))
            retval.append(image_dir)
            logger.info("Model %s Directory: '%s' (%s images)", key, image_dir, len(test))

        return retval

    def _set_timelapse(self) -> bool:
        """Validate timelapse settings

        Returns
        -------
        ``True`` if timelapse is enabled and valid otherwise ``False``
        """
        if (not self._args.timelapse_input_a and
                not self._args.timelapse_input_b and
                not self._args.timelapse_output):
            return False
        if (not self._args.timelapse_input_a or
                not self._args.timelapse_input_b or
                not self._args.timelapse_output):
            raise FaceswapError("To enable the timelapse, you have to supply all the parameters "
                                "(--timelapse-input-A, --timelapse-input-B and "
                                "--timelapse-output).")

        timelapse_folders = [self._args.timelapse_input_a, self._args.timelapse_input_b]
        get_folder(self._args.timelapse_output)

        for idx, folder in enumerate(timelapse_folders):
            side = "a" if idx == 0 else "b"
            if folder is not None and not os.path.isdir(folder):
                raise FaceswapError(f"The Timelapse path '{folder}' does not exist")

            training_folder = getattr(self._args, f"input_{side}")
            if folder == training_folder:
                continue  # Time-lapse folder is training folder

            filenames = [os.path.join(folder, fname) for fname in os.listdir(folder)
                         if os.path.splitext(fname)[-1].lower() == ".png"]
            if not filenames:
                raise FaceswapError(f"The Timelapse path '{folder}' does not contain any valid "
                                    "images")

            self._validate_faceswap_image(filenames[0])
        logger.debug("[Train] Timelapse enabled")
        return True

    def process(self) -> None:
        """The entry point for triggering the Training Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        if self._args.summary:
            self._load_model()
            return
        logger.debug("[Train] Starting Training Process")
        logger.info("Training data directory: %s", self._args.model_dir)
        thread = self._start_thread()
        # from lib.queue_manager import queue_manager; queue_manager.debug_monitor(1)
        err = self._monitor(thread)
        self._end_thread(thread, err)
        logger.debug("[Train] Completed Training Process")

    def _start_thread(self) -> MultiThread:
        """Put the :func:`_training` into a background thread so we can keep control.

        Returns
        -------
        The background thread for running training
        """
        logger.debug("[Train] Launching Trainer thread")
        thread = MultiThread(target=self._training)
        thread.start()
        logger.debug("[Train] Launched Trainer thread")
        return thread

    def _end_thread(self, thread: MultiThread, err: bool) -> None:
        """Output message and join thread back to main on termination.

        Parameters
        ----------
        thread
            The background training thread
        err
            Whether an error has been detected in :func:`_monitor`
        """
        logger.debug("[Train] Ending Training thread")
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
        logger.debug("[Train] Ended training thread")

    def _training(self) -> None:
        """The training process to be run inside a thread."""
        trainer = None
        try:
            sleep(0.5)  # Let preview instructions flush out to logger
            logger.debug("[Train] Commencing Training")
            logger.info("Loading data, this may take a while...")
            model = self._load_model()
            trainer = self._load_trainer(model)
            if trainer.exit_early:
                logger.debug("[Train] Trainer exits early")
                self._stop = True
                return
            self._run_training_cycle(trainer)
        except KeyboardInterrupt:
            try:
                logger.debug("[Train] Keyboard Interrupt Caught. Saving Weights and exiting")
                if trainer is not None:
                    trainer.save(is_exit=True)
            except KeyboardInterrupt:
                logger.info("Saving model weights has been cancelled!")
            sys.exit(0)
        except Exception as err:
            raise err

    def _load_model(self) -> ModelBase:
        """Load the model requested for training.

        Returns
        -------
        The requested model plugin
        """
        logger.debug("[Train] Loading Model")
        model_dir = get_folder(self._args.model_dir)
        model: ModelBase = PluginLoader.get_model(self._args.trainer)(
            model_dir,
            self._args,
            predict=False)
        model.build()
        logger.debug("[Train] Loaded Model")
        return model

    def _load_trainer(self, model: ModelBase) -> Trainer:
        """Load the trainer requested for training.

        Parameters
        ----------
        model
            The requested model plugin

        Returns
        -------
        The model training loop with the requested trainer plugin loaded
        """
        logger.debug("[Train] Loading Trainer")
        trainer = "distributed" if self._args.distributed else "original"
        if trainer == "distributed":
            import torch  # pylint:disable=import-outside-toplevel
            gpu_count = torch.cuda.device_count()
            if gpu_count < 2:
                logger.warning("Distributed selected but fewer than 2 GPUs detected. Switching "
                               "to Original")
                trainer = "original"

        config = TrainConfig(folders=self._images,
                             batch_size=self._args.batch_size,
                             augment_color=not self._args.no_augment_color,
                             flip=not self._args.no_flip,
                             warp=not self._args.no_warp,
                             cache_landmarks=self._args.warp_to_landmarks,
                             lr_finder=self._args.use_lr_finder,
                             snapshot_interval=self._args.snapshot_interval)
        retval = Trainer(PluginLoader.get_trainer(trainer)(model, config),
                         self._args.preview or self._args.write_image or self._args.redirect_gui,
                         timelapse_folders=[self._args.timelapse_input_a,
                                            self._args.timelapse_input_b],
                         timelapse_output=self._args.timelapse_output)
        logger.debug("[Train] Loaded Trainer")
        return retval

    def _run_training_cycle(self, trainer: Trainer) -> None:
        """Perform the training cycle.

        Handles the background training, updating previews/time-lapse on each save interval,
        and saving the model.

        Parameters
        ----------
        trainer
            The requested model trainer plugin
        """
        logger.debug("[Train] Running Training Cycle")
        update_preview_images = False
        if self._args.write_image or self._args.redirect_gui or self._args.preview:
            display_func: Callable | None = self._show
        else:
            display_func = None

        for iteration in range(1, self._args.iterations + 1):
            logger.trace("[Train] Training iteration: %s", iteration)  # type:ignore
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

            trainer.train_one_step(viewer, self._timelapse and save_iteration)

            if viewer is not None and not save_iteration:
                # Ugly spam but required by GUI to know to update window
                print("\x1b[2K", end="\r")  # Clear last line
                logger.info("[Preview Updated]")

            if self._stop:
                logger.debug("[Train] Stop received. Terminating")
                break

            if save_iteration or self._save_now:
                logger.debug("[Train] Saving (save_iterations: %s, save_now: %s) Iteration: "
                             "(iteration: %s)", save_iteration, self._save_now, iteration)
                trainer.save(is_exit=False)
                self._save_now = False
                update_preview_images = True

        logger.debug("[Train] Training cycle complete")
        trainer.save(is_exit=True)
        self._stop = True

    def _output_startup_info(self) -> None:
        """Print the startup information to the console."""
        logger.debug("[Train] Launching Monitor")
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
        """Check if a keypress has been detected.

        Parameters
        ----------
        keypress
            The keypress monitor

        Returns
        -------
        ``True`` if an exit keypress has been detected otherwise ``False``
        """
        retval = False
        try:
            if keypress.kbhit():
                console_key = keypress.getch()
                if console_key in ("\n", "\r"):
                    logger.debug("[Train] Exit requested")
                    retval = True
                if console_key in ("s", "S"):
                    logger.info("Save requested")
                    self._save_now = True
        except ValueError as err:
            if "I/O operation on closed file" in str(err):
                logger.debug("[Train] Error encountered: %s", str(err))
                retval = True
            else:
                raise
        return retval

    def _process_gui_triggers(self) -> dict[T.Literal["mask", "refresh"], bool]:
        """Check whether a file drop has occurred from the GUI to manually update the preview.

        Returns
        -------
        The trigger name as key and boolean as value
        """
        retval: dict[T.Literal["mask", "refresh"], bool] = {key: False
                                                            for key in self._gui_triggers}
        if not self._args.redirect_gui:
            return retval

        for trigger, filename in self._gui_triggers.items():
            if os.path.isfile(filename):
                logger.debug("[Train] GUI Trigger received for: '%s'", trigger)
                retval[trigger] = True
                logger.debug("[Train] Removing gui trigger file: %s", filename)
                os.remove(filename)
                if trigger == "refresh":
                    print("\x1b[2K", end="\r")  # Clear last line
                    logger.info("Refresh preview requested...")
        return retval

    def _monitor(self, thread: MultiThread) -> bool:
        """Monitor the background :func:`_training` thread for key presses and errors.

        Parameters
        ----------
        thread
            The thread containing the training loop

        Returns
        -------
        ``True`` if there has been an error in the background thread otherwise ``False``
        """
        self._output_startup_info()
        keypress = KBHit(is_gui=self._args.redirect_gui)
        err = False
        while True:
            try:
                if thread.has_error:
                    logger.debug("[Train] Thread error detected")
                    err = True
                    break
                if self._stop:
                    logger.debug("[Train] Stop received")
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
                logger.debug("[Train] Keyboard Interrupt received")
                break
        logger.debug("[Train] Closing Monitor")
        self._preview.shutdown()
        keypress.set_normal_term()
        logger.debug("[Train] Closed Monitor")
        return err

    def _show(self, image: np.ndarray, name: str = "") -> None:
        """Generate the preview and write preview file output.

        Handles the output and display of preview images.

        Parameters
        ----------
        image
            The preview image to be displayed and/or written out
        name
            The name of the image for saving or display purposes. If an empty string is passed
            then it will automatically be named. Default: ""
        """
        logger.debug("[Train] Updating preview: (name: %s)", name)
        try:
            script_path = os.path.realpath(os.path.dirname(sys.argv[0]))
            if self._args.write_image:
                logger.debug("[Train] Saving preview to disk")
                img = "training_preview.png"
                img_file = os.path.join(script_path, img)
                cv2.imwrite(img_file, image)  # pylint:disable=no-member
                logger.debug("[Train] Saved preview to: '%s'", img)
            if self._args.redirect_gui:
                logger.debug("[Train] Generating preview for GUI")
                img = TRAINING_PREVIEW
                img_file = os.path.join(script_path, "lib", "gui", ".cache", "preview", img)
                cv2.imwrite(img_file, image)  # pylint:disable=no-member
                logger.debug("[Train] Generated preview for GUI: '%s'", img_file)
            if self._args.preview:
                logger.debug("[Train] Generating preview for display: '%s'", name)
                self._preview.buffer.add_image(name, image)
                logger.debug("[Train] Generated preview for display: '%s'", name)
        except Exception as err:
            logging.error("could not preview sample")
            raise err
        logger.debug("[Train] Updated preview: (name: %s)", name)


class PreviewInterface():
    """Run the preview window in a thread and interface with it

    Parameters
    ----------
    use_preview
        ``True`` if pop-up preview window has been requested otherwise ``False``
    """
    def __init__(self, use_preview: bool) -> None:
        logger.debug(parse_class_init(locals()))
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
        """The thread save preview image object"""
        return self._buffer

    @property
    def should_toggle_mask(self) -> bool:
        """Check whether the mask should be toggled and return the value. If ``True`` is returned
        then resets mask toggle back to ``False``"""
        if not self._active:
            return False
        retval = self._triggers["toggle_mask"].is_set()
        if retval:
            logger.debug("[PreviewInterface] Sending toggle mask")
            self._triggers["toggle_mask"].clear()
        return retval

    @property
    def should_refresh(self) -> bool:
        """Check whether the preview should be updated and return the value. If ``True`` is
        returned then resets the refresh trigger back to ``False``"""
        if not self._active:
            return False
        retval = self._triggers["refresh"].is_set()
        if retval:
            logger.debug("[PreviewInterface] Sending should refresh")
            self._triggers["refresh"].clear()
        return retval

    @property
    def should_save(self) -> bool:
        """Check whether a save request has been made. If ``True`` is returned then save
        trigger is set back to ``False``"""
        if not self._active:
            return False
        retval = self._triggers["save"].is_set()
        if retval:
            logger.debug("[PreviewInterface] Sending should save")
            self._triggers["save"].clear()
        return retval

    @property
    def should_quit(self) -> bool:
        """Check whether an exit request has been made. ``True`` if an exit request has
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
            logger.debug("[PreviewInterface] Sending should stop")
        return retval

    def _launch_thread(self) -> FSThread | None:
        """Launch the preview viewer in it's own thread if preview has been selected

        Returns
        -------
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
        """Send a signal to shutdown the preview window."""
        if not self._active:
            return
        logger.debug("[PreviewInterface] Sending shutdown to preview viewer")
        self._triggers["shutdown"].set()


__all__ = get_module_objects(__name__)
