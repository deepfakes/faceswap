#!/usr/bin python3
""" Main entry point to the training process of FaceSwap """

import logging
import os
import sys

from threading import Lock
from time import sleep
from typing import cast, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from matplotlib import backend_bases, figure, pyplot as plt, rcParams

from lib.image import read_image_meta
from lib.keypress import KBHit
from lib.multithreading import MultiThread
from lib.utils import (deprecation_warning, get_dpi, get_folder, get_image_paths,
                       FaceswapError, _image_extensions)
from plugins.plugin_loader import PluginLoader

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

if TYPE_CHECKING:
    import argparse
    from plugins.train.model._base import ModelBase
    from plugins.train.trainer._base import TrainerBase


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Train():  # pylint:disable=too-few-public-methods
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
    def __init__(self, arguments: "argparse.Namespace") -> None:
        logger.debug("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self._args = arguments
        self._handle_deprecations()

        if self._args.summary:
            # If just outputting summary we don't need to initialize everything
            return

        self._images = self._get_images()
        self._timelapse = self._set_timelapse()
        gui_cache = os.path.join(
            os.path.realpath(os.path.dirname(sys.argv[0])), "lib", "gui", ".cache")
        self._gui_triggers = dict(update=os.path.join(gui_cache, ".preview_trigger"),
                                  mask_toggle=os.path.join(gui_cache, ".preview_mask_toggle"))
        self._stop: bool = False
        self._save_now: bool = False
        self._preview = Preview()

        logger.debug("Initialized %s", self.__class__.__name__)

    def _handle_deprecations(self) -> None:
        """ Handle the update of deprecated arguments and output warnings. """
        if self._args.distributed:
            deprecation_warning("`-d`, `--distributed`",
                                "Please use `-D`, `--distribution-strategy`")
            logger.warning("Setting 'distribution-strategy' to 'mirrored'")
            setattr(self._args, "distribution_strategy", "mirrored")
            del self._args.distributed

    def _get_images(self) -> Dict[Literal["a", "b"], List[str]]:
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
            side = cast(Literal["a", "b"], side)
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
    def _validate_image_counts(cls, images: Dict[Literal["a", "b"], List[str]]) -> None:
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

    def _set_timelapse(self) -> Dict[Literal["input_a", "input_b", "output"], str]:
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
            side = cast(Literal["a", "b"], side)
            folder = getattr(self._args, f"timelapse_input_{side}")
            if folder is not None and not os.path.isdir(folder):
                raise FaceswapError(f"The Timelapse path '{folder}' does not exist")

            training_folder = getattr(self._args, f"input_{side}")
            if folder == training_folder:
                continue  # Time-lapse folder is training folder

            filenames = [fname for fname in os.listdir(folder)
                         if os.path.splitext(fname)[-1].lower() in _image_extensions]
            if not filenames:
                raise FaceswapError(f"The Timelapse path '{folder}' does not contain any valid "
                                    "images")

            # Time-lapse images must appear in the training set, as we need access to alignment and
            # mask info. Check filenames are there to save failing much later in the process.
            training_images = [os.path.basename(img) for img in self._images[side]]
            if not all(img in training_images for img in filenames):
                raise FaceswapError(f"All images in the Timelapse folder '{folder}' must exist in "
                                    f"the training folder '{training_folder}'")

        TKey = Literal["input_a", "input_b", "output"]
        kwargs = {cast(TKey, "input_a"): self._args.timelapse_input_a,
                  cast(TKey, "input_b"): self._args.timelapse_input_b,
                  cast(TKey, "output"): timelapse_output}
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
            self._run_training_cycle(model, trainer)
        except KeyboardInterrupt:
            try:
                logger.debug("Keyboard Interrupt Caught. Saving Weights and exiting")
                model.save(is_exit=True)
                trainer.clear_tensorboard()
            except KeyboardInterrupt:
                logger.info("Saving model weights has been cancelled!")
            sys.exit(0)
        except Exception as err:
            raise err

    def _load_model(self) -> "ModelBase":
        """ Load the model requested for training.

        Returns
        -------
        :file:`plugins.train.model` plugin
            The requested model plugin
        """
        logger.debug("Loading Model")
        model_dir = get_folder(self._args.model_dir)
        model: "ModelBase" = PluginLoader.get_model(self._args.trainer)(
            model_dir,
            self._args,
            predict=False)
        model.build()
        logger.debug("Loaded Model")
        return model

    def _load_trainer(self, model: "ModelBase") -> "TrainerBase":
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
        trainer: "TrainerBase" = base(model,
                                      self._images,
                                      self._args.batch_size,
                                      self._args.configfile)
        logger.debug("Loaded Trainer")
        return trainer

    def _run_training_cycle(self, model: "ModelBase", trainer: "TrainerBase") -> None:
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
        if self._args.write_image or self._args.redirect_gui or self._args.preview:
            display_func: Optional[Callable] = self._show
        else:
            display_func = None

        for iteration in range(1, self._args.iterations + 1):
            logger.trace("Training iteration: %s", iteration)  # type:ignore
            save_iteration = iteration % self._args.save_interval == 0 or iteration == 1

            if self._preview.should_toggle_mask():
                trainer.toggle_mask()
                self._preview.request_refresh()

            if self._preview.should_refresh():
                viewer = display_func
            else:
                viewer = None

            timelapse = self._timelapse if save_iteration else {}
            trainer.train_one_step(viewer, timelapse)

            if viewer is not None and not save_iteration:
                # Spammy but required by GUI to know to update window
                print("\n")
                logger.info("[Preview Updated]")

            if self._stop:
                logger.debug("Stop received. Terminating")
                break

            if save_iteration or self._save_now:
                logger.debug("Saving (save_iterations: %s, save_now: %s) Iteration: "
                             "(iteration: %s)", save_iteration, self._save_now, iteration)
                model.save(is_exit=False)
                self._save_now = False
                self._preview.request_refresh()

        logger.debug("Training cycle complete")
        model.save(is_exit=True)
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
                        "Stop" if self._args.redirect_gui or self._args.colab else "ENTER")
        if not self._args.redirect_gui and not self._args.colab and sys.stdout.isatty():
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

    def _process_gui_triggers(self) -> None:
        """ Check whether a file drop has occurred from the GUI to manually update the preview. """
        if not self._args.redirect_gui:
            return

        parent_flags = dict(mask_toggle="request_mask_toggle", update="request_refresh")
        for trigger in ("mask_toggle", "update"):
            filename = self._gui_triggers[trigger]
            if os.path.isfile(filename):
                logger.debug("GUI Trigger received for: '%s'", trigger)

                logger.debug("Removing gui trigger file: %s", filename)
                os.remove(filename)
                if trigger == "update":
                    print("\n")  # Let log print on different line from loss output
                    logger.info("Refresh preview requested...")
                getattr(self._preview, parent_flags[trigger])()

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
                if self._args.preview:
                    self._preview.display_preview()

                if thread.has_error:
                    logger.debug("Thread error detected")
                    err = True
                    break
                if self._stop:
                    logger.debug("Stop received")
                    break

                # Preview Monitor
                if self._preview.should_quit():
                    break
                if self._preview.should_save():
                    self._save_now = True

                # Console Monitor
                if self._check_keypress(keypress):
                    break  # Exit requested

                # GUI Preview trigger update monitor
                self._process_gui_triggers()

                sleep(1)
            except KeyboardInterrupt:
                logger.debug("Keyboard Interrupt received")
                break
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
            then it will automatically be names. Default: ""
        """
        logger.debug("Updating preview: (name: %s)", name)
        try:
            scriptpath = os.path.realpath(os.path.dirname(sys.argv[0]))
            if self._args.write_image:
                logger.debug("Saving preview to disk")
                img = "training_preview.jpg"
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)  # pylint: disable=no-member
                logger.debug("Saved preview to: '%s'", img)
            if self._args.redirect_gui:
                logger.debug("Generating preview for GUI")
                img = ".gui_training_preview.jpg"
                imgfile = os.path.join(scriptpath, "lib", "gui",
                                       ".cache", "preview", img)
                cv2.imwrite(imgfile, image)  # pylint: disable=no-member
                logger.debug("Generated preview for GUI: '%s'", imgfile)
            if self._args.preview:
                logger.debug("Generating preview for display: '%s'", name)
                self._preview.add_image(name, image)
                logger.debug("Generated preview for display: '%s'", name)
        except Exception as err:
            logging.error("could not preview sample")
            raise err
        logger.debug("Updated preview: (name: %s)", name)


class Preview():
    """ Holds the pop up preview window and options relating to the preview in the window and the
    GUI. Thread safe to take requests from the main thread and the training thread. """
    def __init__(self) -> None:
        self._lock = Lock()
        self._dpi: float = 0.0
        self._triggers: Dict[str, bool] = dict(toggle_mask=False,
                                               full_size=False,
                                               refresh=False,
                                               save=False,
                                               quit=False)
        self._needs_update: bool = False
        self._preview_buffer: Dict[str, np.ndarray] = {}
        self._images: Dict[str, Tuple[figure.Figure, Tuple[float, float]]] = {}
        self._resize_ids: List[Tuple[figure.Figure, int]] = []
        self._callbacks = dict(f="full_size",
                               m="toggle_mask",
                               r="refresh",
                               s="save",
                               enter="quit")
        self._configure_matplotlib()

    def _toggle_size(self) -> None:  # pylint:disable=unused-argument
        """ Toggle between actual size and screen-fit size. """
        self._triggers["full_size"] = not self._triggers["full_size"]
        self._set_resize_callback()

    @classmethod
    def _configure_matplotlib(cls):
        """ Remove `F`, 'S' and 'R' from their default bindings and stop Matplotlib from stealing
        focus """
        rcParams["keymap.fullscreen"] = [k for k in rcParams["keymap.fullscreen"] if k != "f"]
        rcParams["keymap.save"] = [k for k in rcParams["keymap.save"] if k != "s"]
        rcParams["keymap.home"] = [k for k in rcParams["keymap.home"] if k != "r"]
        rcParams["figure.raise_window"] = False

    def should_toggle_mask(self) -> bool:
        """ Check whether the mask should be toggled and return the value. If ``True`` is returned
        then resets mask toggle back to ``False``

        Returns
        -------
        bool
            ``True`` if the mask should be toggled otherwise ``False``. """
        with self._lock:
            retval = self._triggers["toggle_mask"]
            if retval:
                logger.debug("Sending toggle mask")
                self._triggers["toggle_mask"] = False
        return retval

    def should_refresh(self) -> bool:
        """ Check whether the preview should be updated and return the value. If ``True`` is
        returned then resets the refresh trigger back to ``False``

        Returns
        -------
        bool
            ``True`` if the preview should be refreshed otherwise ``False``. """
        with self._lock:
            retval = self._triggers["refresh"]
            if retval:
                logger.debug("Sending should refresh")
                self._triggers["refresh"] = False
            return retval

    def should_save(self) -> bool:
        """ Check whether a save request has been made. If ``True`` is returned then :attr:`_save`
        is set back to ``False``

        Returns
        -------
        bool
            ``True`` if a save has been requested otherwise ``False``. """
        with self._lock:
            retval = self._triggers["save"]
            if retval:
                logger.debug("Sending should save")
                self._triggers["save"] = False
        return retval

    def should_quit(self) -> bool:
        """ Check whether an exit request has been made.

        Returns
        -------
        bool
            ``True`` if an exit request has been made otherwise ``False``. """
        with self._lock:
            retval = self._triggers["quit"]
        if retval:
            logger.debug("Sending should stop")
        return retval

    def request_refresh(self) -> None:
        """ Handle a GUI trigger or a training thread trigger (after a mask toggle) request to set
        the refresh trigger to ``True`` to request a refresh on the next pass of the
        training loop. """
        with self._lock:
            self._triggers["refresh"] = True

    def request_mask_toggle(self) -> None:
        """ Handle a GUI trigger request to set the mask toggle to ``True`` to
        request a mask toggle on next pass of the training loop. """
        logger.verbose("Toggle mask display requested...")  # type:ignore
        with self._lock:
            self._triggers["toggle_mask"] = True

    def add_image(self, name: str, image: np.ndarray) -> None:
        """ Add a preview image to the preview buffer.

        Parameters
        ----------
        name: str
            The name of the preview image to add to the buffer
        image: :class:`numpy.ndarray`
            The preview image to add to the buffer in BGR format.
        """
        with self._lock:
            logger.debug("Adding image '%s' of shape %s to preview buffer", name, image.shape)
            self._preview_buffer[name] = image[..., 2::-1]  # Switch to RGB
            self._needs_update = True

    def display_preview(self) -> None:
        """ Display an image preview in a resizable window. """
        if self._needs_update:
            logger.debug("Updating preview")
            with self._lock:
                for name, image in self._preview_buffer.items():
                    if (name not in self._images or  # new preview or preview was closed
                            not plt.fignum_exists(self._images[name][0].number)):
                        self._create_resizable_window(name, image.shape)
                        if self._triggers["full_size"]:  # Can only be true if preview was closed
                            self._set_resize_callback()
                    plt.figure(name)
                    plt.imshow(image)
                self._needs_update = False
            plt.show(block=False)
            logger.debug("preview updated")  # type: ignore
        plt.pause(0.1)

    def _create_resizable_window(self, name: str, image_shape: tuple) -> None:
        """ Create a resizable Matplotlib window to hold the preview image.

        Parameters
        ----------
        name: str
            The name to display in the window header and for window identification
        shape: tuple
            The (`rows`, `columns`, `channels`) of the image to be displayed
        """
        logger.debug("Creating figure '%s' for image shape %s", name, image_shape)
        if not self._dpi:
            self._dpi = get_dpi()
        height, width = image_shape[:2]
        size = width / self._dpi, height / self._dpi
        fig = plt.figure(name, figsize=size)
        axes = plt.Axes(fig, [0., 0., 1., 1.])  # Remove axes and whitespace
        axes.set_axis_off()
        fig.add_axes(axes)
        fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        fig.canvas.mpl_connect("close_event", self._on_close)
        logger.debug("Created display figure of size: %s", size)
        self._images[name] = (fig, size)

    def _set_resize_callback(self):
        """ Sets the resize callback if displaying preview at actual size or removes it if
        displaying at screen-fit size. """
        if self._triggers["full_size"]:
            logger.debug("Setting resize callback for actual size display")
            for fig, size in self._images.values():
                self._resize_ids.append((fig, fig.canvas.mpl_connect("resize_event",
                                                                     self._on_resize)))
                fig.set_size_inches(size)
        else:
            logger.debug("Removing resize callback for screen-fit display")
            for fig, cid in self._resize_ids:
                fig.canvas.mpl_disconnect(cid)
            self._resize_ids = []

    def _on_key_press(self, event: backend_bases.KeyEvent) -> None:
        """ Callbacks for keypresses to update the requested trigger.

        - `F` (toggle full-size/fit to window)
        - `M` (toggle mask),
        - `R` (refresh preview),
        - `S` (save now)
        - `Enter` (save and exit)

        Parameters
        ----------
        event:
            The key press received
        """
        key = event.key.lower()
        if key not in self._callbacks:
            return

        logger.debug("Preview window keypress '%s' received", key)
        if key == "r":
            print("\n")  # Let log print on different line from loss output
            logger.info("Refresh preview requested...")

        with self._lock:
            if key == "f":
                self._toggle_size()
            else:
                self._triggers[self._callbacks[key]] = True

    def _on_resize(self,
                   event: backend_bases.ResizeEvent) -> None:  # noqa # pylint:disable=unused-argument
        """ If the display is set to `actual size` then the image needs to be resized on any window
        resize event. """
        for fig, size in self._images.values():
            fig.set_size_inches(size)

    def _on_close(self,
                   event: backend_bases.CloseEvent) -> None:  # noqa # pylint:disable=unused-argument
        """ Force an update when the figure has been closed to relaunch it. """
        logger.debug("Preview close detected")
        with self._lock:
            self._needs_update = True
