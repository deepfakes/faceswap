#!/usr/bin python3
""" Main entry point to the training process of FaceSwap """

import logging
import os
import sys

from threading import Lock
from time import sleep

import cv2

from lib.image import read_image_meta
from lib.keypress import KBHit
from lib.multithreading import MultiThread
from lib.utils import (get_folder, get_image_paths, FaceswapError, _image_extensions)
from plugins.plugin_loader import PluginLoader

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
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self._args = arguments
        if self._args.summary:
            # If just outputting summary we don't need to initialize everything
            return

        self._images = self._get_images()
        self._timelapse = self._set_timelapse()
        gui_cache = os.path.join(
            os.path.realpath(os.path.dirname(sys.argv[0])), "lib", "gui", ".cache")
        self._gui_triggers = dict(update=os.path.join(gui_cache, ".preview_trigger"),
                                  mask_toggle=os.path.join(gui_cache, ".preview_mask_toggle"))
        self._stop = False
        self._save_now = False
        self._toggle_preview_mask = False
        self._refresh_preview = False
        self._preview_buffer = dict()
        self._lock = Lock()

        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_images(self):
        """ Check the image folders exist and contains valid extracted faces. Obtain image paths.

        Returns
        -------
        dict
            The image paths for each side. The key is the side, the value is the list of paths
            for that side.
        """
        logger.debug("Getting image paths")
        images = dict()
        for side in ("a", "b"):
            image_dir = getattr(self._args, "input_{}".format(side))
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
    def _validate_image_counts(cls, images):
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

    def _set_timelapse(self):
        """ Set time-lapse paths if requested.

        Returns
        -------
        dict
            The time-lapse keyword arguments for passing to the trainer

        """
        if (not self._args.timelapse_input_a and
                not self._args.timelapse_input_b and
                not self._args.timelapse_output):
            return None
        if (not self._args.timelapse_input_a or
                not self._args.timelapse_input_b or
                not self._args.timelapse_output):
            raise FaceswapError("To enable the timelapse, you have to supply all the parameters "
                                "(--timelapse-input-A, --timelapse-input-B and "
                                "--timelapse-output).")

        timelapse_output = str(get_folder(self._args.timelapse_output))

        for side in ("a", "b"):
            folder = getattr(self._args, "timelapse_input_{}".format(side))
            if folder is not None and not os.path.isdir(folder):
                raise FaceswapError("The Timelapse path '{}' does not exist".format(folder))

            training_folder = getattr(self._args, "input_{}".format(side))
            if folder == training_folder:
                continue  # Time-lapse folder is training folder

            filenames = [fname for fname in os.listdir(folder)
                         if os.path.splitext(fname)[-1].lower() in _image_extensions]
            if not filenames:
                raise FaceswapError("The Timelapse path '{}' does not contain any valid "
                                    "images".format(folder))

            # Time-lapse images must appear in the training set, as we need access to alignment and
            # mask info. Check filenames are there to save failing much later in the process.
            training_images = [os.path.basename(img) for img in self._images[side]]
            if not all(img in training_images for img in filenames):
                raise FaceswapError("All images in the Timelapse folder '{}' must exist in the "
                                    "training folder '{}'".format(folder, training_folder))

        kwargs = {"input_a": self._args.timelapse_input_a,
                  "input_b": self._args.timelapse_input_b,
                  "output": timelapse_output}
        logger.debug("Timelapse enabled: %s", kwargs)
        return kwargs

    def process(self):
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

    def _start_thread(self):
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

    def _end_thread(self, thread, err):
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

    def _training(self):
        """ The training process to be run inside a thread. """
        try:
            sleep(1)  # Let preview instructions flush out to logger
            logger.debug("Commencing Training")
            logger.info("Loading data, this may take a while...")
            model = self._load_model()
            trainer = self._load_trainer(model)
            self._run_training_cycle(model, trainer)
        except KeyboardInterrupt:
            try:
                logger.debug("Keyboard Interrupt Caught. Saving Weights and exiting")
                model.save()
                trainer.clear_tensorboard()
            except KeyboardInterrupt:
                logger.info("Saving model weights has been cancelled!")
            sys.exit(0)
        except Exception as err:
            raise err

    def _load_model(self):
        """ Load the model requested for training.

        Returns
        -------
        :file:`plugins.train.model` plugin
            The requested model plugin
        """
        logger.debug("Loading Model")
        model_dir = str(get_folder(self._args.model_dir))
        model = PluginLoader.get_model(self._args.trainer)(
            model_dir,
            self._args,
            predict=False)
        model.build()
        logger.debug("Loaded Model")
        return model

    def _load_trainer(self, model):
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
        trainer = PluginLoader.get_trainer(model.trainer)
        trainer = trainer(model,
                          self._images,
                          self._args.batch_size,
                          self._args.configfile)
        logger.debug("Loaded Trainer")
        return trainer

    def _run_training_cycle(self, model, trainer):
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
            display_func = self._show
        else:
            display_func = None

        for iteration in range(1, self._args.iterations + 1):
            logger.trace("Training iteration: %s", iteration)
            save_iteration = iteration % self._args.save_interval == 0 or iteration == 1

            if self._toggle_preview_mask:
                trainer.toggle_mask()
                self._toggle_preview_mask = False
                self._refresh_preview = True

            if save_iteration or self._save_now or self._refresh_preview:
                viewer = display_func
            else:
                viewer = None
            timelapse = self._timelapse if save_iteration else None
            trainer.train_one_step(viewer, timelapse)
            if self._stop:
                logger.debug("Stop received. Terminating")
                break

            if self._refresh_preview and viewer is not None:
                if self._args.redirect_gui:  # Remove any gui trigger files following an update
                    print("\n")
                    logger.info("[Preview Updated]")
                self._refresh_preview = False

            if save_iteration:
                logger.debug("Save Iteration: (iteration: %s", iteration)
                model.save()
            elif self._save_now:
                logger.debug("Save Requested: (iteration: %s", iteration)
                model.save()
                self._save_now = False
        logger.debug("Training cycle complete")
        model.save()
        trainer.clear_tensorboard()
        self._stop = True

    def _monitor(self, thread):
        """ Monitor the background :func:`_training` thread for key presses and errors.

        Returns
        -------
        bool
            ``True`` if there has been an error in the background thread otherwise ``False``
        """
        logger.debug("Launching Monitor")
        logger.info("===================================================")
        logger.info("  Starting")
        if self._args.preview:
            logger.info("  Using live preview")
        logger.info("  Press '%s' to save and quit",
                    "Stop" if self._args.redirect_gui or self._args.colab else "ENTER")
        if not self._args.redirect_gui and not self._args.colab:
            logger.info("  Press 'S' to save model weights immediately")
        logger.info("===================================================")

        keypress = KBHit(is_gui=self._args.redirect_gui)
        err = False
        while True:
            try:
                if self._args.preview:
                    with self._lock:
                        for name, image in self._preview_buffer.items():
                            cv2.imshow(name, image)  # pylint: disable=no-member
                    cv_key = cv2.waitKey(1000)  # pylint: disable=no-member
                else:
                    cv_key = None

                if thread.has_error:
                    logger.debug("Thread error detected")
                    err = True
                    break
                if self._stop:
                    logger.debug("Stop received")
                    break

                # Preview Monitor
                if not self._preview_monitor(cv_key):
                    break

                # Console Monitor
                if keypress.kbhit():
                    console_key = keypress.getch()
                    if console_key in ("\n", "\r"):
                        logger.debug("Exit requested")
                        break
                    if console_key in ("s", "S"):
                        logger.info("Save requested")
                        self._save_now = True

                # GUI Preview trigger update monitor
                self._process_gui_triggers()

                sleep(1)
            except KeyboardInterrupt:
                logger.debug("Keyboard Interrupt received")
                break
        keypress.set_normal_term()
        logger.debug("Closed Monitor")
        return err

    def _preview_monitor(self, key_press):
        """ Monitors keyboard presses on the pop-up OpenCV Preview Window.

        Parameters
        ----------
        key_press: str
            The key press received from OpenCV or ``None`` if no press received

        Returns
        -------
        bool
            ``True`` if the process should continue training. ``False`` if an exit has been
            requested and process should terminate
        """
        if not self._args.preview:
            return True

        if key_press == ord("\n") or key_press == ord("\r"):
            logger.debug("Exit requested")
            return False

        if key_press == ord("s"):
            print("\n")
            logger.info("Save requested")
            self._save_now = True
        if key_press == ord("r"):
            print("\n")
            logger.info("Refresh preview requested")
            self._refresh_preview = True
        if key_press == ord("m"):
            print("\n")
            logger.verbose("Toggle mask display requested")
            self._toggle_preview_mask = True

        return True

    def _process_gui_triggers(self):
        """ Check whether a file drop has occurred from the GUI to manually update the preview. """
        if not self._args.redirect_gui:
            return

        parent_flags = dict(mask_toggle="_toggle_preview_mask", update="_refresh_preview")
        for trigger in ("mask_toggle", "update"):
            filename = self._gui_triggers[trigger]
            if os.path.isfile(filename):
                logger.debug("GUI Trigger received for: '%s'", trigger)

                logger.debug("Removing gui trigger file: %s", filename)
                os.remove(filename)

                if trigger == "update":
                    print("\n")
                    logger.info("Refresh preview requested")

                setattr(self, parent_flags[trigger], True)

    def _show(self, image, name=""):
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
                logger.debug("Generated preview for GUI: '%s'", img)
            if self._args.preview:
                logger.debug("Generating preview for display: '%s'", name)
                with self._lock:
                    self._preview_buffer[name] = image
                logger.debug("Generated preview for display: '%s'", name)
        except Exception as err:
            logging.error("could not preview sample")
            raise err
        logger.debug("Updated preview: (name: %s)", name)
