#!/usr/bin python3
""" Main entry point to the training process of FaceSwap """

import logging
import os
import sys

from threading import Lock
from time import sleep

import cv2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from lib.image import read_image
from lib.keypress import KBHit
from lib.multithreading import MultiThread
from lib.utils import (get_folder, get_image_paths, deprecation_warning, FaceswapError,
                       _image_extensions)
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self._args = arguments
        self._timelapse = self._set_timelapse()
        self._images = self._get_images()
        self._stop = False
        self._save_now = False
        self._preview_buffer = dict()
        self._lock = Lock()

        self.trainer_name = self._args.trainer
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _image_size(self):
        """ int: The training image size. Reads the first image in the training folder and returns
        the size. """
        image = read_image(self._images["a"][0], raise_error=True)
        size = image.shape[0]
        logger.debug("Training image size: %s", size)
        return size

    @property
    def _alignments_paths(self):
        """ dict: The alignments paths for each of the source and destination faces. Key is the
            side, value is the path to the alignments file """
        alignments_paths = dict()
        for side in ("a", "b"):
            alignments_path = getattr(self._args, "alignments_path_{}".format(side))
            if not alignments_path:
                image_path = getattr(self._args, "input_{}".format(side))
                alignments_path = os.path.join(image_path, "alignments.fsa")
            alignments_paths[side] = alignments_path
        logger.debug("Alignments paths: %s", alignments_paths)
        return alignments_paths

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

        for folder in (self._args.timelapse_input_a, self._args.timelapse_input_b):
            if folder is not None and not os.path.isdir(folder):
                raise FaceswapError("The Timelapse path '{}' does not exist".format(folder))
            exts = [os.path.splitext(fname)[-1] for fname in os.listdir(folder)]
            if not any(ext in _image_extensions for ext in exts):
                raise FaceswapError("The Timelapse path '{}' does not contain any valid "
                                    "images".format(folder))
        kwargs = {"input_a": self._args.timelapse_input_a,
                  "input_b": self._args.timelapse_input_b,
                  "output": timelapse_output}
        logger.debug("Timelapse enabled: %s", kwargs)
        return kwargs

    def _get_images(self):
        """ Check the image folders exist and contains images and obtain image paths.

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

            images[side] = get_image_paths(image_dir)
            if not images[side]:
                logger.error("Error: '%s' contains no images", image_dir)
                sys.exit(1)

        logger.info("Model A Directory: %s", self._args.input_a)
        logger.info("Model B Directory: %s", self._args.input_b)
        logger.debug("Got image paths: %s", [(key, str(len(val)) + " images")
                                             for key, val in images.items()])
        return images

    def process(self):
        """ The entry point for triggering the Training Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        logger.debug("Starting Training Process")
        logger.info("Training data directory: %s", self._args.model_dir)

        # TODO Move these args to config and remove these deprecation warnings
        if hasattr(self._args, "warp_to_landmarks") and self._args.warp_to_landmarks:
            deprecation_warning("`-wl`, ``--warp-to-landmarks``",
                                additional_info="This option will be available within training "
                                                "config settings (/config/train.ini).")
        if hasattr(self._args, "no_augment_color") and self._args.no_augment_color:
            deprecation_warning("`-nac`, ``--no-augment-color``",
                                additional_info="This option will be available within training "
                                                "config settings (/config/train.ini).")
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

            if self._args.allow_growth:
                self._set_tf_allow_growth()
            model = self._load_model()
            trainer = self._load_trainer(model)
            self._run_training_cycle(model, trainer)
        except KeyboardInterrupt:
            try:
                logger.debug("Keyboard Interrupt Caught. Saving Weights and exiting")
                model.save_models()
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
        model_dir = get_folder(self._args.model_dir)
        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        augment_color = not self._args.no_augment_color
        model = PluginLoader.get_model(self.trainer_name)(
            model_dir,
            gpus=self._args.gpus,
            configfile=configfile,
            snapshot_interval=self._args.snapshot_interval,
            no_logs=self._args.no_logs,
            warp_to_landmarks=self._args.warp_to_landmarks,
            augment_color=augment_color,
            no_flip=self._args.no_flip,
            training_image_size=self._image_size,
            alignments_paths=self._alignments_paths,
            preview_scale=self._args.preview_scale,
            pingpong=self._args.pingpong,
            memory_saving_gradients=self._args.memory_saving_gradients,
            optimizer_savings=self._args.optimizer_savings,
            predict=False)
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

        for iteration in range(0, self._args.iterations):
            logger.trace("Training iteration: %s", iteration)
            save_iteration = iteration % self._args.save_interval == 0
            viewer = display_func if save_iteration or self._save_now else None
            timelapse = self._timelapse if save_iteration else None
            trainer.train_one_step(viewer, timelapse)
            if self._stop:
                logger.debug("Stop received. Terminating")
                break
            if save_iteration:
                logger.trace("Save Iteration: (iteration: %s", iteration)
                if self._args.pingpong:
                    model.save_models()
                    trainer.pingpong.switch()
                else:
                    model.save_models()
            elif self._save_now:
                logger.trace("Save Requested: (iteration: %s", iteration)
                model.save_models()
                self._save_now = False
        logger.debug("Training cycle complete")
        model.save_models()
        trainer.clear_tensorboard()
        self._stop = True

    def _monitor(self, thread):
        """ Monitor the background :func:`_training` thread for key presses and errors.

        Returns
        -------
        bool
            ``True`` if there has been an error in the background thread otherwise ``False``
        """
        is_preview = self._args.preview
        logger.debug("Launching Monitor")
        logger.info("===================================================")
        logger.info("  Starting")
        if is_preview:
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
                if is_preview:
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
                if is_preview and (cv_key == ord("\n") or cv_key == ord("\r")):
                    logger.debug("Exit requested")
                    break
                if is_preview and cv_key == ord("s"):
                    logger.info("Save requested")
                    self._save_now = True

                # Console Monitor
                if keypress.kbhit():
                    console_key = keypress.getch()
                    if console_key in ("\n", "\r"):
                        logger.debug("Exit requested")
                        break
                    if console_key in ("s", "S"):
                        logger.info("Save requested")
                        self._save_now = True

                sleep(1)
            except KeyboardInterrupt:
                logger.debug("Keyboard Interrupt received")
                break
        keypress.set_normal_term()
        logger.debug("Closed Monitor")
        return err

    @staticmethod
    def _set_tf_allow_growth():
        """ Allow TensorFlow to manage VRAM growth.

        Enables the Tensorflow allow_growth option if requested in the command line arguments
        """
        # pylint: disable=no-member
        logger.debug("Setting Tensorflow 'allow_growth' option")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))
        logger.debug("Set Tensorflow 'allow_growth' option")

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
        logger.trace("Updating preview: (name: %s)", name)
        try:
            scriptpath = os.path.realpath(os.path.dirname(sys.argv[0]))
            if self._args.write_image:
                logger.trace("Saving preview to disk")
                img = "training_preview.jpg"
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)  # pylint: disable=no-member
                logger.trace("Saved preview to: '%s'", img)
            if self._args.redirect_gui:
                logger.trace("Generating preview for GUI")
                img = ".gui_training_preview.jpg"
                imgfile = os.path.join(scriptpath, "lib", "gui",
                                       ".cache", "preview", img)
                cv2.imwrite(imgfile, image)  # pylint: disable=no-member
                logger.trace("Generated preview for GUI: '%s'", img)
            if self._args.preview:
                logger.trace("Generating preview for display: '%s'", name)
                with self._lock:
                    self._preview_buffer[name] = image
                logger.trace("Generated preview for display: '%s'", name)
        except Exception as err:
            logging.error("could not preview sample")
            raise err
        logger.trace("Updated preview: (name: %s)", name)
