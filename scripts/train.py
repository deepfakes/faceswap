#!/usr/bin python3
""" The script to run the training process of faceswap """

import logging
import os
import sys

from threading import Lock
from time import sleep

import cv2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from lib.keypress import KBHit
from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager
from lib.utils import (get_folder, get_image_paths, set_system_verbosity)
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Train():
    """ The training process.  """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self.args = arguments
        self.timelapse = self.set_timelapse()
        self.images = self.get_images()
        self.stop = False
        self.save_now = False
        self.preview_buffer = dict()
        self.lock = Lock()

        self.trainer_name = self.args.trainer
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_timelapse(self):
        """ Set timelapse paths if requested """
        if (not self.args.timelapse_input_a and
                not self.args.timelapse_input_b and
                not self.args.timelapse_output):
            return None
        if not self.args.timelapse_input_a or not self.args.timelapse_input_b:
            raise ValueError("To enable the timelapse, you have to supply "
                             "all the parameters (--timelapse-input-A and "
                             "--timelapse-input-B).")

        timelapse_output = None
        if self.args.timelapse_output is not None:
            timelapse_output = str(get_folder(self.args.timelapse_output))

        for folder in (self.args.timelapse_input_a,
                       self.args.timelapse_input_b,
                       timelapse_output):
            if folder is not None and not os.path.isdir(folder):
                raise ValueError("The Timelapse path '{}' does not exist".format(folder))

        kwargs = {"input_a": self.args.timelapse_input_a,
                  "input_b": self.args.timelapse_input_b,
                  "output": timelapse_output}
        logger.debug("Timelapse enabled: %s", kwargs)
        return kwargs

    def get_images(self):
        """ Check the image dirs exist, contain images and return the image
        objects """
        logger.debug("Getting image paths")
        images = dict()
        for side in ("a", "b"):
            image_dir = getattr(self.args, "input_{}".format(side))
            if not os.path.isdir(image_dir):
                logger.error("Error: '%s' does not exist", image_dir)
                exit(1)

            if not os.listdir(image_dir):
                logger.error("Error: '%s' contains no images", image_dir)
                exit(1)

            images[side] = get_image_paths(image_dir)
        logger.info("Model A Directory: %s", self.args.input_a)
        logger.info("Model B Directory: %s", self.args.input_b)
        logger.debug("Got image paths: %s", [(key, str(len(val)) + " images")
                                             for key, val in images.items()])
        return images

    def process(self):
        """ Call the training process object """
        logger.debug("Starting Training Process")
        logger.info("Training data directory: %s", self.args.model_dir)
        set_system_verbosity(self.args.loglevel)
        thread = self.start_thread()
        # queue_manager.debug_monitor(1)

        err = self.monitor(thread)

        self.end_thread(thread, err)
        logger.debug("Completed Training Process")

    def start_thread(self):
        """ Put the training process in a thread so we can keep control """
        logger.debug("Launching Trainer thread")
        thread = MultiThread(target=self.training)
        thread.start()
        logger.debug("Launched Trainer thread")
        return thread

    def end_thread(self, thread, err):
        """ On termination output message and join thread back to main """
        logger.debug("Ending Training thread")
        if err:
            msg = "Error caught! Exiting..."
            log = logger.critical
        else:
            msg = ("Exit requested! The trainer will complete its current cycle, "
                   "save the models and quit (it can take up a couple of seconds "
                   "depending on your training speed). If you want to kill it now, "
                   "press Ctrl + c")
            log = logger.info
        log(msg)
        self.stop = True
        thread.join()
        sys.stdout.flush()
        logger.debug("Ended Training thread")

    def training(self):
        """ The training process to be run inside a thread """
        try:
            sleep(1)  # Let preview instructions flush out to logger
            logger.debug("Commencing Training")
            logger.info("Loading data, this may take a while...")

            if self.args.allow_growth:
                self.set_tf_allow_growth()

            model = self.load_model()
            trainer = self.load_trainer(model)
            self.run_training_cycle(model, trainer)
        except KeyboardInterrupt:
            try:
                logger.debug("Keyboard Interrupt Caught. Saving Weights and exiting")
                model.save_models()
                trainer.clear_tensorboard()
            except KeyboardInterrupt:
                logger.info("Saving model weights has been cancelled!")
            exit(0)
        except Exception as err:
            raise err

    def load_model(self):
        """ Load the model requested for training """
        logger.debug("Loading Model")
        model_dir = get_folder(self.args.model_dir)
        model = PluginLoader.get_model(self.trainer_name)(
            model_dir,
            self.args.gpus,
            no_logs=self.args.no_logs,
            warp_to_landmarks=self.args.warp_to_landmarks,
            no_flip=self.args.no_flip,
            training_image_size=self.image_size,
            alignments_paths=self.alignments_paths,
            preview_scale=self.args.preview_scale,
            pingpong=self.args.pingpong,
            memory_saving_gradients=self.args.memory_saving_gradients,
            predict=False)
        logger.debug("Loaded Model")
        return model

    @property
    def image_size(self):
        """ Get the training set image size for storing in model data """
        image = cv2.imread(self.images["a"][0])  # pylint: disable=no-member
        size = image.shape[0]
        logger.debug("Training image size: %s", size)
        return size

    @property
    def alignments_paths(self):
        """ Set the alignments path to input dirs if not provided """
        alignments_paths = dict()
        for side in ("a", "b"):
            alignments_path = getattr(self.args, "alignments_path_{}".format(side))
            if not alignments_path:
                image_path = getattr(self.args, "input_{}".format(side))
                alignments_path = os.path.join(image_path, "alignments.json")
            alignments_paths[side] = alignments_path
        logger.debug("Alignments paths: %s", alignments_paths)
        return alignments_paths

    def load_trainer(self, model):
        """ Load the trainer requested for training """
        logger.debug("Loading Trainer")
        trainer = PluginLoader.get_trainer(model.trainer)
        trainer = trainer(model,
                          self.images,
                          self.args.batch_size)
        logger.debug("Loaded Trainer")
        return trainer

    def run_training_cycle(self, model, trainer):
        """ Perform the training cycle """
        logger.debug("Running Training Cycle")
        if self.args.write_image or self.args.redirect_gui or self.args.preview:
            display_func = self.show
        else:
            display_func = None

        for iteration in range(0, self.args.iterations):
            logger.trace("Training iteration: %s", iteration)
            save_iteration = iteration % self.args.save_interval == 0
            viewer = display_func if save_iteration or self.save_now else None
            timelapse = self.timelapse if save_iteration else None
            trainer.train_one_step(viewer, timelapse)
            if self.stop:
                logger.debug("Stop received. Terminating")
                break
            elif save_iteration:
                logger.trace("Save Iteration: (iteration: %s", iteration)
                if self.args.pingpong:
                    model.save_models()
                    trainer.pingpong.switch()
                else:
                    model.save_models()
            elif self.save_now:
                logger.trace("Save Requested: (iteration: %s", iteration)
                model.save_models()
                self.save_now = False
        logger.debug("Training cycle complete")
        model.save_models()
        trainer.clear_tensorboard()
        self.stop = True

    def monitor(self, thread):
        """ Monitor the console, and generate + monitor preview if requested """
        is_preview = self.args.preview
        logger.debug("Launching Monitor")
        logger.info("R|===============================================")
        logger.info("R|- Starting                                    -")
        if is_preview:
            logger.info("R|- Using live preview                          -")
        logger.info("R|- Press 'ENTER' to save and quit              -")
        logger.info("R|- Press 'S' to save model weights immediately -")
        logger.info("R|===============================================")

        keypress = KBHit(is_gui=self.args.redirect_gui)
        err = False
        while True:
            try:
                if is_preview:
                    with self.lock:
                        for name, image in self.preview_buffer.items():
                            cv2.imshow(name, image)  # pylint: disable=no-member
                    cv_key = cv2.waitKey(1000)  # pylint: disable=no-member
                else:
                    cv_key = None

                if thread.has_error:
                    logger.debug("Thread error detected")
                    err = True
                    break
                if self.stop:
                    logger.debug("Stop received")
                    break

                # Preview Monitor
                if is_preview and (cv_key == ord("\n") or cv_key == ord("\r")):
                    logger.debug("Exit requested")
                    break
                if is_preview and cv_key == ord("s"):
                    logger.info("Save requested")
                    self.save_now = True

                # Console Monitor
                if keypress.kbhit():
                    console_key = keypress.getch()
                    if console_key in ("\n", "\r"):
                        logger.debug("Exit requested")
                        break
                    if console_key in ("s", "S"):
                        logger.info("Save requested")
                        self.save_now = True

                sleep(1)
            except KeyboardInterrupt:
                logger.debug("Keyboard Interrupt received")
                break
        keypress.set_normal_term()
        logger.debug("Closed Monitor")
        return err

    @staticmethod
    def keypress_monitor(keypress_queue):
        """ Monitor stdin for keypress """
        while True:
            keypress_queue.put(sys.stdin.read(1))

    @staticmethod
    def set_tf_allow_growth():
        """ Allow TensorFlow to manage VRAM growth """
        # pylint: disable=no-member
        logger.debug("Setting Tensorflow 'allow_growth' option")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))
        logger.debug("Set Tensorflow 'allow_growth' option")

    def show(self, image, name=""):
        """ Generate the preview and write preview file output """
        logger.trace("Updating preview: (name: %s)", name)
        try:
            scriptpath = os.path.realpath(os.path.dirname(sys.argv[0]))
            if self.args.write_image:
                logger.trace("Saving preview to disk")
                img = "training_preview.jpg"
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)  # pylint: disable=no-member
                logger.trace("Saved preview to: '%s'", img)
            if self.args.redirect_gui:
                logger.trace("Generating preview for GUI")
                img = ".gui_training_preview.jpg"
                imgfile = os.path.join(scriptpath, "lib", "gui",
                                       ".cache", "preview", img)
                cv2.imwrite(imgfile, image)  # pylint: disable=no-member
                logger.trace("Generated preview for GUI: '%s'", img)
            if self.args.preview:
                logger.trace("Generating preview for display: '%s'", name)
                with self.lock:
                    self.preview_buffer[name] = image
                logger.trace("Generated preview for display: '%s'", name)
        except Exception as err:
            logging.error("could not preview sample")
            raise err
        logger.trace("Updated preview: (name: %s)", name)
