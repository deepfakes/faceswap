#!/usr/bin python3
""" The script to run the training process of faceswap """

import logging
import os
import sys

from threading import Lock

import cv2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from lib.multithreading import MultiThread
from lib.utils import (get_folder, get_image_paths, set_system_verbosity,
                       Timelapse)
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Train():
    """ The training process.  """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s", self.__class__.__name__, arguments)
        self.args = arguments
        self.images = self.get_images()
        self.stop = False
        self.save_now = False
        self.preview_buffer = dict()
        self.lock = Lock()

        # this is so that you can enter case insensitive values for trainer
        trainer_name = self.args.trainer
        self.trainer_name = trainer_name
        self.timelapse = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Call the training process object """
        logger.debug("Starting Training Process")
        logger.info("Training data directory: %s", self.args.model_dir)
        set_system_verbosity()
        thread = self.start_thread()

        if self.args.preview:
            self.monitor_preview()
        else:
            self.monitor_console()

        self.end_thread(thread)
        logger.debug("Completed Training Process")

    def get_images(self):
        """ Check the image dirs exist, contain images and return the image
        objects """
        logger.debug("Getting image paths")
        images = dict()
        for image_group in ("a", "b"):
            image_dir = getattr(self.args, "input_{}".format(image_group))
            if not os.path.isdir(image_dir):
                logger.error("Error: '%s' does not exist", image_dir)
                exit(1)

            if not os.listdir(image_dir):
                logger.error("Error: '%s' contains no images", image_dir)
                exit(1)

            images[image_group] = get_image_paths(image_dir)
        logger.info("Model A Directory: %s", self.args.input_a)
        logger.info("Model B Directory: %s", self.args.input_b)
        logger.debug("Got image paths: %s", [(key, str(len(val)) + " images")
                                             for key, val in images.items()])
        return images

    def start_thread(self):
        """ Put the training process in a thread so we can keep control """
        logger.debug("Launching Trainer thread")
        thread = MultiThread(target=self.training)
        thread.start()
        logger.debug("Launched Trainer thread")
        return thread

    def end_thread(self, thread):
        """ On termination output message and join thread back to main """
        logger.debug("Ending Training thread")
        logger.info("Exit requested! The trainer will complete its current cycle, "
                    "save the models and quit (it can take up a couple of seconds "
                    "depending on your training speed). If you want to kill it now, "
                    "press Ctrl + c")
        self.stop = True
        thread.join()
        sys.stdout.flush()
        logger.debug("Ended Training thread")

    def training(self):
        """ The training process to be run inside a thread """
        try:
            logger.debug("Commencing Training")
            logger.info("Loading data, this may take a while...")

            if self.args.allow_growth:
                self.set_tf_allow_growth()

            model = self.load_model()
            trainer = self.load_trainer(model)

            # TODO Move timelapse out of utils
            self.timelapse = Timelapse.create_timelapse(
                self.args.timelapse_input_a,
                self.args.timelapse_input_b,
                self.args.timelapse_output,
                trainer)

            self.run_training_cycle(model, trainer)
        except KeyboardInterrupt:
            try:
                logger.debug("Keyboard Interrupt Caught. Saving Weights and exiting")
                model.save_weights()
            except KeyboardInterrupt:
                logger.info("Saving model weights has been cancelled!")
            exit(0)
        except Exception as err:
            raise err

    def load_model(self):
        """ Load the model requested for training """
        logger.debug("Loading Model")
        model_dir = get_folder(self.args.model_dir)
        model = PluginLoader.get_model(self.trainer_name)(model_dir,
                                                          self.args.gpus)

        model.load_weights(swapped=False)
        logger.debug("Loaded Model")
        return model

    def load_trainer(self, model):
        """ Load the trainer requested for training """
        logger.debug("Loading Trainer")
        trainer = PluginLoader.get_trainer(model.trainer)
        trainer = trainer(model,
                          self.images,
                          self.args.batch_size,
                          self.args.perceptual_loss)
        logger.debug("Loaded Trainer")
        return trainer

    def run_training_cycle(self, model, trainer):
        """ Perform the training cycle """
        logger.debug("Runnng Training Cycle")
        for iteration in range(0, self.args.iterations):
            logger.trace("Training iteration: %s", iteration)
            save_iteration = iteration % self.args.save_interval == 0
            viewer = self.show if save_iteration or self.save_now else None
            if save_iteration and self.timelapse is not None:
                logger.trace("Updating Timelapse: (iteration: %s", iteration)
                self.timelapse.work()
            trainer.train_one_step(iteration, viewer)
            if self.stop:
                logger.debug("Stop received. Terminating")
                break
            elif save_iteration:
                logger.trace("Save Iteration: (iteration: %s", iteration)
                model.save_weights()
            elif self.save_now:
                logger.trace("Save Requested: (iteration: %s", iteration)
                model.save_weights()
                self.save_now = False
        logger.debug("Training cycle complete")
        model.save_weights()
        self.stop = True

    def monitor_preview(self):
        """ Generate the preview window and wait for keyboard input """
        logger.debug("Launching Preview Monitor")
        logger.info("==================================================================")
        logger.info(" Using live preview")
        logger.info(" Press 'ENTER' on the preview window to save and quit")
        logger.info(" Press 'S' on the preview window to save model weights immediately")
        logger.info("==================================================================")
        while True:
            try:
                with self.lock:
                    for name, image in self.preview_buffer.items():
                        cv2.imshow(name, image)  # pylint: disable=no-member

                key = cv2.waitKey(1000)  # pylint: disable=no-member
                if key == ord("\n") or key == ord("\r"):
                    logger.debug("Exit requested")
                    break
                if key == ord("s"):
                    logger.debug("Save requested")
                    self.save_now = True
                if self.stop:
                    logger.debug("Stop received")
                    break
            except KeyboardInterrupt:
                break
        logger.debug("Closed Preview Monitor")

    @staticmethod
    def monitor_console():
        """ Monitor the console for any input followed by enter or ctrl+c """
        # TODO: how to catch a specific key instead of Enter?
        # there isn't a good multiplatform solution:
        # https://stackoverflow.com/questions/3523174
        # TODO: Find a way to interrupt input() if the target iterations are
        # reached. At the moment, setting a target iteration and using the -p
        # flag is the only guaranteed way to exit the training loop on
        # hitting target iterations.
        logger.debug("Launching Console Monitor")
        logger.info("==============================================")
        logger.info(" Starting")
        logger.info(" Press 'ENTER' to stop training and save model")
        logger.info("==============================================")
        try:
            input()
            logger.debug("Input received")
        except KeyboardInterrupt:
            logger.debug("Keyboard Interrupt received")
        logger.debug("Closed Console Monitor")

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
                img = "_sample_{}.jpg".format(name)
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)  # pylint: disable=no-member
                logger.trace("Saved preview to: '%s'", img)
            if self.args.redirect_gui:
                logger.trace("Generating preview for GUI")
                img = ".gui_preview_{}.jpg".format(name)
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
