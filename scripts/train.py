#!/usr/bin python3
""" The script to run the training process of faceswap """

import logging
import os
import sys

from hashlib import sha1
from threading import Lock
from time import sleep

import cv2
from pathlib import Path
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session

from lib.keypress import KBHit
from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager
from lib.utils import (get_folder, get_image_paths, set_system_verbosity)
from plugins.plugin_loader import PluginLoader
from lib.model.masks import Facehull, Smart, Dummy
from plugins.train.trainer._base import Landmarks

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
        datasets with masks added """
        logger.debug("Getting image paths")

        def dataset_setup(img_list, in_size, batch_size):
            """ Create a mem-mapped image array for training"""

            def loader(img_file, target_size):
                """ Load and resize images with opencv """
                # pylint: disable=no-member
                img = cv2.imread(img_file)
                lm_key = sha1(img).hexdigest()
                img = img.astype('float32')
                height, width, _ = img.shape
                image_size = min(height, width)
                if image_size != target_size:
                    method = cv2.INTER_CUBIC if image_size < target_size else cv2.INTER_AREA
                    img = cv2.resize(img, (target_size, target_size), method)
                return img, lm_key

            filename = str(Path(img_list[0]).parents[0].joinpath(('Images_Batched.npy')))
            batch_num = (len(img_list) + batch_size -1) // batch_size
            img_shape = (batch_num, batch_size, in_size, in_size, 4)
            dataset = np.lib.format.open_memmap(filename, mode='w+', dtype='float32', shape=img_shape)
            hashes = np.empty((batch_num, batch_size), dtype='U40')
            for i, (img, lm_key) in enumerate(loader(img_file, in_size) for img_file in img_list):
                dataset[i // batch_size, i % batch_size, :, :, :3] = img[:, :, :3]
                hashes[i // batch_size, i % batch_size] = lm_key
            means = np.mean(dataset, axis=(0,1,2,3))
            return dataset, filename, means, hashes

        def get_landmarks(side, hashes, alignments, landmark_shape):
            """ Return the landmarks for this face """
            landmarks = Landmarks(alignments).landmarks
            src_points = np.empty(landmark_shape, dtype='float32')
            for src_point_batch, hash_batch in zip(src_points, hashes):
                for src_point, hash in zip(src_point_batch, hash_batch):
                    logger.trace("Retrieving landmarks: (hash: '%s', side: '%s'", hash, side)
                    if hash:
                        try:
                            src_point = landmarks[side][hash]
                        except KeyError:
                            raise Exception("Landmarks not found for hash: '{}'".format(hash))
                        logger.trace("Returning: (src_points: %s)", src_point)
            return src_points

        images = dict()
        img_number = dict()
        mask_args = {None:          (None, Facehull),
                     "none":        (None, Dummy),
                     "components":  (None, Facehull),
                     "dfl_full":    (None, Facehull),
                     "facehull":    (None, Facehull),
                     "vgg_300":     (300, Smart),
                     "vgg_500":     (500, Smart),
                     "unet_256":    (256, Smart)}
        mask_type = self.args.mask_type
        model_in_size, Mask = mask_args[mask_type]
        for side in ("a", "b"):
            image_dir = getattr(self.args, "input_{}".format(side))
            if not os.path.isdir(image_dir):
                logger.error("Error: '%s' does not exist", image_dir)
                exit(1)

            if not os.listdir(image_dir):
                logger.error("Error: '%s' contains no images", image_dir)
                exit(1)

            image_file_list = get_image_paths(image_dir)
            self.image_size = cv2.imread(image_file_list[0]).shape[0]
            if model_in_size is None:
                model_in_size = self.image_size
            logger.debug("Training image size: %s", model_in_size)
            img_dataset, data_file, means, hashes = dataset_setup(image_file_list,
                                                                  model_in_size,
                                                                  self.args.batch_size)
            alignments=dict()
            alignments["training_size"] = self.image_size
            alignments["alignments"] = self.alignments_paths
            if Mask == Facehull:
                landmark_shape = img_dataset.shape[:2] + (68,2)
                landmarks = get_landmarks(side, hashes, alignments, landmark_shape)
            else:
                landmarks = np.empty(landmark_shape, dtype='float32')
            for img_batch, landmark_batch in zip(img_dataset, landmarks):
                img_batch = Mask(mask_type, img_batch, landmark_batch, channels=4).masks
            images[side] = img_dataset
            img_number[side] = len(image_file_list)

        logger.info("Model A Directory: %s", self.args.input_a)
        logger.info("Model B Directory: %s", self.args.input_b)
        logger.debug("Got image paths: %s", [(key, str(val,) + " images")
                                             for key, val in img_number.items()])
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
        # TODO move arguments to trainer as appropriate
        logger.debug("Loaded Model")
        return model

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
                          self.timelapse,
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

        for round in range(self.args.iterations // self.args.save_interval):
            for iteration in range(self.args.save_interval):
                total_iterations = iteration + round * self.args.save_interval
                logger.trace("Training iteration: %s", total_iterations)
                if self.stop:
                    logger.debug("Stop received. Terminating")
                    break
                elif self.save_now:
                    logger.trace("Save Requested: (iteration: %s", total_iterations)
                    self.save_now = False
                    model.save_models()
                    trainer.preview(display_func, None)
                trainer.train_one_step()
            trainer.preview(display_func, self.timelapse)
            logger.trace("Save Iteration: (iteration: %s", total_iterations)
            model.save_models()
            self.save_now = False
            if self.args.pingpong:
                trainer.pingpong.switch()
        logger.debug("Training cycle complete")
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
