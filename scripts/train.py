#!/usr/bin python3
""" The script to run the training process of faceswap """

import os
import sys
import threading

import cv2
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from lib.utils import get_folder, get_image_paths, set_system_verbosity
from plugins.PluginLoader import PluginLoader


class Train(object):
    """ The training process.  """
    def __init__(self, arguments):
        self.args = arguments
        self.images = self.get_images()
        self.stop = False
        self.save_now = False
        self.preview_buffer = dict()
        self.lock = threading.Lock()

        # this is so that you can enter case insensitive values for trainer
        trainer_name = self.args.trainer
        self.trainer_name = "LowMem" if trainer_name.lower() == "lowmem" else trainer_name

    def process(self):
        """ Call the training process object """
        print("Training data directory: {}".format(self.args.model_dir))
        lvl = '0' if self.args.verbose else '2'
        set_system_verbosity(lvl)
        thread = self.start_thread()

        if self.args.preview:
            self.monitor_preview()
        else:
            self.monitor_console()

        self.end_thread(thread)

    def get_images(self):
        """ Check the image dirs exist, contain images and return the image
        objects """
        images = []
        for image_dir in [self.args.input_A, self.args.input_B]:
            if not os.path.isdir(image_dir):
                print('Error: {} does not exist'.format(image_dir))
                exit(1)

            if not os.listdir(image_dir):
                print('Error: {} contains no images'.format(image_dir))
                exit(1)

            images.append(get_image_paths(image_dir))
        print("Model A Directory: {}".format(self.args.input_A))
        print("Model B Directory: {}".format(self.args.input_B))
        return images

    def start_thread(self):
        """ Put the training process in a thread so we can keep control """
        thread = threading.Thread(target=self.process_thread)
        thread.start()
        return thread

    def end_thread(self, thread):
        """ On termination output message and join thread back to main """
        print("Exit requested! The trainer will complete its current cycle, "
              "save the models and quit (it can take up a couple of seconds "
              "depending on your training speed). If you want to kill it now, "
              "press Ctrl + c")
        self.stop = True
        thread.join()
        sys.stdout.flush()

    def process_thread(self):
        """ The training process to be run inside a thread """
        try:
            print("Loading data, this may take a while...")

            if self.args.allow_growth:
                self.set_tf_allow_growth()

            model = self.load_model()
            trainer = self.load_trainer(model)

            self.run_training_cycle(model, trainer)
        except KeyboardInterrupt:
            try:
                model.save_weights()
            except KeyboardInterrupt:
                print("Saving model weights has been cancelled!")
            exit(0)
        except Exception as err:
            raise err

    def load_model(self):
        """ Load the model requested for training """
        model_dir = get_folder(self.args.model_dir)
        model = PluginLoader.get_model(self.trainer_name)(model_dir, self.args.gpus)

        model.load(swapped=False)
        return model

    def load_trainer(self, model):
        """ Load the trainer requested for training """
        images_a, images_b = self.images

        trainer = PluginLoader.get_trainer(self.trainer_name)
        trainer = trainer(model,
                          images_a,
                          images_b,
                          self.args.batch_size,
                          self.args.perceptual_loss)
        return trainer

    def run_training_cycle(self, model, trainer):
        """ Perform the training cycle """
        for iteration in range(0, self.args.iterations):
            save_iteration = iteration % self.args.save_interval == 0
            viewer = self.show if save_iteration or self.save_now else None
            trainer.train_one_step(iteration, viewer)
            if self.stop:
                break
            elif save_iteration:
                model.save_weights()
            elif self.save_now:
                model.save_weights()
                self.save_now = False
        model.save_weights()
        self.stop = True

    def monitor_preview(self):
        """ Generate the preview window and wait for keyboard input """
        print("Using live preview.\n"
              "Press 'ENTER' on the preview window to save and quit.\n"
              "Press 'S' on the preview window to save model weights "
              "immediately")
        while True:
            try:
                with self.lock:
                    for name, image in self.preview_buffer.items():
                        cv2.imshow(name, image)

                key = cv2.waitKey(1000)
                if key == ord("\n") or key == ord("\r"):
                    break
                if key == ord("s"):
                    self.save_now = True
                if self.stop:
                    break
            except KeyboardInterrupt:
                break

    @staticmethod
    def monitor_console():
        """ Monitor the console for any input followed by enter or ctrl+c """
        # TODO: how to catch a specific key instead of Enter?
        # there isn't a good multiplatform solution:
        # https://stackoverflow.com/questions/3523174
        # TODO: Find a way to interrupt input() if the target iterations are reached.
        # At the moment, setting a target iteration and using the -p flag is
        # the only guaranteed way to exit the training loop on hitting target
        # iterations.
        print("Starting. Press 'ENTER' to stop training and save model")
        try:
            input()
        except KeyboardInterrupt:
            pass

    @staticmethod
    def set_tf_allow_growth():
        """ Allow TensorFlow to manage VRAM growth """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))

    def show(self, image, name=""):
        """ Generate the preview and write preview file output """
        try:
            scriptpath = os.path.realpath(os.path.dirname(sys.argv[0]))
            if self.args.write_image:
                img = "_sample_{}.jpg".format(name)
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)
            if self.args.redirect_gui:
                img = ".gui_preview_{}.jpg".format(name)
                imgfile = os.path.join(scriptpath, "lib", "gui", ".cache", "preview", img)
                cv2.imwrite(imgfile, image)
            if self.args.preview:
                with self.lock:
                    self.preview_buffer[name] = image
        except Exception as err:
            print("could not preview sample")
            raise err
