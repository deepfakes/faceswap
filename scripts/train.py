#!/usr/bin python3
""" The script to run the training process of faceswap """

import threading

import cv2

from lib.cli import FullPaths, argparse, os, sys
from lib.utils import get_folder, get_image_paths
from plugins.PluginLoader import PluginLoader

def import_tensorflow_keras():
    """ Import the TensorFlow and keras set_session modules only when they are required """
    import tensorflow as tflow
    from keras.backend.tensorflow_backend import set_session
    return (tflow, set_session)

class TrainingProcessor(object):
    """ Class to parse the command line arguments for training and
        call the training process object """

    def __init__(self, subparser, command, description="default"):
        self.arguments = None

        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()

        self.parse_arguments(description, subparser, command)

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-A", "--input-A"),
                              "action": FullPaths,
                              "dest": "input_A",
                              "default": "input_A",
                              "help": "Input directory. A directory containing training images "
                                      "for face A. Defaults to 'input'"})
        argument_list.append({"opts": ("-B", "--input-B"),
                              "action": FullPaths,
                              "dest": "input_B",
                              "default": "input_B",
                              "help": "Input directory. A directory containing training images "
                                      "for face B Defaults to 'input'"})
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": FullPaths,
                              "dest": "model_dir",
                              "default": "models",
                              "help": "Model directory. This is where the training data will "
                                      "be stored. Defaults to 'model'"})
        argument_list.append({"opts": ("-p", "--preview"),
                              "action": "store_true",
                              "dest": "preview",
                              "default": False,
                              "help": "Show preview output. If not specified, write progress "
                                      "to file."})
        argument_list.append({"opts": ("-v", "--verbose"),
                              "action": "store_true",
                              "dest": "verbose",
                              "default": False,
                              "help": "Show verbose output"})
        argument_list.append({"opts": ("-s", "--save-interval"),
                              "type": int,
                              "dest": "save_interval",
                              "default": 100,
                              "help": "Sets the number of iterations before saving the model."})
        argument_list.append({"opts": ("-w", "--write-image"),
                              "action": "store_true",
                              "dest": "write_image",
                              "default": False,
                              "help": "Writes the training result to a file even on "
                                      "preview mode."})
        argument_list.append({"opts": ("-t", "--trainer"),
                              "type": str,
                              "choices": PluginLoader.get_available_models(),
                              "default": PluginLoader.get_default_model(),
                              "help": "Select which trainer to use, LowMem for cards < 2gb."})
        argument_list.append({"opts": ("-pl", "--use-perceptual-loss"),
                              "action": "store_true",
                              "dest": "perceptual_loss",
                              "default": False,
                              "help": "Use perceptual loss while training"})
        argument_list.append({"opts": ("-bs", "--batch-size"),
                              "type": int,
                              "default": 64,
                              "help": "Batch size, as a power of 2 (64, 128, 256, etc)"})
        argument_list.append({"opts": ("-ag", "--allow-growth"),
                              "action": "store_true",
                              "dest": "allow_growth",
                              "default": False,
                              "help": "Sets allow_growth option of Tensorflow to spare memory "
                                      "on some configs"})
        argument_list.append({"opts": ("-ep", "--epochs"),
                              "type": int,
                              "default": 1000000,
                              "help": "Length of training in epochs."})
        argument_list.append({"opts": ("-g", "--gpus"),
                              "type": int,
                              "default": 1,
                              "help": "Number of GPUs to use for training"})
        # This is a hidden argument to indicate that the GUI is being used,
        # so the preview window should be redirected Accordingly
        argument_list.append({"opts": ("-gui", "--gui"),
                              "action": "store_true",
                              "dest": "redirect_gui",
                              "default": False,
                              "help": argparse.SUPPRESS})
        return argument_list

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        # Override this for custom arguments
        argument_list = []
        return argument_list

    def parse_arguments(self, description, subparser, command):
        """ Parse the arguments passed in from argparse """
        parser = subparser.add_parser(
            command,
            help="This command trains the model for the two faces A and B.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground")

        for option in self.argument_list:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key != "opts"}
            parser.add_argument(*args, **kwargs)

        parser = self.add_optional_arguments(parser)
        parser.set_defaults(func=self.process_arguments)

    def add_optional_arguments(self, parser):
        """ Add any optional arguments passed in from argparse """
        for option in self.optional_arguments:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key != "opts"}
            parser.add_argument(*args, **kwargs)
        return parser

    def process_arguments(self, arguments):
        """ Process the arguments from argparse """
        self.arguments = arguments
        print("Model A Directory: {}".format(self.arguments.input_A))
        print("Model B Directory: {}".format(self.arguments.input_B))
        print("Training data directory: {}".format(self.arguments.model_dir))

        self.process()

    def process(self):
        """ Call the training process object """
        training = Training(self.arguments)
        training.process()

class Training(object):
    """ The training object """
    def __init__(self, opts):
        self.args = opts
        self.stop = False
        self.save_now = False
        self.thread = None
        self.preview_buffer = dict()
        self.lock = threading.Lock()

    def process(self):
        """ Perform the training process """
        self.start_thread()

        if self.args.preview:
            self.monitor_preview()
        else:
            self.monitor_console()

        self.end_thread()

    def start_thread(self):
        """ Put the training process in a thread so we can keep control """
        self.thread = threading.Thread(target=self.process_thread)
        self.thread.start()

    def end_thread(self):
        """ On termination output message and join thread back to main """
        print("Exit requested! The trainer will complete its current cycle, save "
              "the models and quit (it can take up a couple of seconds depending "
              "on your training speed). If you want to kill it now, press Ctrl + c")
        self.stop = True
        self.thread.join()

    def process_thread(self):
        """ The training process to be run inside a thread """
        try:
            # this is so that you can enter case insensitive values for trainer
            trainer_name = self.args.trainer
            trainer_name = "LowMem" if trainer_name.lower() == "lowmem" else trainer_name

            print("Loading data, this may take a while...")

            if self.args.allow_growth:
                self.set_tf_allow_growth()

            model = self.load_model(trainer_name)
            trainer = self.load_trainer(trainer_name, model)

            print("Starting. Press 'ENTER' or 'CTRL+C' to stop training and save model")

            for epoch in range(0, self.args.epochs):
                save_iteration = epoch % self.args.save_interval == 0
                viewer = self.show if save_iteration or self.save_now else None
                trainer.train_one_step(epoch, viewer)

                if self.stop:
                    model.save_weights()
                    exit()
                elif save_iteration:
                    model.save_weights()
                elif self.save_now:
                    model.save_weights()
                    self.save_now = False

        except KeyboardInterrupt:
            try:
                model.save_weights()
            except KeyboardInterrupt:
                print("Saving model weights has been cancelled!")
            exit(0)
        except Exception as err:
            raise err

    def load_model(self, model_name):
        """ Load the model requested for training """
        model_dir = get_folder(self.args.model_dir)

        model = PluginLoader.get_model(model_name)(model_dir, self.args.gpus)

        if not model.load(swapped=False):
            raise ValueError("Model Not Found! A valid model must be provided to continue!")

        return model

    def load_trainer(self, trainer_name, model):
        """ Load the trainer requested for traning """
        images_a = get_image_paths(self.args.input_A)
        images_b = get_image_paths(self.args.input_B)

        trainer = PluginLoader.get_trainer(trainer_name)
        trainer = trainer(model,
                          images_a,
                          images_b,
                          self.args.batch_size,
                          self.args.perceptual_loss)

        return trainer

    def monitor_preview(self):
        """ Generate the preview window and wait for keyboard input """
        print("Using live preview.\n"
              "\tPress 'ENTER' on the preview window to save and quit."
              "\tPress 'S' on the preview window to save model weights immediately")
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
            except KeyboardInterrupt:
                break

    @staticmethod
    def monitor_console():
        """ Monitor the console for any input followed by enter or ctrl+c """
        try:
            # TODO how to catch a specific key instead of Enter?
            # there isnt a good multiplatform solution:
            # https://stackoverflow.com/questions/3523174/raw-input-in-python-without-pressing-enter
            input()
        except KeyboardInterrupt:
            pass

    @staticmethod
    def set_tf_allow_growth():
        """ Allow TensorFlow to manage VRAM growth """
        tflow, set_session = import_tensorflow_keras()
        config = tflow.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        set_session(tflow.Session(config=config))

    def show(self, image, name=""):
        """ Generate the preview and write preview file output """
        try:
            scriptpath = os.path.realpath(os.path.dirname(sys.argv[0]))
            if self.args.write_image:
                img = "_sample_{}.jpg".format(name)
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)

            if self.args.redirect_gui:
                img = ".gui_preview.png"
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)
            elif self.args.preview:
                with self.lock:
                    self.preview_buffer[name] = image
        except Exception as err:
            print("could not preview sample")
            raise err
