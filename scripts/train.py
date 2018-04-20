import cv2
import numpy
import time

import threading
from lib.utils import get_image_paths, get_folder
from lib.cli import DirFullPaths, argparse, os, sys
from plugins.PluginLoader import PluginLoader

tf = None
set_session = None


def import_tensorflow_keras():
    ''' Import the TensorFlow and keras set_session modules only when they
    are required '''
    global tf
    global set_session
    if tf is None or set_session is None:
        import tensorflow
        import keras.backend.tensorflow_backend
        tf = tensorflow
        set_session = keras.backend.tensorflow_backend.set_session


class TrainingProcessor(object):
    arguments = None

    def __init__(self, subparser, command, description='default'):
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        self.parse_arguments(description, subparser, command)
        self.lock = threading.Lock()

    def process_arguments(self, arguments):
        self.arguments = arguments
        print("Model A Directory: {}".format(self.arguments.input_A))
        print("Model B Directory: {}".format(self.arguments.input_B))
        print("Training data directory: {}".format(self.arguments.model_dir))

        self.process()

    @staticmethod
    def get_argument_list():
        ''' Put the arguments in a list so that they are accessible from
        both argparse and gui '''
        argument_list = []
        argument_list.append({"opts": ("-A", "--input-A"),
                              "action": DirFullPaths,
                              "dest": "input_A",
                              "default": "input_A",
                              "help": "Input directory. A directory "
                                      "containing training images for face A.\
                               Defaults to 'input'"})
        argument_list.append({"opts": ("-B", "--input-B"),
                              "action": DirFullPaths,
                              "dest": "input_B",
                              "default": "input_B",
                              "help": "Input directory. A directory "
                                      "containing training images for face B.\
                               Defaults to 'input'"})
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": DirFullPaths,
                              "dest": "model_dir",
                              "default": "models",
                              "help": "Model directory. This is where the "
                                      "training data will \
                               be stored. Defaults to 'model'"})
        argument_list.append({"opts": ("-p", "--preview"),
                              "action": "store_true",
                              "dest": "preview",
                              "default": False,
                              "help": "Show preview output. If not "
                                      "specified, write progress \
                               to file."})
        argument_list.append({"opts": ("-v", "--verbose"),
                              "action": "store_true",
                              "dest": "verbose",
                              "default": False,
                              "help": "Show verbose output"})
        argument_list.append({"opts": ("-s", "--save-interval"),
                              "type": int,
                              "dest": "save_interval",
                              "default": 100,
                              "help": "Sets the number of iterations before "
                                      "saving the model."})
        argument_list.append({"opts": ("-w", "--write-image"),
                              "action": "store_true",
                              "dest": "write_image",
                              "default": False,
                              "help": "Writes the training result to a file "
                                      "even on preview mode."})
        argument_list.append({"opts": ("-t", "--trainer"),
                              "type": str,
                              "choices": PluginLoader.get_available_models(),
                              "default": PluginLoader.get_default_model(),
                              "help": "Select which trainer to use, LowMem "
                                      "for cards < 2gb."})
        argument_list.append({"opts": ("-pl", "--use-perceptual-loss"),
                              "action": "store_true",
                              "dest": "perceptual_loss",
                              "default": False,
                              "help": "Use perceptual loss while training"})
        argument_list.append({"opts": ("-bs", "--batch-size"),
                              "type": int,
                              "default": 64,
                              "help": "Batch size, as a power of 2 (64, 128, "
                                      "256, etc)"})
        argument_list.append({"opts": ("-ag", "--allow-growth"),
                              "action": "store_true",
                              "dest": "allow_growth",
                              "default": False,
                              "help": "Sets allow_growth option of "
                                      "Tensorflow to spare memory on some "
                                      "configs"})
        argument_list.append({"opts": ("-ep", "--epochs"),
                              "type": int,
                              "default": 1000000,
                              "help": "Length of training in epochs."})
        argument_list.append({"opts": ("-g", "--gpus"),
                              "type": int,
                              "default": 1,
                              "help": "Number of GPUs to use for training"})
        # This is a hidden argument to indicate that the GUI is being used,
        # so the preview window
        # should be redirected Accordingly
        argument_list.append({"opts": ("-gui", "--gui"),
                              "action": "store_true",
                              "dest": "redirect_gui",
                              "default": False,
                              "help": argparse.SUPPRESS})
        return argument_list

    @staticmethod
    def get_optional_arguments():
        ''' Put the arguments in a list so that they are accessible from
        both argparse and gui '''
        # Override this for custom arguments
        argument_list = []
        return argument_list

    def parse_arguments(self, description, subparser, command):
        parser = subparser.add_parser(
                command,
                help="This command trains the model for the two faces A and "
                     "B.",
                description=description,
                epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground")

        for option in self.argument_list:
            args = option['opts']
            kwargs = {key: option[key] for key in option.keys() if
                      key != 'opts'}
            parser.add_argument(*args, **kwargs)

        parser = self.add_optional_arguments(parser)
        parser.set_defaults(func=self.process_arguments)

    def add_optional_arguments(self, parser):
        for option in self.optional_arguments:
            args = option['opts']
            kwargs = {key: option[key] for key in option.keys() if
                      key != 'opts'}
            parser.add_argument(*args, **kwargs)
        return parser

    def process(self):
        self.stop = False
        self.save_now = False

        thr = threading.Thread(target=self.processThread, args=(), kwargs={})
        thr.start()

        if self.arguments.preview:
            print('Using live preview')
            while True:
                try:
                    with self.lock:
                        for name, image in self.preview_buffer.items():
                            cv2.imshow(name, image)

                    key = cv2.waitKey(1000)
                    if key == ord('\n') or key == ord('\r'):
                        break
                    if key == ord('s'):
                        self.save_now = True
                except KeyboardInterrupt:
                    break
        else:
            try:
                input()  # TODO how to catch a specific key instead of Enter?
                # there isnt a good multiplatform solution:
                # https://stackoverflow.com/questions/3523174/raw-input-in
                # -python-without-pressing-enter
            except KeyboardInterrupt:
                pass

        print(
            "Exit requested! The trainer will complete its current cycle, "
            "save the models and quit (it can take up a couple of seconds "
            "depending on your training speed). If you want to kill it now, "
            "press Ctrl + c")
        self.stop = True
        thr.join()  # waits until thread finishes

    def processThread(self):
        try:
            if self.arguments.allow_growth:
                self.set_tf_allow_growth()

            print('Loading data, this may take a while...')
            # this is so that you can enter case insensitive values for trainer
            trainer = self.arguments.trainer
            trainer = "LowMem" if trainer.lower() == "lowmem" else trainer
            model = PluginLoader.get_model(trainer)(
                get_folder(self.arguments.model_dir), self.arguments.gpus)
            model.load(swapped=False)

            images_A = get_image_paths(self.arguments.input_A)
            images_B = get_image_paths(self.arguments.input_B)
            trainer = PluginLoader.get_trainer(trainer)
            trainer = trainer(model, images_A, images_B,
                              self.arguments.batch_size,
                              self.arguments.perceptual_loss)

            print('Starting. Press "Enter" to stop training and save model')

            for epoch in range(0, self.arguments.epochs):

                save_iteration = epoch % self.arguments.save_interval == 0

                trainer.train_one_step(epoch, self.show if (
                            save_iteration or self.save_now) else None)

                if save_iteration:
                    model.save_weights()

                if self.stop:
                    break

                if self.save_now:
                    model.save_weights()
                    self.save_now = False

            model.save_weights()
            exit(0)
        except KeyboardInterrupt:
            try:
                model.save_weights()
            except KeyboardInterrupt:
                print('Saving model weights has been cancelled!')
            exit(0)
        except Exception as e:
            raise e
            exit(1)

    def set_tf_allow_growth(self):
        import_tensorflow_keras()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=config))

    preview_buffer = {}

    def show(self, image, name=''):
        try:
            if self.arguments.redirect_gui:
                scriptpath = os.path.realpath(os.path.dirname(sys.argv[0]))
                img = '.gui_preview.png'
                imgfile = os.path.join(scriptpath, img)
                cv2.imwrite(imgfile, image)
            elif self.arguments.preview:
                with self.lock:
                    self.preview_buffer[name] = image
            elif self.arguments.write_image:
                cv2.imwrite('_sample_{}.jpg'.format(name), image)
        except Exception as e:
            print("could not preview sample")
            raise e
