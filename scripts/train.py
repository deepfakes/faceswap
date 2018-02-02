import cv2
import numpy
import time

from lib.utils import get_image_paths
from lib.cli import FullPaths
from plugins.PluginLoader import PluginLoader

class TrainingProcessor(object):
    arguments = None

    def __init__(self, subparser, command, description='default'):
        self.parse_arguments(description, subparser, command)

    def process_arguments(self, arguments):
        self.arguments = arguments
        print("Model A Directory: {}".format(self.arguments.input_A))
        print("Model B Directory: {}".format(self.arguments.input_B))
        print("Training data directory: {}".format(self.arguments.model_dir))

        self.process()

    def parse_arguments(self, description, subparser, command):
        parser = subparser.add_parser(
            command,
            help="This command trains the model for the two faces A and B.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )

        parser.add_argument('-A', '--input-A',
                            action=FullPaths,
                            dest="input_A",
                            default="input_A",
                            help="Input directory. A directory containing training images for face A.\
                             Defaults to 'input'")
        parser.add_argument('-B', '--input-B',
                            action=FullPaths,
                            dest="input_B",
                            default="input_B",
                            help="Input directory. A directory containing training images for face B.\
                             Defaults to 'input'")
        parser.add_argument('-m', '--model-dir',
                            action=FullPaths,
                            dest="model_dir",
                            default="models",
                            help="Model directory. This is where the training data will \
                                be stored. Defaults to 'model'")
        parser.add_argument('-p', '--preview',
                            action="store_true",
                            dest="preview",
                            default=False,
                            help="Show preview output. If not specified, write progress \
                            to file.")
        parser.add_argument('-v', '--verbose',
                            action="store_true",
                            dest="verbose",
                            default=False,
                            help="Show verbose output")
        parser.add_argument('-s', '--save-interval',
                            type=int,
                            dest="save_interval",
                            default=100,
                            help="Sets the number of iterations before saving the model.")
        parser.add_argument('-w', '--write-image',
                            action="store_true",
                            dest="write_image",
                            default=False,
                            help="Writes the training result to a file even on preview mode.")
        parser.add_argument('-t', '--trainer',
                            type=str,
                            choices=("Original", "LowMem"),
                            default="Original",
                            help="Select which trainer to use, LowMem for cards < 2gb.")
        parser.add_argument('-bs', '--batch-size',
                            type=int,
                            default=64,
                            help="Batch size, as a power of 2 (64, 128, 256, etc)")
        parser = self.add_optional_arguments(parser)
        parser.set_defaults(func=self.process_arguments)

    def add_optional_arguments(self, parser):
        # Override this for custom arguments
        return parser

<<<<<<< HEAD
    def save_model_weights(self):
        encoder.save_weights(self.arguments.model_dir + '/encoder.h5')
        decoder_A.save_weights(self.arguments.model_dir + '/decoder_A.h5')
        decoder_B.save_weights(self.arguments.model_dir + '/decoder_B.h5')
        print('save model weights')

    def show_sample(self, test_A, test_B, epoch):
        figure_A = numpy.stack([
            test_A,
            autoencoder_A.predict(test_A),
            autoencoder_B.predict(test_A),
        ], axis=1)
        figure_B = numpy.stack([
            test_B,
            autoencoder_B.predict(test_B),
            autoencoder_A.predict(test_B),
        ], axis=1)

        figure = numpy.concatenate([figure_A, figure_B], axis=0)
        figure = figure.reshape((4, 7) + figure.shape[1:])
        figure = stack_images(figure)

        figure = numpy.clip(figure * 255, 0, 255).astype('uint8')

        if self.arguments.preview is True:
            _file = '/home/rnd/git/faceswap/_protocol/epoch_{0}.png'.format(epoch)
            cv2.imwrite(_file, figure)
            cv2.imshow('', figure)

        if not self.arguments.preview or self.arguments.write_image:
            cv2.imwrite('_sample.jpg', figure)

=======
>>>>>>> 68ef3b992674d87d0c73da9c29a4c5a0e735f04b
    def process(self):
        import threading
        self.stop = False
        self.save_now = False

        thr = threading.Thread(target=self.processThread, args=(), kwargs={})
        thr.start()

        if self.arguments.preview:
            print('Using live preview')
            while True:
                try:
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
            input() # TODO how to catch a specific key instead of Enter?
            # there isnt a good multiplatform solution: https://stackoverflow.com/questions/3523174/raw-input-in-python-without-pressing-enter

        print("Exit requested! The trainer will complete its current cycle, save the models and quit (it can take up a couple of seconds depending on your training speed). If you want to kill it now, press Ctrl + c")
        self.stop = True
        thr.join() # waits until thread finishes

    def processThread(self):
        print('Loading data, this may take a while...')
        # this is so that you can enter case insensitive values for trainer
        trainer = self.arguments.trainer
        trainer = trainer if trainer != "Lowmem" else "LowMem"
        model = PluginLoader.get_model(trainer)(self.arguments.model_dir)
        model.load(swapped=False)

        images_A = get_image_paths(self.arguments.input_A)
        images_B = get_image_paths(self.arguments.input_B)
<<<<<<< HEAD
        images_A = load_images(images_A) / 255.0
        images_B = load_images(images_B) / 255.0

        images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

        print('press "q" to stop training and save model')

        BATCH_SIZE = 64

        for epoch in range(1000000):
            if self.arguments.verbose:
                print("Iteration number {}".format(epoch + 1))
                start_time = time.time()
            warped_A, target_A = get_training_data(images_A, BATCH_SIZE)
            warped_B, target_B = get_training_data(images_B, BATCH_SIZE)

            loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
            loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
            print(loss_A, loss_B)

            if epoch % self.arguments.save_interval == 0:
                self.save_model_weights()
                self.show_sample(target_A[0:14], target_B[0:14], epoch)

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.save_model_weights()
                exit()
            if self.arguments.verbose:
                end_time = time.time()
                time_elapsed = int(round((end_time - start_time)))
                m, s = divmod(time_elapsed, 60)
                h, m = divmod(m, 60)
                print("Iteration done in {:02d}h{:02d}m{:02d}s".format(h, m, s))
=======
        trainer = PluginLoader.get_trainer(trainer)(model,
                                                                   images_A,
                                                                   images_B,
                                                                   batch_size=self.arguments.batch_size)

        try:
            print('Starting. Press "Enter" to stop training and save model')

            for epoch in range(0, 1000000):

                save_iteration = epoch % self.arguments.save_interval == 0

                trainer.train_one_step(epoch, self.show if (save_iteration or self.save_now) else None)

                if save_iteration:
                    model.save_weights()

                if self.stop:
                    model.save_weights()
                    exit()

                if self.save_now:
                    model.save_weights()
                    self.save_now = False

        except KeyboardInterrupt:
            try:
                model.save_weights()
            except KeyboardInterrupt:
                print('Saving model weights has been cancelled!')
            exit(0)

    preview_buffer = {}

    def show(self, image, name=''):
        try:
            if self.arguments.preview:
                self.preview_buffer[name] = image
            elif self.arguments.write_image:
                cv2.imwrite('_sample_{}.jpg'.format(name), image)
        except Exception as e:
            print("could not preview sample")
            print(e)
>>>>>>> 68ef3b992674d87d0c73da9c29a4c5a0e735f04b
