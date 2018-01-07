import cv2
import numpy
import time

from lib.training_data import get_training_data, stack_images
from lib.utils import get_image_paths, get_folder, load_images

from lib.model import autoencoder_A, autoencoder_B
from lib.model import encoder, decoder_A, decoder_B
from lib.cli import FullPaths

class TrainingProcessor(object):
    arguments = None

    def __init__(self, subparser, command, description='default'):
        self.parse_arguments(description, subparser, command)

    def process_arguments(self, arguments):
        self.arguments = arguments
        print("Model A Directory: {}".format(self.arguments.input_A))
        print("Model B Directory: {}".format(self.arguments.input_B))
        print("Training data directory: {}".format(self.arguments.model_dir))

        try:
            encoder.load_weights(self.arguments.model_dir + '/encoder.h5')
            decoder_A.load_weights(self.arguments.model_dir + '/decoder_A.h5')
            decoder_B.load_weights(self.arguments.model_dir + '/decoder_B.h5')
        except Exception as e:
            print('Not loading existing training data.')
            print(e)

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
        parser = self.add_optional_arguments(parser)
        parser.set_defaults(func=self.process_arguments)

    def add_optional_arguments(self, parser):
        # Override this for custom arguments
        return parser

    def save_model_weights(self):
        encoder.save_weights(self.arguments.model_dir + '/encoder.h5')
        decoder_A.save_weights(self.arguments.model_dir + '/decoder_A.h5')
        decoder_B.save_weights(self.arguments.model_dir + '/decoder_B.h5')
        print('save model weights')

    def show_sample(self, test_A, test_B):
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
            cv2.imshow('', figure)
        if not self.arguments.preview or self.arguments.write_image:
            cv2.imwrite('_sample.jpg', figure)

    def process(self):
        print('Starting, this may take a while...')
        images_A = get_image_paths(self.arguments.input_A)
        images_B = get_image_paths(self.arguments.input_B)
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
                self.show_sample(target_A[0:14], target_B[0:14])

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
