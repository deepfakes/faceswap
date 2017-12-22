import argparse
import os
import cv2
import numpy

from lib.utils import get_image_paths, get_folder, load_images, stack_images
from lib.faces_detect import crop_faces
from lib.training_data import get_training_data

from lib.model import autoencoder_A, autoencoder_B
from lib.model import encoder, decoder_A, decoder_B


class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))


class TrainingProcessor(object):
    arguments = None

    def __init__(self, description='default'):
        print('Initializing')
        self.parse_arguments(description)

        print("Model A Directory: {}".format(self.arguments.input_A))
        print("Model B Directory: {}".format(self.arguments.input_B))
        print("Training data directory: {}".format(self.arguments.model_dir))
        print('Starting, this may take a while...')

        try:
            encoder.load_weights(self.arguments.model_dir + '/encoder.h5')
            decoder_A.load_weights(self.arguments.model_dir + '/decoder_A.h5')
            decoder_B.load_weights(self.arguments.model_dir + '/decoder_B.h5')
        except Exception as e:
            print('Not loading existing training data.')

        self.process()

    def parse_arguments(self, description):
        parser = argparse.ArgumentParser(
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
                            default="model",
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
        parser = self.add_optional_arguments(parser)
        self.arguments = parser.parse_args()

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
        else:
            cv2.imwrite('_sample.jpg', figure)

    def process(self):
        images_A = get_image_paths(self.arguments.images_A)
        images_B = get_image_paths(self.arguments.images_B)
        images_A = load_images(images_A) / 255.0
        images_B = load_images(images_B) / 255.0

        images_A += images_B.mean(axis=(0, 1, 2)) - \
            images_A.mean(axis=(0, 1, 2))

        print('press "q" to stop training and save model')

        BATCH_SIZE = 64

        for epoch in range(1000000):
            warped_A, target_A = get_training_data(images_A, BATCH_SIZE)
            warped_B, target_B = get_training_data(images_B, BATCH_SIZE)

            loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
            loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
            print(loss_A, loss_B)

            if epoch % 100 == 0:
                self.save_model_weights()
                self.show_sample(target_A[0:14], target_B[0:14])

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.save_model_weights()
                exit()


class DirectoryProcessor(object):
    '''
    Abstract class that processes a directory of images
    and writes output to the specified folder
    '''
    arguments = None

    input_dir = None
    output_dir = None

    verify_output = False
    images_found = 0
    images_processed = 0
    faces_detected = 0

    def __init__(self, description='default'):
        print('Initializing')
        self.parse_arguments(description)

        print("Input Directory: {}".format(self.arguments.input_dir))
        print("Output Directory: {}".format(self.arguments.output_dir))
        print('Starting, this may take a while...')

        self.output_dir = get_folder(self.arguments.output_dir)
        try:
            self.input_dir = get_image_paths(self.arguments.input_dir)
        except:
            print('Input directory not found. Please ensure it exists.')
            exit(1)

        self.images_found = len(self.input_dir)

        for filename in self.input_dir:
            if self.arguments.verbose:
                print('Processing: {}'.format(os.path.basename(filename)))

            self.process_image(filename)
            self.images_processed = self.images_processed + 1

        self.finalize()

    def parse_arguments(self, description):
        parser = argparse.ArgumentParser(
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )

        parser.add_argument('-i', '--input-dir',
                            action=FullPaths,
                            dest="input_dir",
                            default="input",
                            help="Input directory. A directory containing the files \
                            you wish to process. Defaults to 'input'")
        parser.add_argument('-o', '--output-dir',
                            action=FullPaths,
                            dest="output_dir",
                            default="output",
                            help="Output directory. This is where the converted files will \
                                be stored. Defaults to 'output'")
        parser.add_argument('-v', '--verbose',
                            action="store_true",
                            dest="verbose",
                            default=False,
                            help="Show verbose output")
        parser = self.add_optional_arguments(parser)
        self.arguments = parser.parse_args()

    def add_optional_arguments(self, parser):
        # Override this for custom arguments
        return parser

    def process_image(self, filename):
        try:
            image = cv2.imread(filename)
            for (idx, face) in enumerate(crop_faces(image)):
                if idx > 0 and self.arguments.verbose:
                    print('- Found more than one face!')
                    self.verify_output = True

                self.process_face(face, idx, filename)
                self.faces_detected = self.faces_detected + 1
        except Exception as e:
            print('Failed to extract from image: {}. Reason: {}'.format(filename, e))

    def process_face(self, face, index, filename):
        # implement your face processing!
        raise NotImplementedError()

    def finalize(self):
        print('-------------------------')
        print('Images found:        {}'.format(self.images_found))
        print('Images processed:    {}'.format(self.images_processed))
        print('Faces detected:      {}'.format(self.faces_detected))
        print('-------------------------')

        if self.verify_output:
            print('Note:')
            print('Multiple faces were detected in one or more pictures.')
            print('Double check your results.')
            print('-------------------------')
        print('Done!')
