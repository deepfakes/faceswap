import multiprocessing
from pathlib import Path

import cv2
import os

import time

from lib.cli import MultiProcessDirectoryProcessor, FullPaths
from lib.faces_detect import detect_faces
from lib.model import autoencoder_B
from lib.model import encoder, decoder_A, decoder_B
from plugins.PluginLoader import PluginLoader


class ConvertImage(MultiProcessDirectoryProcessor):
    filename = ''

    def create_parser(self, subparser, command, description):
        self.parser = subparser.add_parser(
            command,
            help="Convert a source image to a new one with the face swapped.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )
    
    def add_optional_arguments(self, parser):
        parser.add_argument('-m', '--model-dir',
                            action=FullPaths,
                            dest="model_dir",
                            default="models",
                            help="Model directory. A directory containing the trained model \
                    you wish to process. Defaults to 'models'")
        parser.add_argument('-s', '--swap-model',
                            action="store_true",
                            dest="swap_model",
                            default=False,
                            help="Swap the model. Instead of A -> B, swap B -> A.")
        return parser

    def process_arguments(self, arguments):
        if not arguments.swap_model:
            self.face_A, self.face_B = ('/decoder_A.h5', '/decoder_B.h5')
        else:
            self.face_A, self.face_B = ('/decoder_B.h5', '/decoder_A.h5')

        model_dir = arguments.model_dir
        encoder.load_weights(model_dir + "/encoder.h5")

        decoder_A.load_weights(model_dir + self.face_A)
        decoder_B.load_weights(model_dir + self.face_B)
        self.converter = PluginLoader.get_converter("Masked")(autoencoder_B)
        super().process_arguments(arguments)

    def process_image(self, filename):
        try:
            image = cv2.imread(filename)
            for (idx, face) in enumerate(detect_faces(image)):
                if idx > 0 and self.arguments.verbose:
                    print('- Found more than one face!')
                    self.verify_output = True

                image = self.converter.patch_image(image, face)
                self.faces_detected = self.faces_detected + 1

            output_file = self.output_dir / Path(filename).name
            cv2.imwrite(str(output_file), image)
        except Exception as e:
            print('Failed to convert image: {}. Reason: {}'.format(filename, e))

    def get_image_face(self, filename):
        try:
            image = cv2.imread(filename)
            for (idx, face) in enumerate(detect_faces(image)):
                if idx > 0 and self.arguments.verbose:
                    print('- Found more than one face!')
                    self.verify_output = True
                return image, face
        except Exception as e:
            print('Failed to convert image: {}. Reason: {}'.format(filename, e))

    def apply_converted_face(self, filename, image, face):
        try:
            image = self.converter.patch_image(image, face)
            self.faces_detected = self.faces_detected + 1
            output_file = self.output_dir / Path(filename).name
            cv2.imwrite(str(output_file), image)
        except Exception as e:
            print('Failed to convert image: {}. Reason: {}'.format(filename, e))

    def process_directory(self):
        # Define some parameters
        max_q_size = 30

        """ Use multiprocessing to generate batches in parallel. """
        try:
            queue = multiprocessing.Queue(maxsize=max_q_size)

            # define producer (putting items into queue)
            def process_image_in_queue(filename, queue):
                try:
                    img_face = self.get_image_face(filename)
                    if self.arguments.verbose:
                        print('Preprocessing: {}'.format(os.path.basename(filename)))

                    if img_face is not None:
                        image, face = img_face
                        # Put the data in a queue
                        queue.put((filename, image, face))
                except Exception as e:
                    print('Failed to process image: {}'.format(e))

            jobs = []

            def process_queue():
                from queue import Empty

                try:
                    while not queue.empty():
                        data = queue.get()
                        filename, image, face = data
                        self.apply_converted_face(filename, image, face)
                except Empty:
                    pass
                except Exception as e:
                    print('Error processing a queue: {}'.format(e))

            # run as consumer (read items from queue, in current thread)
            for filename in self.input_dir:
                p = multiprocessing.Process(target=process_image_in_queue, args=(filename, queue,))
                jobs.append(p)
                p.start()

                while len(jobs) >= self.maximum_jobs_count:
                    for job in jobs:
                        if not job.is_alive():
                            jobs.remove(job)
                    process_queue()


                self.images_processed = self.images_processed + 1

            # Process last jobs left in queue
            process_queue()


        except Exception as e:
            print("Finishing, exception: {}".format(e))
            for th in jobs:
                th.terminate()
            queue.close()
            raise



