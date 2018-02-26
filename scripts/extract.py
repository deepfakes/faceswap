import cv2

from pathlib import Path
from tqdm import tqdm
import os

from lib.cli import DirectoryProcessor
from lib.utils import get_folder
from lib.multithreading import pool_process
from plugins.PluginLoader import PluginLoader

class ExtractTrainingData(DirectoryProcessor):
    def create_parser(self, subparser, command, description):
        self.parser = subparser.add_parser(
            command,
            help="Extract the faces from a pictures.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )

    def add_optional_arguments(self, parser):
        parser.add_argument('-D', '--detector',
                            type=str,
                            choices=("hog", "cnn"), # case sensitive because this is used to load a plugin.
                            default="hog",
                            help="Detector to use. 'cnn' detects much more angles but will be much more resource intensive and may fail on large files.")

        parser.add_argument('-f', '--filter',
                            type=str,
                            dest="filter",
                            default="filter.jpg",
                            help="Reference image for the person you want to process. Should be a front portrait"
                            )

        parser.add_argument('-j', '--processes',
                            type=int,
                            default=1,
                            help="Number of processes to use.")
        return parser

    def process(self):
        extractor_name = "Align" # TODO Pass as argument
        self.extractor = PluginLoader.get_extractor(extractor_name)()
        processes = self.arguments.processes
        try:
            if processes != 1:
                files = list(self.read_directory())
                for filename, faces in tqdm(pool_process(self.processFiles, files, processes=processes), total = len(files)):
                    self.num_faces_detected += 1
                    self.faces_detected[os.path.basename(filename)] = faces
            else:
                try:
                    for filename in tqdm(self.read_directory()):
                        image = cv2.imread(filename)
                        self.faces_detected[os.path.basename(filename)] = self.handleImage(self, image)
                except Exception as e:
                    print('Failed to extract from image: {}. Reason: {}'.format(filename, e))
        finally:
            self.write_alignments()

    def processFiles(self, filename):
        try:
            image = cv2.imread(filename)
            return filename, self.handleImage(image)
        except Exception as e:
            print('Failed to extract from image: {}. Reason: {}'.format(filename, e))

    def handleImage(self, image):
        count = 0

        faces = self.get_faces(image)
        rvals = []
        for idx, face in faces:
            count = idx

            resized_image = self.extractor.extract(image, face, 256)
            output_file = get_folder(self.output_dir) / Path(filename).stem
            cv2.imwrite(str(output_file) + str(idx) + Path(filename).suffix, resized_image)
            f = {
                "x": face.x,
                "w": face.w,
                "y": face.y,
                "h": face.h,
                "landmarksXY": face.landmarksAsXY()
            }
            rvals.append(f)
        return rvals
