import cv2

from pathlib import Path
from lib.cli import DirectoryProcessor
from lib.multithreading import pool_process
from plugins.PluginLoader import PluginLoader
from tqdm import tqdm

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
                            help="Number of processes to use.")
        return parser

    def process(self):
        extractor_name = "Align" # TODO Pass as argument
        self.extractor = PluginLoader.get_extractor(extractor_name)()
        self.faces_detected = 0
        processes = self.arguments.processes
        if processes is None:
            for _ in tqdm(pool_process(self.processFiles, list(self.read_directory()), processes=processes)):
                self.faces_detected +=1
        elif processes > 1:
            for _ in tqdm(pool_process(self.processFiles, list(self.read_directory()), processes=processes)):
                self.faces_detected +=1
        else:
            try:
                for filename in self.read_directory():
                    image = cv2.imread(filename)
                    for idx, face in self.get_faces(image):
                        resized_image = self.extractor.extract(image, face, 256)
                        output_file = self.output_dir / Path(filename).stem
                        cv2.imwrite(str(output_file) + str(idx) + Path(filename).suffix, resized_image)

            except Exception as e:
                print('Failed to extract from image: {}. Reason: {}'.format(filename, e))

        # self.faces_detected = sum()
        # multi threading affects the faces_detected count, so we update it here

    def processFiles(self, filename):
        try:
            image = cv2.imread(filename)
            count = 0
            for idx, face in self.get_faces(image):
                count = idx
                resized_image = self.extractor.extract(image, face, 256)
                output_file = self.output_dir / Path(filename).stem
                cv2.imwrite(str(output_file) + str(idx) + Path(filename).suffix, resized_image)
            return count + 1
        except Exception as e:
            print('Failed to extract from image: {}. Reason: {}'.format(filename, e))
