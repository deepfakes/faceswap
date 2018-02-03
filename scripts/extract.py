import cv2

from pathlib import Path
from lib.cli import DirectoryProcessor
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
        return parser

    def process(self):
        extractor_name = "Align" # TODO Pass as argument
        self.extractor = PluginLoader.get_extractor(extractor_name)()

        self.faces_detected = sum(pool_process(self.processFiles, list(self.read_directory())))
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
