import cv2

from pathlib import Path
from lib.cli import DirectoryProcessor
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
        
    def process(self):
        variant = "Align" # TODO Pass as argument
        extractor = PluginLoader.get_extractor(variant)()

        try:
            for filename in self.read_directory():
                print('Loading %s' % (filename))
                image = cv2.imread(filename)
                for idx, face in self.get_faces(image):
                    resized_image = extractor.extract(image, face, 256)
                    output_file = self.output_dir / Path(filename).stem
                    cv2.imwrite(str(output_file) + str(idx) + Path(filename).suffix, resized_image)
                
        except Exception as e:
            print('Failed to extract from image: {}. Reason: {}'.format(filename, e))
