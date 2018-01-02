import cv2

from pathlib import Path
from lib.cli import DirectoryProcessor
from lib.faces_detect import detect_faces
from plugins.Extract_Align import Extract

class ExtractTrainingData(DirectoryProcessor):
    def create_parser(self, subparser, command, description):
        self.parser = subparser.add_parser(
            command,
            help="Extract the faces from a pictures.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )
        
    def process_image(self, filename):
        extractor = Extract()
        try:
            image = cv2.imread(filename)
            for (idx, face) in enumerate(detect_faces(image)):
                if idx > 0 and self.arguments.verbose:
                    print('- Found more than one face!')
                    self.verify_output = True

                resized_image = extractor.extract(image, face, 256)
                output_file = self.output_dir / Path(filename).stem
                cv2.imwrite(str(output_file) + str(idx) + Path(filename).suffix, resized_image)
                self.faces_detected = self.faces_detected + 1
        except Exception as e:
            print('Failed to extract from image: {}. Reason: {}'.format(filename, e))
