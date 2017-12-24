import cv2
from lib.cli import DirectoryProcessor
from pathlib import Path


class ExtractTrainingData(DirectoryProcessor):
    def create_parser(self, subparser, command, description):
        self.parser = subparser.add_parser(
            command,
            help="Extract the faces from a pictures.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )
        
    def process_face(self, face, index, filename):
        resized_image = cv2.resize(face.image, (256, 256))
        output_file = self.output_dir / Path(filename).stem
        cv2.imwrite(str(output_file) + str(index) + Path(filename).suffix,
                    resized_image)
