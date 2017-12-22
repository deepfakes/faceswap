import cv2
from lib.cli import DirectoryProcessor
from pathlib import Path


class ExtractTrainingData(DirectoryProcessor):
    def process_face(self, face, index, filename):
        resized_image = cv2.resize(face.image, (256, 256))
        output_file = self.output_dir / Path(filename).stem
        cv2.imwrite(str(output_file) + str(index) + Path(filename).suffix,
                    resized_image)


extract_cli = ExtractTrainingData(description='Extracts faces from a collection of pictures \
    and saves them to a separate directory')
