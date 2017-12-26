import cv2
from lib.cli import DirectoryProcessor, FullPaths
from pathlib import Path
from lib.faces_process import convert_one_image
from lib.faces_detect import crop_faces


class ConvertImage(DirectoryProcessor):
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

    def process_image(self, filename):
        try:
            image = cv2.imread(filename)
            for (idx, face) in enumerate(crop_faces(image)):
                if idx > 0 and self.arguments.verbose:
                    print('- Found more than one face!')
                    self.verify_output = True

                new_face = convert_one_image(cv2.resize(face.image, (256, 256)), self.arguments.model_dir, self.arguments.swap_model)
                image[slice(face.y, face.y + face.h), slice(face.x, face.x + face.w)] = cv2.resize(new_face, (face.w, face.h))
                self.faces_detected = self.faces_detected + 1
            output_file = self.output_dir / Path(filename).name
            cv2.imwrite(str(output_file), image)
        except Exception as e:
            print('Failed to extract from image: {}. Reason: {}'.format(filename, e))


