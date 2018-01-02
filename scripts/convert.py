import cv2

from pathlib import Path
from lib.cli import DirectoryProcessor, FullPaths
from lib.faces_detect import detect_faces

from lib.model import autoencoder_A
from lib.model import autoencoder_B
from lib.model import encoder, decoder_A, decoder_B

#from plugins.Convert_Adjust import Convert
from plugins.Convert_Masked import Convert

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
        # TODO move the model load and the converter creation in a method called on init, but after the arg parsing
        (face_A,face_B) = ('/decoder_A.h5', '/decoder_B.h5') if not self.arguments.swap_model else ('/decoder_B.h5', '/decoder_A.h5')

        model_dir = self.arguments.model_dir
        encoder.load_weights(model_dir + "/encoder.h5")
        decoder_A.load_weights(model_dir + face_A)
        decoder_B.load_weights(model_dir + face_B)

        converter = Convert(autoencoder_B)

        try:
            image = cv2.imread(filename)
            for (idx, face) in enumerate(detect_faces(image)):
                if idx > 0 and self.arguments.verbose:
                    print('- Found more than one face!')
                    self.verify_output = True

                image = converter.convert_one_image(image, face)
                self.faces_detected = self.faces_detected + 1

            output_file = self.output_dir / Path(filename).name
            cv2.imwrite(str(output_file), image)
        except Exception as e:
            print('Failed to convert image: {}. Reason: {}'.format(filename, e))


