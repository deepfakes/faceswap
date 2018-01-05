import cv2

from pathlib import Path
from lib.cli import DirectoryProcessor, FullPaths
from lib.faces_detect import detect_faces

from plugins.PluginLoader import PluginLoader

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

    def process(self, reader):
        # Original model goes with Adjust or Masked converter
        # GAN converter & model must go together
        model_name = "GAN" # TODO Pass as argument
        conv_name = "GAN" # TODO Pass as argument

        if conv_name.startswith("GAN"):
            assert model_name.startswith("GAN") is True, "GAN converter can only be used with GAN model!"
        else:
            assert model_name.startswith("GAN") is False, "GAN model can only be used with GAN converter!"

        model = PluginLoader.get_model(model_name)(self.arguments.model_dir)
        model.load(self.arguments.swap_model)

        converter = PluginLoader.get_converter(conv_name)(model.converter(False))

        try:
            for filename in reader():
                image = cv2.imread(filename)
                for (idx, face) in enumerate(detect_faces(image)):
                    if idx > 0 and self.arguments.verbose:
                        print('- Found more than one face!')
                        self.verify_output = True

                    image = converter.patch_image(image, face)
                    self.faces_detected = self.faces_detected + 1

                output_file = self.output_dir / Path(filename).name
                cv2.imwrite(str(output_file), image)
        except Exception as e:
            print('Failed to convert image: {}. Reason: {}'.format(filename, e))


