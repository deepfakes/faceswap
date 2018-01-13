import cv2

from pathlib import Path
from lib.cli import DirectoryProcessor, FullPaths
from lib.utils import BackgroundGenerator
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

    def process(self):
        # Original model goes with Adjust or Masked converter
        model_name = "Original" # TODO Pass as argument
        conv_name = "Masked" # TODO Pass as argument

        model = PluginLoader.get_model(model_name)(self.arguments.model_dir)
        if not model.load(self.arguments.swap_model):
            print('Model Not Found! A valid model must be provided to continue!')
            exit(1)

        converter = PluginLoader.get_converter(conv_name)(model.converter(False))

        batch = BackgroundGenerator(self.prepare_images(), 1)
        for item in batch.iterator():
            self.convert(converter, item)
        
    def convert(self, converter, item):
        try:
            (filename, image, faces) = item
            print('Processing %s' % (filename))
            for idx, face in faces:
                image = converter.patch_image(image, face)

            output_file = self.output_dir / Path(filename).name
            cv2.imwrite(str(output_file), image)
        except Exception as e:
            print('Failed to convert image: {}. Reason: {}'.format(filename, e))

    def prepare_images(self):
        for filename in self.read_directory():
            print('Preparing %s' % (filename))
            image = cv2.imread(filename)
            yield filename, image, self.get_faces(image)
