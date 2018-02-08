import cv2
import re

from pathlib import Path
from tqdm import tqdm

from lib.cli import DirectoryProcessor, FullPaths
from lib.utils import BackgroundGenerator, get_folder

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

        parser.add_argument('-t', '--trainer',
                            type=str,
                            choices=("Original", "LowMem", "GAN"), # case sensitive because this is used to load a plug-in.
                            default="Original",
                            help="Select the trainer that was used to create the model.")

        parser.add_argument('-s', '--swap-model',
                            action="store_true",
                            dest="swap_model",
                            default=False,
                            help="Swap the model. Instead of A -> B, swap B -> A.")

        parser.add_argument('-c', '--converter',
                            type=str,
                            choices=("Masked", "Adjust", "GAN"), # case sensitive because this is used to load a plugin.
                            default="Masked",
                            help="Converter to use.")

        parser.add_argument('-D', '--detector',
                            type=str,
                            choices=("hog", "cnn"), # case sensitive because this is used to load a plugin.
                            default="hog",
                            help="Detector to use. 'cnn' detects much more angles but will be much more resource intensive and may fail on large files.")

        parser.add_argument('-fr', '--frame-ranges',
                            nargs="+",
                            type=str,
                            help="frame ranges to apply transfer to e.g. For frames 10 to 50 and 90 to 100 use --frame-ranges 10-50 90-100. \
                            Files must have the frame-number as the last number in the name!"
                            )

        parser.add_argument('-d', '--discard-frames',
                            action="store_true",
                            dest="discard_frames",
                            default=False,
                            help="When used with --frame-ranges discards frames that are not processed instead of writing them out unchanged."
                            )

        parser.add_argument('-f', '--filter',
                            type=str,
                            dest="filter",
                            default="filter.jpg",
                            help="Reference image for the person you want to process. Should be a front portrait"
                            )

        parser.add_argument('-b', '--blur-size',
                            type=int,
                            default=2,
                            help="Blur size. (Masked converter only)")


        parser.add_argument('-S', '--seamless',
                            action="store_true",
                            dest="seamless_clone",
                            default=False,
                            help="Seamless mode. (Masked converter only)")

        parser.add_argument('-M', '--mask-type',
                            type=str.lower, #lowercase this, because its just a string later on.
                            dest="mask_type",
                            choices=["rect", "facehull", "facehullandrect"],
                            default="facehullandrect",
                            help="Mask to use to replace faces. (Masked converter only)")

        parser.add_argument('-e', '--erosion-kernel-size',
                            dest="erosion_kernel_size",
                            type=int,
                            default=None,
                            help="Erosion kernel size. (Masked converter only)")

        parser.add_argument('-sm', '--smooth-mask',
                            action="store_true",
                            dest="smooth_mask",
                            default=True,
                            help="Smooth mask (Adjust converter only)")

        parser.add_argument('-aca', '--avg-color-adjust',
                            action="store_true",
                            dest="avg_color_adjust",
                            default=True,
                            help="Average color adjust. (Adjust converter only)")
        return parser

    def process(self):
        # Original & LowMem models go with Adjust or Masked converter
        # GAN converter & model must go together
        # Note: GAN prediction outputs a mask + an image, while other predicts only an image
        model_name = self.arguments.trainer
        conv_name = self.arguments.converter

        if conv_name.startswith("GAN"):
            assert model_name.startswith("GAN") is True, "GAN converter can only be used with GAN model!"
        else:
            assert model_name.startswith("GAN") is False, "GAN model can only be used with GAN converter!"

        model = PluginLoader.get_model(model_name)(get_folder(self.arguments.model_dir))
        if not model.load(self.arguments.swap_model):
            print('Model Not Found! A valid model must be provided to continue!')
            exit(1)

        converter = PluginLoader.get_converter(conv_name)(model.converter(False),
            blur_size=self.arguments.blur_size,
            seamless_clone=self.arguments.seamless_clone,
            mask_type=self.arguments.mask_type,
            erosion_kernel_size=self.arguments.erosion_kernel_size,
            smooth_mask=self.arguments.smooth_mask,
            avg_color_adjust=self.arguments.avg_color_adjust
        )

        batch = BackgroundGenerator(self.prepare_images(), 1)

        # frame ranges stuff...
        self.frame_ranges = None

        # split out the frame ranges and parse out "min" and "max" values
        minmax = {
            "min": 0, # never any frames less than 0
            "max": float("inf")
        }

        if self.arguments.frame_ranges:
            self.frame_ranges = [tuple(map(lambda q: minmax[q] if q in minmax.keys() else int(q), v.split("-"))) for v in self.arguments.frame_ranges]

        # last number regex. I know regex is hacky, but its reliablyhacky(tm).
        self.imageidxre = re.compile(r'(\d+)(?!.*\d)')

        for item in batch.iterator():
            self.convert(converter, item)

    def check_skipframe(self, filename):
        try:
            idx = int(self.imageidxre.findall(filename)[0])
            return not any(map(lambda b: b[0]<=idx<=b[1], self.frame_ranges))
        except:
            return False

    def convert(self, converter, item):
        try:
            (filename, image, faces) = item

            skip = self.check_skipframe(filename)
            if self.arguments.discard_frames and skip:
                return

            if not skip: # process as normal
                for idx, face in faces:
                    image = converter.patch_image(image, face)

            output_file = get_folder(self.output_dir) / Path(filename).name
            cv2.imwrite(str(output_file), image)
        except Exception as e:
            print('Failed to convert image: {}. Reason: {}'.format(filename, e))

    def prepare_images(self):
        self.read_alignments()
        for filename in tqdm(self.read_directory()):
            image = cv2.imread(filename)

            if self.have_face(filename):
                faces = self.get_faces_alignments(filename)
            else:
                faces = self.get_faces(image)
            yield filename, image, faces
