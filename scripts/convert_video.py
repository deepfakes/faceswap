import cv2

from pathlib import Path
from lib.cli import DirectoryProcessor, FullPaths
from lib.faces_detect import detect_faces
from lib.utils import get_video_paths, get_folder

from plugins.PluginLoader import PluginLoader

# Src: https://github.com/shaoanlu/faceswap-GAN/blob/master/FaceSwap_GAN_v2_train.ipynb
# # Download ffmpeg if need, which is required by moviepy.
# #import imageio
# #imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

class ConvertVideo(DirectoryProcessor):
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
                            choices=PluginLoader.get_available_models(), # case sensitive because this is used to load a plug-in.
                            default=PluginLoader.get_default_model(),
                            help="Select the trainer that was used to create the model.")
                            
        parser.add_argument('-s', '--swap-model',
                            action="store_true",
                            dest="swap_model",
                            default=False,
                            help="Swap the model. Instead of A -> B, swap B -> A.")

        parser.add_argument('-c', '--converter',
                            type=str,
                            choices=("Masked", "Adjust", "GAN", "GAN128"), # case sensitive because this is used to load a plugin.
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
    
    def start(self):
        self.output_dir = get_folder(self.arguments.output_dir)
        try:
            self.input_dir = get_video_paths(self.arguments.input_dir)
        except:
            print('Input directory not found. Please ensure it exists.')
            exit(1)

        self.filter = self.load_filter()
        self.process()
        self.finalize()

    def process(self):
        for filename in self.read_directory():
            print('Loading %s' % (filename))
            
            input = VideoFileClip(filename)
            clip = input.fl_image(self.frame_processor())#.subclip(11, 13) #NOTE: this function expects color images!!

            output_file = self.output_dir  / Path(filename).stem
            clip.write_videofile(str(output_file) + ".mp4", audio=False)

    #NOTE we recreate converter between each video to reset context (especially if we use Bounding Box that has between-frames context)
    #it also reloads model which may be memory consuming => Multi video conversion is not a high priority! Refactor if needed
    def frame_processor(self):
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
        model.load(self.arguments.swap_model)

        converter = PluginLoader.get_converter(conv_name)(model.converter(False))
        return lambda frame: converter.patch_all(frame, self.get_faces(frame))