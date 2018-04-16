#!/usr/bin python3
""" The script to run the convert process of faceswap """

import re
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from lib.cli import DirectoryProcessor, FullPaths
from lib.utils import BackgroundGenerator, get_folder, get_image_paths, rotate_image

from plugins.PluginLoader import PluginLoader

class ConvertImage(DirectoryProcessor):
    """ Class to parse the command line arguments for conversion and
        run the convert process """

    def __init__(self, subparser, command, description):
        DirectoryProcessor.__init__(self, subparser, command, description)
        self.input_aligned_dir = None

        # frame ranges stuff...
        self.frame_ranges = None
        # last number regex. I know regex is hacky, but its reliablyhacky(tm).
        self.imageidxre = re.compile(r"(\d+)(?!.*\d)")

    filename = ""
    def create_parser(self, subparser, command, description):
        self.optional_arguments = self.get_optional_arguments()
        self.parser = subparser.add_parser(
            command,
            help="Convert a source image to a new one with the face swapped.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": FullPaths,
                              "dest": "model_dir",
                              "default": "models",
                              "help": "Model directory. A directory containing the trained model \
                              you wish to process. Defaults to 'models'"})
        argument_list.append({"opts": ("-a", "--input-aligned-dir"),
                              "action": FullPaths,
                              "dest": "input_aligned_dir",
                              "default": None,
                              "help": "Input \"aligned directory\". A directory that should \
                              contain the aligned faces extracted from the input files. If you \
                              delete faces from this folder, they'll be skipped during \
                              conversion. If no aligned dir is specified, all faces will \
                              be converted."})
        argument_list.append({"opts": ("-t", "--trainer"),
                              "type": str,
                              # case sensitive because this is used to load a plug-in.
                              "choices": PluginLoader.get_available_models(),
                              "default": PluginLoader.get_default_model(),
                              "help": "Select the trainer that was used to create the model."})
        argument_list.append({"opts": ("-s", "--swap-model"),
                              "action": "store_true",
                              "dest": "swap_model",
                              "default": False,
                              "help": "Swap the model. Instead of A -> B, swap B -> A."})
        argument_list.append({"opts": ("-c", "--converter"),
                              "type": str,
                              # case sensitive because this is used to load a plugin.
                              "choices": ("Masked", "Adjust"),
                              "default": "Masked",
                              "help": "Converter to use."})
        argument_list.append({"opts": ("-D", "--detector"),
                              "type": str,
                              # case sensitive because this is used to load a plugin.
                              "choices": ("hog", "cnn"),
                              "default": "hog",
                              "help": "Detector to use. 'cnn' detects much more angles but \
                              will be much more resource intensive and may fail on large files."})
        argument_list.append({"opts": ("-fr", "--frame-ranges"),
                              "nargs": "+",
                              "type": str,
                              "help": "frame ranges to apply transfer to e.g. For frames 10 to \
                              50 and 90 to 100 use --frame-ranges 10-50 90-100. Files must have \
                              the frame-number as the last number in the name!"})
        argument_list.append({"opts": ("-d", "--discard-frames"),
                              "action": "store_true",
                              "dest": "discard_frames",
                              "default": False,
                              "help": "When used with --frame-ranges discards frames that are \
                              not processed instead of writing them out unchanged."})
        argument_list.append({"opts": ("-l", "--ref_threshold"),
                              "type": float,
                              "dest": "ref_threshold",
                              "default": 0.6,
                              "help": "Threshold for positive face recognition"})
        argument_list.append({"opts": ("-n", "--nfilter"),
                              "type": str,
                              "dest": "nfilter",
                              "nargs": "+",
                              "default": "nfilter.jpg",
                              "help": "Reference image for the persons you do not want to \
                              process. Should be a front portrait"})
        argument_list.append({"opts": ("-f", "--filter"),
                              "type": str,
                              "dest": "filter",
                              "nargs": "+",
                              "default": "filter.jpg",
                              "help": "Reference images for the person you want to process. \
                              Should be a front portrait"})
        argument_list.append({"opts": ("-b", "--blur-size"),
                              "type": int,
                              "default": 2,
                              "help": "Blur size. (Masked converter only)"})
        argument_list.append({"opts": ("-S", "--seamless"),
                              "action": "store_true",
                              "dest": "seamless_clone",
                              "default": False,
                              "help": "Use cv2's seamless clone. (Masked converter only)"})
        argument_list.append({"opts": ("-M", "--mask-type"),
                              #lowercase this, because its just a string later on.
                              "type": str.lower,
                              "dest": "mask_type",
                              "choices": ["rect", "facehull", "facehullandrect"],
                              "default": "facehullandrect",
                              "help": "Mask to use to replace faces. (Masked converter only)"})
        argument_list.append({"opts": ("-e", "--erosion-kernel-size"),
                              "dest": "erosion_kernel_size",
                              "type": int,
                              "default": None,
                              "help": "Erosion kernel size. (Masked converter only). Positive \
                              values apply erosion which reduces the edge of the swapped face. \
                              Negative values apply dilation which allows the swapped face to \
                              cover more space."})
        argument_list.append({"opts": ("-mh", "--match-histgoram"),
                              "action": "store_true",
                              "dest": "match_histogram",
                              "default": False,
                              "help": "Use histogram matching. (Masked converter only)"})
        argument_list.append({"opts": ("-sh", ),
                              "type": str.lower,
                              "dest": "sharpen_image",
                              "choices": ["bsharpen", "gsharpen"],
                              "default": None,
                              "help": "Use Sharpen Image - bsharpen = Box Blur, gsharpen = \
                              Gaussian Blur (Masked converter only)"})
        argument_list.append({"opts": ("-sm", "--smooth-mask"),
                              "action": "store_true",
                              "dest": "smooth_mask",
                              "default": True,
                              "help": "Smooth mask (Adjust converter only)"})
        argument_list.append({"opts": ("-aca", "--avg-color-adjust"),
                              "action": "store_true",
                              "dest": "avg_color_adjust",
                              "default": True,
                              "help": "Average color adjust. (Adjust converter only)"})
        argument_list.append({"opts": ("-g", "--gpus"),
                              "type": int,
                              "default": 1,
                              "help": "Number of GPUs to use for conversion"})
        return argument_list

    def process(self):
        """ Original & LowMem models go with Adjust or Masked converter
            Note: GAN prediction outputs a mask + an image, while other predicts only an image """

        self.check_aligned_directory()

        if self.arguments.frame_ranges:
            self.frame_ranges = self.get_frame_ranges()

        model = self.load_model()
        converter = self.load_converter(model)

        batch = BackgroundGenerator(self.prepare_images(), 1)

        for item in batch.iterator():
            self.convert(converter, item)

    def check_aligned_directory(self):
        """ Check for the existence of an aligned directory for identifying
            which faces in the target frames should be swapped """
        input_aligned_dir = self.arguments.input_aligned_dir

        if input_aligned_dir is None:
            print("Aligned directory not specified. All faces listed in the alignments file \
                   will be converted.")
        elif not os.path.isdir(input_aligned_dir):
            print("Aligned directory not found. All faces listed in the alignments file \
                   will be converted.")
        else:
            self.input_aligned_dir = [Path(path) for path in get_image_paths(input_aligned_dir)]
            if not self.input_aligned_dir:
                print("Aligned directory is empty, no faces will be converted!")
            elif len(self.input_aligned_dir) <= len(self.input_dir) / 3:
                print("Aligned directory contains an amount of images much less than the input, \
                        are you sure this is the right directory?")

    def get_frame_ranges(self):
        """ split out the frame ranges and parse out 'min' and 'max' values """
        minmax = {"min": 0, # never any frames less than 0
                  "max": float("inf")}
        rng = [tuple(map(lambda q: minmax[q] if q in minmax.keys() else int(q), v.split("-")))
               for v in self.arguments.frame_ranges]
        return rng

    def load_model(self):
        """ Load the model requested for conversion """
        model_name = self.arguments.trainer
        model_dir = get_folder(self.arguments.model_dir)
        num_gpus = self.arguments.gpus

        model = PluginLoader.get_model(model_name)(model_dir, num_gpus)

        if not model.load(self.arguments.swap_model):
            print("Model Not Found! A valid model must be provided to continue!")
            exit(1)

        return model

    def load_converter(self, model):
        """ Load the requested converter for conversion """
        args = self.arguments
        conv = args.converter

        converter = PluginLoader.get_converter(conv)(model.converter(False),
                                                     trainer=args.trainer,
                                                     blur_size=args.blur_size,
                                                     seamless_clone=args.seamless_clone,
                                                     sharpen_image=args.sharpen_image,
                                                     mask_type=args.mask_type,
                                                     erosion_kernel_size=args.erosion_kernel_size,
                                                     match_histogram=args.match_histogram,
                                                     smooth_mask=args.smooth_mask,
                                                     avg_color_adjust=args.avg_color_adjust)
        return converter

    def check_skipframe(self, filename):
        """ Check whether frame is to be skipped """
        try:
            idx = int(self.imageidxre.findall(filename)[0])
            return not any(map(lambda b: b[0] <= idx <= b[1], self.frame_ranges))
        except:
            return False

    def check_skipface(self, filename, face_idx):
        """ Check whether face is to be skipped """
        face_name = "{}_{}{}".format(Path(filename).stem, face_idx, Path(filename).suffix)
        face_file = Path(self.arguments.input_aligned_dir) / Path(face_name)
        return face_file not in self.input_aligned_dir

    def convert(self, converter, item):
        """ Apply the conversion transferring faces onto frames """
        try:
            (filename, image, faces) = item

            skip = self.check_skipframe(filename)
            if self.arguments.discard_frames and skip:
                return

            if not skip: # process frame as normal
                for idx, face in faces:
                    if self.input_aligned_dir is not None and self.check_skipface(filename, idx):
                        print("face {} for frame {} was deleted, skipping".format(
                            idx, os.path.basename(filename)))
                        continue
                    # Check for image rotations and rotate before mapping face
                    if face.r != 0:
                        height, width = image.shape[:2]
                        image = rotate_image(image, face.r)
                        image = converter.patch_image(
                            image,
                            face,
                            64 if "128" not in self.arguments.trainer else 128)
                        # TODO: This switch between 64 and 128 is a hack for now.
                        # We should have a separate cli option for size
                        image = rotate_image(image,
                                             face.r * -1,
                                             rotated_width=width,
                                             rotated_height=height)
                    else:
                        image = converter.patch_image(
                            image,
                            face,
                            64 if "128" not in self.arguments.trainer else 128)
                        # TODO: This switch between 64 and 128 is a hack for now.
                        # We should have a separate cli option for size

            output_file = get_folder(self.output_dir) / Path(filename).name
            cv2.imwrite(str(output_file), image)
        except Exception as err:
            print("Failed to convert image: {}. Reason: {}".format(filename, err))

    def prepare_images(self):
        """ Prepare the images for conversion """
        self.read_alignments()
        is_have_alignments = self.have_alignments()
        for filename in tqdm(self.read_directory()):
            image = cv2.imread(filename)

            if is_have_alignments:
                if self.have_face(filename):
                    faces = self.get_faces_alignments(filename, image)
                else:
                    tqdm.write("no alignment found for {}, skipping".format(
                        os.path.basename(filename)))
                    continue
            else:
                faces = self.get_faces(image)
            yield filename, image, faces
