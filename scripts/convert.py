#!/usr/bin python3
""" The script to run the convert process of faceswap """

import re
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from lib.cli import DirectoryArgs, FSProcess, FullPaths
from lib.utils import BackgroundGenerator, get_folder, get_image_paths, rotate_image

from plugins.PluginLoader import PluginLoader

class ConvertArgs(DirectoryArgs):
    """ Class to parse the command line arguments for conversion.
        Inherits base options from lib.DirectoryArgs """

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-m", "--model-dir"),
                              "action": FullPaths,
                              "dest": "model_dir",
                              "default": "models",
                              "help": "Model directory. A directory containing the trained model "
                                      "you wish to process. Defaults to 'models'"})
        argument_list.append({"opts": ("-a", "--input-aligned-dir"),
                              "action": FullPaths,
                              "dest": "input_aligned_dir",
                              "default": None,
                              "help": "Input \"aligned directory\". A directory that should "
                                      "contain the aligned faces extracted from the input files. "
                                      "If you delete faces from this folder, they'll be skipped "
                                      "during conversion. If no aligned dir is specified, all "
                                      "faces will be converted."})
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
                              "help": "Detector to use. 'cnn' detects much more angles but "
                                      "will be much more resource intensive and may fail "
                                      "on large files."})
        argument_list.append({"opts": ("-fr", "--frame-ranges"),
                              "nargs": "+",
                              "type": str,
                              "help": "frame ranges to apply transfer to e.g. For frames 10 to "
                                      "50 and 90 to 100 use --frame-ranges 10-50 90-100. Files "
                                      "must have the frame-number as the last number in the "
                                      "name!"})
        argument_list.append({"opts": ("-d", "--discard-frames"),
                              "action": "store_true",
                              "dest": "discard_frames",
                              "default": False,
                              "help": "When used with --frame-ranges discards frames that are "
                                      "not processed instead of writing them out unchanged."})
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
                              "help": "Reference image for the persons you do not want to "
                                      "process. Should be a front portrait"})
        argument_list.append({"opts": ("-f", "--filter"),
                              "type": str,
                              "dest": "filter",
                              "nargs": "+",
                              "default": "filter.jpg",
                              "help": "Reference images for the person you want to process. "
                                      "Should be a front portrait"})
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
                              "help": "Erosion kernel size. (Masked converter only). Positive "
                                      "values apply erosion which reduces the edge of the "
                                      "swapped face. Negative values apply dilation which allows "
                                      "the swapped face to cover more space."})
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
                              "help": "Use Sharpen Image - bsharpen = Box Blur, gsharpen = "
                                      "Gaussian Blur (Masked converter only)"})
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

    def create_parser(self, subparser, command, description):
        self.optional_arguments = self.get_optional_arguments()
        self.parser = subparser.add_parser(
            command,
            help="Convert a source image to a new one with the face swapped.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground")

class Convert(FSProcess):
    """ The convert process. Inherits from cli.FSProcess, including the additional
        classes: images, faces, alignments """
    def __init__(self, arguments):
        FSProcess.__init__(self, arguments)

        self.opts = OptionalActions(self)

    def process(self):
        """ Original & LowMem models go with Adjust or Masked converter
            Note: GAN prediction outputs a mask + an image, while other predicts only an image """
        model = self.load_model()
        converter = self.load_converter(model)

        batch = BackgroundGenerator(self.prepare_images(), 1)

        for item in batch.iterator():
            self.convert(converter, item)

    def load_model(self):
        """ Load the model requested for conversion """
        model_name = self.args.trainer
        model_dir = get_folder(self.args.model_dir)
        num_gpus = self.args.gpus

        model = PluginLoader.get_model(model_name)(model_dir, num_gpus)

        if not model.load(self.args.swap_model):
            print("Model Not Found! A valid model must be provided to continue!")
            exit(1)

        return model

    def load_converter(self, model):
        """ Load the requested converter for conversion """
        args = self.args
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

    def prepare_images(self):
        """ Prepare the images for conversion """
        filename = ""
        have_alignments = self.alignments.have_alignments()
        self.alignments.read_alignments()
        for filename in tqdm(self.images.read_directory()):
            image = cv2.imread(filename)

            if have_alignments:
                faces = self.check_alignments(filename, image)
            else:
                faces = self.faces.get_faces(image)

            if not faces:
                continue

            yield filename, image, faces

    def check_alignments(self, filename, image):
        """ If we have alignments file, but no alignments for this face, skip it """
        faces = None
        if self.faces.have_face(filename):
            faces = self.faces.get_faces_alignments(filename, image)
        else:
            tqdm.write("no alignment found for {}, skipping".format(os.path.basename(filename)))
        return faces

    def convert(self, converter, item):
        """ Apply the conversion transferring faces onto frames """
        try:
            (filename, image, faces) = item
            skip = self.opts.check_skipframe(filename)

            if skip == "discard":
                return
            elif not skip:
                image = (self.convert_one_face(converter,
                                               (filename, image, idx, face))
                         for idx, face in faces)

            output_file = get_folder(self.output_dir) / Path(filename).name
            cv2.imwrite(str(output_file), image)
        except Exception as err:
            print("Failed to convert image: {}. Reason: {}".format(filename, err))

    def convert_one_face(self, converter, imagevars):
        """ Perform the conversion on the given frame for a single face """
        (filename, image, idx, face) = imagevars

        if self.opts.check_skipface(filename, idx):
            return image

        image = self.opts.rotate_image(image, face.r)
        # TODO: This switch between 64 and 128 is a hack for now.
        # We should have a separate cli option for size
        image = converter.patch_image(image,
                                      face,
                                      64 if "128" not in self.args.trainer else 128)
        image = self.opts.rotate_image(image, face.r, reverse=True)
        return image

class OptionalActions(object):
    """ Process the optional actions for convert """

    def __init__(self, convertimage):
        self.args = convertimage.arguments
        self.input_dir = convertimage.input_dir

        self.faces_to_swap = self.get_aligned_directory()

        self.frame_ranges = self.get_frame_ranges()
        self.imageidxre = re.compile(r"(\d+)(?!.*\d)")

        self.rotation_height = 0
        self.rotation_width = 0

    ### SKIP ALIGNMENTS ###
    def get_aligned_directory(self):
        """ Check for the existence of an aligned directory for identifying
            which faces in the target frames should be swapped """
        faces_to_swap = None
        input_aligned_dir = self.args.input_aligned_dir

        if input_aligned_dir is None:
            print("Aligned directory not specified. All faces listed in the alignments file \
                   will be converted.")
        elif not os.path.isdir(input_aligned_dir):
            print("Aligned directory not found. All faces listed in the alignments file \
                   will be converted.")
        else:
            faces_to_swap = [Path(path) for path in get_image_paths(input_aligned_dir)]
            if not faces_to_swap:
                print("Aligned directory is empty, no faces will be converted!")
            elif len(faces_to_swap) <= len(self.input_dir) / 3:
                print("Aligned directory contains an amount of images much less than the input, \
                        are you sure this is the right directory?")
        return faces_to_swap

    ### SKIP FRAME RANGES ###
    def get_frame_ranges(self):
        """ split out the frame ranges and parse out 'min' and 'max' values """
        if not self.args.frame_ranges:
            return None

        minmax = {"min": 0, # never any frames less than 0
                  "max": float("inf")}
        rng = [tuple(map(lambda q: minmax[q] if q in minmax.keys() else int(q), v.split("-")))
               for v in self.args.frame_ranges]
        return rng

    def check_skipframe(self, filename):
        """ Check whether frame is to be skipped """
        idx = int(self.imageidxre.findall(filename)[0])
        skipframe = not any(map(lambda b: b[0] <= idx <= b[1], self.frame_ranges))
        if skipframe and self.args.discard_frames:
            skipframe = "discard"
        return skipframe

    def check_skipface(self, filename, face_idx):
        """ Check whether face is to be skipped """
        if self.faces_to_swap is None:
            return False
        face_name = "{}_{}{}".format(Path(filename).stem, face_idx, Path(filename).suffix)
        face_file = Path(self.args.input_aligned_dir) / Path(face_name)
        skip_face = face_file not in self.faces_to_swap
        if skip_face:
            print("face {} for frame {} was deleted, skipping".format(
                face_idx, os.path.basename(filename)))
        return skip_face

    ### ROTATE IMAGES ###
    def rotate_image(self, image, rotation, reverse=False):
        """ Rotate the image forwards or backwards """
        if rotation != 0:
            if not reverse:
                self.rotation_height, self.rotation_width = image.shape[:2]
                image = rotate_image(image, rotation)
            else:
                image = rotate_image(image,
                                     rotation -1,
                                     rotated_width=self.rotation_width,
                                     rotated_height=self.rotation_height)
        return image

class ConvertImage(object):
    """ TODO: Change this, it shouldn't be a class. 
        It's here to keep compatibility during rewrite """
    def __init__(self, subparser, command, description):
        args = ConvertArgs(subparser, command, description).parser.arguments

        self.process = Convert(args)
        self.process.process()
        self.process.finalize()
