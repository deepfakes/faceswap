#!/usr/bin python3
""" Directory Processing Tasks """

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from lib.detect_blur import is_blurry
from lib import Serializer
from lib.utils import get_folder, get_image_paths, rotate_image
from plugins.PluginLoader import PluginLoader

# DLIB is a GPU Memory hog, so the following modules should only be imported when required
def import_detect_faces():
    """ Import the faces_detect module only when it is required """
    from lib.faces_detect import detect_faces
    return detect_faces

def import_detected_face():
    """ Import the faces_detect module only when it is required """
    from lib.faces_detect import DetectedFace
    return DetectedFace

def import_face_filter():
    """ Import the FaceFilter module only when it is required """
    from lib.FaceFilter import FaceFilter
    return FaceFilter

class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))

class FullHelpArgumentParser(argparse.ArgumentParser):
    """
    Identical to the built-in argument parser, but on error
    it prints full help message instead of just usage information
    """
    def error(self, message):
        self.print_help(sys.stderr)
        args = {"prog": self.prog, "message": message}
        self.exit(2, "%(prog)s: error: %(message)s\n" % args)

class DirectoryArgs(object):
    """ This class is used as a parent class to capture arguments that
        will be used in both the extract and convert process.

        Arguments that can be used in both of these processes should be
        placed here, but no further processing should be done. This class
        just captures arguments """

    def __init__(self, subparser, command, description="default"):

        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        self.parser = self.create_parser(subparser, command, description)
        self.parse_arguments()

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-i", "--input-dir"),
                              "action": FullPaths,
                              "dest": "input_dir",
                              "default": "input",
                              "help": "Input directory. A directory containing the files "
                                      "you wish to process. Defaults to 'input'"})
        argument_list.append({"opts": ("-o", "--output-dir"),
                              "action": FullPaths,
                              "dest": "output_dir",
                              "default": "output",
                              "help": "Output directory. This is where the converted files will "
                                      "be stored. Defaults to 'output'"})
        argument_list.append({"opts": ("--serializer", ),
                              "type": str.lower,
                              "dest": "serializer",
                              "choices": ("yaml", "json", "pickle"),
                              "help": "serializer for alignments file"})
        argument_list.append({"opts": ("--alignments", ),
                              "type": str,
                              "dest": "alignments_path",
                              "help": "optional path to alignments file."})
        argument_list.append({"opts": ("-v", "--verbose"),
                              "action": "store_true",
                              "dest": "verbose",
                              "default": False,
                              "help": "Show verbose output"})
        return argument_list

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        # Override this for custom arguments
        argument_list = []
        return argument_list

    @staticmethod
    def create_parser(subparser, command, description):
        """ Create a directory processing parser
            Overide for extract/convert process """
        parser = subparser.add_parser(
            command,
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )
        return parser

    def parse_arguments(self):
        """ Parse the arguments passed in from argparse """
        for option in self.argument_list:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key != "opts"}
            self.parser.add_argument(*args, **kwargs)

        self.add_optional_arguments()

    def add_optional_arguments(self):
        """ Add any optional arguments passed in from argparse """
        for option in self.optional_arguments:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key != "opts"}
            self.parser.add_argument(*args, **kwargs)

class FSProcess(object):
    """ Class for directory processing.

        This class contains functions that are used in both the extract
        and convert process and in those processes only. If a function
        doesn't meet that criteria then it shouldn't be placed here """

    def __init__(self, arguments):
        print('Starting, this may take a while...')

        self.args = arguments
        self.output_dir = get_folder(self.args.output_dir)

        self.images = Images(self.args)
        self.faces = Faces(self.args)
        self.alignments = Alignments(self.args, self.faces)

        self.extractor = None
        self.export_face = True

    @staticmethod
    def process():
        """ Override with Extract/Convert process """
        raise NotImplementedError

    # EXTRACT ALIGNMENTS FROM IMAGES #
    @staticmethod
    def load_extractor():
        """ Load the requested extractor for extraction """
        # TODO Pass as argument
        extractor_name = "Align"
        extractor = PluginLoader.get_extractor(extractor_name)()

        return extractor

    def extract_face_alignments(self, filename):
        """ Read an image from a file """
        try:
            image = cv2.imread(filename)
            return filename, self.handle_image(image, filename)
        except Exception as err:
            if self.args.verbose:
                print("Failed to extract from image: {}. Reason: {}".format(filename, err))
            return filename, []

    def handle_image(self, image, filename):
        """ Attempt to extract faces from an image """
        faces = self.faces.get_faces(image)
        process_faces = [(idx, face) for idx, face in faces]

        # Run image rotator if requested and no faces found
        if self.images.rotation_angles and not process_faces:
            process_faces, image = self.rotate_frame(image)

        return [self.process_single_face(idx, face, filename, image)
                for idx, face in process_faces]

    def process_single_face(self, idx, face, filename, image):
        """ Perform processing on found faces """
        output_file = self.output_dir / Path(filename).stem if self.export_face else None

        # Draws landmarks for debug
        if self.args.debug_landmarks:
            self.faces.draw_landmarks_on_face(face, image)

        resized_image, t_mat = self.extractor.extract(image,
                                                      face,
                                                      256,
                                                      self.args.align_eyes)

        # Detect blurry images
        if self.args.blur_thresh is not None:
            blurry_file = self.detect_blurry_faces(face, t_mat, resized_image, filename)
            output_file = blurry_file if blurry_file else output_file

        if self.export_face:
            cv2.imwrite("{}_{}{}".format(str(output_file),
                                         str(idx),
                                         Path(filename).suffix),
                        resized_image)

        return {"r": face.r,
                "x": face.x,
                "w": face.w,
                "y": face.y,
                "h": face.h,
                "landmarksXY": face.landmarksAsXY()}

    def rotate_frame(self, image):
        """ Rotate the image to extract more faces if requested """
        for angle in self.images.rotation_angles:
            rotated_image = rotate_image(image, angle)
            faces = self.faces.get_faces(rotated_image, rotation=angle)
            rotated_faces = [(idx, face) for idx, face in faces]
            if rotated_faces and self.args.verbose:
                print("found face(s) by rotating image {} degrees".format(angle))
            if rotated_faces:
                break
            return rotated_faces, rotated_image

    def detect_blurry_faces(self, face, t_mat, resized_image, filename):
        """ Detect and move blurry face """
        blurry_file = None
        aligned_landmarks = self.extractor.transform_points(face.landmarksAsXY(), t_mat, 256, 48)
        feature_mask = self.extractor.get_feature_mask(aligned_landmarks / 256, 256, 48)
        feature_mask = cv2.blur(feature_mask, (10, 10))
        isolated_face = cv2.multiply(feature_mask, resized_image.astype(float)).astype(np.uint8)
        blurry, focus_measure = is_blurry(isolated_face, self.args.blur_thresh)

        if blurry:
            print("{}'s focus measure of {} was below the blur threshold, "
                  "moving to \"blurry\"".format(Path(filename).stem, focus_measure))
            blurry_file = get_folder(Path(self.output_dir) / Path("blurry")) / \
                Path(filename).stem
        return blurry_file

    def finalize(self):
        """ Finalize the image processing """
        print("-------------------------")
        print("Images found:        {}".format(self.images.images_found))
        print("Faces detected:      {}".format(self.faces.num_faces_detected))
        print("-------------------------")

        if self.faces.verify_output:
            print("Note:")
            print("Multiple faces were detected in one or more pictures.")
            print("Double check your results.")
            print("-------------------------")
        print("Done!")

class Images(object):
    """ Holds the full frames/images """
    def __init__(self, arguments):
        self.args = arguments
        self.rotation_angles = self.get_rotation_angles()
        self.already_processed = self.get_already_processed()
        self.input_images = self.get_input_images()

        self.images_found = 0

    def get_rotation_angles(self):
        """ Set the rotation angles. Includes backwards compatibility for the 'on'
            and 'off' options:
                - 'on' - increment 90 degrees
                - 'off' - disable """
        try:
            if not self.args.rotate_images or self.args.rotate_images == "off":
                rotation_angles = None
            elif self.rotation_angles == "on":
                rotation_angles = range(90, 360, 90)
            else:
                rotation_angles = [int(angle)
                                   for angle in self.args.rotate_images.split(",")]
                if len(rotation_angles) == 1:
                    rotation_step_size = rotation_angles[0]
                    rotation_angles = range(rotation_step_size, 360, rotation_step_size)
                elif len(rotation_angles) > 1:
                    rotation_angles = rotation_angles
                else:
                    rotation_angles = None
            return rotation_angles
        except AttributeError:
            pass

    def get_already_processed(self):
        """ Return the images that already exist in the output directory """
        try:
            output_dir = None
            if not os.path.exists(self.args.output_dir):
                print("Output directory {} not found.".format(self.args.input_dir))
                exit(1)

            print("Output Directory: {}".format(self.args.output_dir))

            if self.args.skip_existing:
                output_dir = get_image_paths(self.args.output_dir)
            return output_dir
        except AttributeError:
            pass

    def get_input_images(self):
        """ Return the list of images that are to be processed """
        if not os.path.exists(self.args.input_dir):
            print("Input directory {} not found.".format(self.args.input_dir))
            exit(1)

        print("Input Directory: {}".format(self.args.input_dir))

        if self.args.skip_existing:
            input_images = get_image_paths(self.args.input_dir, self.already_processed)
            print("Excluding %s files" % len(self.already_processed))
        else:
            input_images = get_image_paths(self.args.input_dir)
        return input_images

    def read_directory(self):
        """ Return number of images from directory for tqdm """
        self.images_found = len(self.input_images)
        return self.input_images

class Faces(object):
    """ Holds the faces """
    def __init__(self, arguments):
        self.args = arguments
        self.filter = self.load_face_filter()

        self.faces_detected = dict()
        self.num_faces_detected = 0
        self.verify_output = False

    def have_face(self, filename):
        """ return path of images that have faces """
        return os.path.basename(filename) in self.faces_detected

    def load_face_filter(self):
        """ Load faces to filter out of images """
        facefilter = None
        filter_files = [self.set_face_filter(filter_type)
                        for filter_type in ('filter', 'nfilter')]

        if any(filters for filters in filter_files):
            FaceFilter = import_face_filter()
            facefilter = FaceFilter(filter_files[0], filter_files[1], self.args.ref_threshold)
        return facefilter

    def set_face_filter(self, filter_list):
        """ Set the required filters """
        filter_files = list()
        filter_args = getattr(self.args, filter_list)
        if filter_args:
            print("{}: {}".format(filter_list.title(), filter_args))
            filter_files = filter_args
            if not isinstance(filter_args, list):
                filter_files = [filter_args]
            filter_files = list(filter(lambda fnc: Path(fnc).exists(), filter_files))
        return filter_files

    def get_faces(self, image, rotation=0):
        """ Extract the faces from an image """
        detect_faces = import_detect_faces()
        faces_count = 0
        faces = detect_faces(image, self.args.detector, self.args.verbose, rotation)

        for face in faces:
            if self.filter is not None and not self.filter.check(face):
                if self.args.verbose:
                    print("Skipping not recognized face!")
                continue
            yield faces_count, face

            self.num_faces_detected += 1
            faces_count += 1

        if faces_count > 1 and self.args.verbose:
            self.verify_output = True

    def get_faces_alignments(self, filename, image):
        """ Retrieve the face alignments from an image """
        DetectedFace = import_detected_face()
        faces_count = 0
        faces = self.faces_detected[os.path.basename(filename)]
        for rawface in faces:
            face = DetectedFace(**rawface)
            # Rotate the image if necessary
            if face.r != 0:
                image = rotate_image(image, face.r)
            face.image = image[face.y : face.y + face.h, face.x : face.x + face.w]
            if self.filter and not self.filter.check(face):
                if self.args.verbose:
                    print("Skipping not recognized face!")
                continue

            yield faces_count, face
            self.num_faces_detected += 1
            faces_count += 1
        if faces_count > 1 and self.args.verbose:
            print("Note: Found more than one face in an image! File: %s" % filename)
            self.verify_output = True

    @staticmethod
    def draw_landmarks_on_face(face, image):
        """ Draw debug landmarks on extracted face """
        for (pos_x, pos_y) in face.landmarksAsXY():
            cv2.circle(image, (pos_x, pos_y), 2, (0, 0, 255), -1)

class Alignments(object):
    """ Holds processes pertaining to the alignments file """
    def __init__(self, arguments, faces_detected):
        self.args = arguments
        self.faces_detected = faces_detected

        self.serializer = self.get_serializer()

    def get_serializer(self):
        """ Set the serializer to be used for loading and saving alignments """
        if not self.args.serializer and self.args.alignments_path:
            ext = os.path.splitext(self.args.alignments_path)[-1]
            serializer = Serializer.get_serializer_fromext(ext)
            print(serializer, self.args.alignments_path)
        else:
            serializer = Serializer.get_serializer(self.args.serializer or "json")
        print("Using {} serializer".format(self.serializer.ext))
        return serializer

    def get_alignments_path(self):
        """ Return the path to alignments file """
        if self.args.alignments_path:
            alignfile = self.args.alignments_path
        else:
            alignfile = os.path.join(str(self.args.input_dir),
                                     "alignments.{}".format(self.serializer.ext))
        print("Alignments filepath: %s" % alignfile)
        return alignfile

    def read_alignments(self):
        """ Read the serialized alignments file """
        alignfile = self.get_alignments_path()
        try:
            with open(alignfile, self.serializer.roptions) as align:
                self.faces_detected = self.serializer.unmarshal(align.read())
        except Exception as err:
            print("{} not read!".format(alignfile))
            print(str(err))
            self.faces_detected = dict()

    def write_alignments(self):
        """ Write the serialized alignments file """
        alignfile = self.get_alignments_path()

        if self.args.skip_existing:
            self.load_skip_alignments(alignfile)

        try:
            print("Writing alignments to: {}".format(alignfile))
            with open(alignfile, self.serializer.woptions) as align:
                align.write(self.serializer.marshal(self.faces_detected))
        except Exception as err:
            print("{} not written!".format(alignfile))
            print(str(err))
            self.faces_detected = dict()

    def load_skip_alignments(self, alignfile):
        """ Load existing alignments if skipping existing images """
        if self.have_alignments():
            with open(alignfile, self.serializer.roptions) as inf:
                data = self.serializer.unmarshal(inf.read())
                for key, val in data.items():
                    self.faces_detected[key] = val
        else:
            print("Existing alignments file '%s' not found." % alignfile)

    def have_alignments(self):
        """ Check if an alignments file exists """
        alignfile = self.get_alignments_path()
        return os.path.exists(alignfile)
