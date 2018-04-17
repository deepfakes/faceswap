#!/usr/bin python3
""" The script to run the extract process of faceswap """

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lib.cli import DirectoryArgs, FSProcess, rotate_image

from lib.detect_blur import is_blurry
from lib.multithreading import pool_process
from lib.utils import get_folder
from plugins.PluginLoader import PluginLoader


class ExtractArgs(DirectoryArgs):
    """ Class to parse the command line arguments for extraction. 
        Inherits base options from lib.DirectoryArgs """

    @staticmethod
    def get_optional_arguments():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({"opts": ("-D", "--detector"),
                              "type": str,
                              # case sensitive because this is used to load a plugin.
                              "choices": ("hog", "cnn", "all"),
                              "default": "hog",
                              "help": "Detector to use. 'cnn' detects much more angles but will "
                                      "be much more resource intensive and may fail on large "
                                      "files."})
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
                              "help": "Reference image for the person you want to process. "
                                      "Should be a front portrait"})
        argument_list.append({"opts": ("-j", "--processes"),
                              "type": int,
                              "default": 1,
                              "help": "Number of processes to use."})
        argument_list.append({"opts": ("-s", "--skip-existing"),
                              "action": "store_true",
                              "dest": "skip_existing",
                              "default": False,
                              "help": "Skips frames already extracted."})
        argument_list.append({"opts": ("-dl", "--debug-landmarks"),
                              "action": "store_true",
                              "dest": "debug_landmarks",
                              "default": False,
                              "help": "Draw landmarks for debug."})
        argument_list.append({"opts": ("-r", "--rotate-images"),
                              "type": str,
                              "dest": "rotate_images",
                              "default": None,
                              "help": "If a face isn't found, rotate the images to try to "
                                      "find a face. Can find more faces at the cost of extraction "
                                      "speed. Pass in a single number to use increments of that "
                                      "size up to 360, or pass in a list of numbers to enumerate "
                                      "exactly what angles to check."})
        argument_list.append({"opts": ("-ae", "--align-eyes"),
                              "action": "store_true",
                              "dest": "align_eyes",
                              "default": False,
                              "help": "Perform extra alignment to ensure left/right eyes "
                                      "lie at the same height"})
        argument_list.append({"opts": ("-bt", "--blur-threshold"),
                              "type": int,
                              "dest": "blur_thresh",
                              "default": None,
                              "help": "Automatically discard images blurrier than the specified "
                                      "threshold. Discarded images are moved into a \"blurry\" "
                                      "sub-folder. Lower values allow more blur"})
        return argument_list

    def create_parser(self, subparser, command, description):
        """ Create the extract parser """
        self.optional_arguments = self.get_optional_arguments()
        self.parser = subparser.add_parser(
            command,
            help="Extract the faces from a pictures.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground")

class Extract(FSProcess):
    """ The extract process. Inherits from cli.FSProcess, including the additional
        classes: images, faces, alignments """
    def __init__(self, arguments):
        FSProcess.__init__(self, arguments)

        self.opts = OptionalActions(self)
        self.extractor = self.load_extractor()

    @staticmethod
    def load_extractor():
        """ Load the requested extractor for extraction """
        # TODO Pass as argument
        extractor_name = "Align"
        extractor = PluginLoader.get_extractor(extractor_name)()

        return extractor

    def process(self):
        """ Perform the extraction process """
        processes = self.args.processes

        if processes != 1:
            self.multi_process(processes)
        else:
            self.single_process()

        self.alignments.write_alignments(self.faces.faces_detected)

    def multi_process(self, processes):
        """ Run extraction in a multiple processes """
        files = list(self.images.read_directory())
        for filename, faces in tqdm(pool_process(self.process_files, files, processes=processes),
                                    total=len(files)):
            self.faces.num_faces_detected += 1
            self.faces.faces_detected[os.path.basename(filename)] = faces

    def single_process(self):
        """ Run extraction in a single process """
        for filename in tqdm(self.images.read_directory()):
            filename, faces = self.process_files(filename)
            self.faces.faces_detected[os.path.basename(filename)] = faces

    def process_files(self, filename):
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
        process_faces, image = self.opts.rotate_frame(process_faces, image)

        return [self.process_single_face(idx, face, filename, image)
                for idx, face in process_faces]

    def process_single_face(self, idx, face, filename, image):
        """ Perform processing on found faces """
        output_file = self.output_dir / Path(filename).stem

        # Draws landmarks for debug
        if self.args.debug_landmarks:
            self.opts.draw_landmarks_on_face(face, image)

        resized_image, t_mat = self.extractor.extract(image,
                                                      face,
                                                      256,
                                                      self.args.align_eyes)

        # Detect blurry images
        if self.args.blur_thresh is not None:
            blurry_file = self.opts.detect_blurry_faces(face, t_mat, resized_image, filename)
            output_file = blurry_file if blurry_file else output_file

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

class OptionalActions(object):
    """ Process the optional actions for extract Any actions that 
        are performed because of an optional value should be placed
        here """
    def __init__(self, extract):
        self.args = extract.args

        self.images = extract.images
        self.faces = extract.faces
        self.alignments = extract.alignments

        self.extractor = extract.extractor
        self.output_dir = extract.output_dir

    def rotate_frame(self, process_faces, image):
        """ Rotate the image to extract more faces if requested """
        if self.images.rotation_angles is not None and not process_faces:
            process_faces, image = self.image_rotator(image)
        return process_faces, image

    def get_rotated_image_faces(self, image, angle):
        """ Rotate an image and return the faces with the rotated image """
        rotated_image = rotate_image(image, angle)
        faces = self.faces.get_faces(rotated_image, rotation=angle)
        rotated_faces = [(idx, face) for idx, face in faces]
        return rotated_faces, rotated_image

    def image_rotator(self, image):
        """ rotates the image through rotation_angles to try to find a face """
        for angle in self.images.rotation_angles:
            rotated_faces, rotated_image = self.get_rotated_image_faces(image, angle)
            if rotated_faces:
                if self.args.verbose:
                    print("found face(s) by rotating image {} degrees".format(angle))
                break
        return rotated_faces, rotated_image

    @staticmethod
    def draw_landmarks_on_face(face, image):
        """ Draw debug landmarks on extracted face """
        for (pos_x, pos_y) in face.landmarksAsXY():
            cv2.circle(image, (pos_x, pos_y), 2, (0, 0, 255), -1)

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

class ExtractTrainingData(object):
    """ TODO: Change this, it shouldn't be a class. 
        It's here to keep compatibility during rewrite """
    def __init__(self, subparser, command, description):
        args = ExtractArgs(subparser, command, description).parser.arguments

        self.process = Extract(args)
        self.process.process()
        self.process.finalize()
