#!/usr/bin python3
""" The script to run the extract process of faceswap """

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from lib.cli import DirectoryProcessor, rotate_image
from lib.detect_blur import is_blurry
from lib.multithreading import pool_process
from lib.utils import get_folder
from plugins.PluginLoader import PluginLoader


class ExtractTrainingData(DirectoryProcessor):
    """ Class to parse the command line arguments for extraction and
        run the extract process """
    def __init__(self, subparser, command, description):
        DirectoryProcessor.__init__(subparser, command, description)
        self.extractor = None

    def create_parser(self, subparser, command, description):
        self.optional_arguments = self.get_optional_arguments()
        self.parser = subparser.add_parser(
            command,
            help="Extract the faces from a pictures.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground")

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

    def process(self):
        """ Perform the extraction process """
        self.extractor = self.load_extractor()
        processes = self.arguments.processes

        if processes != 1:
            self.multi_process(processes)
        else:
            self.single_process()

        self.write_alignments()

    @staticmethod
    def load_extractor():
        """ Load the requested extractor for extraction """
        # TODO Pass as argument
        extractor_name = "Align"
        extractor = PluginLoader.get_extractor(extractor_name)()

        return extractor

    def multi_process(self, processes):
        """ Run extraction in a multiple processes """
        files = list(self.read_directory())
        for filename, faces in tqdm(pool_process(self.process_files, files, processes=processes),
                                    total=len(files)):
            self.num_faces_detected += 1
            self.faces_detected[os.path.basename(filename)] = faces

    def single_process(self):
        """ Run extraction in a single process """
        for filename in tqdm(self.read_directory()):
            filename, faces = self.process_files(filename)
            self.faces_detected[os.path.basename(filename)] = faces

    def process_files(self, filename):
        """ Read an image from a file """
        try:
            image = cv2.imread(filename)
            return filename, self.handle_image(image, filename)
        except Exception as err:
            if self.arguments.verbose:
                print("Failed to extract from image: {}. Reason: {}".format(filename, err))
        return filename, []

    def handle_image(self, image, filename):
        """ Attempt to extract faces from an image """
        opts = OptionalActions(self)
        faces = self.get_faces(image)
        process_faces = [(idx, face) for idx, face in faces]

        # Run image rotator if requested and no faces found
        process_faces, image = opts.rotate_image(process_faces, image)

        return [self.process_single_face(opts, (idx, face), (filename, image))
                for idx, face in process_faces]

    def process_single_face(self, opts, facevars, imagevars):
        """ Perform processing on found faces """
        idx, face, = facevars
        filename, image = imagevars
        output_file = get_folder(self.output_dir) / Path(filename).stem

        # Draws landmarks for debug
        if self.arguments.debug_landmarks:
            opts.draw_landmarks_on_face(face, image)

        resized_image, t_mat = self.extractor.extract(image,
                                                      face,
                                                      256,
                                                      self.arguments.align_eyes)

        # Detect blurry images
        if self.arguments.blur_thresh is not None:
            blurry_file = opts.detect_blurry_faces(face, t_mat, resized_image, filename)
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
    """ Process the optional actions for extract """
    def __init__(self, extracttrainingdata):
        self.extract = extracttrainingdata

    def rotate_image(self, process_faces, image):
        """ Rotate the image to extract more faces if requested """
        if self.extract.rotation_angles is not None and not process_faces:
            process_faces, image = self.image_rotator(image)
        return process_faces, image

    def get_rotated_image_faces(self, image, angle):
        """ Rotate an image and return the faces with the rotated image """
        rotated_image = rotate_image(image, angle)
        faces = self.extract.get_faces(rotated_image, rotation=angle)
        rotated_faces = [(idx, face) for idx, face in faces]
        return rotated_faces, rotated_image

    def image_rotator(self, image):
        """ rotates the image through rotation_angles to try to find a face """
        for angle in self.extract.rotation_angles:
            rotated_faces, rotated_image = self.get_rotated_image_faces(image, angle)
            if rotated_faces:
                if self.extract.arguments.verbose:
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
        aligned_landmarks = self.extract.extractor.transform_points(face.landmarksAsXY(),
                                                                    t_mat,
                                                                    256,
                                                                    48)
        feature_mask = self.extract.extractor.get_feature_mask(aligned_landmarks / 256, 256, 48)
        feature_mask = cv2.blur(feature_mask, (10, 10))
        isolated_face = cv2.multiply(feature_mask, resized_image.astype(float)).astype(np.uint8)
        blurry, focus_measure = is_blurry(isolated_face, self.extract.arguments.blur_thresh)

        if blurry:
            print("{}'s focus measure of {} was below the blur threshold, "
                  "moving to \"blurry\"".format(Path(filename).stem, focus_measure))
            blurry_file = get_folder(Path(self.extract.output_dir) / Path("blurry")) / \
                Path(filename).stem
        return blurry_file
