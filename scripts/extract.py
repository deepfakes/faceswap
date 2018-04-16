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
        extractor_name = "Align" # TODO Pass as argument
        self.extractor = PluginLoader.get_extractor(extractor_name)()
        processes = self.arguments.processes
        try:
            if processes != 1:
                files = list(self.read_directory())
                for filename, faces in tqdm(pool_process(self.process_files,
                                                         files,
                                                         processes=processes),
                                            total=len(files)):
                    self.num_faces_detected += 1
                    self.faces_detected[os.path.basename(filename)] = faces
            else:
                for filename in tqdm(self.read_directory()):
                    try:
                        image = cv2.imread(filename)
                        self.faces_detected[os.path.basename(filename)] = \
                            self.handle_image(image, filename)
                    except Exception as err:
                        if self.arguments.verbose:
                            print("Failed to extract from image: {}. Reason: {}".format(filename,
                                                                                        err))
        finally:
            self.write_alignments()

    def process_files(self, filename):
        """ Read an image from a file """
        try:
            image = cv2.imread(filename)
            return filename, self.handle_image(image, filename)
        except Exception as err:
            if self.arguments.verbose:
                print("Failed to extract from image: {}. Reason: {}".format(filename, err))
        return filename, []

    def get_rotated_image_faces(self, image, angle):
        """ Rotate an image and return the faces with the rotated image """
        rotated_image = rotate_image(image, angle)
        faces = self.get_faces(rotated_image, rotation=angle)
        rotated_faces = [(idx, face) for idx, face in faces]
        return rotated_faces, rotated_image

    def image_rotator(self, image):
        """ rotates the image through rotation_angles to try to find a face """
        for angle in self.rotation_angles:
            rotated_faces, rotated_image = self.get_rotated_image_faces(image, angle)
            if rotated_faces:
                if self.arguments.verbose:
                    print("found face(s) by rotating image {} degrees".format(angle))
                break
        return rotated_faces, rotated_image

    def handle_image(self, image, filename):
        """ Attempt to extract faces from an image """
        faces = self.get_faces(image)
        process_faces = [(idx, face) for idx, face in faces]

        # Run image rotator if requested and no faces found
        if self.rotation_angles is not None and process_faces:
            process_faces, image = self.image_rotator(image)

        rvals = []
        for idx, face in process_faces:
            # Draws landmarks for debug
            if self.arguments.debug_landmarks:
                for (pos_x, pos_y) in face.landmarksAsXY():
                    cv2.circle(image, (pos_x, pos_y), 2, (0, 0, 255), -1)

            resized_image, t_mat = self.extractor.extract(image,
                                                          face,
                                                          256,
                                                          self.arguments.align_eyes)
            output_file = get_folder(self.output_dir) / Path(filename).stem

            # Detect blurry images
            if self.arguments.blur_thresh is not None:
                aligned_landmarks = self.extractor.transform_points(face.landmarksAsXY(),
                                                                    t_mat,
                                                                    256,
                                                                    48)
                feature_mask = self.extractor.get_feature_mask(aligned_landmarks / 256, 256, 48)
                feature_mask = cv2.blur(feature_mask, (10, 10))
                isolated_face = cv2.multiply(feature_mask,
                                             resized_image.astype(float)).astype(np.uint8)
                blurry, focus_measure = is_blurry(isolated_face, self.arguments.blur_thresh)
                # print("{} focus measure: {}".format(Path(filename).stem, focus_measure))
                # cv2.imshow("Isolated Face", isolated_face)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if blurry:
                    print("{}'s focus measure of {} was below the blur threshold, "
                          "moving to \"blurry\"".format(Path(filename).stem, focus_measure))
                    output_file = get_folder(Path(self.output_dir) / \
                        Path("blurry")) / Path(filename).stem

            cv2.imwrite("{}_{}{}".format(str(output_file),
                                         str(idx),
                                         Path(filename).suffix),
                        resized_image)
            face_info = {"r": face.r,
                         "x": face.x,
                         "w": face.w,
                         "y": face.y,
                         "h": face.h,
                         "landmarksXY": face.landmarksAsXY()}
            rvals.append(face_info)
        return rvals
