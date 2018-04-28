#!/usr/bin python3
""" Holds the classes for the 3 main Faceswap 'media' objects for input (extract)
    and output (convert) tasks. Those being:
            Images
            Faces
            Alignments"""

import os
from pathlib import Path

import cv2
import numpy as np

from lib.detect_blur import is_blurry
from lib import Serializer
from lib.faces_detect import detect_faces, DetectedFace
from lib.FaceFilter import FaceFilter
from lib.utils import get_folder, get_image_paths, set_system_verbosity
from plugins.PluginLoader import PluginLoader

class Utils(object):
    """ Holds utility functions that are required by more than one media
        object """

    @staticmethod
    def set_verbosity(verbose):
        """ Set the system output verbosity """
        lvl = '0' if verbose else '2'
        set_system_verbosity(lvl)

    @staticmethod
    def rotate_image_by_angle(image, angle, rotated_width=None, rotated_height=None):
        """ Rotate an image by a given angle. From:
            https://stackoverflow.com/questions/22041699
            This is required by both Faces and Images so placed here for now """
        height, width = image.shape[:2]
        image_center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, -1.*angle, 1.)
        if rotated_width is None or rotated_height is None:
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            if rotated_width is None:
                rotated_width = int(height*abs_sin + width*abs_cos)
            if rotated_height is None:
                rotated_height = int(height*abs_cos + width*abs_sin)
        rotation_matrix[0, 2] += rotated_width/2 - image_center[0]
        rotation_matrix[1, 2] += rotated_height/2 - image_center[1]
        return cv2.warpAffine(image, rotation_matrix, (rotated_width, rotated_height))

    @staticmethod
    def cv2_read_write(action, filename, image=None):
        """ Read or write an image using cv2 """
        if action == 'read':
            image = cv2.imread(filename)
        if action == 'write':
            cv2.imwrite(filename, image)
        return image

    @staticmethod
    def finalize(images_found, num_faces_detected, verify_output):
        """ Finalize the image processing """
        print("-------------------------")
        print("Images found:        {}".format(images_found))
        print("Faces detected:      {}".format(num_faces_detected))
        print("-------------------------")

        if verify_output:
            print("Note:")
            print("Multiple faces were detected in one or more pictures.")
            print("Double check your results.")
            print("-------------------------")

        images_found = 0
        num_faces_detected = 0
        print("Done!")
        return images_found, num_faces_detected

class Images(object):
    """ Holds the full frames/images """
    def __init__(self, arguments):
        self.args = arguments
        self.rotation_angles = self.get_rotation_angles()
        self.already_processed = self.get_already_processed()
        self.input_images = self.get_input_images()
        self.images_found = len(self.input_images)

        self.rotation_width = 0
        self.rotation_height = 0

    def get_rotation_angles(self):
        """ Set the rotation angles. Includes backwards compatibility for the 'on'
            and 'off' options:
                - 'on' - increment 90 degrees
                - 'off' - disable
                - 0 is prepended to the list, as whatever happens, we want to scan the image
                  in it's upright state """
        rotation_angles = [0]

        if (not hasattr(self.args, 'rotate_images')
                or not self.args.rotate_images
                or self.args.rotate_images == "off"):
            return rotation_angles

        if self.args.rotate_images == "on":
            rotation_angles.extend(range(90, 360, 90))
        else:
            passed_angles = [int(angle)
                             for angle in self.args.rotate_images.split(",")]
            if len(passed_angles) == 1:
                rotation_step_size = passed_angles[0]
                rotation_angles.extend(range(rotation_step_size, 360, rotation_step_size))
            elif len(rotation_angles) > 1:
                rotation_angles.extend(rotation_angles)

        return rotation_angles

    def get_already_processed(self):
        """ Return the images that already exist in the output directory """
        print("Output Directory: {}".format(self.args.output_dir))

        if not hasattr(self.args, 'skip_existing') or not self.args.skip_existing:
            return None

        return get_image_paths(self.args.output_dir)

    def get_input_images(self):
        """ Return the list of images that are to be processed """
        if not os.path.exists(self.args.input_dir):
            print("Input directory {} not found.".format(self.args.input_dir))
            exit(1)

        print("Input Directory: {}".format(self.args.input_dir))

        if hasattr(self.args, 'skip_existing') and self.args.skip_existing:
            input_images = get_image_paths(self.args.input_dir, self.already_processed)
            print("Excluding %s files" % len(self.already_processed))
        else:
            input_images = get_image_paths(self.args.input_dir)

        return input_images

    def rotate_image(self, image, rotation, reverse=False):
        """ Rotate the image forwards or backwards """
        if rotation != 0:
            if not reverse:
                self.rotation_height, self.rotation_width = image.shape[:2]
                image = Utils.rotate_image_by_angle(image, rotation)
            else:
                image = Utils.rotate_image_by_angle(image,
                                                    rotation * -1,
                                                    rotated_width=self.rotation_width,
                                                    rotated_height=self.rotation_height)
        return image

class Faces(object):
    """ Holds the faces """
    def __init__(self, arguments):
        self.args = arguments
        self.extractor = self.load_extractor()
        self.filter = self.load_face_filter()
        self.align_eyes = self.args.align_eyes if hasattr(self.args, 'align_eyes') else False
        self.output_dir = get_folder(self.args.output_dir)

        self.faces_detected = dict()
        self.num_faces_detected = 0
        self.verify_output = False

    @staticmethod
    def load_extractor():
        """ Load the requested extractor for extraction """
        # TODO Pass as argument
        extractor_name = "Align"
        extractor = PluginLoader.get_extractor(extractor_name)()

        return extractor

    def load_face_filter(self):
        """ Load faces to filter out of images """
        facefilter = None
        filter_files = [self.set_face_filter(filter_type)
                        for filter_type in ('filter', 'nfilter')]

        if any(filters for filters in filter_files):
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

    def have_face(self, filename):
        """ return path of images that have faces """
        return os.path.basename(filename) in self.faces_detected

    def get_faces(self, image, rotation=0):
        """ Extract the faces from an image """
        faces_count = 0
        faces = detect_faces(image, self.args.detector, self.args.verbose, rotation)

        for face in faces:
            if self.filter and not self.filter.check(face):
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
        faces_count = 0
        faces = self.faces_detected[os.path.basename(filename)]
        for rawface in faces:
            face = DetectedFace(**rawface)
            # Rotate the image if necessary
            if face.r != 0:
                image = Utils.rotate_image_by_angle(image, face.r)
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

    def draw_landmarks_on_face(self, face, image):
        """ Draw debug landmarks on extracted face """
        if not hasattr(self.args, 'debug_landmarks') or not self.args.debug_landmarks:
            return

        for (pos_x, pos_y) in face.landmarksAsXY():
            cv2.circle(image, (pos_x, pos_y), 2, (0, 0, 255), -1)

    def detect_blurry_faces(self, face, t_mat, resized_image, filename):
        """ Detect and move blurry face """
        if not hasattr(self.args, 'blur_thresh') or not self.args.blur_thresh:
            return None

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

class Alignments(object):
    """ Holds processes pertaining to the alignments file """
    def __init__(self, arguments):
        self.args = arguments
        self.serializer = self.get_serializer()
        self.alignments_path = self.get_alignments_path()
        self.have_alignments_file = os.path.exists(self.alignments_path)

    def get_serializer(self):
        """ Set the serializer to be used for loading and saving alignments """
        if not self.args.serializer and self.args.alignments_path:
            ext = os.path.splitext(self.args.alignments_path)[-1]
            serializer = Serializer.get_serializer_fromext(ext)
            print("Alignments Output: {}".format(self.args.alignments_path))
        else:
            serializer = Serializer.get_serializer(self.args.serializer or "json")
        print("Using {} serializer".format(serializer.ext))
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
        try:
            with open(self.alignments_path, self.serializer.roptions) as align:
                faces_detected = self.serializer.unmarshal(align.read())
        except Exception as err:
            print("{} not read!".format(self.alignments_path))
            print(str(err))
            faces_detected = dict()
        return faces_detected

    def write_alignments(self, faces_detected):
        """ Write the serialized alignments file """
        if hasattr(self.args, 'skip_existing') and self.args.skip_existing:
            faces_detected = self.load_skip_alignments(self.alignments_path, faces_detected)

        try:
            print("Writing alignments to: {}".format(self.alignments_path))
            with open(self.alignments_path, self.serializer.woptions) as align:
                align.write(self.serializer.marshal(faces_detected))
        except Exception as err:
            print("{} not written!".format(self.alignments_path))
            print(str(err))

    def load_skip_alignments(self, alignfile, faces_detected):
        """ Load existing alignments if skipping existing images """
        if self.have_alignments_file:
            existing_alignments = self.read_alignments()
            for key, val in existing_alignments.items():
                if val:
                    faces_detected[key] = val
        else:
            print("Existing alignments file '%s' not found." % alignfile)
        return faces_detected
