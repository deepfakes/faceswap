#!/usr/bin/env python3
""" Holds the classes for the 3 main Faceswap 'media' objects for
    input (extract) and output (convert) tasks. Those being:
            Images
            Faces
            Alignments"""

import os
import cv2

from lib import Serializer
from lib.faces_detect import DetectedFace
from lib.multithreading import (PoolProcess, SpawnProcess,
                                QueueEmpty, queue_manager)
from lib.utils import (get_folder, get_image_paths, rotate_image_by_angle,
                       set_system_verbosity)
from plugins.plugin_loader import PluginLoader


class Utils():
    """ Holds utility functions that are required by more than one media
        object """

    @staticmethod
    def set_verbosity(verbose):
        """ Set the system output verbosity """
        lvl = '0' if verbose else '2'
        set_system_verbosity(lvl)

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


class Images():
    """ Holds the full frames/images """
    def __init__(self, arguments):
        self.args = arguments
        self.already_processed = self.get_already_processed()
        self.input_images = self.get_input_images()
        self.images_found = len(self.input_images)

        self.rotation_width = 0
        self.rotation_height = 0

    def get_already_processed(self):
        """ Return the images that already exist in the output directory """
        print("Output Directory: {}".format(self.args.output_dir))

        if (not hasattr(self.args, 'skip_existing')
                or not self.args.skip_existing):
            return None

        return get_image_paths(self.args.output_dir)

    def get_input_images(self):
        """ Return the list of images that are to be processed """
        if not os.path.exists(self.args.input_dir):
            print("Input directory {} not found.".format(self.args.input_dir))
            exit(1)

        print("Input Directory: {}".format(self.args.input_dir))

        if hasattr(self.args, 'skip_existing') and self.args.skip_existing:
            input_images = get_image_paths(self.args.input_dir,
                                           self.already_processed)
            print("Excluding %s files" % len(self.already_processed))
        else:
            input_images = get_image_paths(self.args.input_dir)

        return input_images

    def rotate_image(self, image, rotation, reverse=False):
        """ Rotate the image forwards or backwards """
        if rotation == 0:
            return image
        if not reverse:
            self.rotation_height, self.rotation_width = image.shape[:2]
            image, _ = rotate_image_by_angle(image, rotation)
        else:
            image, _ = rotate_image_by_angle(
                image,
                rotation * -1,
                rotated_width=self.rotation_width,
                rotated_height=self.rotation_height)
        return image


class Faces():
    """ Holds the faces """
    def __init__(self, arguments):
        self.args = arguments
        self.output_dir = get_folder(self.args.output_dir)
        self.plugins = self.load_plugins()

        self.faces_detected = dict()
        self.num_faces_detected = 0
        self.verify_output = False

    def load_plugins(self):
        """ Load the requested extractor for extraction """
        detector = self.load_detector()
        aligner = self.load_aligner()
        return {"detector": detector,
                "aligner": aligner}

    def load_detector(self):
        """ Set global arguments and load detector plugin """
        detector_name = self.args.detector.replace("-", "_").lower()

        # Rotation
        rotation = None
        if hasattr(self.args, "rotate_images"):
            rotation = self.args.rotate_images

        detector = PluginLoader.get_detector(detector_name)(
            verbose=self.args.verbose,
            rotation=rotation)

        return detector

    def load_aligner(self):
        """ Set global arguments and load aligner plugin """
        # Add a cli option if other aligner plugins are added
        aligner_name = self.args.aligner.replace("-", "_").lower()

        # Align Eyes
        align_eyes = False
        if hasattr(self.args, 'align_eyes'):
            align_eyes = self.args.align_eyes

        aligner = PluginLoader.get_aligner(aligner_name)(
            verbose=self.args.verbose,
            align_eyes=align_eyes)

        return aligner

    def have_face(self, filename):
        """ return path of images that have faces """
        return os.path.basename(filename) in self.faces_detected

    def detect_faces(self):
        """ Detect faces from in an image """
        self.launch_aligner()
        self.launch_detector()
        out_queue = queue_manager.get_queue("align")

        while True:
            try:
                faces = out_queue.get(True, 1)
                if faces == "EOF":
                    break
            except QueueEmpty:
                continue

            faces_count = len(faces["detected_faces"])
            if self.args.verbose and faces_count == 0:
                print("Warning: No faces were detected in "
                      "image: {}".format(os.path.basename(faces["filename"])))
            self.num_faces_detected += faces_count

            if (self.args.verbose and
                    not self.verify_output and
                    faces_count > 1):
                self.verify_output = True
            yield faces

    def launch_aligner(self):
        """ Launch the face aligner """
        aligner = self.plugins["aligner"]

        out_queue = queue_manager.get_queue("align")
        kwargs = {"in_queue": queue_manager.get_queue("detect"),
                  "out_queue": out_queue}

        align_process = SpawnProcess()
        align_process.in_process(aligner.align, **kwargs)

        while True:
            # Wait for Aligner to take it's VRAM
            try:
                init = out_queue.get(True, 60)
            except QueueEmpty:
                raise ValueError("Error inititalizing Aligner")
            if init != "init":
                raise ValueError("Error inititalizing Aligner")
            break

    def launch_detector(self):
        """ Launch the face detector """
        detector = self.plugins["detector"]

        kwargs = {"in_queue": queue_manager.get_queue("load"),
                  "out_queue": queue_manager.get_queue("detect")}
        if self.args.detector == "mtcnn":
            mtcnn_kwargs = detector.validate_kwargs(self.get_mtcnn_kwargs())
            kwargs["mtcnn_kwargs"] = mtcnn_kwargs

        if detector.parent_is_pool:
            detect_process = PoolProcess(detector.detect_faces,
                                         verbose=self.args.verbose)
        else:
            detect_process = SpawnProcess()

        detect_process.in_process(detector.detect_faces, **kwargs)

    def get_mtcnn_kwargs(self):
        """ Add the mtcnn arguments into a kwargs dictionary """
        mtcnn_threshold = [float(thr.strip())
                           for thr in self.args.mtcnn_threshold]
        return {"minsize": self.args.mtcnn_minsize,
                "threshold": mtcnn_threshold,
                "factor": self.args.mtcnn_scalefactor}

    def get_faces_alignments(self, filename, image):
        """ Retrieve the face alignments from an image """
        faces_count = 0
        faces = self.faces_detected[os.path.basename(filename)]
        for rawface in faces:
            face = DetectedFace(**rawface)
            # Rotate the image if necessary
            # NB: Rotation of landmarks now occurs at extract stage
            # This is here for legacy alignments
            if face.r != 0:
                image, _ = rotate_image_by_angle(image, face.r)
            face.image = image[face.y: face.y + face.h,
                               face.x: face.x + face.w]
            if self.filter and not self.filter.check(face):
                if self.args.verbose:
                    print("Skipping not recognized face!")
                continue

            yield faces_count, face
            self.num_faces_detected += 1
            faces_count += 1
        if faces_count > 1 and self.args.verbose:
            print("Note: Found more than one face in "
                  "an image! File: {}".format(filename))
            self.verify_output = True


class Alignments():
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
            serializer = Serializer.get_serializer_from_ext(ext)
            print("Alignments Output: {}".format(self.args.alignments_path))
        else:
            serializer = Serializer.get_serializer(self.args.serializer)
        print("Using {} serializer".format(serializer.ext))
        return serializer

    def get_alignments_path(self):
        """ Return the path to alignments file """
        if self.args.alignments_path:
            alignfile = self.args.alignments_path
        else:
            alignfile = os.path.join(
                str(self.args.input_dir),
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
            faces_detected = self.load_skip_alignments(self.alignments_path,
                                                       faces_detected)

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
            print("Existing alignments file '{}' not found.".format(alignfile))
        return faces_detected
