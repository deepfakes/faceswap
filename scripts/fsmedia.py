#!/usr/bin/env python3
""" Holds the classes for the 3 main Faceswap 'media' objects for
    input (extract) and output (convert) tasks. Those being:
            Images
            Faces
            Alignments"""

import os
from pathlib import Path

import cv2
import numpy as np

from lib.aligner import Extract as AlignerExtract
from lib.alignments import Alignments as AlignmentsBase
from lib.detect_blur import is_blurry
from lib.FaceFilter import FaceFilter as FilterFunc
from lib.utils import (camel_case_split, get_folder, get_image_paths,
                       set_system_verbosity)


class Utils():
    """ Holds utility functions that are required by more than one media
        object """

    @staticmethod
    def set_verbosity(verbose):
        """ Set the system output verbosity """
        lvl = '0' if verbose else '2'
        set_system_verbosity(lvl)

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

        print("Done!")


class Alignments(AlignmentsBase):
    """ Override main alignments class for extract """
    def __init__(self, arguments, is_extract):
        self.args = arguments
        self.is_extract = is_extract
        folder, filename = self.set_folder_filename()
        serializer = self.set_serializer()
        super().__init__(folder,
                         filename=filename,
                         serializer=serializer,
                         verbose=self.args.verbose)

    def set_folder_filename(self):
        """ Return the folder for the alignments file"""
        if self.args.alignments_path:
            folder, filename = os.path.split(str(self.args.alignments_path))
        else:
            folder = str(self.args.input_dir)
            filename = "alignments"
        return folder, filename

    def set_serializer(self):
        """ Set the serializer to be used for loading and
            saving alignments """
        if hasattr(self.args, "serializer") and self.args.serializer:
            serializer = self.args.serializer
        else:
            # If there is a full filename then this will be overriden
            # by filename extension
            serializer = "json"
        return serializer

    def load(self):
        """ Override  parent loader to handle skip existing on extract """
        data = dict()
        if not self.is_extract:
            data = super().load()
            return data

        skip_existing = bool(hasattr(self.args, 'skip_existing')
                             and self.args.skip_existing)
        skip_faces = bool(hasattr(self.args, 'skip_faces')
                          and self.args.skip_faces)

        if not skip_existing and not skip_faces:
            return data

        if not self.have_alignments_file and (skip_existing or skip_faces):
            print("Skip Existing/Skip Faces selected, but no alignments file "
                  "found!")
            return data

        try:
            with open(self.file, self.serializer.roptions) as align:
                data = self.serializer.unmarshal(align.read())
        except IOError as err:
            print("Error: {} not read: {}".format(self.file, err.strerror))
            exit(1)

        if skip_faces:
            # Remove items from algnments that have no faces so they will
            # be re-detected
            del_keys = [key for key, val in data.items() if not val]
            for key in del_keys:
                if key in data:
                    del data[key]
        return data


class Images():
    """ Holds the full frames/images """
    def __init__(self, arguments):
        self.args = arguments
        self.input_images = self.get_input_images()
        self.images_found = len(self.input_images)

    def get_input_images(self):
        """ Return the list of images that are to be processed """
        if not os.path.exists(self.args.input_dir):
            print("Input directory {} not found.".format(self.args.input_dir))
            exit(1)

        print("Input Directory: {}".format(self.args.input_dir))
        input_images = get_image_paths(self.args.input_dir)

        return input_images

    def load(self):
        """ Load an image and yield it with it's filename """
        for filename in self.input_images:
            yield filename, cv2.imread(filename)  # pylint: disable=no-member

    @staticmethod
    def load_one_image(filename):
        """ load requested image """
        return cv2.imread(filename)  # pylint: disable=no-member


class PostProcess():
    """ Optional post processing tasks """
    def __init__(self, arguments):
        self.args = arguments
        self.verbose = self.args.verbose
        self.actions = self.set_actions()

    def get_items(self):
        """ Set the post processing actions """
        postprocess_items = dict()
        # Debug Landmarks
        if (hasattr(self.args, 'debug_landmarks')
                and self.args.debug_landmarks):
            postprocess_items["DebugLandmarks"] = None

        # Blurry Face
        if hasattr(self.args, 'blur_thresh') and self.args.blur_thresh:
            kwargs = {"blur_thresh": self.args.blur_thresh}
            postprocess_items["BlurryFaceFilter"] = {"kwargs": kwargs}

        # Face Filter post processing
        if ((hasattr(self.args, "filter") and self.args.filter is not None) or
                (hasattr(self.args, "nfilter") and
                 self.args.nfilter is not None)):
            face_filter = dict()
            filter_lists = dict()
            if hasattr(self.args, "ref_threshold"):
                face_filter["ref_threshold"] = self.args.ref_threshold
            for filter_type in ('filter', 'nfilter'):
                filter_args = getattr(self.args, filter_type, None)
                filter_args = None if not filter_args else filter_args
                filter_lists[filter_type] = filter_args
            face_filter["filter_lists"] = filter_lists
            postprocess_items["FaceFilter"] = {"kwargs": face_filter}

        return postprocess_items

    def set_actions(self):
        """ Compile the actions to be performed into a list """
        postprocess_items = self.get_items()
        actions = list()
        for action, options in postprocess_items.items():
            options = dict() if options is None else options
            args = options.get("args", tuple())
            kwargs = options.get("kwargs", dict())
            args = args if isinstance(args, tuple) else tuple()
            kwargs = kwargs if isinstance(kwargs, dict) else dict()
            kwargs["verbose"] = self.verbose
            task = globals()[action](*args, **kwargs)
            actions.append(task)

        for action in actions:
            action_name = camel_case_split(action.__class__.__name__)
            print("Adding post processing item: "
                  "{}".format(" ".join(action_name)))

        return actions

    def do_actions(self, output_item):
        """ Perform the requested post-processing actions """
        for action in self.actions:
            action.process(output_item)


class PostProcessAction():
    """ Parent class for Post Processing Actions
        Usuable in Extract or Convert or both
        depending on context """
    def __init__(self, *args, **kwargs):
        self.verbose = kwargs["verbose"]

    def process(self, output_item):
        """ Override for specific post processing action """
        raise NotImplementedError


class BlurryFaceFilter(PostProcessAction):
    """ Move blurry faces to a different folder
        Extract Only """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blur_thresh = kwargs["blur_thresh"]

    def process(self, output_item):
        """ Detect and move blurry face """
        extractor = AlignerExtract()

        for face in output_item["detected_faces"]:
            aligned_landmarks = face.aligned_landmarks
            resized_face = face.aligned_face
            size = face.aligned["size"]
            feature_mask = extractor.get_feature_mask(
                aligned_landmarks / size,
                size, 48)
            feature_mask = cv2.blur(  # pylint: disable=no-member
                feature_mask, (10, 10))
            isolated_face = cv2.multiply(  # pylint: disable=no-member
                feature_mask,
                resized_face.astype(float)).astype(np.uint8)
            blurry, focus_measure = is_blurry(isolated_face, self.blur_thresh)

            if blurry:
                blur_folder = output_item["output_file"].parts[:-1]
                blur_folder = get_folder(Path(*blur_folder) / Path("blurry"))
                frame_name = output_item["output_file"].parts[-1]
                output_item["output_file"] = blur_folder / Path(frame_name)
                if self.verbose:
                    print("{}'s focus measure of {} was below the blur "
                          "threshold, moving to \"blurry\"".format(
                              frame_name, focus_measure))


class DebugLandmarks(PostProcessAction):
    """ Draw debug landmarks on face
        Extract Only """

    def process(self, output_item):
        """ Draw landmarks on image """
        for face in output_item["detected_faces"]:
            aligned_landmarks = face.aligned_landmarks
            for (pos_x, pos_y) in aligned_landmarks:
                cv2.circle(  # pylint: disable=no-member
                    face.aligned_face,
                    (pos_x, pos_y), 2, (0, 0, 255), -1)


class FaceFilter(PostProcessAction):
    """ Filter in or out faces based on input image(s)
        Extract or Convert """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        filter_lists = kwargs["filter_lists"]
        ref_threshold = kwargs.get("ref_threshold", 0.6)
        self.filter = self.load_face_filter(filter_lists, ref_threshold)

    def load_face_filter(self, filter_lists, ref_threshold):
        """ Load faces to filter out of images """
        if not any(val for val in filter_lists.values()):
            return None

        filter_files = [self.set_face_filter(key, val)
                        for key, val in filter_lists.items()]

        if any(filters for filters in filter_files):
            facefilter = FilterFunc(filter_files[0],
                                    filter_files[1],
                                    ref_threshold)
        return facefilter

    @staticmethod
    def set_face_filter(f_type, f_args):
        """ Set the required filters """
        if not f_args:
            return list()

        print("{}: {}".format(f_type.title(), f_args))
        filter_files = f_args if isinstance(f_args, list) else [f_args]
        filter_files = list(filter(lambda fnc: Path(fnc).exists(),
                                   filter_files))
        return filter_files

    def process(self, output_item):
        """ Filter in/out wanted/unwanted faces """
        if not self.filter:
            return

        detected_faces = output_item["detected_faces"]
        ret_faces = list()
        for face in detected_faces:
            if not self.filter.check(face):
                if self.verbose:
                    print("Skipping not recognized face!")
                continue
            ret_faces.append(face)
        output_item["detected_faces"] = ret_faces
