#!/usr/bin/env python3
""" Holds the classes for the 3 main Faceswap 'media' objects for
    input (extract) and output (convert) tasks. Those being:
            Images
            Faces
            Alignments"""

import logging
import os
from pathlib import Path

import cv2
import imageio
import numpy as np

from lib.aligner import Extract as AlignerExtract
from lib.alignments import Alignments as AlignmentsBase
from lib.face_filter import FaceFilter as FilterFunc
from lib.image import count_frames_and_secs, read_image
from lib.utils import (camel_case_split, get_folder, get_image_paths, set_system_verbosity,
                       _video_extensions)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Utils():
    """ Holds utility functions that are required by more than one media
        object """

    @staticmethod
    def set_verbosity(loglevel):
        """ Set the system output verbosity """
        set_system_verbosity(loglevel)

    @staticmethod
    def finalize(images_found, num_faces_detected, verify_output):
        """ Finalize the image processing """
        logger.info("-------------------------")
        logger.info("Images found:        %s", images_found)
        logger.info("Faces detected:      %s", num_faces_detected)
        logger.info("-------------------------")

        if verify_output:
            logger.info("Note:")
            logger.info("Multiple faces were detected in one or more pictures.")
            logger.info("Double check your results.")
            logger.info("-------------------------")

        logger.info("Process Succesfully Completed. Shutting Down...")


class Alignments(AlignmentsBase):
    """ Override main alignments class for extract """
    def __init__(self, arguments, is_extract, input_is_video=False):
        logger.debug("Initializing %s: (is_extract: %s, input_is_video: %s)",
                     self.__class__.__name__, is_extract, input_is_video)
        self.args = arguments
        self.is_extract = is_extract
        folder, filename = self.set_folder_filename(input_is_video)
        super().__init__(folder,
                         filename=filename)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_folder_filename(self, input_is_video):
        """ Return the folder for the alignments file"""
        if self.args.alignments_path:
            logger.debug("Alignments File provided: '%s'", self.args.alignments_path)
            folder, filename = os.path.split(str(self.args.alignments_path))
        elif input_is_video:
            logger.debug("Alignments from Video File: '%s'", self.args.input_dir)
            folder, filename = os.path.split(self.args.input_dir)
            filename = "{}_alignments".format(os.path.splitext(filename)[0])
        else:
            logger.debug("Alignments from Input Folder: '%s'", self.args.input_dir)
            folder = str(self.args.input_dir)
            filename = "alignments"
        logger.debug("Setting Alignments: (folder: '%s' filename: '%s')", folder, filename)
        return folder, filename

    def load(self):
        """ Override  parent loader to handle skip existing on extract """
        data = dict()
        if not self.is_extract:
            if not self.have_alignments_file:
                return data
            data = super().load()
            return data

        skip_existing = bool(hasattr(self.args, 'skip_existing')
                             and self.args.skip_existing)
        skip_faces = bool(hasattr(self.args, 'skip_faces')
                          and self.args.skip_faces)

        if not skip_existing and not skip_faces:
            logger.debug("No skipping selected. Returning empty dictionary")
            return data

        if not self.have_alignments_file and (skip_existing or skip_faces):
            logger.warning("Skip Existing/Skip Faces selected, but no alignments file found!")
            return data

        data = self.serializer.load(self.file)

        if skip_faces:
            # Remove items from algnments that have no faces so they will
            # be re-detected
            del_keys = [key for key, val in data.items() if not val]
            logger.debug("Frames with no faces selected for redetection: %s", len(del_keys))
            for key in del_keys:
                if key in data:
                    logger.trace("Selected for redetection: '%s'", key)
                    del data[key]
        return data


class Images():
    """ Holds the full frames/images """
    def __init__(self, arguments):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.args = arguments
        self.is_video = self.check_input_folder()
        self.input_images = self.get_input_images()
        self.images_found = self.count_images()
        logger.debug("Initialized %s", self.__class__.__name__)

    def count_images(self):
        """ Number of images or frames """
        if self.is_video:
            retval = int(count_frames_and_secs(self.args.input_dir)[0])
        else:
            retval = len(self.input_images)
        return retval

    def check_input_folder(self):
        """ Check whether the input is a folder or video """
        if not os.path.exists(self.args.input_dir):
            logger.error("Input location %s not found.", self.args.input_dir)
            exit(1)
        if (os.path.isfile(self.args.input_dir) and
                os.path.splitext(self.args.input_dir)[1].lower() in _video_extensions):
            logger.info("Input Video: %s", self.args.input_dir)
            retval = True
        else:
            logger.info("Input Directory: %s", self.args.input_dir)
            retval = False
        return retval

    def get_input_images(self):
        """ Return the list of images or video file that is to be processed """
        if self.is_video:
            input_images = self.args.input_dir
        else:
            input_images = get_image_paths(self.args.input_dir)

        return input_images

    def load(self):
        """ Load an image and yield it with it's filename """
        iterator = self.load_video_frames if self.is_video else self.load_disk_frames
        for filename, image in iterator():
            yield filename, image

    def load_disk_frames(self):
        """ Load frames from disk """
        logger.debug("Input is separate Frames. Loading images")
        for filename in self.input_images:
            image = read_image(filename, raise_error=False)
            if image is None:
                continue
            yield filename, image

    def load_video_frames(self):
        """ Return frames from a video file """
        logger.debug("Input is video. Capturing frames")
        vidname = os.path.splitext(os.path.basename(self.args.input_dir))[0]
        reader = imageio.get_reader(self.args.input_dir, "ffmpeg")
        for i, frame in enumerate(reader):
            # Convert to BGR for cv2 compatibility
            frame = frame[:, :, ::-1]
            filename = "{}_{:06d}.png".format(vidname, i + 1)
            logger.trace("Loading video frame: '%s'", filename)
            yield filename, frame
        reader.close()

    def load_one_image(self, filename):
        """ load requested image """
        logger.trace("Loading image: '%s'", filename)
        if self.is_video:
            if filename.isdigit():
                frame_no = filename
            else:
                frame_no = os.path.splitext(filename)[0][filename.rfind("_") + 1:]
                logger.trace("Extracted frame_no %s from filename '%s'", frame_no, filename)
            retval = self.load_one_video_frame(int(frame_no))
        else:
            retval = read_image(filename, raise_error=True)
        return retval

    def load_one_video_frame(self, frame_no):
        """ Load a single frame from a video file """
        logger.trace("Loading video frame: %s", frame_no)
        reader = imageio.get_reader(self.args.input_dir, "ffmpeg")
        reader.set_image_index(frame_no - 1)
        frame = reader.get_next_data()[:, :, ::-1]
        reader.close()
        return frame


class PostProcess():
    """ Optional post processing tasks """
    def __init__(self, arguments):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.args = arguments
        self.actions = self.set_actions()
        logger.debug("Initialized %s", self.__class__.__name__)

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
            task = globals()[action](*args, **kwargs)
            if task.valid:
                logger.debug("Adding Postprocess action: '%s'", task)
                actions.append(task)

        for action in actions:
            action_name = camel_case_split(action.__class__.__name__)
            logger.info("Adding post processing item: %s", " ".join(action_name))

        return actions

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

            if hasattr(self.args, "detector"):
                detector = self.args.detector.replace("-", "_").lower()
            else:
                detector = "cv2_dnn"
            if hasattr(self.args, "aligner"):
                aligner = self.args.aligner.replace("-", "_").lower()
            else:
                aligner = "cv2_dnn"

            face_filter = dict(detector=detector,
                               aligner=aligner,
                               multiprocess=not self.args.singleprocess)
            filter_lists = dict()
            if hasattr(self.args, "ref_threshold"):
                face_filter["ref_threshold"] = self.args.ref_threshold
            for filter_type in ('filter', 'nfilter'):
                filter_args = getattr(self.args, filter_type, None)
                filter_args = None if not filter_args else filter_args
                filter_lists[filter_type] = filter_args
            face_filter["filter_lists"] = filter_lists
            postprocess_items["FaceFilter"] = {"kwargs": face_filter}

        logger.debug("Postprocess Items: %s", postprocess_items)
        return postprocess_items

    def do_actions(self, output_item):
        """ Perform the requested post-processing actions """
        for action in self.actions:
            logger.debug("Performing postprocess action: '%s'", action.__class__.__name__)
            action.process(output_item)


class PostProcessAction():  # pylint: disable=too-few-public-methods
    """ Parent class for Post Processing Actions
        Usuable in Extract or Convert or both
        depending on context """
    def __init__(self, *args, **kwargs):
        logger.debug("Initializing %s: (args: %s, kwargs: %s)",
                     self.__class__.__name__, args, kwargs)
        self.valid = True  # Set to False if invalid params passed in to disable
        logger.debug("Initialized base class %s", self.__class__.__name__)

    def process(self, output_item):
        """ Override for specific post processing action """
        raise NotImplementedError


class BlurryFaceFilter(PostProcessAction):  # pylint: disable=too-few-public-methods
    """ Move blurry faces to a different folder
        Extract Only """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blur_thresh = kwargs["blur_thresh"]
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self, output_item):
        """ Detect and move blurry face """
        extractor = AlignerExtract()

        for idx, detected_face in enumerate(output_item["detected_faces"]):
            frame_name = detected_face["file_location"].parts[-1]
            face = detected_face["face"]
            logger.trace("Checking for blurriness. Frame: '%s', Face: %s", frame_name, idx)
            aligned_landmarks = face.aligned_landmarks
            resized_face = face.aligned_face
            size = face.aligned["size"]
            padding = int(size * 0.1875)
            feature_mask = extractor.get_feature_mask(
                aligned_landmarks / size,
                size, padding)
            feature_mask = cv2.blur(feature_mask, (10, 10))
            isolated_face = cv2.multiply(feature_mask, resized_face.astype(float)).astype(np.uint8)
            blurry, focus_measure = self.is_blurry(isolated_face)

            if blurry:
                blur_folder = detected_face["file_location"].parts[:-1]
                blur_folder = get_folder(Path(*blur_folder) / Path("blurry"))
                detected_face["file_location"] = blur_folder / Path(frame_name)
                logger.verbose("%s's focus measure of %s was below the blur threshold, "
                               "moving to 'blurry'", frame_name, "{0:.2f}".format(focus_measure))

    def is_blurry(self, image):
        """ Convert to grayscale, and compute the focus measure of the image using the
            Variance of Laplacian method """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        focus_measure = self.variance_of_laplacian(gray)

        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        retval = (focus_measure < self.blur_thresh, focus_measure)
        logger.trace("Returning: (is_blurry: %s, focus_measure %s)", retval[0], retval[1])
        return retval

    @staticmethod
    def variance_of_laplacian(image):
        """ Compute the Laplacian of the image and then return the focus
            measure, which is simply the variance of the Laplacian """
        retval = cv2.Laplacian(image, cv2.CV_64F).var()
        logger.trace("Returning: %s", retval)
        return retval


class DebugLandmarks(PostProcessAction):  # pylint: disable=too-few-public-methods
    """ Draw debug landmarks on face
        Extract Only """

    def process(self, output_item):
        """ Draw landmarks on image """
        for idx, detected_face in enumerate(output_item["detected_faces"]):
            face = detected_face["face"]
            logger.trace("Drawing Landmarks. Frame: '%s'. Face: %s",
                         detected_face["file_location"].parts[-1], idx)
            aligned_landmarks = face.aligned_landmarks
            for (pos_x, pos_y) in aligned_landmarks:
                cv2.circle(face.aligned_face, (pos_x, pos_y), 2, (0, 0, 255), -1)


class FaceFilter(PostProcessAction):
    """ Filter in or out faces based on input image(s)
        Extract or Convert """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Extracting and aligning face for Face Filter...")
        self.filter = self.load_face_filter(**kwargs)
        logger.debug("Initialized %s", self.__class__.__name__)

    def load_face_filter(self, filter_lists, ref_threshold, aligner, detector,
                         multiprocess):
        """ Load faces to filter out of images """
        if not any(val for val in filter_lists.values()):
            return None

        facefilter = None
        filter_files = [self.set_face_filter(f_type, filter_lists[f_type])
                        for f_type in ("filter", "nfilter")]

        if any(filters for filters in filter_files):
            facefilter = FilterFunc(filter_files[0],
                                    filter_files[1],
                                    detector,
                                    aligner,
                                    multiprocess,
                                    ref_threshold)
            logger.debug("Face filter: %s", facefilter)
        else:
            self.valid = False
        return facefilter

    @staticmethod
    def set_face_filter(f_type, f_args):
        """ Set the required filters """
        if not f_args:
            return list()

        logger.info("%s: %s", f_type.title(), f_args)
        filter_files = f_args if isinstance(f_args, list) else [f_args]
        filter_files = list(filter(lambda fpath: Path(fpath).exists(), filter_files))
        if not filter_files:
            logger.warning("Face %s files were requested, but no files could be found. This "
                           "filter will not be applied.", f_type)
        logger.debug("Face Filter files: %s", filter_files)
        return filter_files

    def process(self, output_item):
        """ Filter in/out wanted/unwanted faces """
        if not self.filter:
            return
        ret_faces = list()
        for idx, detect_face in enumerate(output_item["detected_faces"]):
            check_item = detect_face["face"] if isinstance(detect_face, dict) else detect_face
            check_item.load_aligned(output_item["image"])
            if not self.filter.check(check_item):
                logger.verbose("Skipping not recognized face: (Frame: %s Face %s)",
                               output_item["filename"], idx)
                continue
            logger.trace("Accepting recognised face. Frame: %s. Face: %s",
                         output_item["filename"], idx)
            ret_faces.append(detect_face)
        output_item["detected_faces"] = ret_faces
