#!/usr/bin python3
""" The script to run the convert process of faceswap """

import logging
import re
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from scripts.fsmedia import Alignments, Images, PostProcess, Utils
from lib.faces_detect import DetectedFace
from lib.multithreading import BackgroundGenerator, SpawnProcess
from lib.queue_manager import queue_manager
from lib.utils import get_folder, get_image_paths, hash_image_file
from plugins.plugin_loader import PluginLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convert():
    """ The convert process. """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s)", self.__class__.__name__, arguments)
        self.args = arguments
        self.output_dir = get_folder(self.args.output_dir)
        self.extract_faces = False
        self.faces_count = 0

        self.images = Images(self.args)
        self.alignments = Alignments(self.args, False, self.images.is_video)

        # Update Legacy alignments
        Legacy(self.alignments, self.images.input_images, arguments.input_aligned_dir)

        self.post_process = PostProcess(arguments)
        self.verify_output = False

        self.opts = OptionalActions(self.args, self.images.input_images, self.alignments)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Original & LowMem models go with converter

            Note: GAN prediction outputs a mask + an image, while other
            predicts only an image. """
        Utils.set_verbosity(self.args.loglevel)

        if not self.alignments.have_alignments_file:
            self.load_extractor()

        model = self.load_model()
        converter = self.load_converter(model)

        batch = BackgroundGenerator(self.prepare_images(), 1)

        for item in batch.iterator():
            self.convert(converter, item)

        if self.extract_faces:
            queue_manager.terminate_queues()

        Utils.finalize(self.images.images_found,
                       self.faces_count,
                       self.verify_output)

    def load_extractor(self):
        """ Set on the fly extraction """
        logger.warning("No Alignments file found. Extracting on the fly.")
        logger.warning("NB: This will use the inferior dlib-hog for extraction "
                       "and dlib pose predictor for landmarks. It is recommended "
                       "to perfom Extract first for superior results")
        for task in ("load", "detect", "align"):
            queue_manager.add_queue(task, maxsize=0)

        detector = PluginLoader.get_detector("dlib_hog")(loglevel=self.args.loglevel)
        aligner = PluginLoader.get_aligner("dlib")(loglevel=self.args.loglevel)

        d_kwargs = {"in_queue": queue_manager.get_queue("load"),
                    "out_queue": queue_manager.get_queue("detect")}
        a_kwargs = {"in_queue": queue_manager.get_queue("detect"),
                    "out_queue": queue_manager.get_queue("align")}

        d_process = SpawnProcess(detector.run, **d_kwargs)
        d_event = d_process.event
        d_process.start()

        a_process = SpawnProcess(aligner.run, **a_kwargs)
        a_event = a_process.event
        a_process.start()

        d_event.wait(10)
        if not d_event.is_set():
            raise ValueError("Error inititalizing Detector")
        a_event.wait(10)
        if not a_event.is_set():
            raise ValueError("Error inititalizing Aligner")

        self.extract_faces = True

    def load_model(self):
        """ Load the model requested for conversion """
        logger.debug("Loading Model")
        model_dir = get_folder(self.args.model_dir)
        model = PluginLoader.get_model(self.args.trainer)(model_dir, self.args.gpus, predict=True)
        logger.debug("Loaded Model")
        return model

    def load_converter(self, model):
        """ Load the requested converter for conversion """
        conv = self.args.converter
        converter = PluginLoader.get_converter(conv)(
            model.converter(self.args.swap_model),
            arguments=self.args,
            input_size=model.input_shape[0])

        return converter

    def prepare_images(self):
        """ Prepare the images for conversion """
        filename = ""
        for filename, image in tqdm(self.images.load(),
                                    total=self.images.images_found,
                                    file=sys.stdout):

            if (self.args.discard_frames and
                    self.opts.check_skipframe(filename) == "discard"):
                continue

            frame = os.path.basename(filename)
            if self.extract_faces:
                detected_faces = self.detect_faces(filename, image)
            else:
                detected_faces = self.alignments_faces(frame, image)

            faces_count = len(detected_faces)
            if faces_count != 0:
                # Post processing requires a dict with "detected_faces" key
                self.post_process.do_actions(
                    {"detected_faces": detected_faces})
                self.faces_count += faces_count

            if faces_count > 1:
                self.verify_output = True
                logger.verbose("Found more than one face in "
                               "an image! '%s'", frame)

            yield filename, image, detected_faces

    @staticmethod
    def detect_faces(filename, image):
        """ Extract the face from a frame (If not alignments file found) """
        queue_manager.get_queue("load").put((filename, image))
        item = queue_manager.get_queue("align").get()
        detected_faces = item["detected_faces"]
        return detected_faces

    def alignments_faces(self, frame, image):
        """ Get the face from alignments file """
        if not self.check_alignments(frame):
            return list()

        faces = self.alignments.get_faces_in_frame(frame)
        detected_faces = list()

        for rawface in faces:
            face = DetectedFace()
            face.from_alignment(rawface, image=image)
            detected_faces.append(face)
        return detected_faces

    def check_alignments(self, frame):
        """ If we have no alignments for this image, skip it """
        have_alignments = self.alignments.frame_exists(frame)
        if not have_alignments:
            tqdm.write("No alignment found for {}, "
                       "skipping".format(frame))
        return have_alignments

    def convert(self, converter, item):
        """ Apply the conversion transferring faces onto frames """
        try:
            filename, image, faces = item
            skip = self.opts.check_skipframe(filename)

            if not skip:
                for face in faces:
                    image = converter.patch_image(image, face)
                filename = str(self.output_dir / Path(filename).name)
                cv2.imwrite(filename, image)  # pylint: disable=no-member
        except Exception as err:
            logger.error("Failed to convert image: '%s'. Reason: %s", filename, err)
            raise


class OptionalActions():
    """ Process the optional actions for convert """

    def __init__(self, args, input_images, alignments):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.args = args
        self.input_images = input_images
        self.alignments = alignments
        self.frame_ranges = self.get_frame_ranges()
        self.imageidxre = re.compile(r"[^(mp4)](\d+)(?!.*\d)")

        self.remove_skipped_faces()
        logger.debug("Initialized %s", self.__class__.__name__)

    # SKIP FACES #
    def remove_skipped_faces(self):
        """ Remove deleted faces from the loaded alignments """
        logger.debug("Filtering Faces")
        face_hashes = self.get_face_hashes()
        if not face_hashes:
            logger.debug("No face hashes. Not skipping any faces")
            return
        pre_face_count = self.alignments.faces_count
        self.alignments.filter_hashes(face_hashes, filter_out=False)
        logger.info("Faces filtered out: %s", pre_face_count - self.alignments.faces_count)

    def get_face_hashes(self):
        """ Check for the existence of an aligned directory for identifying
            which faces in the target frames should be swapped.
            If it exists, obtain the hashes of the faces in the folder """
        face_hashes = list()
        input_aligned_dir = self.args.input_aligned_dir

        if input_aligned_dir is None:
            logger.verbose("Aligned directory not specified. All faces listed in the "
                           "alignments file will be converted")
        elif not os.path.isdir(input_aligned_dir):
            logger.warning("Aligned directory not found. All faces listed in the "
                           "alignments file will be converted")
        else:
            file_list = [path for path in get_image_paths(input_aligned_dir)]
            logger.info("Getting Face Hashes for selected Aligned Images")
            for face in tqdm(file_list, desc="Hashing Faces"):
                face_hashes.append(hash_image_file(face))
            logger.debug("Face Hashes: %s", (len(face_hashes)))
            if not face_hashes:
                logger.error("Aligned directory is empty, no faces will be converted!")
                exit(1)
            elif len(face_hashes) <= len(self.input_images) / 3:
                logger.warning("Aligned directory contains far fewer images than the input "
                               "directory, are you sure this is the right folder?")
        return face_hashes

    # SKIP FRAME RANGES #
    def get_frame_ranges(self):
        """ split out the frame ranges and parse out 'min' and 'max' values """
        if not self.args.frame_ranges:
            return None

        minmax = {"min": 0,  # never any frames less than 0
                  "max": float("inf")}
        rng = [tuple(map(lambda q: minmax[q] if q in minmax.keys() else int(q),
                         v.split("-")))
               for v in self.args.frame_ranges]
        return rng

    def check_skipframe(self, filename):
        """ Check whether frame is to be skipped """
        if not self.frame_ranges:
            return None
        idx = int(self.imageidxre.findall(filename)[0])
        skipframe = not any(map(lambda b: b[0] <= idx <= b[1],
                                self.frame_ranges))
        if skipframe and self.args.discard_frames:
            skipframe = "discard"
        return skipframe


class Legacy():
    """ Update legacy alignments:

        - Add frame dimensions
        - Rotate landmarks and bounding boxes on legacy alignments
          and remove the 'r' parameter
        - Add face hashes to alignments file
        """
    def __init__(self, alignments, frames, faces_dir):
        self.alignments = alignments
        self.frames = {os.path.basename(frame): frame
                       for frame in frames}
        self.process(faces_dir)

    def process(self, faces_dir):
        """ Run the rotate alignments process """
        no_dims = self.alignments.get_legacy_no_dims()
        rotated = self.alignments.get_legacy_rotation()
        hashes = self.alignments.get_legacy_no_hashes()
        if not no_dims and not rotated and not hashes:
            return
        if no_dims:
            logger.info("Legacy landmarks found. Adding frame dimensions...")
            self.add_dimensions(no_dims)
            self.alignments.save()
        if rotated:
            logger.info("Legacy rotated frames found. Converting...")
            self.rotate_landmarks(rotated)
            self.alignments.save()
        if hashes and faces_dir:
            logger.info("Legacy alignments found. Adding Face Hashes...")
            self.add_hashes(hashes, faces_dir)
            self.alignments.save()

    def add_dimensions(self, no_dims):
        """ Add width and height of original frame to alignments """
        for no_dim in tqdm(no_dims, desc="Adding Frame Dimensions"):
            if no_dim not in self.frames.keys():
                continue
            filename = self.frames[no_dim]
            dims = cv2.imread(filename).shape[:2]  # pylint: disable=no-member
            self.alignments.add_dimensions(no_dim, dims)

    def rotate_landmarks(self, rotated):
        """ Rotate the landmarks """
        for rotate_item in tqdm(rotated, desc="Rotating Landmarks"):
            if rotate_item not in self.frames.keys():
                logger.debug("Skipping missing frame: '%s'", rotate_item)
                continue
            self.alignments.rotate_existing_landmarks(rotate_item)

    def add_hashes(self, hashes, faces_dir):
        """ Add Face Hashes to the alignments file """
        all_faces = dict()
        face_files = sorted(face for face in os.listdir(faces_dir) if "_" in face)
        for face in face_files:
            filename, extension = os.path.splitext(face)
            index = filename[filename.rfind("_") + 1:]
            if not index.isdigit():
                continue
            orig_frame = filename[:filename.rfind("_")] + extension
            all_faces.setdefault(orig_frame, dict())[int(index)] = os.path.join(faces_dir, face)

        for frame in tqdm(hashes):
            if frame not in all_faces.keys():
                logger.warning("Skipping missing frame: '%s'", frame)
                continue
            hash_faces = all_faces[frame]
            for index, face_path in hash_faces.items():
                hash_faces[index] = hash_image_file(face_path)
            self.alignments.add_face_hashes(frame, hash_faces)
