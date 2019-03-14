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
from lib.multithreading import BackgroundGenerator
from lib.queue_manager import queue_manager
from lib.utils import get_folder, get_image_paths, hash_image_file
from plugins.plugin_loader import PluginLoader

from .extract import Plugins as Extractor

import numpy as np
import face_recognition
from PIL import Image
import glob

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convert():
    """ The convert process. """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s)", self.__class__.__name__, arguments)
        self.args = arguments
        self.output_dir = get_folder(self.args.output_dir)
        self.extractor = None
        self.faces_count = 0

        self.images = Images(self.args)
        self.alignments = Alignments(self.args, False, self.images.is_video)

        # Update Legacy alignments
        Legacy(self.alignments, self.images.input_images, arguments.input_aligned_dir)

        self.post_process = PostProcess(arguments)
        self.verify_output = False

        ##initialize last_used face to whichever face is first
        self.matched_face = 0
        ##load images for target face id
        try:
            models_dir = self.args.model_dir
            target_images = [face_recognition.load_image_file(file) for file in glob.glob(models_dir + '**\\*.png', recursive=True)]
        except:
            print("no images found")
            quit()

        ##create target face encodings which will match closest
        try:
            self.target_encodings = [face_recognition.face_encodings(image)[0] for image in target_images]
        except:
            print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
            quit()

        ##load trained models, should have one model per image
        try:
            models = [self.load_model(get_folder(folder)) for folder in glob.glob(models_dir + '*')]
        except:
            print("error finding trained models")
            quit()

        ##create converters for each model
        try:
            self.converters = [self.load_converter(model) for model in models]
        except:
            print("error loading model converters")
            quit()

        self.opts = OptionalActions(self.args, self.images.input_images, self.alignments)
        logger.debug("Initialized %s", self.__class__.__name__)

    def process(self):
        """ Original & LowMem models go with converter

            Note: GAN prediction outputs a mask + an image, while other
            predicts only an image. """
        Utils.set_verbosity(self.args.loglevel)

        if not self.alignments.have_alignments_file:
            self.load_extractor()

        ##model = self.load_model()
        ##converter = self.load_converter(model)

        batch = BackgroundGenerator(self.prepare_images(), 1)

        for item in batch.iterator():
            ##self.convert(converter, item)
            self.convert(item)

        if self.extractor:
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
        extract_args = {"detector": "dlib-hog",
                        "aligner": "dlib",
                        "loglevel": self.args.loglevel}
        self.extractor = Extractor(None, extract_args)
        self.extractor.launch_detector()
        self.extractor.launch_aligner()

    def load_model(self, model_dir):
        """ Load the model requested for conversion """
        logger.debug("Loading Model")
        ##model_dir = get_folder(self.args.model_dir)
        model = PluginLoader.get_model(self.args.trainer)(get_folder(model_dir), self.args.gpus, predict=True)
        logger.debug("Loaded Model")
        return model

    def load_converter(self, model):
        """ Load the requested converter for conversion """
        conv = self.args.converter
        converter = PluginLoader.get_converter(conv)(
            model.converter(self.args.swap_model),
            model=model,
            arguments=self.args)
        return converter

    def prepare_images(self):
        """ Prepare the images for conversion """
        filename = ""
        if self.extractor:
            load_queue = queue_manager.get_queue("load")
        for filename, image in tqdm(self.images.load(),
                                    total=self.images.images_found,
                                    file=sys.stdout):

            ##if (self.args.discard_frames and
            ##        self.opts.check_skipframe(filename) == "discard"):
            ##    continue

            frame = os.path.basename(filename)
            if self.extractor:
                detected_faces = self.detect_faces(load_queue, filename, image)
            else:
                detected_faces = self.alignments_faces(frame, image)
            
            ## if no face was found skip this image
            if not detected_faces:
                continue

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

    def detect_faces(self, load_queue, filename, image):
        """ Extract the face from a frame (If alignments file not found) """
        inp = {"filename": filename,
               "image": image}
        load_queue.put(inp)
        faces = next(self.extractor.detect_faces())

        landmarks = faces["landmarks"]
        detected_faces = faces["detected_faces"]
        final_faces = list()

        for idx, face in enumerate(detected_faces):
            detected_face = DetectedFace()
            detected_face.from_dlib_rect(face)
            detected_face.landmarksXY = landmarks[idx]
            final_faces.append(detected_face)
        return final_faces

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

    ##removed converter, using self.converters
    def convert(self, item):
        """ Apply the conversion transferring faces onto frames """
        try:
            filename, image, faces = item
            ##skip = self.opts.check_skipframe(filename)

            ##if not skip:
            for face in faces:
                ## pick best model and swap
                im = Image.fromarray(face.image)
                try:
                    unknown_face_encoding = face_recognition.face_encodings(np.array(im))[0]
                    results = face_recognition.face_distance(self.target_encodings, unknown_face_encoding)
                    self.matched_face = np.argmin(results)
                    print(self.matched_face)     
                except:
                    pass

                image = self.converters[self.matched_face].patch_image(image, face)
                ##image = converter.patch_image(image, face)

            filename = str(self.output_dir / Path(filename).name)

            if self.args.draw_transparent:
                filename = "{}.png".format(os.path.splitext(filename)[0])
                logger.trace("Set extension to png: `%s`", filename)

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
        rotated = self.alignments.get_legacy_rotation()
        hashes = self.alignments.get_legacy_no_hashes()
        if not rotated and not hashes:
            return
        if rotated:
            logger.info("Legacy rotated frames found. Converting...")
            self.rotate_landmarks(rotated)
            self.alignments.save()
        if hashes and faces_dir:
            logger.info("Legacy alignments found. Adding Face Hashes...")
            self.add_hashes(hashes, faces_dir)
            self.alignments.save()

    def rotate_landmarks(self, rotated):
        """ Rotate the landmarks """
        for rotate_item in tqdm(rotated, desc="Rotating Landmarks"):
            frame = self.frames.get(rotate_item, None)
            if frame is None:
                logger.debug("Skipping missing frame: '%s'", rotate_item)
                continue
            self.alignments.rotate_existing_landmarks(rotate_item, frame)

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
