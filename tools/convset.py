#!/usr/bin/env python3
""" Tool to preview swaps and tweak the config prior to running a convert

Sketch:

    - predict 4 random faces (full set distribution)
    - keep mask + face separate
    - show faces swapped into padded square of final image
    - live update on settings change
    - apply + save config
"""
import logging
import random
import sys
import os

import cv2
import numpy as np

from lib.cli import ConvertArgs
from lib.convert import Converter
from lib.faces_detect import DetectedFace
from lib.utils import set_system_verbosity
from lib.queue_manager import queue_manager
from scripts.fsmedia import Alignments, Images
from scripts.convert import Predict

from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Convset():
    """ Loads up 4 semi-random face swaps and displays them, cropped, in place in the final frame.
        Allows user to live tweak settings, before saving the final config to
        ./config/convert.ini """

    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        set_system_verbosity(arguments.loglevel)
        self.queues = {"patch_in": queue_manager.get_queue("convset_patch_in"),
                       "patch_out": queue_manager.get_queue("convset_patch_out")}

        self.faces = TestSet(arguments, self.queue_patch_in)
        self.arguments = self.generate_converter_arguments(arguments)
        self.configs = self.get_configs()

        print(self.arguments)
        print(self.configs)

        self.converter = Converter(output_dir=None,
                                   output_size=self.faces.predictor.output_size,
                                   output_has_mask=self.faces.predictor.has_predicted_mask,
                                   draw_transparent=False,
                                   pre_encode=None,
                                   arguments=self.arguments)

        # self.display = FacesDisplay(self.faces.faces_source, self.faces.faces_predicted, 256)

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def queue_patch_in(self):
        """ Patch in queue """
        return self.queues["patch_in"]

    @property
    def queue_patch_out(self):
        """ Patch out queue """
        return self.queues["patch_out"]

    @staticmethod
    def generate_converter_arguments(arguments):
        """ Get the default converter arguments """
        converter_arguments = ConvertArgs(None, "convert").get_optional_arguments()
        for item in converter_arguments:
            value = item.get("default", None)
            # Skip options without a default value
            if value is None:
                continue
            option = item.get("dest", item["opts"][1].replace("--", ""))
            # Skip options already in arguments
            if hasattr(arguments, option):
                continue
            # Add option to arguments
            setattr(arguments, option, value)
        logger.debug(arguments)
        return arguments

    def process(self):
        """ The convset process """
        self.faces.generate()
        self.converter.process(self.queue_patch_in, self.queue_patch_out)
        idx = 0
        while idx < self.faces.sample_size:
            logger.debug("Patching image %s of %s", idx + 1, self.faces.sample_size)
            item = self.queue_patch_out.get()
            self.faces.selected_frames[idx]["swapped_image"] = item[1]
            logger.debug("Patched image %s of %s", idx + 1, self.faces.sample_size)
            idx += 1
        logger.debug("Patched faces")
        for frame in self.faces.selected_frames:
            print(frame["filename"],
                  frame["image"].shape,
                  frame["source_face"].shape,
                  frame["swapped_face"].shape,
                  frame["swapped_image"].shape)
        print([key for key in self.faces.selected_frames[0].keys()])
        exit(0)

        cv2.imshow("test", self.display.image)
        cv2.waitKey()
        exit(0)

    def get_configs(self):
        """ Return all of the convert configs """
        modules = self.get_modules()
        configs = {".".join((key, plugin)): Config(".".join((key, plugin)))
                   for key, val in modules.items()
                   for plugin in val}
        logger.debug(configs)
        return configs

    @staticmethod
    def get_modules():
        """ Return all available convert plugins """
        root_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        plugins_path = os.path.join(root_path, "plugins", "convert")
        modules = {os.path.basename(dirpath): [os.path.splitext(module)[0]
                                               for module in filenames
                                               if os.path.splitext(module)[1] == ".py"
                                               and module not in ("__init__.py", "_base.py")]
                   for dirpath, _, filenames in os.walk(plugins_path)
                   if os.path.basename(dirpath) not in ("convert", "__pycache__", "writer")}
        logger.debug("Modules: %s", modules)
        return modules


class TestSet():
    """ Holds 4 random test faces """

    def __init__(self, arguments, patch_queue):
        logger.debug("Initializing %s: (arguments: '%s', patch_queue: %s)",
                     self.__class__.__name__, arguments, patch_queue)
        self.sample_size = 4

        self.images = Images(arguments)
        self.alignments = Alignments(arguments,
                                     is_extract=False,
                                     input_is_video=self.images.is_video)
        self.queues = {"predict_in": queue_manager.get_queue("convset_predict_in"),
                       "patch_in": patch_queue}
        self.indices = self.get_indices()
        self.predictor = Predict(self.queues["predict_in"], 4, arguments)

        self.selected_frames = list()

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def random_choice(self):
        """ Return for random indices from the indices group """
        retval = [random.choice(indices) for indices in self.indices]
        logger.debug(retval)
        return retval

    @property
    def queue_predict_in(self):
        """ Predict in queue """
        return self.queues["predict_in"]

    @property
    def queue_predict_out(self):
        """ Predict in queue """
        return self.predictor.out_queue

    @property
    def queue_patch_in(self):
        """ Patch in queue """
        return self.queues["patch_in"]

    def get_indices(self):
        """ Returns a list of 'self.sample_size' evenly sized partition indices
            pertaining to the file file list """
        # Remove start and end values to get a list divisible by self.sample_size
        crop = self.images.images_found % self.sample_size
        top_tail = list(range(self.images.images_found))[crop // 2:- (crop - (crop // 2))]
        # Partition the indices
        size = len(top_tail)
        retval = [top_tail[start:start + size // self.sample_size]
                  for start in range(0, size, size // self.sample_size)]
        logger.debug(retval)
        return retval

    def generate(self):
        """ Generate a random test set """
        self.load_frames()
        self.predict()

    def load_frames(self):
        """ Load 'self.sample_size' random frames """
        # TODO Only read frames we need rather than all of them and discarding
        # TODO Handle frames without faces
        self.selected_frames = list()
        selection = self.random_choice
        for idx, item in enumerate(self.images.load()):
            if idx not in selection:
                continue
            filename, image = item
            face = self.alignments.get_faces_in_frame(filename)[0]
            detected_face = DetectedFace()
            detected_face.from_alignment(face, image=image)
            self.selected_frames.append({"filename": filename,
                                         "image": image,
                                         "detected_faces": [detected_face]})
        logger.debug("Selected frames: %s", [frame["filename"] for frame in self.selected_frames])

    def predict(self):
        """ Predict from the loaded frames """
        for frame in self.selected_frames:
            self.queue_predict_in.put(frame)
        self.queue_predict_in.put("EOF")
        idx = 0
        while idx < self.sample_size:
            logger.debug("Predicting face %s of %s", idx + 1, self.sample_size)
            item = self.queue_predict_out.get()
            if item == "EOF":
                logger.debug("Received EOF")
                break
            self.queue_patch_in.put(item)
            self.selected_frames[idx]["source_face"] = item["detected_faces"][0].reference_face
            self.selected_frames[idx]["swapped_face"] = item["swapped_faces"][0]
            logger.debug("Predicted face %s of %s", idx + 1, self.sample_size)
            idx += 1
        self.queue_patch_in.put("EOF")
        logger.debug("Predicted faces")


class FacesDisplay():
    """ Compiled faces into a single image """
    def __init__(self, source_faces, predicted_faces, size):
        logger.trace("Initializing %s: (source_faces shape: %s, predicted_faces shape: %s, size: "
                     "%s)",
                     self.__class__.__name__, source_faces.shape, predicted_faces.shape, size)
        self.size = size
        self.faces_src = source_faces
        self.faces_pred = predicted_faces

        self.image = self.build_faces_image()
        logger.trace("Initialized %s", self.__class__.__name__)

    def build_faces_image(self):
        """ Display associated faces """
        assert self.faces_src.shape[0] == self.faces_pred.shape[0]
        total_faces = self.faces_src.shape[0]
        logger.trace("Building faces panel. (total_faces: %s", total_faces)

        source = np.hstack([self.normalize_thumbnail(face) for face in self.faces_src])
        mask = np.hstack([self.normalize_thumbnail(face[:, :, -1]) for face in self.faces_pred])
        predicted = np.hstack([self.normalize_thumbnail(face[:, :, :3])
                               for face in self.faces_pred])
        image = np.vstack((source, mask, predicted))
        logger.debug("source row shape: %s, mask row shape: %s, predicted row shape: %s, final "
                     "image shape: %s", source.shape, mask.shape, predicted.shape, image.shape)
        return image

    def normalize_thumbnail(self, image):
        """ Resize image and draw border """
        if image.shape[0] < self.size:
            interpolation = cv2.INTER_CUBIC  # pylint:disable=no-member
        else:
            interpolation = cv2.INTER_AREA  # pylint:disable=no-member
        output = cv2.resize(image,  # pylint:disable=no-member
                            (self.size, self.size),
                            interpolation=interpolation)
        cv2.rectangle(output,    # pylint:disable=no-member
                      (0, 0),
                      (self.size - 1, self.size - 1),
                      (255, 255, 255),
                      1)
        if output.ndim == 2:
            output = np.expand_dims(output, axis=-1)
        if output.shape[-1] == 1:
            output = np.tile(output, 3)
        output = np.clip(output, 0.0, 255.0)
        return output
