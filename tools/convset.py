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
import tkinter as tk
from tkinter import ttk
import os

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.aligner import Extract as AlignerExtract
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

        self.converter = Converter(output_dir=None,
                                   output_size=self.faces.predictor.output_size,
                                   output_has_mask=self.faces.predictor.has_predicted_mask,
                                   draw_transparent=False,
                                   pre_encode=None,
                                   arguments=self.arguments)

        self.root = tk.Tk()
        self.display = FacesDisplay(256, 64)
        self.root.convset_display = self.display
        self.image_canvas = None
        self.opts_canvas = None
        # TODO Padding + Size dynamic

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
        self.display.build_faces_image(self.faces.selected_frames)
        self.image_canvas = ImagesCanvas(self.root)

        self.root.mainloop()


class ImagesCanvas(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Canvas to hold the images """
    def __init__(self, parent):
        logger.debug("Initializing : %s", self.__class__.__name__)
        super().__init__(parent)
        self.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

        self.display = parent.convset_display
        self.display.update_tk_image()
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.displaycanvas = self.canvas.create_image(0, 0,
                                                      image=self.display.tk_image,
                                                      anchor=tk.NW)
        self.bind("<Configure>", self.resize)
        logger.debug("Initialized %s", self.__class__.__name__)

    def resize(self, event):
        """  Resize the image to fit the frame, maintaining aspect ratio """
        logger.trace("Resizing preview image")
        framesize = (event.width, event.height)
        self.display.display_dims = framesize
        self.display.update_tk_image()
        self.reload()

    def reload(self):
        """ Reload the preview image """
        logger.trace("Reloading preview image")
        self.canvas.itemconfig(self.displaycanvas, image=self.display.tk_image)


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
        top_tail = list(range(self.images.images_found))[
            crop // 2:self.images.images_found - (crop - (crop // 2))]
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
            filename = os.path.basename(filename)
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
    def __init__(self, size, padding):
        logger.trace("Initializing %s: (size: %s, padding: %s)",
                     self.__class__.__name__, size, padding)
        self.size = size
        self.display_dims = (1, 1)
        self.padding = padding
        self.faces_source = None
        self.faces_dest = None
        self.tk_image = None
        logger.trace("Initialized %s", self.__class__.__name__)

    def update_tk_image(self):
        """ Return compiled images images in TK PIL format resized for frame """
        img = np.vstack((self.faces_source, self.faces_dest))
        size = self.get_scale_size(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # pylint:disable=no-member
        img = Image.fromarray(img)
        img = img.resize(size, Image.ANTIALIAS)
        self.tk_image = ImageTk.PhotoImage(img)

    def get_scale_size(self, image):
        """ Return the scale and size for passed in display image """
        frameratio = float(self.display_dims[0]) / float(self.display_dims[1])
        imgratio = float(image.shape[1]) / float(image.shape[0])

        if frameratio <= imgratio:
            scale = self.display_dims[0] / float(image.shape[1])
            size = (self.display_dims[0], max(1, int(image.shape[0] * scale)))
        else:
            scale = self.display_dims[1] / float(image.shape[0])
            size = (max(1, int(image.shape[1] * scale)), self.display_dims[1])
        logger.trace("scale: %s, size: %s", scale, size)
        return size

    def build_faces_image(self, images):
        """ Display associated faces """
        total_faces = len(images)
        faces = self.faces_from_frames(images)
        header = self.header_text(faces["filenames"], total_faces)
        source = np.hstack([self.draw_rect(face) for face in faces["src"]])
        self.faces_dest = np.hstack([self.draw_rect(face) for face in faces["dst"]])
        self.faces_source = np.vstack((header, source))
        logger.debug("source row shape: %s, swapped row shape: %s",
                     self.faces_dest.shape, self.faces_source.shape)

    def faces_from_frames(self, images):
        """ Compile faces from the original images and return a row for each of source and dest """
        # TODO Padding from coverage
        logger.debug("Extracting faces from frames")
        faces = dict()
        for image in images:
            detected_face = image["detected_faces"][0]
            src_img = image["image"]
            swp_img = image["swapped_image"]
            detected_face.load_aligned(src_img, self.size, align_eyes=False)
            faces.setdefault("src", list()).append(AlignerExtract().transform(
                src_img,
                detected_face.aligned["matrix"],
                self.size,
                self.padding))
            faces.setdefault("dst", list()).append(AlignerExtract().transform(
                swp_img,
                detected_face.aligned["matrix"],
                self.size,
                self.padding))
            faces.setdefault("filenames", list()).append(os.path.splitext(image["filename"])[0])
        logger.debug("Extracted faces from frames: %s", {k: len(v) for k, v in faces.items()})
        return faces

    def header_text(self, filenames, total_faces):
        """ Create header text for output image """
        font_scale = self.size / 640
        height = self.size // 8
        font = cv2.FONT_HERSHEY_SIMPLEX  # pylint: disable=no-member
        # Get size of placed text for positioning
        text_sizes = [cv2.getTextSize(filenames[idx],  # pylint: disable=no-member
                                      font,
                                      font_scale,
                                      1)[0]
                      for idx in range(total_faces)]
        # Get X and Y co-ords for each text item
        text_y = int((height + text_sizes[0][1]) / 2)
        text_x = [int((self.size - text_sizes[idx][0]) / 2) + self.size * idx
                  for idx in range(total_faces)]
        logger.debug("filenames: %s, text_sizes: %s, text_x: %s, text_y: %s",
                     filenames, text_sizes, text_x, text_y)
        header_box = np.ones((height, self.size * total_faces, 3), np.uint8) * 255
        for idx, text in enumerate(filenames):
            cv2.putText(header_box,  # pylint: disable=no-member
                        text,
                        (text_x[idx], text_y),
                        font,
                        font_scale,
                        (0, 0, 0),
                        1,
                        lineType=cv2.LINE_AA)  # pylint: disable=no-member
        logger.debug("header_box.shape: %s", header_box.shape)
        return header_box

    def draw_rect(self, image):
        """ draw border """
        cv2.rectangle(image,    # pylint:disable=no-member
                      (0, 0),
                      (self.size - 1, self.size - 1),
                      (255, 255, 255),
                      1)
        image = np.clip(image, 0.0, 255.0)
        return image.astype("uint8")
