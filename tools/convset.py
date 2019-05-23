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
import tkinter as tk
from tkinter import ttk
import os

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.aligner import Extract as AlignerExtract
from lib.cli import ConvertArgs
from lib.gui.utils import ControlBuilder
from lib.convert import Converter
from lib.faces_detect import DetectedFace
from lib.model.masks import get_available_masks
from lib.utils import set_system_verbosity
from lib.queue_manager import queue_manager
from scripts.fsmedia import Alignments, Images
from scripts.convert import Predict

from plugins.plugin_loader import PluginLoader
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
        self.config = Config(None)

        self.converter = Converter(output_dir=None,
                                   output_size=self.faces.predictor.output_size,
                                   output_has_mask=self.faces.predictor.has_predicted_mask,
                                   draw_transparent=False,
                                   pre_encode=None,
                                   arguments=self.arguments)

        self.root = tk.Tk()
        self.display = FacesDisplay(256, 64)
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
        self.build_ui()
        self.root.mainloop()

    def build_ui(self):
        """ Build the UI elements for displaying preview and options """
        container = tk.PanedWindow(self.root, sashrelief=tk.RAISED, orient=tk.VERTICAL)
        container.pack(fill=tk.BOTH, expand=True)
        container.convset_display = self.display
        container.add(ImagesCanvas(container))

        options_frame = ttk.Frame(container)
        ActionFrame(options_frame,
                    self.arguments.color_adjustment.replace("-", "_"),
                    self.arguments.mask_type.replace("-", "_"),
                    self.arguments.scaling.replace("-", "_"))
        OptionsBook(options_frame, self.config)
        container.add(options_frame)


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


class ImagesCanvas(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Canvas to hold the images """
    def __init__(self, parent):
        logger.debug("Initializing %s: (parent: %s)", self.__class__.__name__, parent)
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


class ActionFrame(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ Frame that holds the left hand side options panel """
    def __init__(self, parent, selected_color_method, selected_mask, selected_scaling):
        logger.debug("Initializing %s: (selected_color_method: %s, selected_mask: %s, "
                     "selected_scaling: %s)", self.__class__.__name__, selected_color_method,
                     selected_mask, selected_scaling)
        super().__init__(parent)
        self.pack(side=tk.LEFT, anchor=tk.N, fill=tk.Y)

        self.color_var = tk.StringVar()
        self.color_var.set(self.format_to_display(selected_color_method))

        self.mask_var = tk.StringVar()
        self.mask_var.set(self.format_to_display(selected_mask))

        self.scaling_var = tk.StringVar()
        self.scaling_var.set(self.format_to_display(selected_scaling))

        self.add_color_combobox()
        self.add_mask_combobox()
        self.add_scaling_combobox()
        self.add_refresh_button()

    @staticmethod
    def format_from_display(var):
        """ Format a variable from display version """
        return var.replace(" ", "_").lower()

    @staticmethod
    def format_to_display(var):
        """ Format a variable from display version """
        return var.replace("_", " ").title()

    def add_color_combobox(self):
        """ Add the color adjustment method Combo Box """
        logger.debug("Adding color method Combo Box")
        color_methods = PluginLoader.get_available_convert_plugins("color", True)
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP)
        lbl = ttk.Label(frame, text="Color Adjustment", width=16, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)

        ctl = ttk.Combobox(frame, textvariable=self.color_var, width=12)
        ctl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)
        ctl["values"] = sorted([self.format_to_display(choice) for choice in color_methods])
        logger.debug("Added color method Combo Box")

    def add_mask_combobox(self):
        """ Add the mask Combo Box """
        logger.debug("Adding mask Combo Box")
        masks = get_available_masks() + ["predicted"]
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP)
        lbl = ttk.Label(frame, text="Mask", width=16, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)

        ctl = ttk.Combobox(frame, textvariable=self.mask_var, width=12)
        ctl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)
        ctl["values"] = sorted([self.format_to_display(choice) for choice in masks])
        logger.debug("Added mask Combo Box")

    def add_scaling_combobox(self):
        """ Add the scaling method Combo Box """
        logger.debug("Adding scaling method Combo Box")
        scaling = PluginLoader.get_available_convert_plugins("scaling", True)
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP)
        lbl = ttk.Label(frame, text="Scaling", width=16, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)

        ctl = ttk.Combobox(frame, textvariable=self.scaling_var, width=12)
        ctl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)
        ctl["values"] = sorted([self.format_to_display(choice) for choice in scaling])
        logger.debug("Added scaling Combo Box")

    def add_refresh_button(self):
        """ Add button to refresh the images """
        btn = ttk.Button(self, text="Refresh Images", command=self.callback)
        btn.pack(padx=5, pady=10, side=tk.BOTTOM, fill=tk.X, anchor=tk.S)

    @staticmethod
    def callback():
        # TODO Change this to do something
        print("click!")


class OptionsBook(ttk.Notebook):  # pylint:disable=too-many-ancestors
    """ Convert settings Options Frame """
    def __init__(self, parent, config):
        logger.debug("Initializing %s: (parent: %s, config: %s)",
                     self.__class__.__name__, parent, config)
        super().__init__(parent)
        self.pack(side=tk.RIGHT, anchor=tk.N, fill=tk.BOTH, expand=True)
        self.config = config
        self.config_dicts = self.get_config_dicts(config)

        self.tabs = dict()
        self.build_tabs()
        self.build_sub_tabs()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def sections(self):
        """ Return the sorted unique section names from the configs """
        return sorted(set(plugin.split(".")[0] for plugin in self.config.config.sections()
                          if plugin.split(".")[0] != "writer"))

    @property
    def plugins_dict(self):
        """ Return dict of sections with sorted list of containing plugins """
        return {section: sorted([plugin.split(".")[1] for plugin in self.config.config.sections()
                                 if plugin.split(".")[0] == section])
                for section in self.sections}

    def get_config_dicts(self, config):
        """ Hold a custom config dict for the config """
        config_dicts = dict()
        for section in self.config.config.sections():
            if section == "writer":
                continue
            default_dict = config.defaults[section]
            for key in default_dict.keys():
                if key == "helptext":
                    continue
                default_dict[key]["value"] = config.get(section, key)
            config_dicts[section] = default_dict
        return config_dicts

    def build_tabs(self):
        """ Build the tabs for the relevant section """
        logger.debug("Build Tabs")
        for section in self.sections:
            tab = ttk.Notebook(self)
            self.tabs[section] = {"tab": tab}
            self.add(tab, text=section.replace("_", " ").title())

    def build_sub_tabs(self):
        """ Build the sub tabs for the relevant plugin """
        for section, plugins in self.plugins_dict.items():
            for plugin in plugins:
                config_dict = self.config_dicts[".".join((section, plugin))]
                tab = ConfigFrame(self,
                                  config_dict)
                self.tabs[section][plugin] = tab
                self.tabs[section]["tab"].add(tab, text=plugin.replace("_", " ").title())


class ConfigFrame(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ Config Frame - Holds the Options for config """

    def __init__(self, parent, options):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.options = options

        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.optsframe = ttk.Frame(self.canvas)
        self.optscanvas = self.canvas.create_window((0, 0), window=self.optsframe, anchor=tk.NW)

        self.build_frame()
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_frame(self):
        """ Build the options frame for this command """
        logger.debug("Add Config Frame")
        self.add_scrollbar()
        self.canvas.bind("<Configure>", self.resize_frame)

        for key, val in self.options.items():
            if key == "helptext":
                continue
            value = val.get("value", val["default"])
            ctl = ControlBuilder(self.optsframe,
                                 key,
                                 val["type"],
                                 value,
                                 selected_value=None,
                                 choices=val["choices"],
                                 is_radio=val["gui_radio"],
                                 rounding=val["rounding"],
                                 min_max=val["min_max"],
                                 helptext=val["helptext"],
                                 radio_columns=4)
            val["selected"] = ctl.tk_var
        logger.debug("Added Config Frame")

    def add_scrollbar(self):
        """ Add a scrollbar to the options frame """
        logger.debug("Add Config Scrollbar")
        scrollbar = ttk.Scrollbar(self, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=scrollbar.set)
        self.optsframe.bind("<Configure>", self.update_scrollbar)
        logger.debug("Added Config Scrollbar")

    def update_scrollbar(self, event):  # pylint: disable=unused-argument
        """ Update the options frame scrollbar """
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def resize_frame(self, event):
        """ Resize the options frame to fit the canvas """
        logger.debug("Resize Config Frame")
        canvas_width = event.width
        self.canvas.itemconfig(self.optscanvas, width=canvas_width)
        logger.debug("Resized Config Frame")
