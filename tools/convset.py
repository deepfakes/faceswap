#!/usr/bin/env python3
""" Tool to preview swaps and tweak the config prior to running a convert

Sketch:

    - predict 4 random faces (full set distribution)
    - keep mask + face separate
    - show faces swapped into padded square of final image
    - live update on settings change
    - apply + save config
"""

# TODO put processing in background with indicator when refreshing
# Each time processing triggers, just run with latest value, not with every value passed
import logging
import random
import tkinter as tk
from tkinter import ttk
import os
import sys
from configparser import ConfigParser

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.aligner import Extract as AlignerExtract
from lib.cli import ConvertArgs
from lib.gui.utils import ControlBuilder, get_images, initialize_images
from lib.gui.tooltip import Tooltip
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
        self.config = Config(None)

        self.converter = Converter(output_dir=None,
                                   output_size=self.faces.predictor.output_size,
                                   output_has_mask=self.faces.predictor.has_predicted_mask,
                                   draw_transparent=False,
                                   pre_encode=None,
                                   arguments=self.generate_converter_arguments(arguments))

        self.root = tk.Tk()
        pathscript = os.path.realpath(os.path.dirname(sys.argv[0]))
        pathcache = os.path.join(pathscript, "lib", "gui", ".cache")
        initialize_images(pathcache=pathcache)

        self.display = FacesDisplay(256, 64)
        self.image_canvas = None
        self.opts_book = None
        self.cli_frame = None  # cli frame holds cli options
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
        self.get_random_set(refresh=False)
        self.build_ui()
        self.root.mainloop()

    def get_random_set(self, refresh=True):
        """ Generate a set of face samples from the frames pool """
        logger.debug("Generating a random face set. refresh: %s", refresh)
        self.faces.generate()
        self.patch_faces()
        self.display.build_faces_image(self.faces.selected_frames)
        if refresh:
            self.image_canvas.reload()
        queue_manager.flush_queues()
        logger.debug("Generated a random face set.")

    def patch_faces(self):
        """ Patch faces """
        logger.trace("Patching faces")
        self.converter.process(self.queue_patch_in, self.queue_patch_out)
        idx = 0
        while idx < self.faces.sample_size:
            logger.trace("Patching image %s of %s", idx + 1, self.faces.sample_size)
            item = self.queue_patch_out.get()
            self.faces.selected_frames[idx]["swapped_image"] = item[1]
            logger.trace("Patched image %s of %s", idx + 1, self.faces.sample_size)
            idx += 1
        logger.trace("Patched faces")

    def refresh(self, *args):
        """ Refresh the display """
        # Update converter arguments
        logger.trace("Refreshing swapped faces. args: %s", args)
        for key, val in self.cli_frame.convert_args.items():
            setattr(self.converter.args, key, val)
        self.opts_book.update_config()
        self.converter.reinitialize(config=self.config)
        self.feed_swapped_faces()
        self.patch_faces()
        self.display.refresh_dest_image(self.faces.selected_frames)
        self.image_canvas.reload()
        queue_manager.flush_queues()
        logger.trace("Refreshed swapped faces")

    def feed_swapped_faces(self):
        """ Feed swapped faces to the converter """
        logger.trace("feeding swapped faces to converter")
        for item in self.faces.predicted_items:
            self.queue_patch_in.put(item)
        logger.trace("fed %s swapped faces to converter", len(self.faces.predicted_items))
        logger.trace("Putting EOF to converter")
        self.queue_patch_in.put("EOF")

    def build_ui(self):
        """ Build the UI elements for displaying preview and options """
        container = tk.PanedWindow(self.root, sashrelief=tk.RAISED, orient=tk.VERTICAL)
        container.pack(fill=tk.BOTH, expand=True)
        container.convset_display = self.display
        self.image_canvas = ImagesCanvas(container)
        container.add(self.image_canvas)

        options_frame = ttk.Frame(container)
        self.cli_frame = ActionFrame(options_frame,
                                     self.converter.args.color_adjustment.replace("-", "_"),
                                     self.converter.args.mask_type.replace("-", "_"),
                                     self.converter.args.scaling.replace("-", "_"),
                                     self.refresh,
                                     self.get_random_set)
        self.opts_book = OptionsBook(options_frame, self.config, self.refresh)
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

        self.predicted_items = list()
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
        """ Load a sample of random frames """
        self.selected_frames = list()
        selection = self.random_choice

        for pool, frame_id in enumerate(selection):
            filename, image = self.get_frame_with_face(pool, frame_id)
            face = self.alignments.get_faces_in_frame(filename)[0]
            detected_face = DetectedFace()
            detected_face.from_alignment(face, image=image)
            self.selected_frames.append({"filename": filename,
                                         "image": image,
                                         "detected_faces": [detected_face]})
        logger.debug("Selected frames: %s", [frame["filename"] for frame in self.selected_frames])

    def get_frame_with_face(self, pool, frame_id):
        """ Return the first available frame from current pool that has a face """
        while True:
            filename = frame_id + 1 if self.images.is_video else self.images.input_images[frame_id]
            image = self.images.load_one_image(filename)

            if self.images.is_video:
                # Dummy out a filename for videos
                filename, image = image

            filename = os.path.basename(filename)

            if self.alignments.frame_has_faces(filename):
                return filename, image

            logger.debug("'%s' has no faces, removing from pool", filename)
            location = self.indices[pool].index(frame_id)
            del self.indices[pool][location]

            indices = self.indices[pool]
            new_frame_id = indices[location] if location + 1 <= len(indices) else indices[0]
            logger.debug("Discarded frame_id: %s, new frame_id: %s", frame_id, new_frame_id)
            frame_id = new_frame_id

    def predict(self):
        """ Predict from the loaded frames """
        self.predicted_items = list()
        for frame in self.selected_frames:
            self.queue_predict_in.put(frame)
        idx = 0
        while idx < self.sample_size:
            logger.debug("Predicting face %s of %s", idx + 1, self.sample_size)
            item = self.queue_predict_out.get()
            if item == "EOF":
                logger.debug("Received EOF")
                break
            self.queue_patch_in.put(item)
            self.predicted_items.append(item)
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
        self.faces = dict()
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
        self.faces_from_frames(images)
        header = self.header_text(self.faces["filenames"], total_faces)
        source = np.hstack([self.draw_rect(face) for face in self.faces["src"]])
        self.faces_dest = np.hstack([self.draw_rect(face) for face in self.faces["dst"]])
        self.faces_source = np.vstack((header, source))
        logger.debug("source row shape: %s, swapped row shape: %s",
                     self.faces_dest.shape, self.faces_source.shape)

    def refresh_dest_image(self, images):
        """ Refresh the destination image
            Most times, only the destination image needs to be updated, so kept separate """
        for idx, image in enumerate(images):
            self.faces["dst"][idx] = AlignerExtract().transform(
                image["swapped_image"],
                self.faces["matrix"][idx],
                self.size,
                self.padding)
        self.faces_dest = np.hstack([self.draw_rect(face) for face in self.faces["dst"]])
        logger.debug("swapped row shape: %s", self.faces_source.shape)

    def faces_from_frames(self, images):
        """ Compile faces from the original images and return a row for each of source and dest """
        # TODO Padding from coverage
        logger.debug("Extracting faces from frames: Number images: %s", len(images))
        logger.trace("images keys: %s", [key for key in images[0].keys()])
        self.faces = dict()
        for image in images:
            detected_face = image["detected_faces"][0]
            src_img = image["image"]
            swp_img = image["swapped_image"]
            detected_face.load_aligned(src_img, self.size, align_eyes=False)
            matrix = detected_face.aligned["matrix"]
            self.faces.setdefault("matrix", list()).append(matrix)
            self.faces.setdefault("src", list()).append(AlignerExtract().transform(
                src_img,
                matrix,
                self.size,
                self.padding))
            self.faces.setdefault("dst", list()).append(AlignerExtract().transform(
                swp_img,
                matrix,
                self.size,
                self.padding))
            self.faces.setdefault("filenames",
                                  list()).append(os.path.splitext(image["filename"])[0])
        logger.debug("Extracted faces from frames: %s", {k: len(v) for k, v in self.faces.items()})

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
        self.reload()

    def reload(self):
        """ Reload the preview image """
        logger.trace("Reloading preview image")
        self.display.update_tk_image()
        self.canvas.itemconfig(self.displaycanvas, image=self.display.tk_image)


class ActionFrame(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ Frame that holds the left hand side options panel """
    def __init__(self, parent, selected_color, selected_mask, selected_scaling,
                 patch_callback, refresh_callback):
        logger.debug("Initializing %s: (selected_color: %s, selected_mask: %s, "
                     "selected_scaling: %s, patch_callback: %s, refresh_callback: %s)",
                     self.__class__.__name__, selected_color, selected_mask, selected_scaling,
                     patch_callback, refresh_callback)
        super().__init__(parent)
        self.pack(side=tk.LEFT, anchor=tk.N, fill=tk.Y)
        self.options = ["color", "mask", "scaling"]
        self.tk_vars = dict()

        d_locals = locals()
        defaults = {opt: self.format_to_display(d_locals["selected_{}".format(opt)])
                    for opt in self.options}
        self.add_comboboxes(defaults)
        self.add_refresh_button(refresh_callback)
        self.add_patch_callback(patch_callback)

    @property
    def convert_args(self):
        """ Return a dict of cli arguments for converter based on selected options """
        return {opt if opt != "color" else "color_adjustment":
                self.format_from_display(self.tk_vars[opt].get())
                for opt in self.options}

    @staticmethod
    def format_from_display(var):
        """ Format a variable from display version """
        return var.replace(" ", "_").lower()

    @staticmethod
    def format_to_display(var):
        """ Format a variable from display version """
        return var.replace("_", " ").replace("-", " ").title()

    def add_comboboxes(self, defaults):
        """ Add the comboboxes to the Action Frame """
        for opt in self.options:
            if opt == "mask":
                choices = get_available_masks() + ["predicted"]
            else:
                choices = PluginLoader.get_available_convert_plugins(opt, True)
            choices = [self.format_to_display(choice) for choice in choices]
            ctl = ControlBuilder(self,
                                 opt,
                                 str,
                                 defaults[opt],
                                 choices=choices,
                                 is_radio=False,
                                 label_width=8,
                                 control_width=12)
            self.tk_vars[opt] = ctl.tk_var

    def add_refresh_button(self, refresh_callback):
        """ Add button to refresh the images """
        btn = ttk.Button(self, text="Refresh Images", command=refresh_callback)
        btn.pack(padx=5, pady=10, side=tk.BOTTOM, fill=tk.X, anchor=tk.S)

    def add_patch_callback(self, patch_callback):
        """ Add callback to repatch images on action option change """
        for tk_var in self.tk_vars.values():
            tk_var.trace("w", patch_callback)


class OptionsBook(ttk.Notebook):  # pylint:disable=too-many-ancestors
    """ Convert settings Options Frame """
    def __init__(self, parent, config, patch_callback):
        logger.debug("Initializing %s: (parent: %s, config: %s)",
                     self.__class__.__name__, parent, config)
        super().__init__(parent)
        self.pack(side=tk.RIGHT, anchor=tk.N, fill=tk.BOTH, expand=True)
        self.config = config
        self.config_dicts = self.get_config_dicts(config)
        self.tk_vars = dict()

        self.tabs = dict()
        self.build_tabs()
        self.build_sub_tabs()
        self.add_patch_callback(patch_callback)
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

    def update_config(self):
        """ Update config with selected values """
        for section, items in self.tk_vars.items():
            for item, value in items.items():
                new_value = str(value.get())
                old_value = self.config.config[section][item]
                if new_value != old_value:
                    logger.trace("Updating config: %s, %s from %s to %s",
                                 section, item, old_value, new_value)
                    self.config.config[section][item] = str(value.get())

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
                config_key = ".".join((section, plugin))
                config_dict = self.config_dicts[config_key]
                tab = ConfigFrame(self,
                                  config_key,
                                  config_dict)
                self.tabs[section][plugin] = tab
                self.tabs[section]["tab"].add(tab, text=plugin.replace("_", " ").title())

    def add_patch_callback(self, patch_callback):
        """ Add callback to repatch images on config option change """
        for plugins in self.tk_vars.values():
            for tk_var in plugins.values():
                tk_var.trace("w", patch_callback)

    def reset_config_saved(self, section):
        """ Reset config to saved values """
        logger.debug("Resetting to saved config: %s", section)
        for item, options in self.config_dicts[section].items():
            if item == "helptext":
                continue
            val = options["value"]
            if val != self.tk_vars[section][item].get():
                self.tk_vars[section][item].set(val)
                logger.debug("Setting %s - %s to saved value %s", section, item, val)
        logger.debug("Reset to saved config: %s", section)

    def reset_config_default(self, section):
        """ Reset config to default values """
        logger.debug("Resetting to default: %s", section)
        for item, options in self.config.defaults[section].items():
            if item == "helptext":
                continue
            default = options["default"]
            if default != self.tk_vars[section][item].get():
                self.tk_vars[section][item].set(default)
                logger.debug("Setting %s - %s to default value %s", section, item, default)
        logger.debug("Reset to default: %s", section)

    def save_config(self, section):
        """ Save config """
        logger.debug("Saving %s config", section)
        new_config = ConfigParser(allow_no_value=True)
        for config_section, items in self.config_dicts.items():
            logger.debug("Adding section: '%s')", config_section)
            self.config.insert_config_section(config_section, items["helptext"], config=new_config)
            for item, options in items.items():
                if item == "helptext":
                    continue
                if config_section != section:
                    new_opt = options["value"]  # Keep saved item for other sections
                    logger.debug("Retaining option: (item: '%s', value: '%s')", item, new_opt)
                else:
                    new_opt = self.tk_vars[section][item].get()
                    logger.debug("Setting option: (item: '%s', value: '%s')", item, new_opt)
                helptext = options["helptext"]
                helptext = self.config.format_help(helptext, is_section=False)
                new_config.set(config_section, helptext)
                new_config.set(config_section, item, str(new_opt))
        self.config.config = new_config
        self.config.save_config()
        print("Saved config: '{}'".format(self.config.configfile))
        logger.debug("Saved config")


class ConfigFrame(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ Config Frame - Holds the Options for config """

    def __init__(self, parent, config_key, options):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.options = options

        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.optsframe = ttk.Frame(self.canvas)
        self.optscanvas = self.canvas.create_window((0, 0), window=self.optsframe, anchor=tk.NW)

        self.build_frame(parent, config_key)
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_frame(self, parent, config_key):
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
            parent.tk_vars.setdefault(config_key, dict())[key] = ctl.tk_var
        self.add_frame_separator()
        self.add_actions(parent, config_key)
        logger.debug("Added Config Frame")

    def add_scrollbar(self):
        """ Add a scrollbar to the options frame """
        logger.debug("Add Config Scrollbar")
        scrollbar = ttk.Scrollbar(self.canvas_frame, command=self.canvas.yview)
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

    def add_frame_separator(self):
        """ Add a separator between top and bottom frames """
        logger.debug("Add frame seperator")
        sep = ttk.Frame(self, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.TOP)
        logger.debug("Added frame seperator")

    def add_actions(self, parent, config_key):
        """ Add Actio Buttons """
        logger.debug("Adding util buttons")
        action_frame = ttk.Frame(self)
        action_frame.pack(padx=5, pady=5, side=tk.BOTTOM, fill=tk.Y, expand=True, anchor=tk.E)

        title = config_key.split(".")[1].replace("_", " ").title()
        for utl in ("save", "clear", "reset"):
            logger.debug("Adding button: '%s'", utl)
            img = get_images().icons[utl]
            if utl == "save":
                text = "Save {} config".format(title)
                action = parent.save_config
            elif utl == "clear":
                text = "Reset {} config to default values".format(title)
                action = parent.reset_config_default
            elif utl == "reset":
                text = "Reset {} config to saved values".format(title)
                action = parent.reset_config_saved

            btnutl = ttk.Button(action_frame,
                                image=img,
                                command=lambda cmd=action: cmd(config_key))
            btnutl.pack(padx=2, side=tk.RIGHT)
            Tooltip(btnutl, text=text, wraplength=200)
        logger.debug("Added util buttons")
