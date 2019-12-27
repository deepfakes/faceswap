#!/usr/bin/env python3
""" Tool to preview swaps and tweak the config prior to running a convert """

import logging
import random
import tkinter as tk
from tkinter import ttk
import os

from configparser import ConfigParser
from threading import Event, Lock

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.aligner import Extract as AlignerExtract
from lib.cli import ConvertArgs
from lib.gui.custom_widgets import ContextMenu
from lib.gui.utils import get_images, initialize_config, initialize_images
from lib.gui.custom_widgets import Tooltip
from lib.gui.control_helper import set_slider_rounding
from lib.convert import Converter
from lib.faces_detect import DetectedFace
from lib.model.masks import get_available_masks
from lib.multithreading import MultiThread
from lib.utils import FaceswapError
from lib.queue_manager import queue_manager
from scripts.fsmedia import Alignments, Images
from scripts.convert import Predict

from plugins.plugin_loader import PluginLoader
from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Preview():
    """ Loads up 5 semi-random face swaps and displays them, cropped, in place in the final frame.
        Allows user to live tweak settings, before saving the final config to
        ./config/convert.ini """

    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        self.config_tools = ConfigTools()
        self.lock = Lock()
        self.trigger_patch = Event()

        self.root = tk.Tk()
        self.scaling = self.get_scaling()

        self.tk_vars = dict(refresh=tk.BooleanVar(), busy=tk.BooleanVar())
        for val in self.tk_vars.values():
            val.set(False)
        self.display = FacesDisplay(256, 64, self.tk_vars)
        self.samples = Samples(arguments, 5, self.display, self.lock, self.trigger_patch)
        self.patch = Patch(arguments,
                           self.samples,
                           self.display,
                           self.lock,
                           self.trigger_patch,
                           self.config_tools,
                           self.tk_vars)

        self.initialize_tkinter()
        self.image_canvas = None
        self.opts_book = None
        self.cli_frame = None  # cli frame holds cli options
        logger.debug("Initialized %s", self.__class__.__name__)

    def initialize_tkinter(self):
        """ Initialize tkinter for standalone or GUI """
        logger.debug("Initializing tkinter")
        initialize_config(self.root, None, None, None)
        initialize_images()
        self.set_geometry()
        self.root.title("Faceswap.py - Convert Settings")
        self.root.tk.call(
            "wm",
            "iconphoto",
            self.root._w, get_images().icons["favicon"])  # pylint:disable=protected-access
        logger.debug("Initialized tkinter")

    def get_scaling(self):
        """ Get dpi and update scaling for the display """
        dpi = self.root.winfo_fpixels("1i")
        scaling = dpi / 72.0
        logger.debug("dpi: %s, scaling: %s'", dpi, scaling)
        return scaling

    def set_geometry(self):
        """ Set GUI geometry """
        self.root.tk.call("tk", "scaling", self.scaling)
        width = int(940 * self.scaling)
        height = int(600 * self.scaling)
        logger.debug("Geometry: %sx%s", width, height)
        self.root.geometry("{}x{}+80+80".format(str(width), str(height)))

    def process(self):
        """ The preview process """
        self.build_ui()
        self.root.mainloop()

    def refresh(self, *args):
        """ Refresh the display """
        logger.trace("Refreshing swapped faces. args: %s", args)
        self.tk_vars["busy"].set(True)
        self.config_tools.update_config()
        with self.lock:
            self.patch.converter_arguments = self.cli_frame.convert_args
            self.patch.current_config = self.config_tools.config
        self.patch.trigger.set()
        logger.trace("Refreshed swapped faces")

    def build_ui(self):
        """ Build the UI elements for displaying preview and options """
        container = tk.PanedWindow(self.root,
                                   sashrelief=tk.RIDGE,
                                   sashwidth=4,
                                   sashpad=8,
                                   orient=tk.VERTICAL)
        container.pack(fill=tk.BOTH, expand=True)
        container.preview_display = self.display
        self.image_canvas = ImagesCanvas(container, self.tk_vars)
        container.add(self.image_canvas, height=400 * self.scaling)

        options_frame = ttk.Frame(container)
        self.cli_frame = ActionFrame(options_frame,
                                     self.patch.converter.args.color_adjustment.replace("-", "_"),
                                     self.patch.converter.args.mask_type.replace("-", "_"),
                                     self.patch.converter.args.scaling.replace("-", "_"),
                                     self.config_tools,
                                     self.refresh,
                                     self.samples.generate,
                                     self.tk_vars)
        self.opts_book = OptionsBook(options_frame, self.config_tools, self.refresh, self.scaling)
        container.add(options_frame)


class Samples():
    """ Holds 5 random test faces """

    def __init__(self, arguments, sample_size, display, lock, trigger_patch):
        logger.debug("Initializing %s: (arguments: '%s', sample_size: %s, display: %s, lock: %s, "
                     "trigger_patch: %s)", self.__class__.__name__, arguments, sample_size,
                     display, lock, trigger_patch)
        self.sample_size = sample_size
        self.display = display
        self.lock = lock
        self.trigger_patch = trigger_patch
        self.input_images = list()
        self.predicted_images = list()

        self.images = Images(arguments)
        self.alignments = Alignments(arguments,
                                     is_extract=False,
                                     input_is_video=self.images.is_video)
        if not self.alignments.have_alignments_file:
            logger.error("Alignments file not found at: '%s'", self.alignments.file)
            exit(1)
        self.filelist = self.get_filelist()
        self.indices = self.get_indices()

        self.predictor = Predict(queue_manager.get_queue("preview_predict_in"),
                                 sample_size,
                                 arguments)
        self.generate()

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def random_choice(self):
        """ Return for random indices from the indices group """
        retval = [random.choice(indices) for indices in self.indices]
        logger.debug(retval)
        return retval

    def get_filelist(self):
        """ Return a list of files, filtering out those frames which do not contain faces """
        logger.debug("Filtering file list to frames with faces")
        if self.images.is_video:
            filelist = ["{}_{:06d}.png".format(os.path.splitext(self.images.input_images)[0],
                                               frame_no)
                        for frame_no in range(1, self.images.images_found + 1)]
        else:
            filelist = self.images.input_images

        retval = [filename for filename in filelist
                  if self.alignments.frame_has_faces(os.path.basename(filename))]
        logger.debug("Filtered out frames: %s", self.images.images_found - len(retval))
        try:
            assert retval
        except AssertionError as err:
            msg = ("No faces were found in any of the frames passed in. Make sure you are passing "
                   "in a frames source rather than extracted faces, and that you have provided "
                   "the correct alignments file.")
            raise FaceswapError(msg) from err
        return retval

    def get_indices(self):
        """ Returns a list of 'self.sample_size' evenly sized partition indices
            pertaining to the filtered file list """
        # Remove start and end values to get a list divisible by self.sample_size
        no_files = len(self.filelist)
        crop = no_files % self.sample_size
        top_tail = list(range(no_files))[
            crop // 2:no_files - (crop - (crop // 2))]
        # Partition the indices
        size = len(top_tail)
        retval = [top_tail[start:start + size // self.sample_size]
                  for start in range(0, size, size // self.sample_size)]
        logger.debug("Indices pools: %s", ["{}: (start: {}, end: {}, size: {})".format(idx,
                                                                                       min(pool),
                                                                                       max(pool),
                                                                                       len(pool))
                                           for idx, pool in enumerate(retval)])
        return retval

    def generate(self):
        """ Generate a random test set """
        self.load_frames()
        self.predict()
        self.trigger_patch.set()

    def load_frames(self):
        """ Load a sample of random frames """
        self.input_images = list()
        for selection in self.random_choice:
            filename = os.path.basename(self.filelist[selection])
            image = self.images.load_one_image(self.filelist[selection])
            # Get first face only
            face = self.alignments.get_faces_in_frame(filename)[0]
            detected_face = DetectedFace()
            detected_face.from_alignment(face, image=image)
            self.input_images.append({"filename": filename,
                                      "image": image,
                                      "detected_faces": [detected_face]})
        self.display.source = self.input_images
        self.display.update_source = True
        logger.debug("Selected frames: %s", [frame["filename"] for frame in self.input_images])

    def predict(self):
        """ Predict from the loaded frames """
        with self.lock:
            self.predicted_images = list()
            for frame in self.input_images:
                self.predictor.in_queue.put(frame)
            idx = 0
            while idx < self.sample_size:
                logger.debug("Predicting face %s of %s", idx + 1, self.sample_size)
                items = self.predictor.out_queue.get()
                if items == "EOF":
                    logger.debug("Received EOF")
                    break
                for item in items:
                    self.predicted_images.append(item)
                    logger.debug("Predicted face %s of %s", idx + 1, self.sample_size)
                    idx += 1
        logger.debug("Predicted faces")


class Patch():
    """ The patch pipeline
        To be run within it's own thread """
    def __init__(self, arguments, samples, display, lock, trigger, config_tools, tk_vars):
        logger.debug("Initializing %s: (arguments: '%s', samples: %s: display: %s, lock: %s,"
                     " trigger: %s, config_tools: %s, tk_vars %s)", self.__class__.__name__,
                     arguments, samples, display, lock, trigger, config_tools, tk_vars)
        self.samples = samples
        self.queue_patch_in = queue_manager.get_queue("preview_patch_in")
        self.display = display
        self.lock = lock
        self.trigger = trigger
        self.current_config = config_tools.config
        self.converter_arguments = None  # Updated converter arguments dict

        configfile = arguments.configfile if hasattr(arguments, "configfile") else None
        self.converter = Converter(output_dir=None,
                                   output_size=self.samples.predictor.output_size,
                                   output_has_mask=self.samples.predictor.has_predicted_mask,
                                   draw_transparent=False,
                                   pre_encode=None,
                                   configfile=configfile,
                                   arguments=self.generate_converter_arguments(arguments))

        self.shutdown = Event()

        self.thread = MultiThread(self.process,
                                  self.trigger,
                                  self.shutdown,
                                  self.queue_patch_in,
                                  self.samples,
                                  tk_vars,
                                  thread_count=1,
                                  name="patch_thread")
        self.thread.start()

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

    def process(self, trigger_event, shutdown_event, patch_queue_in, samples, tk_vars):
        """ Wait for event trigger and run when process when set """
        patch_queue_out = queue_manager.get_queue("preview_patch_out")
        while True:
            trigger = trigger_event.wait(1)
            if shutdown_event.is_set():
                logger.debug("Shutdown received")
                break
            if not trigger:
                continue
            # Clear trigger so calling process can set it during this run
            trigger_event.clear()
            tk_vars["busy"].set(True)
            queue_manager.flush_queue("preview_patch_in")
            self.feed_swapped_faces(patch_queue_in, samples)
            with self.lock:
                self.update_converter_arguments()
                self.converter.reinitialize(config=self.current_config)
            swapped = self.patch_faces(patch_queue_in, patch_queue_out, samples.sample_size)
            with self.lock:
                self.display.destination = swapped
            tk_vars["refresh"].set(True)
            tk_vars["busy"].set(False)

    def update_converter_arguments(self):
        """ Update the converter arguments """
        logger.debug("Updating Converter cli arguments")
        if self.converter_arguments is None:
            logger.debug("No arguments to update")
            return
        for key, val in self.converter_arguments.items():
            logger.debug("Updating %s to %s", key, val)
            setattr(self.converter.args, key, val)
        logger.debug("Updated Converter cli arguments")

    @staticmethod
    def feed_swapped_faces(patch_queue_in, samples):
        """ Feed swapped faces to the converter and trigger a run """
        logger.trace("feeding swapped faces to converter")
        for item in samples.predicted_images:
            patch_queue_in.put(item)
        logger.trace("fed %s swapped faces to converter", len(samples.predicted_images))
        logger.trace("Putting EOF to converter")
        patch_queue_in.put("EOF")

    def patch_faces(self, queue_in, queue_out, sample_size):
        """ Patch faces """
        logger.trace("Patching faces")
        self.converter.process(queue_in, queue_out)
        swapped = list()
        idx = 0
        while idx < sample_size:
            logger.trace("Patching image %s of %s", idx + 1, sample_size)
            item = queue_out.get()
            swapped.append(item[1])
            logger.trace("Patched image %s of %s", idx + 1, sample_size)
            idx += 1
        logger.trace("Patched faces")
        return swapped


class FacesDisplay():
    """ Compiled faces into a single image """
    def __init__(self, size, padding, tk_vars):
        logger.trace("Initializing %s: (size: %s, padding: %s, tk_vars: %s)",
                     self.__class__.__name__, size, padding, tk_vars)
        self.size = size
        self.display_dims = (1, 1)
        self.tk_vars = tk_vars
        self.padding = padding

        # Set from Samples
        self.update_source = False
        self.source = list()  # Source images, filenames + detected faces
        # Set from Patch
        self.destination = list()  # Swapped + patched images

        self.faces = dict()
        self.faces_source = None
        self.faces_dest = None
        self.tk_image = None
        logger.trace("Initialized %s", self.__class__.__name__)

    @property
    def total_columns(self):
        """ Return the total number of images that are being displayed """
        return len(self.source)

    def update_tk_image(self):
        """ Return compiled images images in TK PIL format resized for frame """
        logger.trace("Updating tk image")
        self.build_faces_image()
        img = np.vstack((self.faces_source, self.faces_dest))
        size = self.get_scale_size(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize(size, Image.ANTIALIAS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.tk_vars["refresh"].set(False)
        logger.trace("Updated tk image")

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

    def build_faces_image(self):
        """ Display associated faces """
        logger.trace("Building Faces Image")
        update_all = self.update_source
        self.faces_from_frames()
        if update_all:
            header = self.header_text()
            source = np.hstack([self.draw_rect(face) for face in self.faces["src"]])
            self.faces_source = np.vstack((header, source))
        self.faces_dest = np.hstack([self.draw_rect(face) for face in self.faces["dst"]])
        logger.debug("source row shape: %s, swapped row shape: %s",
                     self.faces_dest.shape, self.faces_source.shape)

    def faces_from_frames(self):
        """ Compile faces from the original images and return a row for each of source and dest """
        logger.debug("Extracting faces from frames: Number images: %s", len(self.source))
        if self.update_source:
            self.crop_source_faces()
        self.crop_destination_faces()
        logger.debug("Extracted faces from frames: %s", {k: len(v) for k, v in self.faces.items()})

    def crop_source_faces(self):
        """ Update the main faces dict with new source faces and matrices """
        logger.debug("Updating source faces")
        self.faces = dict()
        for image in self.source:
            detected_face = image["detected_faces"][0]
            src_img = image["image"]
            detected_face.load_aligned(src_img, self.size)
            matrix = detected_face.aligned["matrix"]
            self.faces.setdefault("filenames",
                                  list()).append(os.path.splitext(image["filename"])[0])
            self.faces.setdefault("matrix", list()).append(matrix)
            self.faces.setdefault("src", list()).append(AlignerExtract().transform(
                src_img,
                matrix,
                self.size,
                self.padding))
        self.update_source = False
        logger.debug("Updated source faces")

    def crop_destination_faces(self):
        """ Update the main faces dict with new destination faces based on source matrices """
        logger.debug("Updating destination faces")
        self.faces["dst"] = list()
        destination = self.destination if self.destination else [np.ones_like(src["image"])
                                                                 for src in self.source]
        for idx, image in enumerate(destination):
            self.faces["dst"].append(AlignerExtract().transform(
                image,
                self.faces["matrix"][idx],
                self.size,
                self.padding))
        logger.debug("Updated destination faces")

    def header_text(self):
        """ Create header text for output image """
        font_scale = self.size / 640
        height = self.size // 8
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Get size of placed text for positioning
        text_sizes = [cv2.getTextSize(self.faces["filenames"][idx],
                                      font,
                                      font_scale,
                                      1)[0]
                      for idx in range(self.total_columns)]
        # Get X and Y co-ords for each text item
        text_y = int((height + text_sizes[0][1]) / 2)
        text_x = [int((self.size - text_sizes[idx][0]) / 2) + self.size * idx
                  for idx in range(self.total_columns)]
        logger.debug("filenames: %s, text_sizes: %s, text_x: %s, text_y: %s",
                     self.faces["filenames"], text_sizes, text_x, text_y)
        header_box = np.ones((height, self.size * self.total_columns, 3), np.uint8) * 255
        for idx, text in enumerate(self.faces["filenames"]):
            cv2.putText(header_box,
                        text,
                        (text_x[idx], text_y),
                        font,
                        font_scale,
                        (0, 0, 0),
                        1,
                        lineType=cv2.LINE_AA)
        logger.debug("header_box.shape: %s", header_box.shape)
        return header_box

    def draw_rect(self, image):
        """ draw border """
        cv2.rectangle(image, (0, 0), (self.size - 1, self.size - 1), (255, 255, 255), 1)
        image = np.clip(image, 0.0, 255.0)
        return image.astype("uint8")


class ConfigTools():
    """ Saving and resetting config values and stores selected variables """
    def __init__(self):
        self.config = Config(None)
        self.config_dicts = self.get_config_dicts()  # Holds currently saved config
        self.tk_vars = dict()

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
                try:
                    new_value = str(value.get())
                except tk.TclError as err:
                    # When manually filling in text fields, blank values will
                    # raise an error on numeric datatypes so return 0
                    logger.debug("Error getting value. Defaulting to 0. Error: %s", str(err))
                    new_value = str(0)
                old_value = self.config.config[section][item]
                if new_value != old_value:
                    logger.trace("Updating config: %s, %s from %s to %s",
                                 section, item, old_value, new_value)
                    self.config.config[section][item] = new_value

    def get_config_dicts(self):
        """ Hold a custom config dict for the config """
        config_dicts = dict()
        for section in self.config.config.sections():
            if section == "writer":
                continue
            default_dict = self.config.defaults[section]
            for key in default_dict.keys():
                if key == "helptext":
                    continue
                default_dict[key]["value"] = self.config.get(section, key)
            config_dicts[section] = default_dict
        return config_dicts

    def reset_config_saved(self, section=None):
        """ Reset config to saved values """
        logger.debug("Resetting to saved config: %s", section)
        sections = [section] if section is not None else list(self.tk_vars.keys())
        for config_section in sections:
            for item, options in self.config_dicts[config_section].items():
                if item == "helptext":
                    continue
                val = options["value"]
                if val != self.tk_vars[config_section][item].get():
                    self.tk_vars[config_section][item].set(val)
                    logger.debug("Setting %s - %s to saved value %s", config_section, item, val)
        logger.debug("Reset to saved config: %s", section)

    def reset_config_default(self, section=None):
        """ Reset config to default values """
        logger.debug("Resetting to default: %s", section)
        sections = [section] if section is not None else list(self.tk_vars.keys())
        for config_section in sections:
            for item, options in self.config.defaults[config_section].items():
                if item == "helptext":
                    continue
                default = options["default"]
                if default != self.tk_vars[config_section][item].get():
                    self.tk_vars[config_section][item].set(default)
                    logger.debug("Setting %s - %s to default value %s",
                                 config_section, item, default)
        logger.debug("Reset to default: %s", section)

    def save_config(self, section=None):
        """ Save config """
        logger.debug("Saving %s config", section)
        new_config = ConfigParser(allow_no_value=True)
        for config_section, items in self.config_dicts.items():
            logger.debug("Adding section: '%s')", config_section)
            self.config.insert_config_section(config_section, items["helptext"], config=new_config)
            for item, options in items.items():
                if item == "helptext":
                    continue
                if ((section is not None and config_section != section)
                        or config_section not in self.tk_vars):
                    new_opt = options["value"]  # Keep saved item for other sections
                    logger.debug("Retaining option: (item: '%s', value: '%s')", item, new_opt)
                else:
                    new_opt = self.tk_vars[config_section][item].get()
                    logger.debug("Setting option: (item: '%s', value: '%s')", item, new_opt)
                helptext = options["helptext"]
                helptext = self.config.format_help(helptext, is_section=False)
                new_config.set(config_section, helptext)
                new_config.set(config_section, item, str(new_opt))
        self.config.config = new_config
        self.config.save_config()
        print("Saved config: '{}'".format(self.config.configfile))
        # Update config dict to newly saved
        self.config_dicts = self.get_config_dicts()
        logger.debug("Saved config")


class ImagesCanvas(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Canvas to hold the images """
    def __init__(self, parent, tk_vars):
        logger.debug("Initializing %s: (parent: %s,  tk_vars: %s)",
                     self.__class__.__name__, parent, tk_vars)
        super().__init__(parent)
        self.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

        self.refresh_display_trigger = tk_vars["refresh"]
        self.refresh_display_trigger.trace("w", self.refresh_display_callback)
        self.display = parent.preview_display
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.displaycanvas = self.canvas.create_image(0, 0,
                                                      image=self.display.tk_image,
                                                      anchor=tk.NW)
        self.bind("<Configure>", self.resize)
        logger.debug("Initialized %s", self.__class__.__name__)

    def refresh_display_callback(self, *args):
        """ Add a trace to refresh display on callback """
        if not self.refresh_display_trigger.get():
            return
        logger.trace("Refresh display trigger received: %s", args)
        self.reload()

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
    def __init__(self, parent, selected_color, selected_mask_type, selected_scaling,
                 config_tools, patch_callback, refresh_callback, tk_vars):
        logger.debug("Initializing %s: (selected_color: %s, selected_mask_type: %s, "
                     "selected_scaling: %s, config_tools, patch_callback: %s, "
                     "refresh_callback: %s, tk_vars: %s)", self.__class__.__name__, selected_color,
                     selected_mask_type, selected_scaling, patch_callback, refresh_callback,
                     tk_vars)
        self.config_tools = config_tools

        super().__init__(parent)
        self.pack(side=tk.LEFT, anchor=tk.N, fill=tk.Y)
        self.options = ["color", "mask_type", "scaling"]
        self.busy_tkvar = tk_vars["busy"]
        self.tk_vars = dict()

        d_locals = locals()
        defaults = {opt: self.format_to_display(d_locals["selected_{}".format(opt)])
                    for opt in self.options}
        self.busy_indicator = self.build_frame(defaults, refresh_callback, patch_callback)

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

    def build_frame(self, defaults, refresh_callback, patch_callback):
        """ Build the action frame """
        logger.debug("Building Action frame")
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, anchor=tk.N, expand=True)
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, anchor=tk.S)

        self.add_comboboxes(top_frame, defaults)
        busy_indicator = self.add_busy_indicator(top_frame)
        self.add_refresh_button(top_frame, refresh_callback)
        self.add_patch_callback(patch_callback)
        self.add_actions(bottom_frame)
        logger.debug("Built Action frame")
        return busy_indicator

    def add_comboboxes(self, parent, defaults):
        """ Add the comboboxes to the Action Frame """
        for opt in self.options:
            if opt == "mask_type":
                choices = get_available_masks() + ["predicted"]
            else:
                choices = PluginLoader.get_available_convert_plugins(opt, True)
            choices = [self.format_to_display(choice) for choice in choices]
            ctl = ControlBuilder(parent,
                                 opt,
                                 str,
                                 defaults[opt],
                                 choices=choices,
                                 is_radio=False,
                                 label_width=10,
                                 control_width=12)
            self.tk_vars[opt] = ctl.tk_var

    @staticmethod
    def add_refresh_button(parent, refresh_callback):
        """ Add button to refresh the images """
        btn = ttk.Button(parent, text="Update Samples", command=refresh_callback)
        btn.pack(padx=5, pady=5, side=tk.BOTTOM, fill=tk.X, anchor=tk.S)

    def add_patch_callback(self, patch_callback):
        """ Add callback to repatch images on action option change """
        for tk_var in self.tk_vars.values():
            tk_var.trace("w", patch_callback)

    def add_busy_indicator(self, parent):
        """ Place progress bar into bottom bar to indicate when processing """
        logger.debug("Placing busy indicator")
        pbar = ttk.Progressbar(parent, mode="indeterminate")
        pbar.pack(side=tk.BOTTOM, padx=5, pady=5, fill=tk.X)
        pbar.pack_forget()
        self.busy_tkvar.trace("w", self.busy_indicator_trace)
        return pbar

    def busy_indicator_trace(self, *args):
        """ Show or hide busy indicator """
        logger.trace("Busy indicator trace: %s", args)
        if self.busy_tkvar.get():
            self.start_busy_indicator()
        else:
            self.stop_busy_indicator()

    def stop_busy_indicator(self):
        """ Stop and hide progress bar """
        logger.debug("Stopping busy indicator")
        self.busy_indicator.stop()
        self.busy_indicator.pack_forget()

    def start_busy_indicator(self):
        """ Start and display progress bar """
        logger.debug("Starting busy indicator")
        self.busy_indicator.pack(side=tk.BOTTOM, padx=5, pady=5, fill=tk.X)
        self.busy_indicator.start()

    def add_actions(self, parent):
        """ Add Action Buttons """
        logger.debug("Adding util buttons")
        frame = ttk.Frame(parent)
        frame.pack(padx=5, pady=(5, 10), side=tk.BOTTOM, fill=tk.X, anchor=tk.E)

        for utl in ("save", "clear", "reload"):
            logger.debug("Adding button: '%s'", utl)
            img = get_images().icons[utl]
            if utl == "save":
                text = "Save full config"
                action = self.config_tools.save_config
            elif utl == "clear":
                text = "Reset full config to default values"
                action = self.config_tools.reset_config_default
            elif utl == "reload":
                text = "Reset full config to saved values"
                action = self.config_tools.reset_config_saved

            btnutl = ttk.Button(frame,
                                image=img,
                                command=action)
            btnutl.pack(padx=2, side=tk.RIGHT)
            Tooltip(btnutl, text=text, wraplength=200)
        logger.debug("Added util buttons")


class OptionsBook(ttk.Notebook):  # pylint:disable=too-many-ancestors
    """ Convert settings Options Frame """
    def __init__(self, parent, config_tools, patch_callback, scaling):
        logger.debug("Initializing %s: (parent: %s, config: %s, scaling: %s)",
                     self.__class__.__name__, parent, config_tools, scaling)
        super().__init__(parent)
        self.pack(side=tk.RIGHT, anchor=tk.N, fill=tk.BOTH, expand=True)
        self.config_tools = config_tools
        self.scaling = scaling

        self.tabs = dict()
        self.build_tabs()
        self.build_sub_tabs()
        self.add_patch_callback(patch_callback)
        logger.debug("Initialized %s", self.__class__.__name__)

    def build_tabs(self):
        """ Build the tabs for the relevant section """
        logger.debug("Build Tabs")
        for section in self.config_tools.sections:
            tab = ttk.Notebook(self)
            self.tabs[section] = {"tab": tab}
            self.add(tab, text=section.replace("_", " ").title())

    def build_sub_tabs(self):
        """ Build the sub tabs for the relevant plugin """
        for section, plugins in self.config_tools.plugins_dict.items():
            for plugin in plugins:
                config_key = ".".join((section, plugin))
                config_dict = self.config_tools.config_dicts[config_key]
                tab = ConfigFrame(self,
                                  config_key,
                                  config_dict)
                self.tabs[section][plugin] = tab
                self.tabs[section]["tab"].add(tab, text=plugin.replace("_", " ").title())

    def add_patch_callback(self, patch_callback):
        """ Add callback to repatch images on config option change """
        for plugins in self.config_tools.tk_vars.values():
            for tk_var in plugins.values():
                tk_var.trace("w", patch_callback)


class ConfigFrame(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ Config Frame - Holds the Options for config """

    def __init__(self, parent, config_key, options):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.options = options
        self.static_dims = [0, 0]

        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.Y)

        self.optsframe = ttk.Frame(self.canvas)
        self.optscanvas = self.canvas.create_window((0, 0), window=self.optsframe, anchor=tk.NW)

        self.scrollbar = self.add_scrollbar()

        self.frame_separator = self.add_frame_separator()
        self.action_frame = ttk.Frame(self)
        self.action_frame.pack(padx=5, pady=5, side=tk.BOTTOM, fill=tk.X, anchor=tk.E)

        self.build_frame(parent, config_key)

        self.bind("<Configure>", self.resize_frame)

        logger.debug("Initialized %s", self.__class__.__name__)

    def build_frame(self, parent, config_key):
        """ Build the options frame for this command """
        logger.debug("Add Config Frame")

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
            parent.config_tools.tk_vars.setdefault(config_key, dict())[key] = ctl.tk_var
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
        return scrollbar

    def update_scrollbar(self, event):  # pylint: disable=unused-argument
        """ Update the options frame scrollbar """
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def resize_frame(self, event):
        """ Resize the options frame to fit the canvas """
        logger.debug("Resize Config Frame")
        canvas_width = event.width - self.scrollbar.winfo_reqwidth()
        canvas_height = event.height - (self.action_frame.winfo_reqheight() +
                                        self.frame_separator.winfo_reqheight() + 16)
        self.canvas.configure(width=canvas_width, height=canvas_height)
        self.canvas.itemconfig(self.optscanvas, width=canvas_width, height=canvas_height)
        logger.debug("Resized Config Frame")

    def add_frame_separator(self):
        """ Add a separator between top and bottom frames """
        logger.debug("Add frame seperator")
        sep = ttk.Frame(self, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.TOP)
        logger.debug("Added frame seperator")
        return sep

    def add_actions(self, parent, config_key):
        """ Add Action Buttons """
        logger.debug("Adding util buttons")

        title = config_key.split(".")[1].replace("_", " ").title()
        for utl in ("save", "clear", "reload"):
            logger.debug("Adding button: '%s'", utl)
            img = get_images().icons[utl]
            if utl == "save":
                text = "Save {} config".format(title)
                action = parent.config_tools.save_config
            elif utl == "clear":
                text = "Reset {} config to default values".format(title)
                action = parent.config_tools.reset_config_default
            elif utl == "reload":
                text = "Reset {} config to saved values".format(title)
                action = parent.config_tools.reset_config_saved

            btnutl = ttk.Button(self.action_frame,
                                image=img,
                                command=lambda cmd=action: cmd(config_key))
            btnutl.pack(padx=2, side=tk.RIGHT)
            Tooltip(btnutl, text=text, wraplength=200)
        logger.debug("Added util buttons")


class ControlBuilder():
    """
    Builds and returns a frame containing a tkinter control with label

    Currently only setup for config items

    Parameters
    ----------
    parent: tkinter object
        Parent tkinter object
    title: str
        Title of the control. Will be used for label text
    dtype: datatype object
        Datatype of the control
    default: str
        Default value for the control
    selected_value: str, optional
        Selected value for the control. If None, default will be used
    choices: list or tuple, object
        Used for combo boxes and radio control option setting
    is_radio: bool, optional
        Specifies to use a Radio control instead of combobox if choices are passed
    rounding: int or float, optional
        For slider controls. Sets the stepping
    min_max: int or float, optional
        For slider controls. Sets the min and max values
    helptext: str, optional
        Sets the tooltip text
    radio_columns: int, optional
        Sets the number of columns to use for grouping radio buttons
    label_width: int, optional
        Sets the width of the control label. Defaults to 20
    control_width: int, optional
        Sets the width of the control. Default is to auto expand
    """
    def __init__(self, parent, title, dtype, default,
                 selected_value=None, choices=None, is_radio=False, rounding=None,
                 min_max=None, helptext=None, radio_columns=3, label_width=20, control_width=None):
        logger.debug("Initializing %s: (parent: %s, title: %s, dtype: %s, default: %s, "
                     "selected_value: %s, choices: %s, is_radio: %s, rounding: %s, min_max: %s, "
                     "helptext: %s, radio_columns: %s, label_width: %s, control_width: %s)",
                     self.__class__.__name__, parent, title, dtype, default, selected_value,
                     choices, is_radio, rounding, min_max, helptext, radio_columns, label_width,
                     control_width)

        self.title = title
        self.default = default

        self.frame = self.control_frame(parent, helptext)
        self.control = self.set_control(dtype, choices, is_radio)
        self.tk_var = self.set_tk_var(dtype, selected_value)

        self.build_control(choices,
                           dtype,
                           rounding,
                           min_max,
                           radio_columns,
                           label_width,
                           control_width)
        logger.debug("Initialized: %s", self.__class__.__name__)

    # Frame, control type and varable
    def control_frame(self, parent, helptext):
        """ Frame to hold control and it's label """
        logger.debug("Build control frame")
        frame = ttk.Frame(parent)
        frame.pack(side=tk.TOP, fill=tk.X)
        if helptext is not None:
            helptext = self.format_helptext(helptext)
            Tooltip(frame, text=helptext, wraplength=720)
        logger.debug("Built control frame")
        return frame

    def format_helptext(self, helptext):
        """ Format the help text for tooltips """
        logger.debug("Format control help: '%s'", self.title)
        helptext = helptext.replace("\n\t", "\n  - ").replace("%%", "%")
        helptext = self.title + " - " + helptext
        logger.debug("Formatted control help: (title: '%s', help: '%s'", self.title, helptext)
        return helptext

    def set_control(self, dtype, choices, is_radio):
        """ Set the correct control type based on the datatype or for this option """
        if choices and is_radio:
            control = ttk.Radiobutton
        elif choices:
            control = ttk.Combobox
        elif dtype == bool:
            control = ttk.Checkbutton
        elif dtype in (int, float):
            control = ttk.Scale
        else:
            control = ttk.Entry
        logger.debug("Setting control '%s' to %s", self.title, control)
        return control

    def set_tk_var(self, dtype, selected_value):
        """ Correct variable type for control """
        logger.debug("Setting tk variable: (title: '%s', dtype: %s, selected_value: %s)",
                     self.title, dtype, selected_value)
        if dtype == bool:
            var = tk.BooleanVar
        elif dtype == int:
            var = tk.IntVar
        elif dtype == float:
            var = tk.DoubleVar
        else:
            var = tk.StringVar
        var = var(self.frame)
        val = self.default if selected_value is None else selected_value
        var.set(val)
        logger.debug("Set tk variable: (title: '%s', type: %s, value: '%s')",
                     self.title, type(var), val)
        return var

    # Build the full control
    def build_control(self, choices, dtype, rounding, min_max, radio_columns,
                      label_width, control_width):
        """ Build the correct control type for the option passed through """
        logger.debug("Build confog option control")
        self.build_control_label(label_width)
        self.build_one_control(choices, dtype, rounding, min_max, radio_columns, control_width)
        logger.debug("Built option control")

    def build_control_label(self, label_width):
        """ Label for control """
        logger.debug("Build control label: (title: '%s', label_width: %s)",
                     self.title, label_width)
        title = self.title.replace("_", " ").title()
        lbl = ttk.Label(self.frame, text=title, width=label_width, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)
        logger.debug("Built control label: '%s'", self.title)

    def build_one_control(self, choices, dtype, rounding, min_max, radio_columns, control_width):
        """ Build and place the option controls """
        logger.debug("Build control: (title: '%s', control: %s, choices: %s, dtype: %s, "
                     "rounding: %s, min_max: %s: radio_columns: %s, control_width: %s)",
                     self.title, self.control, choices, dtype, rounding, min_max, radio_columns,
                     control_width)
        if self.control == ttk.Scale:
            ctl = self.slider_control(dtype, rounding, min_max)
        elif self.control == ttk.Radiobutton:
            ctl = self.radio_control(choices, radio_columns)
        else:
            ctl = self.control_to_optionsframe(choices)
        self.set_control_width(ctl, control_width)
        ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)
        logger.debug("Built control: '%s'", self.title)

    @staticmethod
    def set_control_width(ctl, control_width):
        """ Set the control width if required """
        if control_width is not None:
            ctl.config(width=control_width)

    def radio_control(self, choices, columns):
        """ Create a group of radio buttons """
        logger.debug("Adding radio group: %s", self.title)
        ctl = ttk.Frame(self.frame)
        frames = list()
        for _ in range(columns):
            frame = ttk.Frame(ctl)
            frame.pack(padx=5, pady=5, fill=tk.X, expand=True, side=tk.LEFT, anchor=tk.N)
            frames.append(frame)

        for idx, choice in enumerate(choices):
            frame_id = idx % columns
            radio = ttk.Radiobutton(frames[frame_id],
                                    text=choice.title(),
                                    value=choice,
                                    variable=self.tk_var)
            radio.pack(anchor=tk.W)
            logger.debug("Adding radio option %s to column %s", choice, frame_id)
        logger.debug("Added radio group: '%s'", self.title)
        return ctl

    def slider_control(self, dtype, rounding, min_max):
        """ A slider control with corresponding Entry box """
        logger.debug("Add slider control to Options Frame: (title: '%s', dtype: %s, rounding: %s, "
                     "min_max: %s)", self.title, dtype, rounding, min_max)
        tbox = ttk.Entry(self.frame, width=8, textvariable=self.tk_var, justify=tk.RIGHT)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)
        ctl = self.control(
            self.frame,
            variable=self.tk_var,
            command=lambda val, var=self.tk_var, dt=dtype, rn=rounding, mm=min_max:
            set_slider_rounding(val, var, dt, rn, mm))
        rc_menu = ContextMenu(tbox)
        rc_menu.cm_bind()
        ctl["from_"] = min_max[0]
        ctl["to"] = min_max[1]
        logger.debug("Added slider control to Options Frame: %s", self.title)
        return ctl

    def control_to_optionsframe(self, choices):
        """ Standard non-check buttons sit in the main options frame """
        logger.debug("Add control to Options Frame: (title: '%s', control: %s, choices: %s)",
                     self.title, self.control, choices)
        if self.control == ttk.Checkbutton:
            ctl = self.control(self.frame, variable=self.tk_var, text=None)
        else:
            ctl = self.control(self.frame, textvariable=self.tk_var)
            rc_menu = ContextMenu(ctl)
            rc_menu.cm_bind()
        if choices:
            logger.debug("Adding combo choices: %s", choices)
            ctl["values"] = [choice for choice in choices]
        logger.debug("Added control to Options Frame: %s", self.title)
        return ctl
