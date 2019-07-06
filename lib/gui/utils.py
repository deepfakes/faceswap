#!/usr/bin/env python3
""" Utility functions for the GUI """
import logging
import os
import platform
import re
import sys
import tkinter as tk
from tkinter import filedialog, ttk
from threading import Event, Thread
from queue import Queue
import numpy as np

from PIL import Image, ImageDraw, ImageTk

from lib.Serializer import JSONSerializer
from .tooltip import Tooltip

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_CONFIG = None
_IMAGES = None


def initialize_config(root, cli_opts, scaling_factor, pathcache, statusbar, session):
    """ Initialize the config and add to global constant """
    global _CONFIG  # pylint: disable=global-statement
    if _CONFIG is not None:
        return
    logger.debug("Initializing config: (root: %s, cli_opts: %s, tk_vars: %s, pathcache: %s, "
                 "statusbar: %s, session: %s)", root, cli_opts, scaling_factor, pathcache,
                 statusbar, session)
    _CONFIG = Config(root, cli_opts, scaling_factor, pathcache, statusbar, session)


def get_config():
    """ return the _CONFIG constant """
    return _CONFIG


def initialize_images(pathcache=None):
    """ Initialize the images and add to global constant """
    global _IMAGES  # pylint: disable=global-statement
    if _IMAGES is not None:
        return
    logger.debug("Initializing images")
    _IMAGES = Images(pathcache)


def get_images():
    """ return the _CONFIG constant """
    return _IMAGES


def set_slider_rounding(value, var, d_type, round_to, min_max):
    """ Set the underlying variable to correct number based on slider rounding """
    if d_type == float:
        var.set(round(float(value), round_to))
    else:
        steps = range(min_max[0], min_max[1] + round_to, round_to)
        value = min(steps, key=lambda x: abs(x - int(float(value))))
        var.set(value)


def adjust_wraplength(event):
    """ dynamically adjust the wraplength of a label on event """
    label = event.widget
    label.configure(wraplength=event.width - 1)


class FileHandler():
    """ Raise a filedialog box and capture input """

    def __init__(self, handletype, filetype, command=None, action=None,
                 variable=None):
        logger.debug("Initializing %s: (Handletype: '%s', filetype: '%s', command: '%s', action: "
                     "'%s', variable: %s)", self.__class__.__name__, handletype, filetype, command,
                     action, variable)
        self.handletype = handletype
        self.contexts = {
            "effmpeg": {
                "input": {"extract": "filename",
                          "gen-vid": "dir",
                          "get-fps": "filename",
                          "get-info": "filename",
                          "mux-audio": "filename",
                          "rescale": "filename",
                          "rotate": "filename",
                          "slice": "filename"},
                "output": {"extract": "dir",
                           "gen-vid": "savefilename",
                           "get-fps": "nothing",
                           "get-info": "nothing",
                           "mux-audio": "savefilename",
                           "rescale": "savefilename",
                           "rotate": "savefilename",
                           "slice": "savefilename"}
            }
        }
        self.defaults = self.set_defaults()
        self.kwargs = self.set_kwargs(filetype, command, action, variable)
        self.retfile = getattr(self, self.handletype.lower())()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def filetypes(self):
        """ Set the filetypes for opening/saving """
        all_files = ("All files", "*.*")
        filetypes = {"default": (all_files,),
                     "alignments": [("JSON", "*.json"),
                                    ("Pickle", "*.p"),
                                    ("YAML", "*.yaml" "*.yml"),  # pylint: disable=W1403
                                    all_files],
                     "config": [("Faceswap GUI config files", "*.fsw"), all_files],
                     "csv": [("Comma separated values", "*.csv"), all_files],
                     "image": [("Bitmap", "*.bmp"),
                               ("JPG", "*.jpeg" "*.jpg"),  # pylint: disable=W1403
                               ("PNG", "*.png"),
                               ("TIFF", "*.tif" "*.tiff"),  # pylint: disable=W1403
                               all_files],
                     "ini": [("Faceswap config files", "*.ini"), all_files],
                     "state": [("State files", "*.json"), all_files],
                     "log": [("Log files", "*.log"), all_files],
                     "video": [("Audio Video Interleave", "*.avi"),
                               ("Flash Video", "*.flv"),
                               ("Matroska", "*.mkv"),
                               ("MOV", "*.mov"),
                               ("MP4", "*.mp4"),
                               ("MPEG", "*.mpeg"),
                               ("WebM", "*.webm"),
                               all_files]}
        # Add in multi-select options
        for key, val in filetypes.items():
            if len(val) < 3:
                continue
            multi = ["{} Files".format(key.title())]
            multi.append(" ".join([ftype[1] for ftype in val if ftype[0] != "All files"]))
            val.insert(0, tuple(multi))
        return filetypes

    def set_defaults(self):
        """ Set the default filetype to be first in list of filetypes,
            or set a custom filetype if the first is not correct """
        defaults = {key: val[0][1].replace("*", "")
                    for key, val in self.filetypes.items()}
        defaults["default"] = None
        defaults["video"] = ".mp4"
        defaults["image"] = ".png"
        logger.debug(defaults)
        return defaults

    def set_kwargs(self, filetype, command, action, variable=None):
        """ Generate the required kwargs for the requested browser """
        logger.debug("Setting Kwargs: (filetype: '%s', command: '%s': action: '%s', "
                     "variable: '%s')", filetype, command, action, variable)
        kwargs = dict()
        if self.handletype.lower() == "context":
            self.set_context_handletype(command, action, variable)

        if self.handletype.lower() in (
                "open", "save", "filename", "filename_multi", "savefilename"):
            kwargs["filetypes"] = self.filetypes[filetype]
            if self.defaults.get(filetype, None):
                kwargs['defaultextension'] = self.defaults[filetype]
        if self.handletype.lower() == "save":
            kwargs["mode"] = "w"
        if self.handletype.lower() == "open":
            kwargs["mode"] = "r"
        logger.debug("Set Kwargs: %s", kwargs)
        return kwargs

    def set_context_handletype(self, command, action, variable):
        """ Choose the correct file browser action based on context """
        if self.contexts[command].get(variable, None) is not None:
            handletype = self.contexts[command][variable][action]
        else:
            handletype = self.contexts[command][action]
        logger.debug(handletype)
        self.handletype = handletype

    def open(self):
        """ Open a file """
        logger.debug("Popping Open browser")
        return filedialog.askopenfile(**self.kwargs)

    def save(self):
        """ Save a file """
        logger.debug("Popping Save browser")
        return filedialog.asksaveasfile(**self.kwargs)

    def dir(self):
        """ Get a directory location """
        logger.debug("Popping Dir browser")
        return filedialog.askdirectory(**self.kwargs)

    def savedir(self):
        """ Get a save dir location """
        logger.debug("Popping SaveDir browser")
        return filedialog.askdirectory(**self.kwargs)

    def filename(self):
        """ Get an existing file location """
        logger.debug("Popping Filename browser")
        return filedialog.askopenfilename(**self.kwargs)

    def filename_multi(self):
        """ Get multiple existing file locations """
        logger.debug("Popping Filename browser")
        return filedialog.askopenfilenames(**self.kwargs)

    def savefilename(self):
        """ Get a save file location """
        logger.debug("Popping SaveFilename browser")
        return filedialog.asksaveasfilename(**self.kwargs)

    @staticmethod
    def nothing():  # pylint: disable=useless-return
        """ Method that does nothing, used for disabling open/save pop up  """
        logger.debug("Popping Nothing browser")
        return


class Images():
    """ Holds locations of images and actual images

        Don't call directly. Call get_images()
    """

    def __init__(self, pathcache=None):
        logger.debug("Initializing %s", self.__class__.__name__)
        pathcache = get_config().pathcache if pathcache is None else pathcache
        self.pathicons = os.path.join(pathcache, "icons")
        self.pathpreview = os.path.join(pathcache, "preview")
        self.pathoutput = None
        self.previewoutput = None
        self.previewtrain = dict()
        self.errcount = 0
        self.icons = dict()
        self.icons["folder"] = ImageTk.PhotoImage(file=os.path.join(
            self.pathicons, "open_folder.png"))
        self.icons["load"] = ImageTk.PhotoImage(file=os.path.join(
            self.pathicons, "open_file.png"))
        self.icons["load_multi"] = ImageTk.PhotoImage(file=os.path.join(
            self.pathicons, "open_file.png"))
        self.icons["context"] = ImageTk.PhotoImage(file=os.path.join(
            self.pathicons, "open_file.png"))
        self.icons["save"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "save.png"))
        self.icons["reset"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "reset.png"))
        self.icons["clear"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "clear.png"))
        self.icons["graph"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "graph.png"))
        self.icons["zoom"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "zoom.png"))
        self.icons["move"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "move.png"))
        self.icons["favicon"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "logo.png"))
        logger.debug("Initialized %s: (icons: %s)", self.__class__.__name__, self.icons)

    def delete_preview(self):
        """ Delete the preview files """
        logger.debug("Deleting previews")
        for item in os.listdir(self.pathpreview):
            if item.startswith(".gui_training_preview") and item.endswith(".jpg"):
                fullitem = os.path.join(self.pathpreview, item)
                logger.debug("Deleting: '%s'", fullitem)
                os.remove(fullitem)
        self.clear_image_cache()

    def clear_image_cache(self):
        """ Clear all cached images """
        logger.debug("Clearing image cache")
        self.pathoutput = None
        self.previewoutput = None
        self.previewtrain = dict()

    @staticmethod
    def get_images(imgpath):
        """ Get the images stored within the given directory """
        logger.trace("Getting images: '%s'", imgpath)
        if not os.path.isdir(imgpath):
            logger.debug("Folder does not exist")
            return None
        files = [os.path.join(imgpath, f)
                 for f in os.listdir(imgpath) if f.lower().endswith((".png", ".jpg"))]
        logger.trace("Image files: %s", files)
        return files

    def load_latest_preview(self, thumbnail_size, frame_dims):
        """ Load the latest preview image for extract and convert """
        logger.trace("Loading preview image: (thumbnail_size: %s, frame_dims: %s)",
                     thumbnail_size, frame_dims)
        imagefiles = self.get_images(self.pathoutput)
        if not imagefiles or len(imagefiles) == 1:
            logger.debug("No preview to display")
            self.previewoutput = None
            return
        logger.trace("Image Files: %s", len(imagefiles))
        num_images = (frame_dims[0] // thumbnail_size) * (frame_dims[1] // thumbnail_size)
        if num_images > len(imagefiles):
            logger.debug("Not enough images to generate display. (to display: %s, available "
                         "images: %s)", num_images, len(imagefiles))
            self.previewoutput = None
            return
        filenames, samples = self.get_preview_samples(imagefiles, num_images, thumbnail_size)
        show_image = self.place_previews(samples, frame_dims)
        logger.trace("Displaying preview: '%s'", filenames)
        self.previewoutput = (show_image, ImageTk.PhotoImage(show_image))

    @staticmethod
    def get_preview_samples(imagefiles, num_images, thumbnail_size):
        """ Return a subset of the imagefiles images
            Exclude final file so we don't accidentally load a file that is being saved """
        logger.trace("num_images: %s", num_images)
        samples = list()
        start_idx = len(imagefiles) - (num_images + 1)
        end_idx = len(imagefiles) - 1
        logger.trace("start_idx: %s, end_idx: %s", start_idx, end_idx)
        show_files = sorted(imagefiles, key=os.path.getctime)[start_idx: end_idx]
        for fname in show_files:
            img = Image.open(fname)
            width, height = img.size
            scaling = thumbnail_size / max(width, height)
            logger.debug("image width: %s, height: %s, scaling: %s", width, height, scaling)
            img = img.resize((int(width * scaling), int(height * scaling)))
            if img.size[0] != img.size[1]:
                # Pad to square
                new_img = Image.new("RGB", (thumbnail_size, thumbnail_size))
                new_img.paste(img, ((thumbnail_size - img.size[0])//2,
                                    (thumbnail_size - img.size[1])//2))
                img = new_img
            draw = ImageDraw.Draw(img)
            draw.rectangle(((0, 0), (thumbnail_size, thumbnail_size)), outline="#E5E5E5", width=1)
            samples.append(np.array(img))
        samples = np.array(samples)
        logger.trace("Samples shape: %s", samples.shape)
        return show_files, samples

    @staticmethod
    def place_previews(samples, frame_dims):
        """ Stack the preview images to fit display """
        num_images, thumbnail_size = samples.shape[:2]
        logger.trace("num_images: %s, thumbnail_size: %s", num_images, thumbnail_size)
        cols, rows = frame_dims[0] // thumbnail_size, frame_dims[1] // thumbnail_size
        logger.trace("cols: %s, rows: %s", cols, rows)
        display = np.vstack([np.hstack(samples[row * cols: (row + 1) * cols])
                             for row in range(rows)])
        logger.trace("display shape: %s", display.shape)
        return Image.fromarray(display)

    def load_training_preview(self):
        """ Load the training preview images """
        logger.trace("Loading Training preview images")
        imagefiles = self.get_images(self.pathpreview)
        modified = None
        if not imagefiles:
            logger.debug("No preview to display")
            self.previewtrain = dict()
            return
        for img in imagefiles:
            modified = os.path.getmtime(img) if modified is None else modified
            name = os.path.basename(img)
            name = os.path.splitext(name)[0]
            name = name[name.rfind("_") + 1:].title()
            try:
                logger.trace("Displaying preview: '%s'", img)
                size = self.get_current_size(name)
                self.previewtrain[name] = [Image.open(img), None, modified]
                self.resize_image(name, size)
                self.errcount = 0
            except ValueError:
                # This is probably an error reading the file whilst it's
                # being saved  so ignore it for now and only pick up if
                # there have been multiple consecutive fails
                logger.warning("Unable to display preview: (image: '%s', attempt: %s)",
                               img, self.errcount)
                if self.errcount < 10:
                    self.errcount += 1
                else:
                    logger.error("Error reading the preview file for '%s'", img)
                    print("Error reading the preview file for {}".format(name))
                    self.previewtrain[name] = None

    def get_current_size(self, name):
        """ Return the size of the currently displayed image """
        logger.trace("Getting size: '%s'", name)
        if not self.previewtrain.get(name, None):
            return None
        img = self.previewtrain[name][1]
        if not img:
            return None
        logger.trace("Got size: (name: '%s', width: '%s', height: '%s')",
                     name, img.width(), img.height())
        return img.width(), img.height()

    def resize_image(self, name, framesize):
        """ Resize the training preview image
            based on the passed in frame size """
        logger.trace("Resizing image: (name: '%s', framesize: %s", name, framesize)
        displayimg = self.previewtrain[name][0]
        if framesize:
            frameratio = float(framesize[0]) / float(framesize[1])
            imgratio = float(displayimg.size[0]) / float(displayimg.size[1])

            if frameratio <= imgratio:
                scale = framesize[0] / float(displayimg.size[0])
                size = (framesize[0], int(displayimg.size[1] * scale))
            else:
                scale = framesize[1] / float(displayimg.size[1])
                size = (int(displayimg.size[0] * scale), framesize[1])
            logger.trace("Scaling: (scale: %s, size: %s", scale, size)

            # Hacky fix to force a reload if it happens to find corrupted
            # data, probably due to reading the image whilst it is partially
            # saved. If it continues to fail, then eventually raise.
            for i in range(0, 1000):
                try:
                    displayimg = displayimg.resize(size, Image.ANTIALIAS)
                except OSError:
                    if i == 999:
                        raise
                    continue
                break

        self.previewtrain[name][1] = ImageTk.PhotoImage(displayimg)


class ContextMenu(tk.Menu):  # pylint: disable=too-many-ancestors
    """ Pop up menu """
    def __init__(self, widget):
        logger.debug("Initializing %s: (widget_class: '%s')",
                     self.__class__.__name__, widget.winfo_class())
        super().__init__(tearoff=0)
        self.widget = widget
        self.standard_actions()
        logger.debug("Initialized %s", self.__class__.__name__)

    def standard_actions(self):
        """ Standard menu actions """
        self.add_command(label="Cut", command=lambda: self.widget.event_generate("<<Cut>>"))
        self.add_command(label="Copy", command=lambda: self.widget.event_generate("<<Copy>>"))
        self.add_command(label="Paste", command=lambda: self.widget.event_generate("<<Paste>>"))
        self.add_separator()
        self.add_command(label="Select all", command=self.select_all)

    def cm_bind(self):
        """ Bind the menu to the widget's Right Click event """
        button = "<Button-2>" if platform.system() == "Darwin" else "<Button-3>"
        logger.debug("Binding '%s' to '%s'", button, self.widget.winfo_class())
        scaling_factor = get_config().scaling_factor if get_config() is not None else 1.0
        x_offset = int(34 * scaling_factor)
        self.widget.bind(button,
                         lambda event: self.tk_popup(event.x_root + x_offset, event.y_root, 0))

    def select_all(self):
        """ Select all for Text or Entry widgets """
        logger.debug("Selecting all for '%s'", self.widget.winfo_class())
        if self.widget.winfo_class() == "Text":
            self.widget.focus_force()
            self.widget.tag_add("sel", "1.0", "end")
        else:
            self.widget.focus_force()
            self.widget.select_range(0, tk.END)


class ConsoleOut(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ The Console out section of the GUI """

    def __init__(self, parent, debug):
        logger.debug("Initializing %s: (parent: %s, debug: %s)",
                     self.__class__.__name__, parent, debug)
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(2, 0),
                  fill=tk.BOTH, expand=True)
        self.console = tk.Text(self)
        rc_menu = ContextMenu(self.console)
        rc_menu.cm_bind()
        self.console_clear = get_config().tk_vars['consoleclear']
        self.set_console_clear_var_trace()
        self.debug = debug
        self.build_console()
        self.add_tags()
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_console_clear_var_trace(self):
        """ Set the trigger actions for the clear console var
            when it has been triggered from elsewhere """
        logger.debug("Set clear trace")
        self.console_clear.trace("w", self.clear)

    def build_console(self):
        """ Build and place the console """
        logger.debug("Build console")
        self.console.config(width=100, height=6, bg="gray90", fg="black")
        self.console.pack(side=tk.LEFT, anchor=tk.N, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self, command=self.console.yview)
        scrollbar.pack(side=tk.LEFT, fill="y")
        self.console.configure(yscrollcommand=scrollbar.set)

        self.redirect_console()
        logger.debug("Built console")

    def add_tags(self):
        """ Add tags to text widget to color based on output """
        logger.debug("Adding text color tags")
        self.console.tag_config("default", foreground="#1E1E1E")
        self.console.tag_config("stderr", foreground="#E25056")
        self.console.tag_config("info", foreground="#2B445E")
        self.console.tag_config("verbose", foreground="#008140")
        self.console.tag_config("warning", foreground="#F77B00")
        self.console.tag_config("critical", foreground="red")
        self.console.tag_config("error", foreground="red")

    def redirect_console(self):
        """ Redirect stdout/stderr to console frame """
        logger.debug("Redirect console")
        if self.debug:
            logger.info("Console debug activated. Outputting to main terminal")
        else:
            sys.stdout = SysOutRouter(console=self.console, out_type="stdout")
            sys.stderr = SysOutRouter(console=self.console, out_type="stderr")
        logger.debug("Redirected console")

    def clear(self, *args):  # pylint: disable=unused-argument
        """ Clear the console output screen """
        logger.debug("Clear console")
        if not self.console_clear.get():
            logger.debug("Console not set for clearing. Skipping")
            return
        self.console.delete(1.0, tk.END)
        self.console_clear.set(False)
        logger.debug("Cleared console")


class SysOutRouter():
    """ Route stdout/stderr to the console window """

    def __init__(self, console=None, out_type=None):
        logger.debug("Initializing %s: (console: %s, out_type: '%s')",
                     self.__class__.__name__, console, out_type)
        self.console = console
        self.out_type = out_type
        self.recolor = re.compile(r".+?(\s\d+:\d+:\d+\s)(?P<lvl>[A-Z]+)\s")
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_tag(self, string):
        """ Set the tag based on regex of log output """
        if self.out_type == "stderr":
            # Output all stderr in red
            return self.out_type

        output = self.recolor.match(string)
        if not output:
            return "default"
        tag = output.groupdict()["lvl"].strip().lower()
        return tag

    def write(self, string):
        """ Capture stdout/stderr """
        self.console.insert(tk.END, string, self.get_tag(string))
        self.console.see(tk.END)

    @staticmethod
    def flush():
        """ If flush is forced, send it to normal terminal """
        sys.__stdout__.flush()


class Config():
    """ Global configuration settings

        Don't call directly. Call get_config()
    """

    def __init__(self, root, cli_opts, scaling_factor, pathcache, statusbar, session):
        logger.debug("Initializing %s: (root %s, cli_opts: %s, scaling_factor: %s, pathcache: %s, "
                     "statusbar: %s, session: %s)", self.__class__.__name__, root, cli_opts,
                     scaling_factor, pathcache, statusbar, session)
        self.root = root
        self.cli_opts = cli_opts
        self.scaling_factor = scaling_factor
        self.pathcache = pathcache
        self.statusbar = statusbar
        self.serializer = JSONSerializer
        self.tk_vars = self.set_tk_vars()
        self.command_notebook = None  # set in command.py
        self.session = session
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def command_tabs(self):
        """ Return dict of command tab titles with their IDs """
        return {self.command_notebook.tab(tab_id, "text").lower(): tab_id
                for tab_id in range(0, self.command_notebook.index("end"))}

    @property
    def tools_command_tabs(self):
        """ Return dict of tools command tab titles with their IDs """
        return {self.command_notebook.tools_notebook.tab(tab_id, "text").lower(): tab_id
                for tab_id in range(0, self.command_notebook.tools_notebook.index("end"))}

    def set_cursor_busy(self, widget=None):
        """ Set the root or widget cursor to busy """
        logger.debug("Setting cursor to busy. widget: %s", widget)
        widget = self.root if widget is None else widget
        widget.config(cursor="watch")
        widget.update_idletasks()

    def set_cursor_default(self, widget=None):
        """ Set the root or widget cursor to default """
        logger.debug("Setting cursor to default. widget: %s", widget)
        widget = self.root if widget is None else widget
        widget.config(cursor="")
        widget.update_idletasks()

    @staticmethod
    def set_tk_vars():
        """ TK Variables to be triggered by to indicate
            what state various parts of the GUI should be in """
        display = tk.StringVar()
        display.set(None)

        runningtask = tk.BooleanVar()
        runningtask.set(False)

        istraining = tk.BooleanVar()
        istraining.set(False)

        actioncommand = tk.StringVar()
        actioncommand.set(None)

        generatecommand = tk.StringVar()
        generatecommand.set(None)

        consoleclear = tk.BooleanVar()
        consoleclear.set(False)

        refreshgraph = tk.BooleanVar()
        refreshgraph.set(False)

        smoothgraph = tk.DoubleVar()
        smoothgraph.set(0.90)

        updatepreview = tk.BooleanVar()
        updatepreview.set(False)

        traintimeout = tk.IntVar()
        traintimeout.set(120)

        tk_vars = {"display": display,
                   "runningtask": runningtask,
                   "istraining": istraining,
                   "action": actioncommand,
                   "generate": generatecommand,
                   "consoleclear": consoleclear,
                   "refreshgraph": refreshgraph,
                   "smoothgraph": smoothgraph,
                   "updatepreview": updatepreview,
                   "traintimeout": traintimeout}
        logger.debug(tk_vars)
        return tk_vars

    def load(self, command=None, filename=None):
        """ Pop up load dialog for a saved config file """
        logger.debug("Loading config: (command: '%s')", command)
        if filename:
            if not os.path.isfile(filename):
                msg = "File does not exist: '{}'".format(filename)
                logger.error(msg)
                return
            with open(filename, "r") as cfgfile:
                cfg = self.serializer.unmarshal(cfgfile.read())
        else:
            cfgfile = FileHandler("open", "config").retfile
            if not cfgfile:
                return
            cfg = self.serializer.unmarshal(cfgfile.read())

        if not command and len(cfg.keys()) == 1:
            command = list(cfg.keys())[0]

        opts = self.get_command_options(cfg, command) if command else cfg
        if not opts:
            return

        for cmd, opts in opts.items():
            self.set_command_args(cmd, opts)

        if command:
            if command in self.command_tabs:
                self.command_notebook.select(self.command_tabs[command])
            else:
                self.command_notebook.select(self.command_tabs["tools"])
                self.command_notebook.tools_notebook.select(self.tools_command_tabs[command])
        self.add_to_recent(cfgfile.name, command)
        logger.debug("Loaded config: (command: '%s', cfgfile: '%s')", command, cfgfile)

    def get_command_options(self, cfg, command):
        """ return the saved options for the requested
            command, if not loading global options """
        opts = cfg.get(command, None)
        retval = {command: opts}
        if not opts:
            self.tk_vars["consoleclear"].set(True)
            print("No {} section found in file".format(command))
            logger.info("No  %s section found in file", command)
            retval = None
        logger.debug(retval)
        return retval

    def set_command_args(self, command, options):
        """ Pass the saved config items back to the CliOptions """
        if not options:
            return
        for srcopt, srcval in options.items():
            optvar = self.cli_opts.get_one_option_variable(command, srcopt)
            if not optvar:
                continue
            optvar.set(srcval)

    def save(self, command=None):
        """ Save the current GUI state to a config file in json format """
        logger.debug("Saving config: (command: '%s')", command)
        cfgfile = FileHandler("save", "config").retfile
        if not cfgfile:
            return
        cfg = self.cli_opts.get_option_values(command)
        cfgfile.write(self.serializer.marshal(cfg))
        cfgfile.close()
        self.add_to_recent(cfgfile.name, command)
        logger.debug("Saved config: (command: '%s', cfgfile: '%s')", command, cfgfile)

    def add_to_recent(self, filename, command):
        """ Add to recent files """
        recent_filename = os.path.join(self.pathcache, ".recent.json")
        logger.debug("Adding to recent files '%s': (%s, %s)", recent_filename, filename, command)
        if not os.path.exists(recent_filename) or os.path.getsize(recent_filename) == 0:
            recent_files = list()
        else:
            with open(recent_filename, "rb") as inp:
                recent_files = self.serializer.unmarshal(inp.read().decode("utf-8"))
        logger.debug("Initial recent files: %s", recent_files)
        filenames = [recent[0] for recent in recent_files]
        if filename in filenames:
            idx = filenames.index(filename)
            del recent_files[idx]
        recent_files.insert(0, (filename, command))
        recent_files = recent_files[:20]
        logger.debug("Final recent files: %s", recent_files)
        recent_json = self.serializer.marshal(recent_files)
        with open(recent_filename, "wb") as out:
            out.write(recent_json.encode("utf-8"))


class ControlBuilder():
    # TODO Expand out for cli options
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


class LongRunningTask(Thread):
    """ For long running tasks, to stop the GUI becoming unresponsive
        Run in a thread and handle cursor events """
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=True,
                 widget=None):
        logger.debug("Initializing %s: (group: %s, target: %s, name: %s, args: %s, kwargs: %s, "
                     "daemon: %s)", self.__class__.__name__, group, target, name, args, kwargs,
                     daemon)
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs,
                         daemon=daemon)
        self.err = None
        self.widget = widget
        self._config = get_config()
        self._config.set_cursor_busy(widget=self.widget)
        self.complete = Event()
        self._queue = Queue()
        logger.debug("Initialized %s", self.__class__.__name__,)

    def run(self):
        """ Run the target in a thread """
        try:
            if self._target:
                retval = self._target(*self._args, **self._kwargs)
                self._queue.put(retval)
        except Exception:  # pylint: disable=broad-except
            self.err = sys.exc_info()
            logger.debug("Error in thread (%s): %s", self._name,
                         self.err[1].with_traceback(self.err[2]))
        finally:
            self.complete.set()
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def get_result(self):
        """ Return the result from the queue """
        if not self.complete.is_set():
            logger.warning("Aborting attempt to retrieve result from a LongRunningTask that is "
                           "still running")
            return None
        if self.err:
            logger.debug("Error caught in thread")
            self._config.set_cursor_default(widget=self.widget)
            raise self.err[1].with_traceback(self.err[2])

        logger.debug("Getting result from thread")
        retval = self._queue.get()
        logger.debug("Got result from thread")
        self._config.set_cursor_default(widget=self.widget)
        return retval
