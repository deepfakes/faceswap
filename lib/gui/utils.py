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

from ._config import Config as UserConfig
from ._redirector import WidgetRedirector
from .project import Project, Tasks

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
    return _CONFIG


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


class FileHandler():
    """ Raise a file dialog box and capture input """

    def __init__(self, handletype, filetype, title=None, initialdir=None, command=None,
                 action=None, variable=None):
        logger.debug("Initializing %s: (Handletype: '%s', filetype: '%s', title: '%s', "
                     "initialdir: '%s, 'command: '%s', action: '%s', variable: %s)",
                     self.__class__.__name__, handletype, filetype, title, initialdir, command,
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
        self.kwargs = self.set_kwargs(title, initialdir, filetype, command, action, variable)
        self.retfile = getattr(self, self.handletype.lower())()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def filetypes(self):
        """ Set the filetypes for opening/saving """
        all_files = ("All files", "*.*")
        filetypes = {"default": (all_files,),
                     "alignments": [("Faceswap Alignments", "*.fsa *.json"),
                                    all_files],
                     "config_project": [("Faceswap Project files", "*.fsw"), all_files],
                     "config_task": [("Faceswap Task files", "*.fst"), all_files],
                     "config_all": [("Faceswap Project and Task files", "*.fst *.fsw"), all_files],
                     "csv": [("Comma separated values", "*.csv"), all_files],
                     "image": [("Bitmap", "*.bmp"),
                               ("JPG", "*.jpeg *.jpg"),
                               ("PNG", "*.png"),
                               ("TIFF", "*.tif *.tiff"),
                               all_files],
                     "ini": [("Faceswap config files", "*.ini"), all_files],
                     "state": [("State files", "*.json"), all_files],
                     "log": [("Log files", "*.log"), all_files],
                     "video": [("Audio Video Interleave", "*.avi"),
                               ("Flash Video", "*.flv"),
                               ("Matroska", "*.mkv"),
                               ("MOV", "*.mov"),
                               ("MP4", "*.mp4"),
                               ("MPEG", "*.mpeg *.mpg"),
                               ("WebM", "*.webm"),
                               ("Windows Media Video", "*.wmv"),
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
        """ Set the default file type to be first in list of file types,
            or set a custom file type if the first is not correct """
        defaults = {key: val[0][1].replace("*", "")
                    for key, val in self.filetypes.items()}
        defaults["default"] = None
        defaults["video"] = ".mp4"
        defaults["image"] = ".png"
        logger.debug(defaults)
        return defaults

    def set_kwargs(self, title, initialdir, filetype, command, action, variable=None):
        """ Generate the required kwargs for the requested browser """
        logger.debug("Setting Kwargs: (title: %s, initialdir: %s, filetype: '%s', "
                     "command: '%s': action: '%s', variable: '%s')",
                     title, initialdir, filetype, command, action, variable)
        kwargs = dict()
        if self.handletype.lower() == "context":
            self.set_context_handletype(command, action, variable)

        if title is not None:
            kwargs["title"] = title

        if initialdir is not None:
            kwargs["initialdir"] = initialdir

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
        """ Get a save directory location """
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
        self.pathpreview = os.path.join(pathcache, "preview")
        self.pathoutput = None
        self.previewoutput = None
        self.previewtrain = dict()
        self.previewcache = dict(modified=None,  # cache for extract and convert
                                 images=None,
                                 filenames=list(),
                                 placeholder=None)
        self.errcount = 0
        self.icons = self._load_icons(pathcache)
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def _load_icons(pathcache):
        size = get_config().user_config_dict.get("icon_size", 16)
        size = int(round(size * get_config().scaling_factor))
        icons = dict()
        pathicons = os.path.join(pathcache, "icons")
        for fname in os.listdir(pathicons):
            name, ext = os.path.splitext(fname)
            if ext != ".png":
                continue
            img = Image.open(os.path.join(pathicons, fname))
            img = ImageTk.PhotoImage(img.resize((size, size), resample=Image.HAMMING))
            icons[name] = img
        logger.debug(icons)
        return icons

    def delete_preview(self):
        """ Delete the preview files """
        logger.debug("Deleting previews")
        for item in os.listdir(self.pathpreview):
            if item.startswith(".gui_training_preview") and item.endswith(".jpg"):
                fullitem = os.path.join(self.pathpreview, item)
                logger.debug("Deleting: '%s'", fullitem)
                os.remove(fullitem)
        for fname in self.previewcache["filenames"]:
            if os.path.basename(fname) == ".gui_preview.jpg":
                logger.debug("Deleting: '%s'", fname)
                try:
                    os.remove(fname)
                except FileNotFoundError:
                    logger.debug("File does not exist: %s", fname)
        self.clear_image_cache()

    def clear_image_cache(self):
        """ Clear all cached images """
        logger.debug("Clearing image cache")
        self.pathoutput = None
        self.previewoutput = None
        self.previewtrain = dict()
        self.previewcache = dict(modified=None,  # cache for extract and convert
                                 images=None,
                                 filenames=list(),
                                 placeholder=None)

    @staticmethod
    def get_images(imgpath):
        """ Get the images stored within the given directory """
        logger.debug("Getting images: '%s'", imgpath)
        if not os.path.isdir(imgpath):
            logger.debug("Folder does not exist")
            return None
        files = [os.path.join(imgpath, f)
                 for f in os.listdir(imgpath) if f.lower().endswith((".png", ".jpg"))]
        logger.debug("Image files: %s", files)
        return files

    def load_latest_preview(self, thumbnail_size, frame_dims):
        """ Load the latest preview image for extract and convert """
        logger.debug("Loading preview image: (thumbnail_size: %s, frame_dims: %s)",
                     thumbnail_size, frame_dims)
        image_files = self.get_images(self.pathoutput)
        gui_preview = os.path.join(self.pathoutput, ".gui_preview.jpg")
        if not image_files or (len(image_files) == 1 and gui_preview not in image_files):
            logger.debug("No preview to display")
            self.previewoutput = None
            return
        # Filter to just the gui_preview if it exists in folder output
        image_files = [gui_preview] if gui_preview in image_files else image_files
        logger.debug("Image Files: %s", len(image_files))

        image_files = self.get_newest_filenames(image_files)
        if not image_files:
            return

        self.load_images_to_cache(image_files, frame_dims, thumbnail_size)
        if image_files == [gui_preview]:
            # Delete the preview image so that the main scripts know to output another
            logger.debug("Deleting preview image")
            os.remove(image_files[0])
        show_image = self.place_previews(frame_dims)
        if not show_image:
            self.previewoutput = None
            return
        logger.debug("Displaying preview: %s", self.previewcache["filenames"])
        self.previewoutput = (show_image, ImageTk.PhotoImage(show_image))

    def get_newest_filenames(self, image_files):
        """ Return image filenames that have been modified since the last check """
        if self.previewcache["modified"] is None:
            retval = image_files
        else:
            retval = [fname for fname in image_files
                      if os.path.getmtime(fname) > self.previewcache["modified"]]
        if not retval:
            logger.debug("No new images in output folder")
        else:
            self.previewcache["modified"] = max([os.path.getmtime(img) for img in retval])
            logger.debug("Number new images: %s, Last Modified: %s",
                         len(retval), self.previewcache["modified"])
        return retval

    def load_images_to_cache(self, image_files, frame_dims, thumbnail_size):
        """ Load new images and append to cache, filtering to the number of display images """
        logger.debug("Number image_files: %s, frame_dims: %s, thumbnail_size: %s",
                     len(image_files), frame_dims, thumbnail_size)
        num_images = (frame_dims[0] // thumbnail_size) * (frame_dims[1] // thumbnail_size)
        logger.debug("num_images: %s", num_images)
        if num_images == 0:
            return
        samples = list()
        start_idx = len(image_files) - num_images if len(image_files) > num_images else 0
        show_files = sorted(image_files, key=os.path.getctime)[start_idx:]
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
        self.previewcache["filenames"] = (self.previewcache["filenames"] +
                                          show_files)[-num_images:]
        cache = self.previewcache["images"]
        if cache is None:
            logger.debug("Creating new cache")
            cache = samples[-num_images:]
        else:
            logger.debug("Appending to existing cache")
            cache = np.concatenate((cache, samples))[-num_images:]
        self.previewcache["images"] = cache
        logger.debug("Cache shape: %s", self.previewcache["images"].shape)

    @staticmethod
    def get_preview_samples(image_files, num_images, thumbnail_size):
        """ Return a subset of the ``image_files`` images
            Exclude final file so we don't accidentally load a file that is being saved """
        logger.debug("num_images: %s", num_images)
        samples = list()
        start_idx = len(image_files) - (num_images + 1)
        end_idx = len(image_files) - 1
        logger.debug("start_idx: %s, end_idx: %s", start_idx, end_idx)
        show_files = sorted(image_files, key=os.path.getctime)[start_idx: end_idx]
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
        logger.debug("Samples shape: %s", samples.shape)
        return show_files, samples

    def place_previews(self, frame_dims):
        """ Stack the preview images to fit display """
        if self.previewcache.get("images", None) is None:
            logger.debug("No images in cache. Returning None")
            return None
        samples = self.previewcache["images"].copy()
        num_images, thumbnail_size = samples.shape[:2]
        if self.previewcache["placeholder"] is None:
            self.create_placeholder(thumbnail_size)

        logger.debug("num_images: %s, thumbnail_size: %s", num_images, thumbnail_size)
        cols, rows = frame_dims[0] // thumbnail_size, frame_dims[1] // thumbnail_size
        logger.debug("cols: %s, rows: %s", cols, rows)
        if cols == 0 or rows == 0:
            logger.debug("Cols or Rows is zero. No items to display")
            return None
        remainder = (cols * rows) - num_images
        if remainder != 0:
            logger.debug("Padding sample display. Remainder: %s", remainder)
            placeholder = np.concatenate([np.expand_dims(self.previewcache["placeholder"],
                                                         0)] * remainder)
            samples = np.concatenate((samples, placeholder))

        display = np.vstack([np.hstack(samples[row * cols: (row + 1) * cols])
                             for row in range(rows)])
        logger.debug("display shape: %s", display.shape)
        return Image.fromarray(display)

    def create_placeholder(self, thumbnail_size):
        """ Create a placeholder image for when there are fewer samples available
            then columns to display them """
        logger.debug("Creating placeholder. thumbnail_size: %s", thumbnail_size)
        placeholder = Image.new("RGB", (thumbnail_size, thumbnail_size))
        draw = ImageDraw.Draw(placeholder)
        draw.rectangle(((0, 0), (thumbnail_size, thumbnail_size)), outline="#E5E5E5", width=1)
        placeholder = np.array(placeholder)
        self.previewcache["placeholder"] = placeholder
        logger.debug("Created placeholder. shape: %s", placeholder.shape)

    def load_training_preview(self):
        """ Load the training preview images """
        logger.debug("Loading Training preview images")
        image_files = self.get_images(self.pathpreview)
        modified = None
        if not image_files:
            logger.debug("No preview to display")
            self.previewtrain = dict()
            return
        for img in image_files:
            modified = os.path.getmtime(img) if modified is None else modified
            name = os.path.basename(img)
            name = os.path.splitext(name)[0]
            name = name[name.rfind("_") + 1:].title()
            try:
                logger.debug("Displaying preview: '%s'", img)
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
        logger.debug("Getting size: '%s'", name)
        if not self.previewtrain.get(name, None):
            return None
        img = self.previewtrain[name][1]
        if not img:
            return None
        logger.debug("Got size: (name: '%s', width: '%s', height: '%s')",
                     name, img.width(), img.height())
        return img.width(), img.height()

    def resize_image(self, name, framesize):
        """ Resize the training preview image
            based on the passed in frame size """
        logger.debug("Resizing image: (name: '%s', framesize: %s", name, framesize)
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
            logger.debug("Scaling: (scale: %s, size: %s", scale, size)

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


class ReadOnlyText(tk.Text):  # pylint: disable=too-many-ancestors
    """ A read only text widget that redirects a standard tkinter.Text widgets insert and delete
    attributes.
    Source: https://stackoverflow.com/questions/3842155
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redirector = WidgetRedirector(self)
        self.insert = self.redirector.register("insert", lambda *args, **kw: "break")
        self.delete = self.redirector.register("delete", lambda *args, **kw: "break")


class ConsoleOut(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ The Console out section of the GUI """

    def __init__(self, parent, debug):
        logger.debug("Initializing %s: (parent: %s, debug: %s)",
                     self.__class__.__name__, parent, debug)
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(2, 0),
                  fill=tk.BOTH, expand=True)
        self.console = ReadOnlyText(self)
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
        self._constants = dict(pathcache=pathcache,
                               root=root,
                               scaling_factor=scaling_factor,
                               status_bar=statusbar)
        self._cli_opts = cli_opts
        self._statusbar = statusbar
        self._tk_vars = self.set_tk_vars()
        self._user_config = UserConfig(None)
        self._project = Project(self, FileHandler)
        self._tasks = Tasks(self, FileHandler)
        self._command_notebook = None  # set in command.py
        self.default_options = None
        self.session = session
        self._default_font = tk.font.nametofont("TkDefaultFont").configure()["family"]
        logger.debug("Initialized %s", self.__class__.__name__)

    # Constants
    @property
    def pathcache(self):
        """ str: The path to the GUI cache folder """
        return self._constants["pathcache"]

    @property
    def root(self):
        """ :class:`tkinter.Tk`: The root tkinter window """
        return self._constants["root"]

    @property
    def scaling_factor(self):
        """ float: The scaling factor for current display """
        return self._constants["scaling_factor"]

    @property
    def statusbar(self):
        """ :class:`lib.gui.statusbar.StatusBar`: The GUI StatusBar :class:`tkinter.ttk.Frame` """
        return self._constants["scaling_factor"]

    # GUI tkinter Objects
    @property
    def command_notebook(self):
        """ :class:`lib.gui.command.CommandNoteboook`: The main Faceswap Command Notebook """
        return self._command_notebook

    @property
    def tools_notebook(self):
        """ :class:`lib.gui.command.ToolsNotebook`: The Faceswap Tools sub-Notebook """
        return self._command_notebook.tools_notebook

    @property
    def tk_vars(self):
        """ dict: The global tkinter variables """
        return self._tk_vars

    @property
    def modified_vars(self):
        """ dict: The command tab modified tkinter variables """
        return self._command_notebook.modified_vars

    @property
    def command_tabs(self):
        """ dict: Command tab titles with their IDs """
        return self._command_notebook.tab_names

    @property
    def tools_tabs(self):
        """ dict: Tools command tab titles with their IDs """
        return self._command_notebook.tools_tab_names

    # Config
    @property
    def user_config(self):
        """ dict: The GUI config in dict form """
        return self._user_config

    @property
    def user_config_dict(self):
        """ dict: The GUI config in dict form """
        return self._user_config.config_dict

    @property
    def default_font(self):
        """ Return the selected font """
        font = self.user_config_dict["font"]
        font = self._default_font if font == "default" else font
        return (font, self.user_config_dict["font_size"])

    # GUI Properties
    @property
    def cli_opts(self):
        """ :class:`lib.gui.options.CliOptions`: The command line options for this GUI Session """
        return self._cli_opts

    @property
    def project(self):
        """ :class:`lib.gui.project.Project`: The project session handler """
        return self._project

    @property
    def tasks(self):
        """ :class:`lib.gui.project.Tasks`: The session tasks handler """
        return self._tasks

    def set_command_notebook(self, notebook):
        """ Set the command notebook to the :attr:`command_notebook` attribute "

        Parameters
        ----------
        notebook: :class:`lib.gui.command.CommandNotebook`
            The main command notebook for the Faceswap GUI
        """
        self._command_notebook = notebook
        self._project.set_modified_callback()

    def get_active_tab_name(self):
        """ Return the active tab from :attr:`command_notebook`

        Returns
        -------
        str:
            The name of the currently active :class:`lib.gui.command.CommandNotebook` tab
        """
        command = self._command_notebook.tab(self._command_notebook.select(), "text").lower()
        if command == "tools":
            command = self.tools_notebook.tab(self.tools_notebook.select(), "text").lower()
        logger.debug("Active tab: %s", command)
        return command

    def set_active_tab_by_name(self, name):
        """ Sets the :attr:`command_notebook` or :attr:`tools_notebook` to active based on given
        name

        Parameters
        ----------
        name: str
            The name of the tab to set active
        """
        name = name.lower()
        if name in self.command_tabs:
            tab_id = self.command_tabs[name]
            logger.debug("Setting active tab to: (name: %s, id: %s)", name, tab_id)
            self._command_notebook.select(tab_id)
        elif name in self.tools_tabs:
            self._command_notebook.select(self.command_tabs["tools"])
            tab_id = self.tools_tabs[name]
            logger.debug("Setting active Tools tab to: (name: %s, id: %s)", name, tab_id)
            self.tools_notebook.select()
        else:
            logger.debug("Name couldn't be found. Setting to id 0: %s", name)
            self._command_notebook.select(0)

    def set_modified_true(self, command):
        """ Set the modified variable to True for the given command """
        tkvar = self.modified_vars.get(command, None)
        if tkvar is None:
            logger.debug("No tkvar for command: '%s'", command)
            return
        tkvar.set(True)
        logger.debug("Set modified var to True for: '%s'", command)

    def reset_modified_vars(self):
        """ Set the modified variable to True for the given command """
        for command, var in self.modified_vars.items():
            var.set(False)
            logger.debug("Set modified var to False for `%s`", command)

    def refresh_config(self):
        """ Reload the user config """
        self._user_config = UserConfig(None)

    def set_default_options(self):
        """ Store the default options """
        self.default_options = self.cli_opts.get_option_values()

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
        """ tkinter variables to be triggered by to indicate
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

        tk_vars = {"display": display,
                   "runningtask": runningtask,
                   "istraining": istraining,
                   "action": actioncommand,
                   "generate": generatecommand,
                   "consoleclear": consoleclear,
                   "refreshgraph": refreshgraph,
                   "smoothgraph": smoothgraph,
                   "updatepreview": updatepreview}
        logger.debug(tk_vars)
        return tk_vars

    def set_root_title(self, text=None):
        """ Set the main title text for Faceswap """
        title = "Faceswap.py"
        title += " - {}".format(text) if text is not None and text else ""
        self.root.title(title)


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
        """ Bind the menu to the widgets Right Click event """
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
            # Avoid a ref-cycle if the thread is running a function with
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
