#!/usr/bin/env python3
""" Utility functions for the GUI """
import logging
import os
import sys
import tkinter as tk
from tkinter import filedialog
from threading import Event, Thread
from queue import Queue
import numpy as np

from PIL import Image, ImageDraw, ImageTk

from ._config import Config as UserConfig
from .project import Project, Tasks

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_CONFIG = None
_IMAGES = None
PATHCACHE = os.path.join(os.path.realpath(os.path.dirname(sys.argv[0])), "lib", "gui", ".cache")


def initialize_config(root, cli_opts, statusbar, session):
    """ Initialize the GUI Master :class:`Config` and add to global constant.

    This should only be called once on first GUI startup. Future access to :class:`Config`
    should only be executed through :func:`get_config`.

    Parameters
    ----------
    root: :class:`tkinter.Tk`
        The root Tkinter object
    cli_opts: :class:`lib.gui.options.CliOpts`
        The command line options object
    statusbar: :class:`lib.gui.custom_widgets.StatusBar`
        The GUI Status bar
    session: :class:`lib.gui.stats.Session`
        The current training Session
    """
    global _CONFIG  # pylint: disable=global-statement
    if _CONFIG is not None:
        return None
    logger.debug("Initializing config: (root: %s, cli_opts: %s, "
                 "statusbar: %s, session: %s)", root, cli_opts, statusbar, session)
    _CONFIG = Config(root, cli_opts, statusbar, session)
    return _CONFIG


def get_config():
    """ Get the Master GUI configuration.

    Returns
    -------
    :class:`Config`
        The Master GUI Config
    """
    return _CONFIG


def initialize_images():
    """ Initialize the :class:`Images` handler  and add to global constant.

    This should only be called once on first GUI startup. Future access to :class:`Images`
    handler should only be executed through :func:`get_images`.
    """
    global _IMAGES  # pylint: disable=global-statement
    if _IMAGES is not None:
        return
    logger.debug("Initializing images")
    _IMAGES = Images()


def get_images():
    """ Get the Master GUI Images handler.

    Returns
    -------
    :class:`Images`
        The Master GUI Images handler
    """
    return _IMAGES


class FileHandler():  # pylint:disable=too-few-public-methods
    """ Handles all GUI File Dialog actions and tasks.

    Parameters
    ----------
    handletype: ['open', 'save', 'filename', 'filename_multi', 'savefilename', 'context']
        The type of file dialog to return. `open` and `save` will perform the open and save actions
        and return the file. `filename` returns the filename from an `open` dialog.
        `filename_multi` allows for multi-selection of files and returns a list of files selected.
        `savefilename` returns the filename from a `save as` dialog. `context` is a context
        sensitive parameter that returns a certain dialog based on the current options
    filetype: ['default', 'alignments', 'config_project', 'config_task', 'config_all', 'csv', \
               'image', 'ini', 'state', 'log', 'video']
        The type of file that this dialog is for. `default` allows selection of any files. Other
        options limit the file type selection
    title: str, optional
        The title to display on the file dialog. If `None` then the default title will be used.
        Default: ``None``
    initialdir: str, optional
        The folder to initially open with the file dialog. If `None` then tkinter will decide.
        Default: ``None``
    command: str, optional
        Required for context handling file dialog, otherwise unused. Default: ``None``
    action: str, optional
        Required for context handling file dialog, otherwise unused. Default: ``None``
    variable: :class:`tkinter.StringVar`, optional
        Required for context handling file dialog, otherwise unused. The variable to associate
        with this file dialog. Default: ``None``

    Attributes
    ----------
    retfile: str or object
        The return value from the file dialog

    Example
    -------
    >>> handler = FileHandler('filename', 'video', title='Select a video...')
    >>> video_file = handler.retfile
    >>> print(video_file)
    '/path/to/selected/video.mp4'
    """

    def __init__(self, handletype, filetype, title=None, initialdir=None, command=None,
                 action=None, variable=None):
        logger.debug("Initializing %s: (Handletype: '%s', filetype: '%s', title: '%s', "
                     "initialdir: '%s, 'command: '%s', action: '%s', variable: %s)",
                     self.__class__.__name__, handletype, filetype, title, initialdir, command,
                     action, variable)
        self._handletype = handletype
        self._defaults = self._set_defaults()
        self._kwargs = self._set_kwargs(title, initialdir, filetype, command, action, variable)
        self.retfile = getattr(self, "_{}".format(self._handletype.lower()))()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _filetypes(self):
        """ dict: The accepted extensions for each file type for opening/saving """
        all_files = ("All files", "*.*")
        filetypes = {"default": (all_files,),
                     "alignments": [("Faceswap Alignments", "*.fsa"),
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
                               ("MPEG", "*.mpeg *.mpg *.ts *.vob"),
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

    @property
    def _contexts(self):
        """dict: Mapping of commands, actions and their corresponding file dialog for context
        handle types. """
        return {
            "effmpeg": {
                "input": {
                    "extract": "filename",
                    "gen-vid": "dir",
                    "get-fps": "filename",
                    "get-info": "filename",
                    "mux-audio": "filename",
                    "rescale": "filename",
                    "rotate": "filename",
                    "slice": "filename"},
                "output": {
                    "extract": "dir",
                    "gen-vid": "savefilename",
                    "get-fps": "nothing",
                    "get-info": "nothing",
                    "mux-audio": "savefilename",
                    "rescale": "savefilename",
                    "rotate": "savefilename",
                    "slice": "savefilename"}
                }
            }

    def _set_defaults(self):
        """ Set the default file type for the file dialog. Generally the first found file type
        will be used, but this is overridden if it is not appropriate.

        Returns
        -------
        dict:
            The default file extension for each file type
        """
        defaults = {key: val[0][1].replace("*", "")
                    for key, val in self._filetypes.items()}
        defaults["default"] = None
        defaults["video"] = ".mp4"
        defaults["image"] = ".png"
        logger.debug(defaults)
        return defaults

    def _set_kwargs(self, title, initialdir, filetype, command, action, variable=None):
        """ Generate the required kwargs for the requested file dialog browser.

        Returns
        -------
        dict:
            The key word arguments for the file dialog to be launched
        """
        logger.debug("Setting Kwargs: (title: %s, initialdir: %s, filetype: '%s', "
                     "command: '%s': action: '%s', variable: '%s')",
                     title, initialdir, filetype, command, action, variable)
        kwargs = dict()
        if self._handletype.lower() == "context":
            self._set_context_handletype(command, action, variable)

        if title is not None:
            kwargs["title"] = title

        if initialdir is not None:
            kwargs["initialdir"] = initialdir

        if self._handletype.lower() in (
                "open", "save", "filename", "filename_multi", "savefilename"):
            kwargs["filetypes"] = self._filetypes[filetype]
            if self._defaults.get(filetype, None):
                kwargs['defaultextension'] = self._defaults[filetype]
        if self._handletype.lower() == "save":
            kwargs["mode"] = "w"
        if self._handletype.lower() == "open":
            kwargs["mode"] = "r"
        logger.debug("Set Kwargs: %s", kwargs)
        return kwargs

    def _set_context_handletype(self, command, action, variable):
        """ Sets the correct handle type  based on context.

        Parameters
        ----------
        command: str
            The command that is being executed. Used to look up the context actions
        action: str
            The action that is being performed. Used to look up the correct file dialog
        variable: :class:`tkinter.StringVar`
            The variable associated with this file dialog
        """
        if self._contexts[command].get(variable, None) is not None:
            handletype = self._contexts[command][variable][action]
        else:
            handletype = self._contexts[command][action]
        logger.debug(handletype)
        self._handletype = handletype

    def _open(self):
        """ Open a file. """
        logger.debug("Popping Open browser")
        return filedialog.askopenfile(**self._kwargs)

    def _save(self):
        """ Save a file. """
        logger.debug("Popping Save browser")
        return filedialog.asksaveasfile(**self._kwargs)

    def _dir(self):
        """ Get a directory location. """
        logger.debug("Popping Dir browser")
        return filedialog.askdirectory(**self._kwargs)

    def _savedir(self):
        """ Get a save directory location. """
        logger.debug("Popping SaveDir browser")
        return filedialog.askdirectory(**self._kwargs)

    def _filename(self):
        """ Get an existing file location. """
        logger.debug("Popping Filename browser")
        return filedialog.askopenfilename(**self._kwargs)

    def _filename_multi(self):
        """ Get multiple existing file locations. """
        logger.debug("Popping Filename browser")
        return filedialog.askopenfilenames(**self._kwargs)

    def _savefilename(self):
        """ Get a save file location. """
        logger.debug("Popping SaveFilename browser")
        return filedialog.asksaveasfilename(**self._kwargs)

    @staticmethod
    def _nothing():  # pylint: disable=useless-return
        """ Method that does nothing, used for disabling open/save pop up.  """
        logger.debug("Popping Nothing browser")
        return


class Images():
    """ The centralized image repository for holding all icons and images required by the GUI.

    This class should be initialized on GUI startup through :func:`initialize_images`. Any further
    access to this class should be through :func:`get_images`.
    """
    def __init__(self):
        logger.debug("Initializing %s", self.__class__.__name__)
        self._pathpreview = os.path.join(PATHCACHE, "preview")
        self._pathoutput = None
        self._previewoutput = None
        self._previewtrain = dict()
        self._previewcache = dict(modified=None,  # cache for extract and convert
                                  images=None,
                                  filenames=list(),
                                  placeholder=None)
        self._errcount = 0
        self._icons = self._load_icons()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def previewoutput(self):
        """ Tuple or ``None``: First item in the tuple is the extract or convert preview image
        (:class:`PIL.Image`), the second item is the image in a format that tkinter can display
        (:class:`PIL.ImageTK.PhotoImage`).

        The value of the property is ``None`` if no extract or convert task is running or there are
        no files available in the output folder. """
        return self._previewoutput

    @property
    def previewtrain(self):
        """ dict or ``None``: The training preview images. Dictionary key is the image name
        (`str`). Dictionary values are a `list` of the training image (:class:`PIL.Image`), the
        image formatted for tkinter display (:class:`PIL.ImageTK.PhotoImage`), the last
        modification time of the image (`float`).

        The value of this property is ``None`` if training is not running or there are no preview
        images available.
        """
        return self._previewtrain

    @property
    def icons(self):
        """ dict: The faceswap icons for all parts of the GUI. The dictionary key is the icon
        name (`str`) the value is the icon sized and formatted for display
        (:class:`PIL.ImageTK.PhotoImage`).

        Example
        -------
        >>> icons = get_images().icons
        >>> save = icons["save"]
        >>> button = ttk.Button(parent, image=save)
        >>> button.pack()
        """
        return self._icons

    @staticmethod
    def _load_icons():
        """ Scan the icons cache folder and load the icons into :attr:`icons` for retrieval
        throughout the GUI.

        Returns
        -------
        dict:
            The icons formatted as described in :attr:`icons`

        """
        size = get_config().user_config_dict.get("icon_size", 16)
        size = int(round(size * get_config().scaling_factor))
        icons = dict()
        pathicons = os.path.join(PATHCACHE, "icons")
        for fname in os.listdir(pathicons):
            name, ext = os.path.splitext(fname)
            if ext != ".png":
                continue
            img = Image.open(os.path.join(pathicons, fname))
            img = ImageTk.PhotoImage(img.resize((size, size), resample=Image.HAMMING))
            icons[name] = img
        logger.debug(icons)
        return icons

    def set_faceswap_output_path(self, location):
        """ Set the path that will contain the output from an Extract or Convert task.

        Required so that the GUI can fetch output images to display for return in
        :attr:`previewoutput`.

        Parameters
        ----------
        location: str
            The output location that has been specified for an Extract or Convert task
        """
        self._pathoutput = location

    def delete_preview(self):
        """ Delete the preview files in the cache folder and reset the image cache.

        Should be called when terminating tasks, or when Faceswap starts up or shuts down.
        """
        logger.debug("Deleting previews")
        for item in os.listdir(self._pathpreview):
            if item.startswith(".gui_training_preview") and item.endswith(".jpg"):
                fullitem = os.path.join(self._pathpreview, item)
                logger.debug("Deleting: '%s'", fullitem)
                os.remove(fullitem)
        for fname in self._previewcache["filenames"]:
            if os.path.basename(fname) == ".gui_preview.jpg":
                logger.debug("Deleting: '%s'", fname)
                try:
                    os.remove(fname)
                except FileNotFoundError:
                    logger.debug("File does not exist: %s", fname)
        self._clear_image_cache()

    def _clear_image_cache(self):
        """ Clear all cached images. """
        logger.debug("Clearing image cache")
        self._pathoutput = None
        self._previewoutput = None
        self._previewtrain = dict()
        self._previewcache = dict(modified=None,  # cache for extract and convert
                                  images=None,
                                  filenames=list(),
                                  placeholder=None)

    @staticmethod
    def _get_images(image_path):
        """ Get the images stored within the given directory.

        Parameters
        ----------
        image_path: str
            The folder containing images to be scanned

        Returns
        -------
        list:
            The image filenames stored within the given folder

        """
        logger.debug("Getting images: '%s'", image_path)
        if not os.path.isdir(image_path):
            logger.debug("Folder does not exist")
            return None
        files = [os.path.join(image_path, f)
                 for f in os.listdir(image_path) if f.lower().endswith((".png", ".jpg"))]
        logger.debug("Image files: %s", files)
        return files

    def load_latest_preview(self, thumbnail_size, frame_dims):
        """ Load the latest preview image for extract and convert.

        Retrieves the latest preview images from the faceswap output folder, resizes to thumbnails
        and lays out for display. Places the images into :attr:`previewoutput` for loading into
        the display panel.

        Parameters
        ----------
        thumbnail_size: int
            The size of each thumbnail that should be created
        frame_dims: tuple
            The (width (`int`), height (`int`)) of the display panel that will display the preview
        """
        logger.debug("Loading preview image: (thumbnail_size: %s, frame_dims: %s)",
                     thumbnail_size, frame_dims)
        image_files = self._get_images(self._pathoutput)
        gui_preview = os.path.join(self._pathoutput, ".gui_preview.jpg")
        if not image_files or (len(image_files) == 1 and gui_preview not in image_files):
            logger.debug("No preview to display")
            self._previewoutput = None
            return
        # Filter to just the gui_preview if it exists in folder output
        image_files = [gui_preview] if gui_preview in image_files else image_files
        logger.debug("Image Files: %s", len(image_files))

        image_files = self._get_newest_filenames(image_files)
        if not image_files:
            return

        self._load_images_to_cache(image_files, frame_dims, thumbnail_size)
        if image_files == [gui_preview]:
            # Delete the preview image so that the main scripts know to output another
            logger.debug("Deleting preview image")
            os.remove(image_files[0])
        show_image = self._place_previews(frame_dims)
        if not show_image:
            self._previewoutput = None
            return
        logger.debug("Displaying preview: %s", self._previewcache["filenames"])
        self._previewoutput = (show_image, ImageTk.PhotoImage(show_image))

    def _get_newest_filenames(self, image_files):
        """ Return image filenames that have been modified since the last check.

        Parameters
        ----------
        image_files: list
            The list of image files to check the modification date for

        Returns
        -------
        list:
            A list of images that have been modified since the last check
        """
        if self._previewcache["modified"] is None:
            retval = image_files
        else:
            retval = [fname for fname in image_files
                      if os.path.getmtime(fname) > self._previewcache["modified"]]
        if not retval:
            logger.debug("No new images in output folder")
        else:
            self._previewcache["modified"] = max([os.path.getmtime(img) for img in retval])
            logger.debug("Number new images: %s, Last Modified: %s",
                         len(retval), self._previewcache["modified"])
        return retval

    def _load_images_to_cache(self, image_files, frame_dims, thumbnail_size):
        """ Load preview images to the image cache.

        Load new images and append to cache, filtering the cache the number of thumbnails that will
        fit  inside the display panel.

        Parameters
        ----------
        image_files: list
            A list of new image files that have been modified since the last check
        frame_dims: tuple
            The (width (`int`), height (`int`)) of the display panel that will display the preview
        thumbnail_size: int
            The size of each thumbnail that should be created
        """
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
        self._previewcache["filenames"] = (self._previewcache["filenames"] +
                                           show_files)[-num_images:]
        cache = self._previewcache["images"]
        if cache is None:
            logger.debug("Creating new cache")
            cache = samples[-num_images:]
        else:
            logger.debug("Appending to existing cache")
            cache = np.concatenate((cache, samples))[-num_images:]
        self._previewcache["images"] = cache
        logger.debug("Cache shape: %s", self._previewcache["images"].shape)

    def _place_previews(self, frame_dims):
        """ Format the preview thumbnails stored in the cache into a grid fitting the display
        panel.

        Parameters
        ----------
        frame_dims: tuple
            The (width (`int`), height (`int`)) of the display panel that will display the preview

        Returns
        :class:`PIL.Image`:
            The final preview display image
        """
        if self._previewcache.get("images", None) is None:
            logger.debug("No images in cache. Returning None")
            return None
        samples = self._previewcache["images"].copy()
        num_images, thumbnail_size = samples.shape[:2]
        if self._previewcache["placeholder"] is None:
            self._create_placeholder(thumbnail_size)

        logger.debug("num_images: %s, thumbnail_size: %s", num_images, thumbnail_size)
        cols, rows = frame_dims[0] // thumbnail_size, frame_dims[1] // thumbnail_size
        logger.debug("cols: %s, rows: %s", cols, rows)
        if cols == 0 or rows == 0:
            logger.debug("Cols or Rows is zero. No items to display")
            return None
        remainder = (cols * rows) - num_images
        if remainder != 0:
            logger.debug("Padding sample display. Remainder: %s", remainder)
            placeholder = np.concatenate([np.expand_dims(self._previewcache["placeholder"],
                                                         0)] * remainder)
            samples = np.concatenate((samples, placeholder))

        display = np.vstack([np.hstack(samples[row * cols: (row + 1) * cols])
                             for row in range(rows)])
        logger.debug("display shape: %s", display.shape)
        return Image.fromarray(display)

    def _create_placeholder(self, thumbnail_size):
        """ Create a placeholder image for when there are fewer thumbnails available
        than columns to display them.

        Parameters
        ----------
        thumbnail_size: int
            The size of the thumbnail that the placeholder should replicate
        """
        logger.debug("Creating placeholder. thumbnail_size: %s", thumbnail_size)
        placeholder = Image.new("RGB", (thumbnail_size, thumbnail_size))
        draw = ImageDraw.Draw(placeholder)
        draw.rectangle(((0, 0), (thumbnail_size, thumbnail_size)), outline="#E5E5E5", width=1)
        placeholder = np.array(placeholder)
        self._previewcache["placeholder"] = placeholder
        logger.debug("Created placeholder. shape: %s", placeholder.shape)

    def load_training_preview(self):
        """ Load the training preview images.

        Reads the training image currently stored in the cache folder and loads them to
        :attr:`previewtrain` for retrieval in the GUI.
        """
        logger.debug("Loading Training preview images")
        image_files = self._get_images(self._pathpreview)
        modified = None
        if not image_files:
            logger.debug("No preview to display")
            self._previewtrain = dict()
            return
        for img in image_files:
            modified = os.path.getmtime(img) if modified is None else modified
            name = os.path.basename(img)
            name = os.path.splitext(name)[0]
            name = name[name.rfind("_") + 1:].title()
            try:
                logger.debug("Displaying preview: '%s'", img)
                size = self._get_current_size(name)
                self._previewtrain[name] = [Image.open(img), None, modified]
                self.resize_image(name, size)
                self._errcount = 0
            except ValueError:
                # This is probably an error reading the file whilst it's
                # being saved  so ignore it for now and only pick up if
                # there have been multiple consecutive fails
                logger.warning("Unable to display preview: (image: '%s', attempt: %s)",
                               img, self._errcount)
                if self._errcount < 10:
                    self._errcount += 1
                else:
                    logger.error("Error reading the preview file for '%s'", img)
                    print("Error reading the preview file for {}".format(name))
                    self._previewtrain[name] = None

    def _get_current_size(self, name):
        """ Return the size of the currently displayed training preview image.

        Parameters
        ----------
        name: str
            The name of the training image to get the size for

        Returns
        -------
        width: int
            The width of the training image
        height: int
            The height of the training image
        """
        logger.debug("Getting size: '%s'", name)
        if not self._previewtrain.get(name, None):
            return None
        img = self._previewtrain[name][1]
        if not img:
            return None
        logger.debug("Got size: (name: '%s', width: '%s', height: '%s')",
                     name, img.width(), img.height())
        return img.width(), img.height()

    def resize_image(self, name, frame_dims):
        """ Resize the training preview image based on the passed in frame size.

        If the canvas that holds the preview image changes, update the image size
        to fit the new canvas and refresh :attr:`previewtrain`.

        Parameters
        ----------
        name: str
            The name of the training image to be resized
        frame_dims: tuple
            The (width (`int`), height (`int`)) of the display panel that will display the preview
        """
        logger.debug("Resizing image: (name: '%s', frame_dims: %s", name, frame_dims)
        displayimg = self._previewtrain[name][0]
        if frame_dims:
            frameratio = float(frame_dims[0]) / float(frame_dims[1])
            imgratio = float(displayimg.size[0]) / float(displayimg.size[1])

            if frameratio <= imgratio:
                scale = frame_dims[0] / float(displayimg.size[0])
                size = (frame_dims[0], int(displayimg.size[1] * scale))
            else:
                scale = frame_dims[1] / float(displayimg.size[1])
                size = (int(displayimg.size[0] * scale), frame_dims[1])
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
        self._previewtrain[name][1] = ImageTk.PhotoImage(displayimg)


class Config():
    """ The centralized configuration class for holding items that should be made available to all
    parts of the GUI.

    This class should be initialized on GUI startup through :func:`initialize_config`. Any further
    access to this class should be through :func:`get_config`.

    Parameters
    ----------
    root: :class:`tkinter.Tk`
        The root Tkinter object
    cli_opts: :class:`lib.gui.options.CliOpts`
        The command line options object
    statusbar: :class:`lib.gui.custom_widgets.StatusBar`
        The GUI Status bar
    session: :class:`lib.gui.stats.Session`
        The current training Session
    """
    def __init__(self, root, cli_opts, statusbar, session):
        logger.debug("Initializing %s: (root %s, cli_opts: %s, statusbar: %s, session: %s)",
                     self.__class__.__name__, root, cli_opts, statusbar, session)
        self._constants = dict(
            root=root,
            scaling_factor=self._get_scaling(root),
            default_font=tk.font.nametofont("TkDefaultFont").configure()["family"])
        self._gui_objects = dict(
            cli_opts=cli_opts,
            tk_vars=self._set_tk_vars(),
            project=Project(self, FileHandler),
            tasks=Tasks(self, FileHandler),
            default_options=None,
            status_bar=statusbar,
            command_notebook=None)  # set in command.py
        self._user_config = UserConfig(None)
        self.session = session
        self._default_font = tk.font.nametofont("TkDefaultFont").configure()["family"]
        logger.debug("Initialized %s", self.__class__.__name__)

    # Constants
    @property
    def root(self):
        """ :class:`tkinter.Tk`: The root tkinter window. """
        return self._constants["root"]

    @property
    def scaling_factor(self):
        """ float: The scaling factor for current display. """
        return self._constants["scaling_factor"]

    @property
    def pathcache(self):
        """ str: The path to the GUI cache folder """
        return PATHCACHE

    # GUI Objects
    @property
    def cli_opts(self):
        """ :class:`lib.gui.options.CliOptions`: The command line options for this GUI Session. """
        return self._gui_objects["cli_opts"]

    @property
    def tk_vars(self):
        """ dict: The global tkinter variables. """
        return self._gui_objects["tk_vars"]

    @property
    def project(self):
        """ :class:`lib.gui.project.Project`: The project session handler. """
        return self._gui_objects["project"]

    @property
    def tasks(self):
        """ :class:`lib.gui.project.Tasks`: The session tasks handler. """
        return self._gui_objects["tasks"]

    @property
    def default_options(self):
        """ dict: The default options for all tabs """
        return self._gui_objects["default_options"]

    @property
    def statusbar(self):
        """ :class:`lib.gui.custom_widgets.StatusBar`: The GUI StatusBar
        :class:`tkinter.ttk.Frame`. """
        return self._gui_objects["status_bar"]

    @property
    def command_notebook(self):
        """ :class:`lib.gui.command.CommandNoteboook`: The main Faceswap Command Notebook. """
        return self._gui_objects["command_notebook"]

    # Convenience GUI Objects
    @property
    def tools_notebook(self):
        """ :class:`lib.gui.command.ToolsNotebook`: The Faceswap Tools sub-Notebook. """
        return self.command_notebook.tools_notebook

    @property
    def modified_vars(self):
        """ dict: The command notebook modified tkinter variables. """
        return self.command_notebook.modified_vars

    @property
    def _command_tabs(self):
        """ dict: Command tab titles with their IDs. """
        return self.command_notebook.tab_names

    @property
    def _tools_tabs(self):
        """ dict: Tools command tab titles with their IDs. """
        return self.command_notebook.tools_tab_names

    # Config
    @property
    def user_config(self):
        """ dict: The GUI config in dict form. """
        return self._user_config

    @property
    def user_config_dict(self):
        """ dict: The GUI config in dict form. """
        return self._user_config.config_dict

    @property
    def default_font(self):
        """ tuple: The selected font as configured in user settings. First item is the font (`str`)
        second item the font size (`int`). """
        font = self.user_config_dict["font"]
        font = self._default_font if font == "default" else font
        return (font, self.user_config_dict["font_size"])

    @staticmethod
    def _get_scaling(root):
        """ Get the display DPI.

        Returns
        -------
        float:
            The scaling factor
        """
        dpi = root.winfo_fpixels("1i")
        scaling = dpi / 72.0
        logger.debug("dpi: %s, scaling: %s'", dpi, scaling)
        return scaling

    def set_default_options(self):
        """ Set the default options for :mod:`lib.gui.projects`

        The Default GUI options are stored on Faceswap startup.

        Exposed as the :attr:`_default_opts` for a project cannot be set until after the main
        Command Tabs have been loaded.
        """
        default = self.cli_opts.get_option_values()
        logger.debug(default)
        self._gui_objects["default_options"] = default
        self.project.set_default_options()

    def set_command_notebook(self, notebook):
        """ Set the command notebook to the :attr:`command_notebook` attribute
        and enable the modified callback for :attr:`project`.

        Parameters
        ----------
        notebook: :class:`lib.gui.command.CommandNotebook`
            The main command notebook for the Faceswap GUI
        """
        logger.debug("Setting commane notebook: %s", notebook)
        self._gui_objects["command_notebook"] = notebook
        self.project.set_modified_callback()

    def set_active_tab_by_name(self, name):
        """ Sets the :attr:`command_notebook` or :attr:`tools_notebook` to active based on given
        name.

        Parameters
        ----------
        name: str
            The name of the tab to set active
        """
        name = name.lower()
        if name in self._command_tabs:
            tab_id = self._command_tabs[name]
            logger.debug("Setting active tab to: (name: %s, id: %s)", name, tab_id)
            self.command_notebook.select(tab_id)
        elif name in self._tools_tabs:
            self.command_notebook.select(self._command_tabs["tools"])
            tab_id = self._tools_tabs[name]
            logger.debug("Setting active Tools tab to: (name: %s, id: %s)", name, tab_id)
            self.tools_notebook.select()
        else:
            logger.debug("Name couldn't be found. Setting to id 0: %s", name)
            self.command_notebook.select(0)

    def set_modified_true(self, command):
        """ Set the modified variable to ``True`` for the given command in :attr:`modified_vars`.

        Parameters
        ----------
        command: str
            The command to set the modified state to ``True``

        """
        tkvar = self.modified_vars.get(command, None)
        if tkvar is None:
            logger.debug("No tkvar for command: '%s'", command)
            return
        tkvar.set(True)
        logger.debug("Set modified var to True for: '%s'", command)

    def refresh_config(self):
        """ Reload the user config from file. """
        self._user_config = UserConfig(None)

    def set_cursor_busy(self, widget=None):
        """ Set the root or widget cursor to busy.

        Parameters
        ----------
        widget: tkinter object, optional
            The widget to set busy cursor for. If the provided value is ``None`` then sets the
            cursor busy for the whole of the GUI. Default: ``None``.
        """
        logger.debug("Setting cursor to busy. widget: %s", widget)
        widget = self.root if widget is None else widget
        widget.config(cursor="watch")
        widget.update_idletasks()

    def set_cursor_default(self, widget=None):
        """ Set the root or widget cursor to default.

        Parameters
        ----------
        widget: tkinter object, optional
            The widget to set default cursor for. If the provided value is ``None`` then sets the
            cursor busy for the whole of the GUI. Default: ``None``
        """
        logger.debug("Setting cursor to default. widget: %s", widget)
        widget = self.root if widget is None else widget
        widget.config(cursor="")
        widget.update_idletasks()

    @staticmethod
    def _set_tk_vars():
        """ Set the global tkinter variables stored for easy access in :class:`Config`.

        The variables are available through :attr:`tk_vars`.
        """
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

        analysis_folder = tk.StringVar()
        analysis_folder.set(None)

        tk_vars = {"display": display,
                   "runningtask": runningtask,
                   "istraining": istraining,
                   "action": actioncommand,
                   "generate": generatecommand,
                   "consoleclear": consoleclear,
                   "refreshgraph": refreshgraph,
                   "smoothgraph": smoothgraph,
                   "updatepreview": updatepreview,
                   "analysis_folder": analysis_folder}
        logger.debug(tk_vars)
        return tk_vars

    def set_root_title(self, text=None):
        """ Set the main title text for Faceswap.

        The title will always begin with 'Faceswap.py'. Additional text can be appended.

        Parameters
        ----------
        text: str, optional
            Additional text to be appended to the GUI title bar. Default: ``None``
        """
        title = "Faceswap.py"
        title += " - {}".format(text) if text is not None and text else ""
        self.root.title(title)

    def set_geometry(self, width, height, fullscreen=False):
        """ Set the geometry for the root tkinter object.

        Parameters
        ----------
        width: int
            The width to set the window to (prior to scaling)
        height: int
            The height to set the window to (prior to scaling)
        fullscreen: bool, optional
            Whether to set the window to full-screen mode. If ``True`` then :attr:`width` and
            :attr:`height` are ignored. Default: ``False``
        """
        self.root.tk.call("tk", "scaling", self.scaling_factor)
        if fullscreen:
            initial_dimensions = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        else:
            initial_dimensions = (round(width * self.scaling_factor),
                                  round(height * self.scaling_factor))

        if fullscreen and sys.platform == "win32":
            self.root.state('zoomed')
        elif fullscreen:
            self.root.attributes('-zoomed', True)
        else:
            self.root.geometry("{}x{}+80+80".format(str(initial_dimensions[0]),
                                                    str(initial_dimensions[1])))
        logger.debug("Geometry: %sx%s", *initial_dimensions)


class LongRunningTask(Thread):
    """ Runs long running tasks in a background thread to prevent the GUI from becoming
    unresponsive.

    This is sub-classed from :class:`Threading.Thread` so check documentation there for base
    parameters. Additional parameters listed below.

    Parameters
    ----------
    widget: tkinter object, optional
        The widget that this :class:`LongRunningTask` is associated with. Used for setting the busy
        cursor in the correct location. Default: ``None``.
    """
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=True,
                 widget=None):
        logger.debug("Initializing %s: (group: %s, target: %s, name: %s, args: %s, kwargs: %s, "
                     "daemon: %s)", self.__class__.__name__, group, target, name, args, kwargs,
                     daemon)
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs,
                         daemon=daemon)
        self.err = None
        self._widget = widget
        self._config = get_config()
        self._config.set_cursor_busy(widget=self._widget)
        self._complete = Event()
        self._queue = Queue()
        logger.debug("Initialized %s", self.__class__.__name__,)

    @property
    def complete(self):
        """ :class:`threading.Event`:  Event is set if the thread has completed its task,
        otherwise it is unset.
        """
        return self._complete

    def run(self):
        """ Commence the given task in a background thread. """
        try:
            if self._target:
                retval = self._target(*self._args, **self._kwargs)
                self._queue.put(retval)
        except Exception:  # pylint: disable=broad-except
            self.err = sys.exc_info()
            logger.debug("Error in thread (%s): %s", self._name,
                         self.err[1].with_traceback(self.err[2]))
        finally:
            self._complete.set()
            # Avoid a ref-cycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def get_result(self):
        """ Return the result from the given task.

        Returns
        -------
        varies:
            The result of the thread will depend on the given task. If a call is made to
            :func:`get_result` prior to the thread completing its task then ``None`` will be
            returned
        """
        if not self._complete.is_set():
            logger.warning("Aborting attempt to retrieve result from a LongRunningTask that is "
                           "still running")
            return None
        if self.err:
            logger.debug("Error caught in thread")
            self._config.set_cursor_default(widget=self._widget)
            raise self.err[1].with_traceback(self.err[2])

        logger.debug("Getting result from thread")
        retval = self._queue.get()
        logger.debug("Got result from thread")
        self._config.set_cursor_default(widget=self._widget)
        return retval
