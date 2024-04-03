#!/usr/bin python3
""" Utilities for handling images in the Faceswap GUI """
from __future__ import annotations
import logging
import os
import typing as T

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from lib.training.preview_cv import PreviewBuffer

from .config import get_config, PATHCACHE

if T.TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
_IMAGES: "Images" | None = None
_PREVIEW_TRIGGER: "PreviewTrigger" | None = None
TRAININGPREVIEW = ".gui_training_preview.png"


def initialize_images() -> None:
    """ Initialize the :class:`Images` handler  and add to global constant.

    This should only be called once on first GUI startup. Future access to :class:`Images`
    handler should only be executed through :func:`get_images`.
    """
    global _IMAGES  # pylint:disable=global-statement
    if _IMAGES is not None:
        return
    logger.debug("Initializing images")
    _IMAGES = Images()


def get_images() -> "Images":
    """ Get the Master GUI Images handler.

    Returns
    -------
    :class:`Images`
        The Master GUI Images handler
    """
    assert _IMAGES is not None
    return _IMAGES


def _get_previews(image_path: str) -> list[str]:
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
        return []
    files = [os.path.join(image_path, f)
             for f in os.listdir(image_path) if f.lower().endswith((".png", ".jpg"))]
    logger.debug("Image files: %s", files)
    return files


class PreviewTrain():
    """ Handles the loading of the training preview image(s) and adding to the display buffer

    Parameters
    ----------
    cache_path: str
        Full path to the cache folder that contains the preview images
    """
    def __init__(self, cache_path: str) -> None:
        logger.debug("Initializing %s: (cache_path: '%s')", self.__class__.__name__, cache_path)
        self._buffer = PreviewBuffer()
        self._cache_path = cache_path
        self._modified: float = 0.0
        self._error_count: int = 0
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def buffer(self) -> PreviewBuffer:
        """ :class:`~lib.training.PreviewBuffer` The preview buffer for the training preview
        image. """
        return self._buffer

    def load(self) -> bool:
        """ Load the latest training preview image(s) from disk and add to :attr:`buffer` """
        logger.trace("Loading Training preview images")  # type:ignore
        image_files = _get_previews(self._cache_path)
        filename = next((fname for fname in image_files
                         if os.path.basename(fname) == TRAININGPREVIEW), "")
        if not filename:
            logger.trace("No preview to display")  # type:ignore
            return False
        try:
            modified = os.path.getmtime(filename)
            if modified <= self._modified:
                logger.trace("preview '%s' not updated. Current timestamp: %s, "  # type:ignore
                             "existing timestamp: %s", filename, modified, self._modified)
                return False

            logger.debug("Loading preview: '%s'", filename)
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            assert img is not None
            self._modified = modified
            self._buffer.add_image(os.path.basename(filename), img)
            self._error_count = 0
        except (ValueError, AssertionError):
            # This is probably an error reading the file whilst it's being saved so ignore it
            # for now and only pick up if there have been multiple consecutive fails
            logger.debug("Unable to display preview: (image: '%s', attempt: %s)",
                         img, self._error_count)
            if self._error_count < 10:
                self._error_count += 1
            else:
                logger.error("Error reading the preview file for '%s'", filename)
            return False

        logger.debug("Loaded preview: '%s' (%s)", filename, img.shape)
        return True

    def reset(self) -> None:
        """ Reset the preview buffer when the display page has been disabled.

        Notes
        -----
        The buffer requires resetting, otherwise the re-enabled preview window hangs waiting for a
        training image that has already been marked as processed
        """
        logger.debug("Resetting training preview")
        del self._buffer
        self._buffer = PreviewBuffer()
        self._modified = 0.0
        self._error_count = 0


class PreviewExtract():
    """ Handles the loading of preview images for extract and convert

    Parameters
    ----------
    cache_path: str
        Full path to the cache folder that contains the preview images
    """
    def __init__(self, cache_path: str) -> None:
        logger.debug("Initializing %s: (cache_path: '%s')", self.__class__.__name__, cache_path)
        self._cache_path = cache_path

        self._batch_mode = False
        self._output_path = ""

        self._modified: float = 0.0
        self._filenames: list[str] = []
        self._images: np.ndarray | None = None
        self._placeholder: np.ndarray | None = None

        self._preview_image: Image.Image | None = None
        self._preview_image_tk: ImageTk.PhotoImage | None = None

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def image(self) -> ImageTk.PhotoImage:
        """:class:`PIL.ImageTk.PhotoImage` The preview image for displaying in a tkinter canvas """
        assert self._preview_image_tk is not None
        return self._preview_image_tk

    def save(self, filename: str) -> None:
        """ Save the currently displaying preview image to the given location

        Parameters
        ----------
        filename: str
            The full path to the filename to save the preview image to
        """
        logger.debug("Saving preview to %s", filename)
        assert self._preview_image is not None
        self._preview_image.save(filename)

    def set_faceswap_output_path(self, location: str, batch_mode: bool = False) -> None:
        """ Set the path that will contain the output from an Extract or Convert task.

        Required so that the GUI can fetch output images to display for return in
        :attr:`preview_image`.

        Parameters
        ----------
        location: str
            The output location that has been specified for an Extract or Convert task
        batch_mode: bool
            ``True`` if extracting in batch mode otherwise False
        """
        self._output_path = location
        self._batch_mode = batch_mode

    def _get_newest_folder(self) -> str:
        """ Obtain the most recent folder created in the extraction output folder when processing
        in batch mode.

        Returns
        -------
        str
            The most recently modified folder within the parent output folder. If no folders have
            been created, returns the parent output folder

        """
        folders = [] if not os.path.exists(self._output_path) else [
            os.path.join(self._output_path, folder)
            for folder in os.listdir(self._output_path)
            if os.path.isdir(os.path.join(self._output_path, folder))]

        folders.sort(key=os.path.getmtime)
        retval = folders[-1] if folders else self._output_path
        logger.debug("sorted folders: %s, return value: %s", folders, retval)
        return retval

    def _get_newest_filenames(self, image_files: list[str]) -> list[str]:
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
        if not self._modified:
            retval = image_files
        else:
            retval = [fname for fname in image_files
                      if os.path.getmtime(fname) > self._modified]
        if not retval:
            logger.debug("No new images in output folder")
        else:
            self._modified = max(os.path.getmtime(img) for img in retval)
            logger.debug("Number new images: %s, Last Modified: %s",
                         len(retval), self._modified)
        return retval

    def _pad_and_border(self, image: Image.Image, size: int) -> np.ndarray:
        """ Pad rectangle images to a square and draw borders

        Parameters
        ----------
        image: :class:`PIL.Image`
            The image to process
        size: int
            The size of the image as it should be displayed

        Returns
        -------
        :class:`numpy.ndarray`:
            The processed image
        """
        if image.size[0] != image.size[1]:
            # Pad to square
            new_img = Image.new("RGB", (size, size))
            new_img.paste(image, ((size - image.size[0]) // 2, (size - image.size[1]) // 2))
            image = new_img
        draw = ImageDraw.Draw(image)
        draw.rectangle(((0, 0), (size, size)), outline="#E5E5E5", width=1)
        retval = np.array(image)
        logger.trace("image shape: %s", retval.shape)  # type: ignore
        return retval

    def _process_samples(self,
                         samples: list[np.ndarray],
                         filenames: list[str],
                         num_images: int) -> bool:
        """ Process the latest sample images into a displayable image.

        Parameters
        ----------
        samples: list
            The list of extract/convert preview images to display
        filenames: list
            The full path to the filenames corresponding to the images
        num_images: int
            The number of images that should be displayed

        Returns
        -------
        bool
            ``True`` if samples succesfully compiled otherwise ``False``
        """
        asamples = np.array(samples)
        if not np.any(asamples):
            logger.debug("No preview images collected.")
            return False

        self._filenames = (self._filenames + filenames)[-num_images:]
        cache = self._images

        if cache is None:
            logger.debug("Creating new cache")
            cache = asamples[-num_images:]
        else:
            logger.debug("Appending to existing cache")
            cache = np.concatenate((cache, asamples))[-num_images:]

        self._images = cache
        assert self._images is not None
        logger.debug("Cache shape: %s", self._images.shape)
        return True

    def _load_images_to_cache(self,
                              image_files: list[str],
                              frame_dims: tuple[int, int],
                              thumbnail_size: int) -> bool:
        """ Load preview images to the image cache.

        Load new images and append to cache, filtering the cache to the number of thumbnails that
        will fit inside the display panel.

        Parameters
        ----------
        image_files: list
            A list of new image files that have been modified since the last check
        frame_dims: tuple
            The (width (`int`), height (`int`)) of the display panel that will display the preview
        thumbnail_size: int
            The size of each thumbnail that should be created

        Returns
        -------
        bool
            ``True`` if images were successfully loaded to cache otherwise ``False``
        """
        logger.debug("Number image_files: %s, frame_dims: %s, thumbnail_size: %s",
                     len(image_files), frame_dims, thumbnail_size)
        num_images = (frame_dims[0] // thumbnail_size) * (frame_dims[1] // thumbnail_size)
        logger.debug("num_images: %s", num_images)
        if num_images == 0:
            return False
        samples: list[np.ndarray] = []
        start_idx = len(image_files) - num_images if len(image_files) > num_images else 0
        show_files = sorted(image_files, key=os.path.getctime)[start_idx:]
        dropped_files = []
        for fname in show_files:
            try:
                img = Image.open(fname)
            except PermissionError as err:
                logger.debug("Permission error opening preview file: '%s'. Original error: %s",
                             fname, str(err))
                dropped_files.append(fname)
                continue
            except Exception as err:  # pylint:disable=broad-except
                # Swallow any issues with opening an image rather than spamming console
                # Can happen when trying to read partially saved images
                logger.debug("Error opening preview file: '%s'. Original error: %s",
                             fname, str(err))
                dropped_files.append(fname)
                continue

            width, height = img.size
            scaling = thumbnail_size / max(width, height)
            logger.debug("image width: %s, height: %s, scaling: %s", width, height, scaling)

            try:
                img = img.resize((int(width * scaling), int(height * scaling)))
            except OSError as err:
                # Image only gets loaded when we call a method, so may error on partial loads
                logger.debug("OS Error resizing preview image: '%s'. Original error: %s",
                             fname, err)
                dropped_files.append(fname)
                continue

            samples.append(self._pad_and_border(img, thumbnail_size))

        return self._process_samples(samples,
                                     [fname for fname in show_files if fname not in dropped_files],
                                     num_images)

    def _create_placeholder(self, thumbnail_size: int) -> None:
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
        self._placeholder = placeholder
        logger.debug("Created placeholder. shape: %s", placeholder.shape)

    def _place_previews(self, frame_dims: tuple[int, int]) -> Image.Image:
        """ Format the preview thumbnails stored in the cache into a grid fitting the display
        panel.

        Parameters
        ----------
        frame_dims: tuple
            The (width (`int`), height (`int`)) of the display panel that will display the preview

        Returns
        -------
        :class:`PIL.Image`:
            The final preview display image
        """
        if self._images is None:
            logger.debug("No images in cache. Returning None")
            return None
        samples = self._images.copy()
        num_images, thumbnail_size = samples.shape[:2]
        if self._placeholder is None:
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
            assert self._placeholder is not None
            placeholder = np.concatenate([np.expand_dims(self._placeholder, 0)] * remainder)
            samples = np.concatenate((samples, placeholder))

        display = np.vstack([np.hstack(T.cast("Sequence", samples[row * cols: (row + 1) * cols]))
                             for row in range(rows)])
        logger.debug("display shape: %s", display.shape)
        return Image.fromarray(display)

    def load_latest_preview(self, thumbnail_size: int, frame_dims: tuple[int, int]) -> bool:
        """ Load the latest preview image for extract and convert.

        Retrieves the latest preview images from the faceswap output folder, resizes to thumbnails
        and lays out for display. Places the images into :attr:`preview_image` for loading into
        the display panel.

        Parameters
        ----------
        thumbnail_size: int
            The size of each thumbnail that should be created
        frame_dims: tuple
            The (width (`int`), height (`int`)) of the display panel that will display the preview

        Returns
        -------
        bool
            ``True`` if a preview was succesfully loaded otherwise ``False``
        """
        logger.debug("Loading preview image: (thumbnail_size: %s, frame_dims: %s)",
                     thumbnail_size, frame_dims)
        image_path = self._get_newest_folder() if self._batch_mode else self._output_path
        image_files = _get_previews(image_path)
        gui_preview = os.path.join(self._output_path, ".gui_preview.jpg")
        if not image_files or (len(image_files) == 1 and gui_preview not in image_files):
            logger.debug("No preview to display")
            return False
        # Filter to just the gui_preview if it exists in folder output
        image_files = [gui_preview] if gui_preview in image_files else image_files
        logger.debug("Image Files: %s", len(image_files))

        image_files = self._get_newest_filenames(image_files)
        if not image_files:
            return False

        if not self._load_images_to_cache(image_files, frame_dims, thumbnail_size):
            logger.debug("Failed to load any preview images")
            if gui_preview in image_files:
                # Reset last modified for failed loading of a gui preview image so it is picked
                # up next time
                self._modified = 0.0
            return False

        if image_files == [gui_preview]:
            # Delete the preview image so that the main scripts know to output another
            logger.debug("Deleting preview image")
            os.remove(image_files[0])
        show_image = self._place_previews(frame_dims)
        if not show_image:
            self._preview_image = None
            self._preview_image_tk = None
            return False

        logger.debug("Displaying preview: %s", self._filenames)
        self._preview_image = show_image
        self._preview_image_tk = ImageTk.PhotoImage(show_image)
        return True

    def delete_previews(self) -> None:
        """ Remove any image preview files """
        for fname in self._filenames:
            if os.path.basename(fname) == ".gui_preview.jpg":
                logger.debug("Deleting: '%s'", fname)
                try:
                    os.remove(fname)
                except FileNotFoundError:
                    logger.debug("File does not exist: %s", fname)


class Images():
    """ The centralized image repository for holding all icons and images required by the GUI.

    This class should be initialized on GUI startup through :func:`initialize_images`. Any further
    access to this class should be through :func:`get_images`.
    """
    def __init__(self) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        self._pathpreview = os.path.join(PATHCACHE, "preview")
        self._pathoutput: str | None = None
        self._batch_mode = False
        self._preview_train = PreviewTrain(self._pathpreview)
        self._preview_extract = PreviewExtract(self._pathpreview)
        self._icons = self._load_icons()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def preview_train(self) -> PreviewTrain:
        """ :class:`PreviewTrain` The object handling the training preview images """
        return self._preview_train

    @property
    def preview_extract(self) -> PreviewExtract:
        """ :class:`PreviewTrain` The object handling the training preview images """
        return self._preview_extract

    @property
    def icons(self) -> dict[str, ImageTk.PhotoImage]:
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
    def _load_icons() -> dict[str, ImageTk.PhotoImage]:
        """ Scan the icons cache folder and load the icons into :attr:`icons` for retrieval
        throughout the GUI.

        Returns
        -------
        dict:
            The icons formatted as described in :attr:`icons`

        """
        size = get_config().user_config_dict.get("icon_size", 16)
        size = int(round(size * get_config().scaling_factor))
        icons: dict[str, ImageTk.PhotoImage] = {}
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

    def delete_preview(self) -> None:
        """ Delete the preview files in the cache folder and reset the image cache.

        Should be called when terminating tasks, or when Faceswap starts up or shuts down.
        """
        logger.debug("Deleting previews")
        for item in os.listdir(self._pathpreview):
            if item.startswith(os.path.splitext(TRAININGPREVIEW)[0]) and item.endswith((".jpg",
                                                                                        ".png")):
                fullitem = os.path.join(self._pathpreview, item)
                logger.debug("Deleting: '%s'", fullitem)
                os.remove(fullitem)

        self._preview_extract.delete_previews()
        del self._preview_train
        del self._preview_extract
        self._preview_train = PreviewTrain(self._pathpreview)
        self._preview_extract = PreviewExtract(self._pathpreview)


class PreviewTrigger():
    """ Triggers to indicate to underlying Faceswap process that the preview image should
    be updated.

    Writes a file to the cache folder that is picked up by the main process.
    """
    def __init__(self) -> None:
        logger.debug("Initializing: %s", self.__class__.__name__)
        self._trigger_files = {"update": os.path.join(PATHCACHE, ".preview_trigger"),
                               "mask_toggle": os.path.join(PATHCACHE, ".preview_mask_toggle")}
        logger.debug("Initialized: %s (trigger_files: %s)",
                     self.__class__.__name__, self._trigger_files)

    def set(self, trigger_type: T.Literal["update", "mask_toggle"]):
        """ Place the trigger file into the cache folder

        Parameters
        ----------
        trigger_type: ["update", "mask_toggle"]
            The type of action to trigger. 'update': Full preview update. 'mask_toggle': toggle
            mask on and off
         """
        trigger = self._trigger_files[trigger_type]
        if not os.path.isfile(trigger):
            with open(trigger, "w", encoding="utf8"):
                pass
            logger.debug("Set preview trigger: %s", trigger)

    def clear(self, trigger_type: T.Literal["update", "mask_toggle"] | None = None) -> None:
        """ Remove the trigger file from the cache folder.

        Parameters
        ----------
        trigger_type: ["update", "mask_toggle", ``None``], optional
            The trigger to clear. 'update': Full preview update. 'mask_toggle': toggle mask on
            and off. ``None`` - clear all triggers. Default: ``None``
        """
        if trigger_type is None:
            triggers = list(self._trigger_files.values())
        else:
            triggers = [self._trigger_files[trigger_type]]
        for trigger in triggers:
            if os.path.isfile(trigger):
                os.remove(trigger)
                logger.debug("Removed preview trigger: %s", trigger)


def preview_trigger() -> PreviewTrigger:
    """ Set the global preview trigger if it has not already been set and return.

    Returns
    -------
    :class:`PreviewTrigger`
        The trigger to indicate to the main faceswap process that it should perform a training
        preview update
    """
    global _PREVIEW_TRIGGER  # pylint:disable=global-statement
    if _PREVIEW_TRIGGER is None:
        _PREVIEW_TRIGGER = PreviewTrigger()
    return _PREVIEW_TRIGGER
