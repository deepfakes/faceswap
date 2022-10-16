#!/usr/bin python3
""" Utilities for handling images in the Faceswap GUI """

import logging
import os
import sys
from typing import cast, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .config import get_config, PATHCACHE

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

_IMAGES: Optional["Images"] = None
_PREVIEW_TRIGGER: Optional["PreviewTrigger"] = None


def initialize_images() -> None:
    """ Initialize the :class:`Images` handler  and add to global constant.

    This should only be called once on first GUI startup. Future access to :class:`Images`
    handler should only be executed through :func:`get_images`.
    """
    global _IMAGES  # pylint: disable=global-statement
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


class Images():
    """ The centralized image repository for holding all icons and images required by the GUI.

    This class should be initialized on GUI startup through :func:`initialize_images`. Any further
    access to this class should be through :func:`get_images`.
    """
    def __init__(self) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        self._pathpreview = os.path.join(PATHCACHE, "preview")
        self._pathoutput: Optional[str] = None
        self._batch_mode = False
        self._previewoutput: Optional[Tuple[Image.Image, ImageTk.PhotoImage]] = None
        self._previewtrain: Dict[str, List[Union[Image.Image,
                                                 ImageTk.PhotoImage,
                                                 None,
                                                 float]]] = {}
        self._previewcache: Dict[str, Union[None, float, np.ndarray, List[str]]] = dict(
            modified=None,  # cache for extract and convert
            images=None,
            filenames=[],
            placeholder=None)
        self._errcount = 0
        self._icons = self._load_icons()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def previewoutput(self) -> Optional[Tuple[Image.Image, ImageTk.PhotoImage]]:
        """ Tuple or ``None``: First item in the tuple is the extract or convert preview image
        (:class:`PIL.Image`), the second item is the image in a format that tkinter can display
        (:class:`PIL.ImageTK.PhotoImage`).

        The value of the property is ``None`` if no extract or convert task is running or there are
        no files available in the output folder. """
        return self._previewoutput

    @property
    def previewtrain(self) -> Dict[str, List[Union[Image.Image, ImageTk.PhotoImage, None, float]]]:
        """ dict or ``None``: The training preview images. Dictionary key is the image name
        (`str`). Dictionary values are a `list` of the training image (:class:`PIL.Image`), the
        image formatted for tkinter display (:class:`PIL.ImageTK.PhotoImage`), the last
        modification time of the image (`float`).

        The value of this property is ``None`` if training is not running or there are no preview
        images available.
        """
        return self._previewtrain

    @property
    def icons(self) -> Dict[str, ImageTk.PhotoImage]:
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
    def _load_icons() -> Dict[str, ImageTk.PhotoImage]:
        """ Scan the icons cache folder and load the icons into :attr:`icons` for retrieval
        throughout the GUI.

        Returns
        -------
        dict:
            The icons formatted as described in :attr:`icons`

        """
        size = get_config().user_config_dict.get("icon_size", 16)
        size = int(round(size * get_config().scaling_factor))
        icons: Dict[str, ImageTk.PhotoImage] = {}
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

    def set_faceswap_output_path(self, location: str, batch_mode: bool = False) -> None:
        """ Set the path that will contain the output from an Extract or Convert task.

        Required so that the GUI can fetch output images to display for return in
        :attr:`previewoutput`.

        Parameters
        ----------
        location: str
            The output location that has been specified for an Extract or Convert task
        batch_mode: bool
            ``True`` if extracting in batch mode otherwise False
        """
        self._pathoutput = location
        self._batch_mode = batch_mode

    def delete_preview(self) -> None:
        """ Delete the preview files in the cache folder and reset the image cache.

        Should be called when terminating tasks, or when Faceswap starts up or shuts down.
        """
        logger.debug("Deleting previews")
        for item in os.listdir(self._pathpreview):
            if item.startswith(".gui_training_preview") and item.endswith((".jpg", ".png")):
                fullitem = os.path.join(self._pathpreview, item)
                logger.debug("Deleting: '%s'", fullitem)
                os.remove(fullitem)
        for fname in cast(List[str], self._previewcache["filenames"]):
            if os.path.basename(fname) == ".gui_preview.jpg":
                logger.debug("Deleting: '%s'", fname)
                try:
                    os.remove(fname)
                except FileNotFoundError:
                    logger.debug("File does not exist: %s", fname)
        self._clear_image_cache()

    def _clear_image_cache(self) -> None:
        """ Clear all cached images. """
        logger.debug("Clearing image cache")
        self._pathoutput = None
        self._batch_mode = False
        self._previewoutput = None
        self._previewtrain = {}
        self._previewcache = dict(modified=None,  # cache for extract and convert
                                  images=None,
                                  filenames=[],
                                  placeholder=None)

    @staticmethod
    def _get_images(image_path: str) -> List[str]:
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

    def load_latest_preview(self, thumbnail_size: int, frame_dims: Tuple[int, int]) -> None:
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
        assert self._pathoutput is not None
        image_path = self._get_newest_folder() if self._batch_mode else self._pathoutput
        image_files = self._get_images(image_path)
        gui_preview = os.path.join(self._pathoutput, ".gui_preview.jpg")
        if not image_files or (len(image_files) == 1 and gui_preview not in image_files):
            logger.debug("No preview to display")
            return
        # Filter to just the gui_preview if it exists in folder output
        image_files = [gui_preview] if gui_preview in image_files else image_files
        logger.debug("Image Files: %s", len(image_files))

        image_files = self._get_newest_filenames(image_files)
        if not image_files:
            return

        if not self._load_images_to_cache(image_files, frame_dims, thumbnail_size):
            logger.debug("Failed to load any preview images")
            if gui_preview in image_files:
                # Reset last modified for failed loading of a gui preview image so it is picked
                # up next time
                self._previewcache["modified"] = None
            return

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

    def _get_newest_folder(self) -> str:
        """ Obtain the most recent folder created in the extraction output folder when processing
        in batch mode.

        Returns
        -------
        str
            The most recently modified folder within the parent output folder. If no folders have
            been created, returns the parent output folder

        """
        assert self._pathoutput is not None
        folders = [] if not os.path.exists(self._pathoutput) else [
            os.path.join(self._pathoutput, folder)
            for folder in os.listdir(self._pathoutput)
            if os.path.isdir(os.path.join(self._pathoutput, folder))]

        folders.sort(key=os.path.getmtime)
        retval = folders[-1] if folders else self._pathoutput
        logger.debug("sorted folders: %s, return value: %s", folders, retval)
        return retval

    def _get_newest_filenames(self, image_files: List[str]) -> List[str]:
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
                      if os.path.getmtime(fname) > cast(float, self._previewcache["modified"])]
        if not retval:
            logger.debug("No new images in output folder")
        else:
            self._previewcache["modified"] = max(os.path.getmtime(img) for img in retval)
            logger.debug("Number new images: %s, Last Modified: %s",
                         len(retval), self._previewcache["modified"])
        return retval

    def _load_images_to_cache(self,
                              image_files: List[str],
                              frame_dims: Tuple[int, int],
                              thumbnail_size: int) -> bool:
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
        samples: List[np.ndarray] = []
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
        :class:`PIL.Image`:
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
                         samples: List[np.ndarray],
                         filenames: List[str],
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

        self._previewcache["filenames"] = (cast(List[str], self._previewcache["filenames"]) +
                                           filenames)[-num_images:]
        cache = cast(Optional[np.ndarray], self._previewcache["images"])
        if cache is None:
            logger.debug("Creating new cache")
            cache = asamples[-num_images:]
        else:
            logger.debug("Appending to existing cache")
            cache = np.concatenate((cache, asamples))[-num_images:]
        self._previewcache["images"] = cache
        logger.debug("Cache shape: %s", cast(np.ndarray, self._previewcache["images"]).shape)
        return True

    def _place_previews(self, frame_dims: Tuple[int, int]) -> Image.Image:
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
        if self._previewcache.get("images", None) is None:
            logger.debug("No images in cache. Returning None")
            return None
        samples = cast(np.ndarray, self._previewcache["images"]).copy()
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
            placeholder = np.concatenate([np.expand_dims(
                cast(np.ndarray, self._previewcache["placeholder"]), 0)] * remainder)
            samples = np.concatenate((samples, placeholder))

        display = np.vstack([np.hstack(cast(Sequence, samples[row * cols: (row + 1) * cols]))
                             for row in range(rows)])
        logger.debug("display shape: %s", display.shape)
        return Image.fromarray(display)

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
        self._previewcache["placeholder"] = placeholder
        logger.debug("Created placeholder. shape: %s", placeholder.shape)

    def load_training_preview(self) -> None:
        """ Load the training preview images.

        Reads the training image currently stored in the cache folder and loads them to
        :attr:`previewtrain` for retrieval in the GUI.
        """
        logger.debug("Loading Training preview images")
        image_files = self._get_images(self._pathpreview)
        modified = None
        if not image_files:
            logger.debug("No preview to display")
            self._previewtrain = {}
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
                    print(f"Error reading the preview file for {name}")
                    del self._previewtrain[name]

    def _get_current_size(self, name: str) -> Optional[Tuple[int, int]]:
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
        if not self._previewtrain.get(name):
            return None
        img = cast(Image.Image, self._previewtrain[name][1])
        if not img:
            return None
        logger.debug("Got size: (name: '%s', width: '%s', height: '%s')",
                     name, img.width(), img.height())
        return img.width(), img.height()

    def resize_image(self, name: str, frame_dims: Optional[Tuple[int, int]]) -> None:
        """ Resize the training preview image based on the passed in frame size.

        If the canvas that holds the preview image changes, update the image size
        to fit the new canvas and refresh :attr:`previewtrain`.

        Parameters
        ----------
        name: str
            The name of the training image to be resized
        frame_dims: tuple, optional
            The (width (`int`), height (`int`)) of the display panel that will display the preview.
            ``None`` if the frame dimensions are not known.
        """
        logger.debug("Resizing image: (name: '%s', frame_dims: %s", name, frame_dims)
        displayimg = cast(Image.Image, self._previewtrain[name][0])
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


class PreviewTrigger():
    """ Triggers to indicate to underlying Faceswap process that the preview image should
    be updated.

    Writes a file to the cache folder that is picked up by the main process.
    """
    def __init__(self) -> None:
        logger.debug("Initializing: %s", self.__class__.__name__)
        self._trigger_files = dict(update=os.path.join(PATHCACHE, ".preview_trigger"),
                                   mask_toggle=os.path.join(PATHCACHE, ".preview_mask_toggle"))
        logger.debug("Initialized: %s (trigger_files: %s)",
                     self.__class__.__name__, self._trigger_files)

    def set(self, trigger_type: Literal["update", "mask_toggle"]):
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

    def clear(self, trigger_type: Optional[Literal["update", "mask_toggle"]] = None) -> None:
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
