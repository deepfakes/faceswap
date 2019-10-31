#!/usr/bin python3
""" Utilities for working with images and videos """

import logging
import subprocess
import os

from concurrent import futures
from hashlib import sha1

import cv2
import imageio
import imageio_ffmpeg as im_ffm
import numpy as np
from tqdm import tqdm

from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager, QueueEmpty
from lib.utils import convert_to_secs, FaceswapError, _video_extensions, get_image_paths

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name

# ################### #
# <<< IMAGE UTILS >>> #
# ################### #


# <<< IMAGE IO >>> #

def read_image(filename, raise_error=False, with_hash=False):
    """ Read an image file from a file location.

    Extends the functionality of :func:`cv2.imread()` by ensuring that an image was actually
    loaded. Errors can be logged and ignored so that the process can continue on an image load
    failure.

    Parameters
    ----------
    filename: str
        Full path to the image to be loaded.
    raise_error: bool, optional
        If ``True`` then any failures (including the returned image being ``None``) will be
        raised. If ``False`` then an error message will be logged, but the error will not be
        raised. Default: ``False``
    with_hash: bool, optional
        If ``True`` then returns the image's sha1 hash with the image. Default: ``False``

    Returns
    -------
    numpy.ndarray or tuple
        If :attr:`with_hash` is ``False`` then returns a `numpy.ndarray` of the image in `BGR`
        channel order. If :attr:`with_hash` is ``True`` then returns a `tuple` of (`numpy.ndarray`"
        of the image in `BGR`, `str` of sha` hash of image)
    Example
    -------
    >>> image_file = "/path/to/image.png"
    >>> try:
    >>>    image = read_image(image_file, raise_error=True, with_hash=False)
    >>> except:
    >>>     raise ValueError("There was an error")
    """
    logger.trace("Requested image: '%s'", filename)
    success = True
    image = None
    try:
        image = cv2.imread(filename)
        if image is None:
            raise ValueError
    except TypeError:
        success = False
        msg = "Error while reading image (TypeError): '{}'".format(filename)
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    except ValueError:
        success = False
        msg = ("Error while reading image. This is most likely caused by special characters in "
               "the filename: '{}'".format(filename))
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    except Exception as err:  # pylint:disable=broad-except
        success = False
        msg = "Failed to load image '{}'. Original Error: {}".format(filename, str(err))
        logger.error(msg)
        if raise_error:
            raise Exception(msg)
    logger.trace("Loaded image: '%s'. Success: %s", filename, success)
    retval = (image, sha1(image).hexdigest()) if with_hash else image
    return retval


def read_image_batch(filenames):
    """ Load a batch of images from the given file locations.

    Leverages multi-threading to load multiple images from disk at the same time
    leading to vastly reduced image read times.

    Parameters
    ----------
    filenames: list
        A list of ``str`` full paths to the images to be loaded.

    Returns
    -------
    numpy.ndarray
        The batch of images in `BGR` channel order.

    Notes
    -----
    As the images are compiled into a batch, they must be all of the same dimensions.

    Example
    -------
    >>> image_filenames = ["/path/to/image_1.png", "/path/to/image_2.png", "/path/to/image_3.png"]
    >>> images = read_image_batch(image_filenames)
    """
    logger.trace("Requested batch: '%s'", filenames)
    executor = futures.ThreadPoolExecutor()
    with executor:
        images = [executor.submit(read_image, filename, raise_error=True)
                  for filename in filenames]
        batch = np.array([future.result() for future in futures.as_completed(images)])
    logger.trace("Returning images: %s", batch.shape)
    return batch


def read_image_hash(filename):
    """ Return the `sha1` hash of an image saved on disk.

    Parameters
    ----------
    filename: str
        Full path to the image to be loaded.

    Returns
    -------
    str
        The :func:`hashlib.hexdigest()` representation of the `sha1` hash of the given image.
    Example
    -------
    >>> image_file = "/path/to/image.png"
    >>> image_hash = read_image_hash(image_file)
    """
    img = read_image(filename, raise_error=True)
    image_hash = sha1(img).hexdigest()
    logger.trace("filename: '%s', hash: %s", filename, image_hash)
    return image_hash


def read_image_hash_batch(filenames):
    """ Return the `sha` hash of a batch of images

    Leverages multi-threading to load multiple images from disk at the same time
    leading to vastly reduced image read times. Creates a generator to retrieve filenames
    with their hashes as they are calculated.

    Notes
    -----
    The order of returned values is non-deterministic so will most likely not be returned in the
    same order as the filenames

    Parameters
    ----------
    filenames: list
        A list of ``str`` full paths to the images to be loaded.
    show_progress: bool, optional
        Display a progress bar. Default: False

    Yields
    -------
    tuple: (`filename`, :func:`hashlib.hexdigest()` representation of the `sha1` hash of the image)
    Example
    -------
    >>> image_filenames = ["/path/to/image_1.png", "/path/to/image_2.png", "/path/to/image_3.png"]
    >>> for filename, hash in read_image_hash_batch(image_filenames):
    >>>         <do something>
    """
    logger.trace("Requested batch: '%s'", filenames)
    executor = futures.ThreadPoolExecutor()
    with executor:
        read_hashes = {executor.submit(read_image_hash, filename): filename
                       for filename in filenames}
        for future in futures.as_completed(read_hashes):
            retval = (read_hashes[future], future.result())
            logger.trace("Yielding: %s", retval)
            yield retval


def encode_image_with_hash(image, extension):
    """ Encode an image, and get the encoded image back with its `sha1` hash.

    Parameters
    ----------
    image: numpy.ndarray
        The image to be encoded in `BGR` channel order.
    extension: str
        A compatible `cv2` image file extension that the final image is to be saved to.

    Returns
    -------
    image_hash: str
        The :func:`hashlib.hexdigest()` representation of the `sha1` hash of the encoded image
    encoded_image: bytes
        The image encoded into the correct file format

    Example
    -------
    >>> image_file = "/path/to/image.png"
    >>> image = read_image(image_file)
    >>> image_hash, encoded_image = encode_image_with_hash(image, ".jpg")
    """
    encoded_image = cv2.imencode(extension, image)[1]
    image_hash = sha1(cv2.imdecode(encoded_image, cv2.IMREAD_UNCHANGED)).hexdigest()
    return image_hash, encoded_image


def batch_convert_color(batch, colorspace):
    """ Convert a batch of images from one color space to another.

    Converts a batch of images by reshaping the batch prior to conversion rather than iterating
    over the images. This leads to a significant speed up in the convert process.

    Parameters
    ----------
    batch: numpy.ndarray
        A batch of images.
    colorspace: str
        The OpenCV Color Conversion Code suffix. For example for BGR to LAB this would be
        ``'BGR2LAB'``.
        See https://docs.opencv.org/4.1.1/d8/d01/group__imgproc__color__conversions.html for a full
        list of color codes.

    Returns
    -------
    numpy.ndarray
        The batch converted to the requested color space.

    Example
    -------
    >>> images_bgr = numpy.array([image1, image2, image3])
    >>> images_lab = batch_convert_color(images_bgr, "BGR2LAB")

    Notes
    -----
    This function is only compatible for color space conversions that have the same image shape
    for source and destination color spaces.

    If you use :func:`batch_convert_color` with 8-bit images, the conversion will have some
    information lost. For many cases, this will not be noticeable but it is recommended
    to use 32-bit images in cases that need the full range of colors or that convert an image
    before an operation and then convert back.
    """
    logger.trace("Batch converting: (batch shape: %s, colorspace: %s)", batch.shape, colorspace)
    original_shape = batch.shape
    batch = batch.reshape((original_shape[0] * original_shape[1], *original_shape[2:]))
    batch = cv2.cvtColor(batch, getattr(cv2, "COLOR_{}".format(colorspace)))
    return batch.reshape(original_shape)


# ################### #
# <<< VIDEO UTILS >>> #
# ################### #

def count_frames(filename, fast=False):
    """ Count the number of frames in a video file

    There is no guaranteed accurate way to get a count of video frames without iterating through
    a video and decoding every frame.

    :func:`count_frames` can return an accurate count (albeit fairly slowly) or a possibly less
    accurate count, depending on the :attr:`fast` parameter. A progress bar is displayed.

    Parameters
    ----------
    filename: str
        Full path to the video to return the frame count from.
    fast: bool, optional
        Whether to count the frames without decoding them. This is significantly faster but
        accuracy is not guaranteed. Default: ``False``.

    Returns
    -------
    int:
        The number of frames in the given video file.

    Example
    -------
    >>> filename = "/path/to/video.mp4"
    >>> frame_count = count_frames(filename)
    """
    logger.debug("filename: %s, fast: %s", filename, fast)
    assert isinstance(filename, str), "Video path must be a string"

    cmd = [im_ffm.get_ffmpeg_exe(), "-i", filename, "-map", "0:v:0"]
    if fast:
        cmd.extend(["-c", "copy"])
    cmd.extend(["-f", "null", "-"])

    logger.debug("FFMPEG Command: '%s'", " ".join(cmd))
    process = subprocess.Popen(cmd,
                               stderr=subprocess.STDOUT,
                               stdout=subprocess.PIPE,
                               universal_newlines=True)
    pbar = None
    duration = None
    init_tqdm = False
    update = 0
    frames = 0
    while True:
        output = process.stdout.readline().strip()
        if output == "" and process.poll() is not None:
            break

        if output.startswith("Duration:"):
            logger.debug("Duration line: %s", output)
            idx = output.find("Duration:") + len("Duration:")
            duration = int(convert_to_secs(*output[idx:].split(",", 1)[0].strip().split(":")))
            logger.debug("duration: %s", duration)
        if output.startswith("frame="):
            logger.debug("frame line: %s", output)
            if not init_tqdm:
                logger.debug("Initializing tqdm")
                pbar = tqdm(desc="Counting Video Frames", total=duration, unit="secs")
                init_tqdm = True
            time_idx = output.find("time=") + len("time=")
            frame_idx = output.find("frame=") + len("frame=")
            frames = int(output[frame_idx:].strip().split(" ")[0].strip())
            vid_time = int(convert_to_secs(*output[time_idx:].split(" ")[0].strip().split(":")))
            logger.debug("frames: %s, vid_time: %s", frames, vid_time)
            prev_update = update
            update = vid_time
            pbar.update(update - prev_update)
    if pbar is not None:
        pbar.close()
    return_code = process.poll()
    logger.debug("Return code: %s, frames: %s", return_code, frames)
    return frames


class BackgroundIO():
    """ Perform disk IO for images or videos in a background thread.

    Loads images or videos from a given location in a background thread.
    Saves images to the given location in a background thread.
    Images/Videos will be loaded or saved in deterministic order.

    Parameters
    ----------
    path: str
        The path to load or save images to/from. For loading this can be a folder which contains
        images or a video file. For saving this must be an existing folder.
    task: {'load', 'save'}
        The task to be performed. ``'load'`` to load images/video frames, ``'save'`` to save
        images.
    load_with_hash: bool, optional
        When loading images, set to ``True`` to return the sha1 hash of the image along with the
        image. Default: ``False``.
    queue_size: int, optional
        The amount of images to hold in the internal buffer. Default: 16.

    Examples
    --------
    Loading from a video file:

    >>> loader = BackgroundIO('/path/to/video.mp4', 'load')
    >>> for filename, image in loader.load():
    >>>     <do processing>

    Loading faces with their sha1 hash:

    >>> loader = BackgroundIO('/path/to/faces/folder', 'load', load_with_hash=True)
    >>> for filename, image, sha1_hash in loader.load():
    >>>     <do processing>

    Saving out images:

    >>> saver = BackgroundIO('/path/to/save/folder', 'save')
    >>> for filename, image in <image_iterator>:
    >>>     saver.save(filename, image)
    >>> saver.close()
    """

    def __init__(self, path, task, load_with_hash=False, queue_size=16):
        logger.debug("Initializing %s: (path: %s, task: %s, load_with_hash: %s, queue_size: %s)",
                     self.__class__.__name__, path, task, load_with_hash, queue_size)
        self._location = path

        self._task = task.lower()
        self._is_video = self._check_input()
        self._input = self.location if self._is_video else get_image_paths(self.location)
        self._count = count_frames(self._input) if self._is_video else len(self._input)
        self._queue = queue_manager.get_queue(name="{}_{}".format(self.__class__.__name__,
                                                                  self._task),
                                              maxsize=queue_size)
        self._thread = self._set_thread(io_args=(load_with_hash, ))
        self._thread.start()

    @property
    def count(self):
        """ int: The number of images or video frames to be processed """
        return self._count

    @property
    def location(self):
        """ str: The folder or video that was passed in as the :attr:`path` parameter. """
        return self._location

    def _check_input(self):
        """ Check whether the input path is valid and return if it is a video.

        Returns
        -------
        bool: 'True' if input is a video 'False' if it is a folder.
        """
        if not os.path.exists(self.location):
            raise FaceswapError("The location '{}' does not exist".format(self.location))

        if self._task == "save" and not os.path.isdir(self.location):
            raise FaceswapError("The output location '{}' is not a folder".format(self.location))

        is_video = (self._task == "load" and
                    os.path.isfile(self.location) and
                    os.path.splitext(self.location)[1].lower() in _video_extensions)
        if is_video:
            logger.debug("Input is video")
        else:
            logger.debug("Input is folder")
        return is_video

    def _set_thread(self, io_args=None):
        """ Set the load/save thread

        Parameters
        ----------
        io_args: tuple, optional
            The arguments to be passed to the load or save thread. Default: `None`.

        Returns
        -------
        :class:`lib.multithreading.MultiThread`: Thread containing the load/save function.
        """
        io_args = (self._queue) if io_args is None else (self._queue, *io_args)
        retval = MultiThread(getattr(self, "_{}".format(self._task)), *io_args, thread_count=1)
        logger.trace(retval)
        return retval

    # LOADING #
    def _load(self, *args):
        """ The load thread.

        Loads from a folder of images or from a video and puts to a queue

        Parameters
        ----------
        args: tuple
            The arguments to be passed to the load iterator
        """
        queue = args[0]
        io_args = args[1:]
        iterator = self._load_video if self._is_video else self._load_images
        logger.debug("Load iterator: %s", iterator)
        for retval in iterator(*io_args):
            logger.trace("Putting to queue: %s", [v.shape if isinstance(v, np.ndarray) else v
                                                  for v in retval])
            queue.put(retval)
        logger.trace("Putting EOF")
        queue.put("EOF")

    def _load_video(self, *args):  # pylint:disable=unused-argument
        """ Generator for loading frames from a video

        Parameters
        ----------
        args: tuple
            Unused

        Yields
        ------
        filename: str
            The dummy filename of the loaded video frame.
        image: numpy.ndarray
            The loaded video frame.
        """
        logger.debug("Loading frames from video: '%s'", self._input)
        vidname = os.path.splitext(os.path.basename(self._input))[0]
        reader = imageio.get_reader(self._input, "ffmpeg")
        for i, frame in enumerate(reader):
            # Convert to BGR for cv2 compatibility
            frame = frame[:, :, ::-1]
            filename = "{}_{:06d}.png".format(vidname, i + 1)
            logger.trace("Loading video frame: '%s'", filename)
            yield filename, frame
        reader.close()

    def _load_images(self, with_hash):
        """ Generator for loading images from a folder

        Parameters
        ----------
        with_hash: bool
            If ``True`` adds the sha1 hash to the output tuple as the final item.

        Yields
        ------
        filename: str
            The filename of the loaded image.
        image: numpy.ndarray
            The loaded image.
        sha1_hash: str, optional
            The sha1 hash of the loaded image. Only yielded if :class:`BackgroundIO` was
            initialized with :attr:`load_with_hash` set to ``True`` and the :attr:`location`
            is a folder of images.
        """
        logger.debug("Loading images from folder: '%s'", self._input)
        for filename in self._input:
            image_read = read_image(filename, raise_error=False, with_hash=with_hash)
            if with_hash:
                retval = filename, *image_read
            else:
                retval = filename, image_read
            if retval[1] is None:
                logger.debug("Image not loaded: '%s'", filename)
                continue
            yield retval

    def load(self):
        """ Generator for loading images from the given :attr:`location`

        If :class:`BackgroundIO` was initialized with :attr:`load_with_hash` set to ``True`` then
        the sha1 hash of the image is added as the final item in the output `tuple`.

        Yields
        ------
        filename: str
            The filename of the loaded image.
        image: numpy.ndarray
            The loaded image.
        sha1_hash: str, optional
            The sha1 hash of the loaded image. Only yielded if :class:`BackgroundIO` was
            initialized with :attr:`load_with_hash` set to ``True`` and the :attr:`location`
            is a folder of images.
        """
        while True:
            self._thread.check_and_raise_error()
            try:
                retval = self._queue.get(True, 1)
            except QueueEmpty:
                continue
            if retval == "EOF":
                logger.trace("Got EOF")
                break
            logger.trace("Yielding: %s", [v.shape if isinstance(v, np.ndarray) else v
                                          for v in retval])
            yield retval
        self._thread.join()

    # SAVING #
    @staticmethod
    def _save(*args):
        """ Saves images from the save queue to the given :attr:`location` inside a thread.

        Parameters
        ----------
        args: tuple
            The save arguments
        """
        queue = args[0]
        while True:
            item = queue.get()
            if item == "EOF":
                logger.debug("EOF received")
                break
            filename, image = item
            logger.trace("Saving image: '%s'", filename)
            cv2.imwrite(filename, image)

    def save(self, filename, image):
        """ Save the given image in the background thread

        Ensure that :func:`close` is called once all save operations are complete.

        Parameters
        ----------
        filename: str
            The filename of the image to be saved
        image: numpy.ndarray
            The image to be saved
        """
        logger.trace("Putting to save queue: '%s'", filename)
        self._queue.put((filename, image))

    def close(self):
        """ Closes down and joins the internal threads

        Must be called after a :func:`save` operation to ensure all items are saved before the
        parent process exits.
        """
        logger.debug("Received Close")
        if self._task == "save":
            logger.debug("Putting EOF to save queue")
            self._queue.put("EOF")
        self._thread.join()
        logger.debug("Closed")
