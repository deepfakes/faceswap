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


class ImageIO():
    """ Perform disk IO for images or videos in a background thread.

    This is the parent thread for :class:`ImagesLoader` and :class:`ImagesSaver` and should not
    be called directly.

    Parameters
    ----------
    path: str or list
        The path to load or save images to/from. For loading this can be a folder which contains
        images, video file or a list of image files. For saving this must be an existing folder.
    queue_size: int
        The amount of images to hold in the internal buffer.
    args: tuple, optional
        The arguments to be passed to the loader or saver thread. Default: ``None``

    See Also
    --------
    lib.image.ImagesLoader : Background Image Loader inheriting from this class.
    lib.image.ImagesSaver : Background Image Saver inheriting from this class.
    """

    def __init__(self, path, queue_size, args=None):
        logger.debug("Initializing %s: (path: %s, queue_size: %s, args: %s)",
                     self.__class__.__name__, path, queue_size, args)

        self._args = tuple() if args is None else args

        self._location = path
        self._check_location_exists()

        self._queue = queue_manager.get_queue(name=self.__class__.__name__, maxsize=queue_size)
        self._thread = None

    @property
    def location(self):
        """ str: The folder or video that was passed in as the :attr:`path` parameter. """
        return self._location

    def _check_location_exists(self):
        """ Check whether the input location exists.

        Raises
        ------
        FaceswapError
            If the given location does not exist
        """
        if isinstance(self.location, str) and not os.path.exists(self.location):
            raise FaceswapError("The location '{}' does not exist".format(self.location))
        if isinstance(self.location, (list, tuple)) and not all(os.path.exists(location)
                                                                for location in self.location):
            raise FaceswapError("Not all locations in the input list exist")

    def _set_thread(self):
        """ Set the load/save thread """
        logger.debug("Setting thread")
        if self._thread is not None and self._thread.is_alive():
            logger.debug("Thread pre-exists and is alive: %s", self._thread)
            return
        self._thread = MultiThread(self._process,
                                   self._queue,
                                   name=self.__class__.__name__,
                                   thread_count=1)
        logger.debug("Set thread: %s", self._thread)
        self._thread.start()

    def _process(self, queue):
        """ Image IO process to be run in a thread. Override for loader/saver process.

        Parameters
        ----------
        queue: queue.Queue()
            The ImageIO Queue
        """
        raise NotImplementedError

    def close(self):
        """ Closes down and joins the internal threads """
        logger.debug("Received Close")
        if self._thread is not None:
            self._thread.join()
        logger.debug("Closed")


class ImagesLoader(ImageIO):
    """ Perform image loading from a folder of images or a video.

    Images will be loaded and returned in the order that they appear in the folder, or in the video
    to ensure deterministic ordering. Loading occurs in a background thread, caching 8 images at a
    time so that other processes do not need to wait on disk reads.

    See also :class:`ImageIO` for additional attributes.

    Parameters
    ----------
    path: str or list
        The path to load images from. This can be a folder which contains images a video file or a
        list of image files.
    queue_size: int, optional
        The amount of images to hold in the internal buffer. Default: 8.
    load_with_hash: bool, optional
        Set to ``True`` to return the sha1 hash of the image along with the image.
        Default: ``False``.
    fast_count: bool, optional
        When loading from video, the video needs to be parsed frame by frame to get an accurate
        count. This can be done quite quickly without guaranteed accuracy, or slower with
        guaranteed accuracy. Set to ``True`` to count quickly, or ``False`` to count slower
        but accurately. Default: ``True``.
    skip_list: list, optional
        Optional list of frame/image indices to not load. Any indices provided here will be skipped
        when reading images from the given location. Default: ``None``

    Examples
    --------
    Loading from a video file:

    >>> loader = ImagesLoader('/path/to/video.mp4')
    >>> for filename, image in loader.load():
    >>>     <do processing>

    Loading faces with their sha1 hash:

    >>> loader = ImagesLoader('/path/to/faces/folder', load_with_hash=True)
    >>> for filename, image, sha1_hash in loader.load():
    >>>     <do processing>
    """

    def __init__(self, path, queue_size=8, load_with_hash=False, fast_count=True, skip_list=None):
        logger.debug("Initializing %s: (path: %s, queue_size: %s, load_with_hash: %s, "
                     "fast_count: %s)", self.__class__.__name__, path, queue_size,
                     load_with_hash, fast_count)

        args = (load_with_hash, )
        super().__init__(path, queue_size=queue_size, args=args)
        self._skip_list = set() if skip_list is None else set(skip_list)

        self._is_video = self._check_for_video()

        self._count = None
        self._file_list = None
        self._get_count_and_filelist(fast_count)

    @property
    def count(self):
        """ int: The number of images or video frames in the source location. This count includes
        any files that will ultimately be skipped if a :attr:`skip_list` has been provided. See
        also: :attr:`process_count`"""
        return self._count

    @property
    def process_count(self):
        """ int: The number of images or video frames to be processed (IE the total count less
        items that are to be skipped from the :attr:`skip_list`)"""
        return self._count - len(self._skip_list)

    @property
    def is_video(self):
        """ bool: ``True`` if the input is a video, ``False`` if it is not """
        return self._is_video

    @property
    def file_list(self):
        """ list: A full list of files in the source location. This includes any files that will
        ultimately be skipped if a :attr:`skip_list` has been provided. If the input is a video
        then this is a list of dummy filenames as corresponding to an alignments file """
        return self._file_list

    def add_skip_list(self, skip_list):
        """ Add a skip list to this :class:`ImagesLoader`

        Parameters
        ----------
        skip_list: list
            A list of indices corresponding to the frame indices that should be skipped
        """
        logger.debug(skip_list)
        self._skip_list = set(skip_list)

    def _check_for_video(self):
        """ Check whether the input is a video

        Returns
        -------
        bool: 'True' if input is a video 'False' if it is a folder.

        Raises
        ------
        FaceswapError
            If the given location is a file and does not have a valid video extension.

        """
        if os.path.isdir(self.location):
            retval = False
        elif os.path.splitext(self.location)[1].lower() in _video_extensions:
            retval = True
        else:
            raise FaceswapError("The input file '{}' is not a valid video".format(self.location))
        logger.debug("Input '%s' is_video: %s", self.location, retval)
        return retval

    def _get_count_and_filelist(self, fast_count):
        """ Set the count of images to be processed and set the file list

            If the input is a video, a dummy file list is created for checking against an
            alignments file, otherwise it will be a list of full filenames.

        Parameters
        ----------
        fast_count: bool
            When loading from video, the video needs to be parsed frame by frame to get an accurate
            count. This can be done quite quickly without guaranteed accuracy, or slower with
            guaranteed accuracy. Set to ``True`` to count quickly, or ``False`` to count slower
            but accurately.
        """
        if self._is_video:
            self._count = int(count_frames(self.location, fast=fast_count))
            self._file_list = [self._dummy_video_framename(i + 1) for i in range(self.count)]
        else:
            if isinstance(self.location, (list, tuple)):
                self._file_list = self.location
            else:
                self._file_list = get_image_paths(self.location)
            self._count = len(self.file_list)

        logger.debug("count: %s", self.count)
        logger.trace("filelist: %s", self.file_list)

    def _process(self, queue):
        """ The load thread.

        Loads from a folder of images or from a video and puts to a queue

        Parameters
        ----------
        queue: queue.Queue()
            The ImageIO Queue
        """
        iterator = self._from_video if self._is_video else self._from_folder
        logger.debug("Load iterator: %s", iterator)
        for retval in iterator():
            filename, image = retval[:2]
            if image is None or (not image.any() and image.ndim not in (2, 3)):
                # All black frames will return not numpy.any() so check dims too
                logger.warning("Unable to open image. Skipping: '%s'", filename)
                continue
            logger.trace("Putting to queue: %s", [v.shape if isinstance(v, np.ndarray) else v
                                                  for v in retval])
            queue.put(retval)
        logger.trace("Putting EOF")
        queue.put("EOF")

    def _from_video(self):
        """ Generator for loading frames from a video

        Yields
        ------
        filename: str
            The dummy filename of the loaded video frame.
        image: numpy.ndarray
            The loaded video frame.
        """
        logger.debug("Loading frames from video: '%s'", self.location)
        reader = imageio.get_reader(self.location, "ffmpeg")
        for idx, frame in enumerate(reader):
            if idx in self._skip_list:
                logger.trace("Skipping frame %s due to skip list")
                continue
            # Convert to BGR for cv2 compatibility
            frame = frame[:, :, ::-1]
            filename = self._dummy_video_framename(idx + 1)
            logger.trace("Loading video frame: '%s'", filename)
            yield filename, frame
        reader.close()

    def _dummy_video_framename(self, frame_no):
        """ Return a dummy filename for video files

        Parameters
        ----------
        frame_no: int
            The frame number for the video frame

        Returns
        -------
        str: A dummied filename for a video frame """
        vidname = os.path.splitext(os.path.basename(self.location))[0]
        return "{}_{:06d}.png".format(vidname, frame_no + 1)

    def _from_folder(self):
        """ Generator for loading images from a folder

        Yields
        ------
        filename: str
            The filename of the loaded image.
        image: numpy.ndarray
            The loaded image.
        sha1_hash: str, optional
            The sha1 hash of the loaded image. Only yielded if :class:`ImageIO` was
            initialized with :attr:`load_with_hash` set to ``True`` and the :attr:`location`
            is a folder of images.
        """
        with_hash = self._args[0]
        logger.debug("Loading images from folder: '%s'. with_hash: %s", self.location, with_hash)
        for idx, filename in enumerate(self.file_list):
            if idx in self._skip_list:
                logger.trace("Skipping frame %s due to skip list")
                continue
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

        If :class:`ImageIO` was initialized with :attr:`load_with_hash` set to ``True`` then
        the sha1 hash of the image is added as the final item in the output `tuple`.

        Yields
        ------
        filename: str
            The filename of the loaded image.
        image: numpy.ndarray
            The loaded image.
        sha1_hash: str, optional
            The sha1 hash of the loaded image. Only yielded if :class:`ImageIO` was
            initialized with :attr:`load_with_hash` set to ``True`` and the :attr:`location`
            is a folder of images.
        """
        logger.debug("Initializing Load Generator")
        self._set_thread()
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
        logger.debug("Closing Load Generator")
        self._thread.join()


class ImagesSaver(ImageIO):
    """ Perform image saving to a destination folder.

    Images are saved in a background ThreadPoolExecutor to allow for concurrent saving.
    See also :class:`ImageIO` for additional attributes.

    Parameters
    ----------
    path: str
        The folder to save images to. This must be an existing folder.
    queue_size: int, optional
        The amount of images to hold in the internal buffer. Default: 8.
    as_bytes: bool, optional
        ``True`` if the image is already encoded to bytes, ``False`` if the image is a
        :class:`numpy.ndarray`. Default: ``False``.

    Examples
    --------

    >>> saver = ImagesSaver('/path/to/save/folder')
    >>> for filename, image in <image_iterator>:
    >>>     saver.save(filename, image)
    >>> saver.close()
    """

    def __init__(self, path, queue_size=8, as_bytes=False):
        logger.debug("Initializing %s: (path: %s, load_with_hash: %s, as_bytes: %s)",
                     self.__class__.__name__, path, queue_size, as_bytes)

        super().__init__(path, queue_size=queue_size)
        self._as_bytes = as_bytes

    def _check_location_exists(self):
        """ Check whether the output location exists and is a folder

        Raises
        ------
        FaceswapError
            If the given location does not exist or the location is not a folder
        """
        if not isinstance(self.location, str):
            raise FaceswapError("The output location must be a string not a "
                                "{}".format(type(self.location)))
        super()._check_location_exists()
        if not os.path.isdir(self.location):
            raise FaceswapError("The output location '{}' is not a folder".format(self.location))

    def _process(self, queue):
        """ Saves images from the save queue to the given :attr:`location` inside a thread.

        Parameters
        ----------
        queue: queue.Queue()
            The ImageIO Queue
        """
        executor = futures.ThreadPoolExecutor(thread_name_prefix=self.__class__.__name__)
        while True:
            item = queue.get()
            if item == "EOF":
                logger.debug("EOF received")
                break
            logger.trace("Submitting: '%s'", item[0])
            executor.submit(self._save, *item)
        executor.shutdown()

    def _save(self, filename, image):
        """ Save a single image inside a ThreadPoolExecutor

        Parameters
        ----------
        filename: str
            The filename of the image to be saved. Can include or exclude the folder location.
        image: numpy.ndarray
            The image to be saved
        """
        if not os.path.commonprefix([self.location, filename]):
            filename = os.path.join(self.location, filename)
        try:
            if self._as_bytes:
                with open(filename, "wb") as out_file:
                    out_file.write(image)
            else:
                cv2.imwrite(filename, image)
            logger.trace("Saved image: '%s'", filename)
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Failed to save image '%s'. Original Error: %s", filename, err)

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
        self._set_thread()
        logger.trace("Putting to save queue: '%s'", filename)
        self._queue.put((filename, image))

    def close(self):
        """ Signal to the Save Threads that they should be closed and cleanly shutdown
        the saver """
        logger.debug("Putting EOF to save queue")
        self._queue.put("EOF")
        super().close()
