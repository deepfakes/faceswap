#!/usr/bin python3
""" Utilities for working with images and videos """

import logging
import re
import subprocess
import os
import sys

from bisect import bisect
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

class FfmpegReader(imageio.plugins.ffmpeg.FfmpegFormat.Reader):
    """ Monkey patch imageio ffmpeg to use keyframes whilst seeking """
    def __init__(self, format, request):
        super().__init__(format, request)
        self._frame_pts = None
        self._keyframes = None
        self.use_patch = False

    def get_frame_info(self, frame_pts=None, keyframes=None):
        """ Store the source video's keyframes in :attr:`_frame_info" for the current video for use
        in :func:`initialize`.

        Parameters
        ----------
        frame_pts: list, optional
            A list corresponding to the video frame count of the pts_time per frame. If this and
            `keyframes` are provided, then analyzing the video is skipped and the values from the
            given lists are used. Default: ``None``
        keyframes: list, optional
            A list containing the frame numbers of each key frame. if this and `frame_pts` are
            provided, then analyzing the video is skipped and the values from the given lists are
            used. Default: ``None``
        """
        if frame_pts is not None and keyframes is not None:
            logger.debug("Video meta information provided. Not analyzing video")
            self._frame_pts = frame_pts
            self._keyframes = keyframes
            return len(frame_pts), dict(pts_time=self._frame_pts, keyframes=self._keyframes)

        assert isinstance(self._filename, str), "Video path must be a string"
        cmd = [im_ffm.get_ffmpeg_exe(),
               "-hide_banner",
               "-copyts",
               "-i", self._filename,
               "-vf", "showinfo",
               "-start_number", "0",
               "-an",
               "-f", "null",
               "-"]
        logger.debug("FFMPEG Command: '%s'", " ".join(cmd))
        process = subprocess.Popen(cmd,
                                   stderr=subprocess.STDOUT,
                                   stdout=subprocess.PIPE,
                                   universal_newlines=True)
        frame_pts = []
        key_frames = []
        last_update = 0
        pbar = tqdm(desc="Analyzing Video",
                    leave=False,
                    total=int(self._meta["duration"]),
                    unit="secs")
        while True:
            output = process.stdout.readline().strip()
            if output == "" and process.poll() is not None:
                break
            if "iskey" not in output:
                continue
            logger.trace("Keyframe line: %s", output)
            line = re.split(r"\s+|:\s*", output)
            pts_time = float(line[line.index("pts_time") + 1])
            frame_no = int(line[line.index("n") + 1])
            frame_pts.append(pts_time)
            if "iskey:1" in output:
                key_frames.append(frame_no)

            logger.trace("pts_time: %s, frame_no: %s", pts_time, frame_no)
            if int(pts_time) == last_update:
                # Floating points make TQDM display poorly, so only update on full
                # second increments
                continue
            pbar.update(int(pts_time) - last_update)
            last_update = int(pts_time)
        pbar.close()
        return_code = process.poll()
        frame_count = len(frame_pts)
        logger.debug("Return code: %s, frame_pts: %s, keyframes: %s, frame_count: %s",
                     return_code, frame_pts, key_frames, frame_count)

        self._frame_pts = frame_pts
        self._keyframes = key_frames
        return frame_count, dict(pts_time=self._frame_pts, keyframes=self._keyframes)

    def _previous_keyframe_info(self, index=0):
        """ Return the previous keyframe's pts_time and frame number """
        prev_keyframe_idx = bisect(self._keyframes, index) - 1
        prev_keyframe = self._keyframes[prev_keyframe_idx]
        prev_pts_time = self._frame_pts[prev_keyframe]
        logger.trace("keyframe pts_time: %s, keyframe: %s", prev_pts_time, prev_keyframe)
        return prev_pts_time, prev_keyframe

    def _initialize(self, index=0):
        """ Replace ImageIO _initialize with a version that explictly uses keyframes.

        Notes
        -----
        This introduces a minor change by seeking fast to the previous keyframe and then discarding
        subsequent frames until the desired frame is reached. In testing, setting -ss flag either
        prior to input, or both prior (fast) and after (slow) would not always bring back the
        correct frame for all videos. Navigating to the previous keyframe then discarding frames
        until the correct frame is reached appears to work well.
        """
        # pylint: disable-all
        if self._read_gen is not None:
            self._read_gen.close()

        iargs = []
        oargs = []
        skip_frames = 0

        # Create input args
        iargs += self._arg_input_params
        if self.request._video:
            iargs += ["-f", CAM_FORMAT]  # noqa
            if self._arg_pixelformat:
                iargs += ["-pix_fmt", self._arg_pixelformat]
            if self._arg_size:
                iargs += ["-s", self._arg_size]
        elif index > 0:  # re-initialize  / seek
            # Note: only works if we initialized earlier, and now have meta. Some info here:
            # https://trac.ffmpeg.org/wiki/Seeking
            # There are two ways to seek, one before -i (input_params) and after (output_params).
            # The former is fast, because it uses keyframes, the latter is slow but accurate.
            # According to the article above, the fast method should also be accurate from ffmpeg
            # version 2.1, however in version 4.1 our tests start failing again. Not sure why, but
            # we can solve this by combining slow and fast.
            # Further note: The old method would go back 10 seconds and then seek slow. This was
            # still somewhat unresponsive and did not always land on the correct frame. This monkey
            # patched version goes to the previous keyframe then discards frames until the correct
            # frame is landed on.
            if self.use_patch and self._frame_pts is None:
                self.get_frame_info()

            if self.use_patch:
                keyframe_pts, keyframe = self._previous_keyframe_info(index)
                seek_fast = keyframe_pts
                skip_frames = index - keyframe
            else:
                starttime = index / self._meta["fps"]
                seek_slow = min(10, starttime)
                seek_fast = starttime - seek_slow

            # We used to have this epsilon earlier, when we did not use
            # the slow seek. I don't think we need it anymore.
            # epsilon = -1 / self._meta["fps"] * 0.1
            iargs += ["-ss", "%.06f" % (seek_fast)]
            if not self.use_patch:
                oargs += ["-ss", "%.06f" % (seek_slow)]

        # Output args, for writing to pipe
        if self._arg_size:
            oargs += ["-s", self._arg_size]
        if self.request.kwargs.get("fps", None):
            fps = float(self.request.kwargs["fps"])
            oargs += ["-r", "%.02f" % fps]
        oargs += self._arg_output_params

        # Get pixelformat and bytes per pixel
        pix_fmt = self._pix_fmt
        bpp = self._depth * self._bytes_per_channel

        # Create generator
        rf = self._ffmpeg_api.read_frames
        self._read_gen = rf(
            self._filename, pix_fmt, bpp, input_params=iargs, output_params=oargs
        )

        # Read meta data. This start the generator (and ffmpeg subprocess)
        if self.request._video:
            # With cameras, catch error and turn into IndexError
            try:
                meta = self._read_gen.__next__()
            except IOError as err:
                err_text = str(err)
                if "darwin" in sys.platform:
                    if "Unknown input format: 'avfoundation'" in err_text:
                        err_text += (
                            "Try installing FFMPEG using "
                            "home brew to get a version with "
                            "support for cameras."
                        )
                raise IndexError(
                    "No camera at {}.\n\n{}".format(self.request._video, err_text)
                )
            else:
                self._meta.update(meta)
        elif index == 0:
            self._meta.update(self._read_gen.__next__())
        else:
            if self.use_patch:
                frames_skipped = 0
                while skip_frames != frames_skipped:
                    # Skip frames that are not the desired frame
                    _ = self._read_gen.__next__()
                    frames_skipped += 1
            self._read_gen.__next__()  # we already have meta data


imageio.plugins.ffmpeg.FfmpegFormat.Reader = FfmpegReader


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

    Leverages multi-threading to load multiple images from disk at the same time leading to vastly
    reduced image read times.

    Parameters
    ----------
    filenames: list
        A list of ``str`` full paths to the images to be loaded.

    Returns
    -------
    numpy.ndarray
        The batch of images in `BGR` channel order returned in the order of :attr:`filenames`

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
        images = {executor.submit(read_image, filename, raise_error=True): filename
                  for filename in filenames}
        batch = [None for _ in range(len(filenames))]
        # There is no guarantee that the same filename will not be passed through multiple times
        # (and when shuffle is true this can definitely happen), so we can't just call
        # filenames.index().
        return_indices = {filename: [idx for idx, fname in enumerate(filenames)
                                     if fname == filename]
                          for filename in set(filenames)}
        for future in futures.as_completed(images):
            batch[return_indices[images[future]].pop()] = future.result()
    batch = np.array(batch)
    logger.trace("Returning images: (filenames: %s, batch shape: %s)", filenames, batch.shape)
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
        logger.debug("Submitting %s items to executor", len(filenames))
        read_hashes = {executor.submit(read_image_hash, filename): filename
                       for filename in filenames}
        logger.debug("Succesfully submitted %s items to executor", len(filenames))
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


def hex_to_rgb(hexcode):
    """ Convert a hex number to it's RGB counterpart.

    Parameters
    ----------
    hexcode: str
        The hex code to convert (e.g. `"#0d25ac"`)

    Returns
    -------
    tuple
        The hex code as a 3 integer (`R`, `G`, `B`) tuple
    """
    value = hexcode.lstrip("#")
    chars = len(value)
    return tuple(int(value[i:i + chars // 3], 16) for i in range(0, chars, chars // 3))


def rgb_to_hex(rgb):
    """ Convert an RGB tuple to it's hex counterpart.

    Parameters
    ----------
    rgb: tuple
        The (`R`, `G`, `B`) integer values to convert (e.g. `(0, 255, 255)`)

    Returns
    -------
    str:
        The 6 digit hex code with leading `#` applied
    """
    return "#{:02x}{:02x}{:02x}".format(*rgb)


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
                pbar = tqdm(desc="Analyzing Video", leave=False, total=duration, unit="secs")
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
        """ Set the background thread for the load and save iterators and launch it. """
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
        self._thread = None
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
    fast_count: bool, optional
        When loading from video, the video needs to be parsed frame by frame to get an accurate
        count. This can be done quite quickly without guaranteed accuracy, or slower with
        guaranteed accuracy. Set to ``True`` to count quickly, or ``False`` to count slower
        but accurately. Default: ``True``.
    skip_list: list, optional
        Optional list of frame/image indices to not load. Any indices provided here will be skipped
        when executing the :func:`load` function from the given location. Default: ``None``
    count: int, optional
        If the number of images that the loader will encounter is already known, it can be passed
        in here to skip the image counting step, which can save time at launch. Set to ``None`` if
        the count is not already known. Default: ``None``

    Examples
    --------
    Loading from a video file:

    >>> loader = ImagesLoader('/path/to/video.mp4')
    >>> for filename, image in loader.load():
    >>>     <do processing>
    """

    def __init__(self,
                 path,
                 queue_size=8,
                 fast_count=True,
                 skip_list=None,
                 count=None):
        logger.debug("Initializing %s: (path: %s, queue_size: %s, fast_count: %s, skip_list: %s, "
                     "count: %s)", self.__class__.__name__, path, queue_size, fast_count,
                     skip_list, count)

        super().__init__(path, queue_size=queue_size)
        self._skip_list = set() if skip_list is None else set(skip_list)
        self._is_video = self._check_for_video()
        self._fps = self._get_fps()

        self._count = None
        self._file_list = None
        self._get_count_and_filelist(fast_count, count)

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
    def fps(self):
        """ float: For an input folder of images, this will always return 25fps. If the input is a
        video, then the fps of the video will be returned. """
        return self._fps

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
            A list of indices corresponding to the frame indices that should be skipped by the
            :func:`load` function.
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

    def _get_fps(self):
        """ Get the Frames per Second.

        If the input is a folder of images than 25.0 will be returned, as it is not possible to
        calculate the fps just from frames alone. For video files the correct FPS will be returned.

        Returns
        -------
        float: The Frames per Second of the input sources
        """
        if self._is_video:
            reader = imageio.get_reader(self.location, "ffmpeg")
            retval = reader.get_meta_data()["fps"]
            reader.close()
        else:
            retval = 25.0
        logger.debug(retval)
        return retval

    def _get_count_and_filelist(self, fast_count, count):
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
        count: int
            The number of images that the loader will encounter if already known, otherwise
            ``None``
        """
        if self._is_video:
            self._count = int(count_frames(self.location,
                                           fast=fast_count)) if count is None else count
            self._file_list = [self._dummy_video_framename(i) for i in range(self.count)]
        else:
            if isinstance(self.location, (list, tuple)):
                self._file_list = self.location
            else:
                self._file_list = get_image_paths(self.location)
            self._count = len(self.file_list) if count is None else count

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
                logger.trace("Skipping frame %s due to skip list", idx)
                continue
            # Convert to BGR for cv2 compatibility
            frame = frame[:, :, ::-1]
            filename = self._dummy_video_framename(idx)
            logger.trace("Loading video frame: '%s'", filename)
            yield filename, frame
        reader.close()

    def _dummy_video_framename(self, index):
        """ Return a dummy filename for video files

        Parameters
        ----------
        index: int
            The index number for the frame in the video file

        Notes
        -----
        Indexes start at 0, frame numbers start at 1, so index is incremented by 1
        when creating the filename

        Returns
        -------
        str: A dummied filename for a video frame """
        vidname = os.path.splitext(os.path.basename(self.location))[0]
        return "{}_{:06d}.png".format(vidname, index + 1)

    def _from_folder(self):
        """ Generator for loading images from a folder

        Yields
        ------
        filename: str
            The filename of the loaded image.
        image: numpy.ndarray
            The loaded image.
        """
        logger.debug("Loading frames from folder: '%s'", self.location)
        for idx, filename in enumerate(self.file_list):
            if idx in self._skip_list:
                logger.trace("Skipping frame %s due to skip list")
                continue
            image_read = read_image(filename, raise_error=False, with_hash=False)
            retval = filename, image_read
            if retval[1] is None:
                logger.warning("Frame not loaded: '%s'", filename)
                continue
            yield retval

    def load(self):
        """ Generator for loading images from the given :attr:`location`

        If :class:`FacesLoader` is in use then the sha1 hash of the image is added as the final
        item in the output `tuple`.

        Yields
        ------
        filename: str
            The filename of the loaded image.
        image: numpy.ndarray
            The loaded image.
        sha1_hash: str, (:class:`FacesLoader` only)
            The sha1 hash of the loaded image. Only yielded if :class:`FacesLoader` is being
            executed.
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
        self.close()


class FacesLoader(ImagesLoader):
    """ Loads faces from a faces folder along with the face's hash.

    Examples
    --------
    Loading faces with their sha1 hash:

    >>> loader = FacesLoader('/path/to/faces/folder')
    >>> for filename, face, sha1_hash in loader.load():
    >>>     <do processing>
    """
    def __init__(self, path, skip_list=None, count=None):
        logger.debug("Initializing %s: (path: %s, count: %s)", self.__class__.__name__,
                     path, count)
        super().__init__(path, queue_size=8, skip_list=skip_list, count=count)

    def _from_folder(self):
        """ Generator for loading images from a folder
        Faces will only ever be loaded from a folder, so this is the only function requiring
        an override

        Yields
        ------
        filename: str
            The filename of the loaded image.
        image: numpy.ndarray
            The loaded image.
        sha1_hash: str
            The sha1 hash of the loaded image.
        """
        logger.debug("Loading images from folder: '%s'", self.location)
        for idx, filename in enumerate(self.file_list):
            if idx in self._skip_list:
                logger.trace("Skipping face %s due to skip list")
                continue
            image_read = read_image(filename, raise_error=False, with_hash=True)
            retval = filename, *image_read
            if retval[1] is None:
                logger.warning("Face not loaded: '%s'", filename)
                continue
            yield retval


class SingleFrameLoader(ImagesLoader):
    """ Allows direct access to a frame by filename or frame index.

    As we are interested in instant access to frames, there is no requirement to process in a
    background thread, as either way we need to wait for the frame to load.

    Parameters
    ----------
    video_meta_data: dict, optional
        Existing video meta information containing the pts_time and iskey flags for the given
        video. Used in conjunction with single_frame_reader for faster seeks. Providing this means
        that the video does not need to be scanned again. Set to ``None`` if the video is to be
        scanned. Default: ``None``
     """
    def __init__(self, path, video_meta_data=None):
        logger.debug("Initializing %s: (path: %s, video_meta_data: %s)",
                     self.__class__.__name__, path, video_meta_data)
        self._video_meta_data = dict() if video_meta_data is None else video_meta_data
        self._reader = None
        super().__init__(path, queue_size=1, fast_count=False)

    @property
    def video_meta_data(self):
        """ dict: For videos contains the keys `frame_pts` holding a list of time stamps for each
        frame and `keyframes` holding the frame index of each key frame.

        Notes
        -----
        Only populated if the input is a video and single frame reader is being used, otherwise
        returns ``None``.
        """
        return self._video_meta_data

    def _get_count_and_filelist(self, fast_count, count):
        if self._is_video:
            self._reader = imageio.get_reader(self.location, "ffmpeg")
            self._reader.use_patch = True
            count, video_meta_data = self._reader.get_frame_info(
                frame_pts=self._video_meta_data.get("pts_time", None),
                keyframes=self._video_meta_data.get("keyframes", None))
            self._video_meta_data = video_meta_data
        super()._get_count_and_filelist(fast_count, count)

    def image_from_index(self, index):
        """ Return a single image from :attr:`file_list` for the given index.

        Parameters
        ----------
        index: int
            The index number (frame number) of the frame to retrieve. NB: The first frame is
            index `0`

        Returns
        -------
        filename: str
            The filename of the returned image
        image: :class:`numpy.ndarray`
            The image for the given index

        Notes
        -----
        Retrieving frames from video files can be slow as the whole video file needs to be
        iterated to retrieve the requested frame. If a frame has already been retrieved, then
        retrieving frames of a higher index will be quicker than retrieving frames of a lower
        index, as iteration needs to start from the beginning again when navigating backwards.

        We do not use a background thread for this task, as it is assumed that requesting an image
        by index will be done when required.
        """
        if self.is_video:
            image = self._reader.get_data(index)[..., ::-1]
            filename = self._dummy_video_framename(index)
        else:
            filename = self.file_list[index]
            image = read_image(filename, raise_error=True)
            filename = os.path.basename(filename)
        logger.trace("index: %s, filename: %s image shape: %s", index, filename, image.shape)
        return filename, image


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
        logger.debug("Initializing %s: (path: %s, queue_size: %s, as_bytes: %s)",
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
            The filename of the image to be saved. NB: Any folders passed in with the filename
            will be stripped and replaced with :attr:`location`.
        image: numpy.ndarray
            The image to be saved
        """
        filename = os.path.join(self.location, os.path.basename(filename))
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
