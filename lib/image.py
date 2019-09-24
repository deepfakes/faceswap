#!/usr/bin python3
""" Utilities for working with images and videos """

import logging
import subprocess
import sys

from concurrent import futures
from hashlib import sha1

import cv2
import imageio_ffmpeg as im_ffm
import numpy as np

from lib.utils import convert_to_secs, FaceswapError

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name

# ################### #
# <<< IMAGE UTILS >>> #
# ################### #


# <<< IMAGE IO >>> #

def read_image(filename, raise_error=False):
    """ Read an image file from a file location.

    Extends the functionality of :func:`cv2.imread()` by ensuring that an image was actually
    loaded. Errors can be logged and ignored so that the process can continue on an image load
    failure.

    Parameters
    ----------
    filename: str
        Full path to the image to be loaded.
    raise_error: bool, optional
        If ``True``, then any failures (including the returned image being ``None``) will be
        raised. If ``False`` then an error message will be logged, but the error will not be
        raised. Default: ``False``

    Returns
    -------
    numpy.ndarray
        The image in `BGR` channel order.

    Example
    -------
    >>> image_file = "/path/to/image.png"
    >>> try:
    >>>    image = read_image(image_file, raise_error=True)
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
    return image


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

def count_frames_and_secs(filename, timeout=60):
    """ Count the number of frames and seconds in a video file.

    Adapted From :mod:`ffmpeg_imageio` to handle the issue of ffmpeg occasionally hanging
    inside a subprocess.

    If the operation times out then the process will try to read the data again, up to a total
    of 3 times. If the data still cannot be read then an exception will be raised.

    Note that this operation can be quite slow for large files.

    Parameters
    ----------
    filename: str
        Full path to the video to be analyzed.
    timeout: str, optional
        The amount of time in seconds to wait for the video data before aborting.
        Default: ``60``

    Returns
    -------
    nframes: int
        The number of frames in the given video file.
    nsecs: float
        The duration, in seconds, of the given video file.

    Example
    -------
    >>> video = "/path/to/video.mp4"
    >>> frames, secs = count_frames_and_secs(video)
    """
    # https://stackoverflow.com/questions/2017843/fetch-frame-count-with-ffmpeg

    assert isinstance(filename, str), "Video path must be a string"
    exe = im_ffm.get_ffmpeg_exe()
    iswin = sys.platform.startswith("win")
    logger.debug("iswin: '%s'", iswin)
    cmd = [exe, "-i", filename, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"]
    logger.debug("FFMPEG Command: '%s'", " ".join(cmd))
    attempts = 3
    for attempt in range(attempts):
        try:
            logger.debug("attempt: %s of %s", attempt + 1, attempts)
            out = subprocess.check_output(cmd,
                                          stderr=subprocess.STDOUT,
                                          shell=iswin,
                                          timeout=timeout)
            logger.debug("Succesfully communicated with FFMPEG")
            break
        except subprocess.CalledProcessError as err:
            out = err.output.decode(errors="ignore")
            raise RuntimeError("FFMEG call failed with {}:\n{}".format(err.returncode, out))
        except subprocess.TimeoutExpired as err:
            this_attempt = attempt + 1
            if this_attempt == attempts:
                msg = ("FFMPEG hung while attempting to obtain the frame count. "
                       "Sometimes this issue resolves itself, so you can try running again. "
                       "Otherwise use the Effmpeg Tool to extract the frames from your video into "
                       "a folder, and then run the requested Faceswap process on that folder.")
                raise FaceswapError(msg) from err
            logger.warning("FFMPEG hung while attempting to obtain the frame count. "
                           "Retrying %s of %s", this_attempt + 1, attempts)
            continue

    # Note that other than with the subprocess calls below, ffmpeg wont hang here.
    # Worst case Python will stop/crash and ffmpeg will continue running until done.

    nframes = nsecs = None
    for line in reversed(out.splitlines()):
        if not line.startswith(b"frame="):
            continue
        line = line.decode(errors="ignore")
        logger.debug("frame line: '%s'", line)
        idx = line.find("frame=")
        if idx >= 0:
            splitframes = line[idx:].split("=", 1)[-1].lstrip().split(" ", 1)[0].strip()
            nframes = int(splitframes)
        idx = line.find("time=")
        if idx >= 0:
            splittime = line[idx:].split("=", 1)[-1].lstrip().split(" ", 1)[0].strip()
            nsecs = convert_to_secs(*splittime.split(":"))
        logger.debug("nframes: %s, nsecs: %s", nframes, nsecs)
        return nframes, nsecs

    raise RuntimeError("Could not get number of frames")  # pragma: no cover
