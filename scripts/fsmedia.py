#!/usr/bin/env python3
""" Helper functions for :mod:`~scripts.extract` and :mod:`~scripts.convert`.

Holds the classes for the 2 main Faceswap 'media' objects: Images and Alignments.

Holds optional pre/post processing functions for convert and extract.
"""
from __future__ import annotations
import logging
import os
import sys
import typing as T

from collections.abc import Iterator

import cv2
import numpy as np
import imageio

from lib.align import Alignments as AlignmentsBase, get_centered_size
from lib.image import count_frames, read_image
from lib.utils import (camel_case_split, get_image_paths, VIDEO_EXTENSIONS)

if T.TYPE_CHECKING:
    from collections.abc import Generator
    from argparse import Namespace
    from lib.align import AlignedFace
    from plugins.extract import ExtractMedia

logger = logging.getLogger(__name__)


def finalize(images_found: int, num_faces_detected: int, verify_output: bool) -> None:
    """ Output summary statistics at the end of the extract or convert processes.

    Parameters
    ----------
    images_found: int
        The number of images/frames that were processed
    num_faces_detected: int
        The number of faces that have been detected
    verify_output: bool
        ``True`` if multiple faces were detected in frames otherwise ``False``.
     """
    logger.info("-------------------------")
    logger.info("Images found:        %s", images_found)
    logger.info("Faces detected:      %s", num_faces_detected)
    logger.info("-------------------------")

    if verify_output:
        logger.info("Note:")
        logger.info("Multiple faces were detected in one or more pictures.")
        logger.info("Double check your results.")
        logger.info("-------------------------")

    logger.info("Process Successfully Completed. Shutting Down...")


class Alignments(AlignmentsBase):
    """ Override :class:`lib.align.Alignments` to add custom loading based on command
    line arguments.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The command line arguments that were passed to Faceswap
    is_extract: bool
        ``True`` if the process calling this class is extraction otherwise ``False``
    input_is_video: bool, optional
        ``True`` if the input to the process is a video, ``False`` if it is a folder of images.
        Default: False
    """
    def __init__(self,
                 arguments: Namespace,
                 is_extract: bool,
                 input_is_video: bool = False) -> None:
        logger.debug("Initializing %s: (is_extract: %s, input_is_video: %s)",
                     self.__class__.__name__, is_extract, input_is_video)
        self._args = arguments
        self._is_extract = is_extract
        folder, filename = self._set_folder_filename(input_is_video)
        super().__init__(folder, filename=filename)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _set_folder_filename(self, input_is_video: bool) -> tuple[str, str]:
        """ Return the folder and the filename for the alignments file.

        If the input is a video, the alignments file will be stored in the same folder
        as the video, with filename `<videoname>_alignments`.

        If the input is a folder of images, the alignments file will be stored in folder with
        the images and just be called 'alignments'

        Parameters
        ----------
        input_is_video: bool, optional
            ``True`` if the input to the process is a video, ``False`` if it is a folder of images.

        Returns
        -------
        folder: str
            The folder where the alignments file will be stored
        filename: str
            The filename of the alignments file
        """
        if self._args.alignments_path:
            logger.debug("Alignments File provided: '%s'", self._args.alignments_path)
            folder, filename = os.path.split(str(self._args.alignments_path))
        elif input_is_video:
            logger.debug("Alignments from Video File: '%s'", self._args.input_dir)
            folder, filename = os.path.split(self._args.input_dir)
            filename = f"{os.path.splitext(filename)[0]}_alignments.fsa"
        else:
            logger.debug("Alignments from Input Folder: '%s'", self._args.input_dir)
            folder = str(self._args.input_dir)
            filename = "alignments"
        logger.debug("Setting Alignments: (folder: '%s' filename: '%s')", folder, filename)
        return folder, filename

    def _load(self) -> dict[str, T.Any]:
        """ Override the parent :func:`~lib.align.Alignments._load` to handle skip existing
        frames and faces on extract.

        If skip existing has been selected, existing alignments are loaded and returned to the
        calling script.

        Returns
        -------
        dict
            Any alignments that have already been extracted if skip existing has been selected
            otherwise an empty dictionary
        """
        data: dict[str, T.Any] = {}
        if not self._is_extract and not self.have_alignments_file:
            return data
        if not self._is_extract:
            data = super()._load()
            return data

        skip_existing = hasattr(self._args, 'skip_existing') and self._args.skip_existing
        skip_faces = hasattr(self._args, 'skip_faces') and self._args.skip_faces

        if not skip_existing and not skip_faces:
            logger.debug("No skipping selected. Returning empty dictionary")
            return data

        if not self.have_alignments_file and (skip_existing or skip_faces):
            logger.warning("Skip Existing/Skip Faces selected, but no alignments file found!")
            return data

        data = super()._load()

        if skip_faces:
            # Remove items from alignments that have no faces so they will
            # be re-detected
            del_keys = [key for key, val in data.items() if not val["faces"]]
            logger.debug("Frames with no faces selected for redetection: %s", len(del_keys))
            for key in del_keys:
                if key in data:
                    logger.trace("Selected for redetection: '%s'",  # type:ignore[attr-defined]
                                 key)
                    del data[key]
        return data


class Images():
    """ Handles the loading of frames from a folder of images or a video file for extract
    and convert processes.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The command line arguments that were passed to Faceswap
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        self._args = arguments
        self._is_video = self._check_input_folder()
        self._input_images = self._get_input_images()
        self._images_found = self._count_images()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_video(self) -> bool:
        """bool: ``True`` if the input is a video file otherwise ``False``. """
        return self._is_video

    @property
    def input_images(self) -> str | list[str]:
        """str or list: Path to the video file if the input is a video otherwise list of
        image paths. """
        return self._input_images

    @property
    def images_found(self) -> int:
        """int: The number of frames that exist in the video file, or the folder of images. """
        return self._images_found

    def _count_images(self) -> int:
        """ Get the number of Frames from a video file or folder of images.

        Returns
        -------
        int
            The number of frames in the image source
        """
        if self._is_video:
            retval = int(count_frames(self._args.input_dir, fast=True))
        else:
            retval = len(self._input_images)
        return retval

    def _check_input_folder(self) -> bool:
        """ Check whether the input is a folder or video.

        Returns
        -------
        bool
            ``True`` if the input is a video otherwise ``False``
        """
        if not os.path.exists(self._args.input_dir):
            logger.error("Input location %s not found.", self._args.input_dir)
            sys.exit(1)
        if (os.path.isfile(self._args.input_dir) and
                os.path.splitext(self._args.input_dir)[1].lower() in VIDEO_EXTENSIONS):
            logger.info("Input Video: %s", self._args.input_dir)
            retval = True
        else:
            logger.info("Input Directory: %s", self._args.input_dir)
            retval = False
        return retval

    def _get_input_images(self) -> str | list[str]:
        """ Return the list of images or path to video file that is to be processed.

        Returns
        -------
        str or list
            Path to the video file if the input is a video otherwise list of image paths.
        """
        if self._is_video:
            input_images = self._args.input_dir
        else:
            input_images = get_image_paths(self._args.input_dir)

        return input_images

    def load(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """ Generator to load frames from a folder of images or from a video file.

        Yields
        ------
        filename: str
            The filename of the current frame
        image: :class:`numpy.ndarray`
            A single frame
        """
        iterator = self._load_video_frames if self._is_video else self._load_disk_frames
        for filename, image in iterator():
            yield filename, image

    def _load_disk_frames(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """ Generator to load frames from a folder of images.

        Yields
        ------
        filename: str
            The filename of the current frame
        image: :class:`numpy.ndarray`
            A single frame
        """
        logger.debug("Input is separate Frames. Loading images")
        for filename in self._input_images:
            image = read_image(filename, raise_error=False)
            if image is None:
                continue
            yield filename, image

    def _load_video_frames(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """ Generator to load frames from a video file.

        Yields
        ------
        filename: str
            The filename of the current frame
        image: :class:`numpy.ndarray`
            A single frame
        """
        logger.debug("Input is video. Capturing frames")
        vidname, ext = os.path.splitext(os.path.basename(self._args.input_dir))
        reader = imageio.get_reader(self._args.input_dir, "ffmpeg")  # type:ignore[arg-type]
        for i, frame in enumerate(T.cast(Iterator[np.ndarray], reader)):
            # Convert to BGR for cv2 compatibility
            frame = frame[:, :, ::-1]
            filename = f"{vidname}_{i + 1:06d}{ext}"
            logger.trace("Loading video frame: '%s'", filename)  # type:ignore[attr-defined]
            yield filename, frame
        reader.close()

    def load_one_image(self, filename) -> np.ndarray:
        """ Obtain a single image for the given filename.

        Parameters
        ----------
        filename: str
            The filename to return the image for

        Returns
        ------
        :class:`numpy.ndarray`
            The image for the requested filename,

        """
        logger.trace("Loading image: '%s'", filename)  # type:ignore[attr-defined]
        if self._is_video:
            if filename.isdigit():
                frame_no = filename
            else:
                frame_no = os.path.splitext(filename)[0][filename.rfind("_") + 1:]
                logger.trace(  # type:ignore[attr-defined]
                    "Extracted frame_no %s from filename '%s'", frame_no, filename)
            retval = self._load_one_video_frame(int(frame_no))
        else:
            retval = read_image(filename, raise_error=True)
        return retval

    def _load_one_video_frame(self, frame_no: int) -> np.ndarray:
        """ Obtain a single frame from a video file.

        Parameters
        ----------
        frame_no: int
            The frame index for the required frame

        Returns
        ------
        :class:`numpy.ndarray`
            The image for the requested frame index,
        """
        logger.trace("Loading video frame: %s", frame_no)  # type:ignore[attr-defined]
        reader = imageio.get_reader(self._args.input_dir, "ffmpeg")  # type:ignore[arg-type]
        reader.set_image_index(frame_no - 1)
        frame = reader.get_next_data()[:, :, ::-1]  # type:ignore[index]
        reader.close()
        return frame


class PostProcess():
    """ Optional pre/post processing tasks for convert and extract.

    Builds a pipeline of actions that have optionally been requested to be performed
    in this session.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The command line arguments that were passed to Faceswap
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        self._args = arguments
        self._actions = self._set_actions()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _set_actions(self) -> list[PostProcessAction]:
        """ Compile the requested actions to be performed into a list

        Returns
        -------
        list
            The list of :class:`PostProcessAction` to be performed
        """
        postprocess_items = self._get_items()
        actions: list["PostProcessAction"] = []
        for action, options in postprocess_items.items():
            options = {} if options is None else options
            args = options.get("args", tuple())
            kwargs = options.get("kwargs", {})
            args = args if isinstance(args, tuple) else tuple()
            kwargs = kwargs if isinstance(kwargs, dict) else {}
            task = globals()[action](*args, **kwargs)
            if task.valid:
                logger.debug("Adding Postprocess action: '%s'", task)
                actions.append(task)

        for ppaction in actions:
            action_name = camel_case_split(ppaction.__class__.__name__)
            logger.info("Adding post processing item: %s", " ".join(action_name))

        return actions

    def _get_items(self) -> dict[str, dict[str, tuple | dict] | None]:
        """ Check the passed in command line arguments for requested actions,

        For any requested actions, add the item to the actions list along with
        any relevant arguments and keyword arguments.

        Returns
        -------
        dict
            The name of the action to be performed as the key. Any action specific
            arguments and keyword arguments as the value.
        """
        postprocess_items: dict[str, dict[str, tuple | dict] | None] = {}
        # Debug Landmarks
        if (hasattr(self._args, 'debug_landmarks') and self._args.debug_landmarks):
            postprocess_items["DebugLandmarks"] = None

        logger.debug("Postprocess Items: %s", postprocess_items)
        return postprocess_items

    def do_actions(self, extract_media: ExtractMedia) -> None:
        """ Perform the requested optional post-processing actions on the given image.

        Parameters
        ----------
        extract_media: :class:`~plugins.extract.extract_media.ExtractMedia`
            The :class:`~plugins.extract.extract_media.ExtractMedia` object to perform the
            action on.

        Returns
        -------
        :class:`~plugins.extract.extract_media.ExtractMedia`
            The original :class:`~plugins.extract.extract_media.ExtractMedia` with any actions
            applied
        """
        for action in self._actions:
            logger.debug("Performing postprocess action: '%s'", action.__class__.__name__)
            action.process(extract_media)


class PostProcessAction():
    """ Parent class for Post Processing Actions.

    Usable in Extract or Convert or both depending on context. Any post-processing actions should
    inherit from this class.

    Parameters
    -----------
    args: tuple
        Varies for specific post process action
    kwargs: dict
        Varies for specific post process action
    """
    def __init__(self, *args, **kwargs) -> None:
        logger.debug("Initializing %s: (args: %s, kwargs: %s)",
                     self.__class__.__name__, args, kwargs)
        self._valid = True  # Set to False if invalid parameters passed in to disable
        logger.debug("Initialized base class %s", self.__class__.__name__)

    @property
    def valid(self) -> bool:
        """bool: ``True`` if the action if the parameters passed in for this action are valid,
        otherwise ``False`` """
        return self._valid

    def process(self, extract_media: ExtractMedia) -> None:
        """ Override for specific post processing action

        Parameters
        ----------
        extract_media: :class:`~plugins.extract.extract_media.ExtractMedia`
            The :class:`~plugins.extract.extract_media.ExtractMedia` object to perform the
            action on.
        """
        raise NotImplementedError


class DebugLandmarks(PostProcessAction):
    """ Draw debug landmarks on face output. Extract Only """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        self._face_size = 0
        self._legacy_size = 0
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.0
        self._font_pad = 0

    def _initialize_font(self, size: int) -> None:
        """ Set the font scaling sizes on first call

        Parameters
        ----------
        size: int
            The pixel size of the saved aligned face
        """
        self._font_scale = size / 512
        self._font_pad = size // 64

    def _border_text(self,
                     image: np.ndarray,
                     text: str,
                     color: tuple[int, int, int],
                     position: tuple[int, int]) -> None:
        """ Create text on an image with a black border

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The image to put bordered text on to
        text: str
            The text to place the image
        color: tuple
            The color of the text
        position: tuple
            The (x, y) co-ordinates to place the text
        """
        thickness = 2
        for idx in range(2):
            text_color = (0, 0, 0) if idx == 0 else color
            cv2.putText(image,
                        text,
                        position,
                        self._font,
                        self._font_scale,
                        text_color,
                        thickness,
                        lineType=cv2.LINE_AA)
            thickness //= 2

    def _annotate_face_box(self, face: AlignedFace) -> None:
        """ Annotate the face extract box and print the original size in pixels

        face: :class:`~lib.align.AlignedFace`
            The object containing the aligned face to annotate
        """
        assert face.face is not None
        color = (0, 255, 0)
        roi = face.get_cropped_roi(face.size, self._face_size, "face")
        cv2.rectangle(face.face, tuple(roi[:2]), tuple(roi[2:]), color, 1)

        # Size in top right corner
        roi_pnts = np.array([[roi[0], roi[1]],
                             [roi[0], roi[3]],
                             [roi[2], roi[3]],
                             [roi[2], roi[1]]])
        orig_roi = face.transform_points(roi_pnts, invert=True)
        size = int(round(((orig_roi[1][0] - orig_roi[0][0]) ** 2 +
                          (orig_roi[1][1] - orig_roi[0][1]) ** 2) ** 0.5))
        text_img = face.face.copy()
        text = f"{size}px"
        text_size = cv2.getTextSize(text, self._font, self._font_scale, 1)[0]
        pos_x = roi[2] - (text_size[0] + self._font_pad)
        pos_y = roi[1] + text_size[1] + self._font_pad

        self._border_text(text_img, text, color, (pos_x, pos_y))
        cv2.addWeighted(text_img, 0.75, face.face, 0.25, 0, face.face)

    def _print_stats(self, face: AlignedFace) -> None:
        """ Print various metrics on the output face images

        Parameters
        ----------
        face: :class:`~lib.align.AlignedFace`
            The loaded aligned face
        """
        assert face.face is not None
        text_image = face.face.copy()
        texts = [f"pitch: {face.pose.pitch:.2f}",
                 f"yaw: {face.pose.yaw:.2f}",
                 f"roll: {face.pose.roll: .2f}",
                 f"distance: {face.average_distance:.2f}"]
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 255)]
        text_sizes = [cv2.getTextSize(text, self._font, self._font_scale, 1)[0] for text in texts]

        final_y = face.size - text_sizes[-1][1]
        pos_y = [(size[1] + self._font_pad) * (idx + 1)
                 for idx, size in enumerate(text_sizes)][:-1] + [final_y]
        pos_x = self._font_pad

        for idx, text in enumerate(texts):
            self._border_text(text_image, text, colors[idx], (pos_x, pos_y[idx]))

        # Apply text to face
        cv2.addWeighted(text_image, 0.75, face.face, 0.25, 0, face.face)

    def process(self, extract_media: ExtractMedia) -> None:
        """ Draw landmarks on a face.

        Parameters
        ----------
        extract_media: :class:`~plugins.extract.extract_media.ExtractMedia`
            The :class:`~plugins.extract.extract_media.ExtractMedia` object that contains the faces
            to draw the landmarks on to
        """
        frame = os.path.splitext(os.path.basename(extract_media.filename))[0]
        for idx, face in enumerate(extract_media.detected_faces):
            if not self._face_size:
                self._face_size = get_centered_size(face.aligned.centering,
                                                    "face",
                                                    face.aligned.size)
                logger.debug("set face size: %s", self._face_size)
            if not self._legacy_size:
                self._legacy_size = get_centered_size(face.aligned.centering,
                                                      "legacy",
                                                      face.aligned.size)
                logger.debug("set legacy size: %s", self._legacy_size)
            if not self._font_scale:
                self._initialize_font(face.aligned.size)

            logger.trace("Drawing Landmarks. Frame: '%s'. Face: %s",  # type:ignore[attr-defined]
                         frame, idx)
            # Landmarks
            assert face.aligned.face is not None
            for (pos_x, pos_y) in face.aligned.landmarks.astype("int32"):
                cv2.circle(face.aligned.face, (pos_x, pos_y), 1, (0, 255, 255), -1)
            # Pose
            center = (face.aligned.size // 2, face.aligned.size // 2)
            points = (face.aligned.pose.xyz_2d * face.aligned.size).astype("int32")
            cv2.line(face.aligned.face, center, tuple(points[1]), (0, 255, 0), 1)
            cv2.line(face.aligned.face, center, tuple(points[0]), (255, 0, 0), 1)
            cv2.line(face.aligned.face, center, tuple(points[2]), (0, 0, 255), 1)
            # Face centering
            self._annotate_face_box(face.aligned)
            # Legacy centering
            roi = face.aligned.get_cropped_roi(face.aligned.size, self._legacy_size, "legacy")
            cv2.rectangle(face.aligned.face, tuple(roi[:2]), tuple(roi[2:]), (0, 0, 255), 1)
            self._print_stats(face.aligned)
