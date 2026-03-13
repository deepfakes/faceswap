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

import numpy as np
import imageio

from lib.align import Alignments as AlignmentsBase
from lib.image import count_frames, read_image
from lib.logger import parse_class_init
from lib.serializer import get_serializer
from lib.utils import get_image_paths, get_module_objects, VIDEO_EXTENSIONS

if T.TYPE_CHECKING:
    from collections.abc import Generator
    from argparse import Namespace
    from lib.align.alignments import AlignmentFileDict

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
    if verify_output:
        logger.info("Note: Multiple faces were detected in one or more pictures. "
                    "Double check your results.")
    logger.info("-------------------------")


class Alignments(AlignmentsBase):
    """Override :class:`lib.align.Alignments` to add custom loading based on command
    line arguments.

    Parameters
    ----------
    location
        Full path to the alignments file. ``None`` to derive from the source file location
    source_location
        Full path to the source media for the alignments file. Either a folder of images or a video
        file
    arguments
        The command line arguments that were passed to Faceswap
    is_extract
        ``True`` if the process calling this class is extraction. Default: ``False``
    skip_existing_frames
        For extracting, indicates that 'skip existing' frames has been selected. Default: ``False``
    skip_existing_faces
        For extracting, indicates that 'skip existing faces' has been selected. Default: ``False``
    plugin_is_file
        ``True`` if 'File' has been selected for either/or a detector or aligner, indicating that
        information may be being loaded from a json file
    save_alignments
        ``True`` if the alignments are to be saved at the end of the running process, based on
        the selected extraction plugins
    input_is_video
        ``True`` if the input to the process is a video, ``False`` if it is a folder of images.
        Default: ``False``
    """
    def __init__(self,
                 location: str | None,
                 source_location: str,
                 is_extract: bool,
                 skip_existing_frames: bool = False,
                 skip_existing_faces: bool = False,
                 plugin_is_file: bool = False,
                 save_alignments: bool = False,
                 input_is_video: bool = False) -> None:
        logger.debug(parse_class_init(locals()))
        self._is_extract = is_extract
        self._skip_existing_frames = skip_existing_frames
        self._skip_existing_faces = skip_existing_faces
        self._plugin_is_file = plugin_is_file
        self._save_alignments = save_alignments
        folder, filename, self._import_json = self._set_folder_filename(location,
                                                                        source_location,
                                                                        input_is_video)
        super().__init__(folder, filename=filename)
        self._import_from_json()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def save_alignments(self) -> bool:
        """``True`` if the alignments should be saved at the end of the running process"""
        return self._save_alignments or self._import_json

    def _set_folder_filename(self,
                             location: str | None,
                             source_location: str,
                             input_is_video: bool) -> tuple[str, str, bool]:
        """Return the folder and the filename for the alignments file.

        If the location is not provided then for videos, the alignments file will be stored in the
        same folder as the video, with filename `<video_name>_alignments`. For a folder of images,
        the alignments file will be stored in folder with the images and just be called
        'alignments'

        Parameters
        ----------
        location
            Full path to the alignments file. ``None`` to derive from the source file location
        source_location
            Full path to the source media for the alignments file. Either a folder of images or a
            video file
        input_is_video:
            ``True`` if the input to the process is a video, ``False`` if it is a folder of images.

        Returns
        -------
        folder
            The folder where the alignments file will be stored
        filename
            The filename of the alignments file
        needs_import
            ``True`` if a 'file' plugin is being used for detect/align and the provided file is a
            .json file
        """
        if location:
            logger.debug("Alignments File provided: '%s'", location)
            folder, filename = os.path.split(str(location))
            if not self._plugin_is_file and os.path.splitext(filename)[-1].lower() == ".json":
                logger.error("Json files are only valid with 'File' detect/align plugins.")
                sys.exit(1)
        elif input_is_video:
            logger.debug("Alignments from Video File: '%s'", source_location)
            folder, filename = os.path.split(source_location)
            filename = f"{os.path.splitext(filename)[0]}_alignments"
        else:
            logger.debug("Alignments from Input Folder: '%s'", source_location)
            folder = str(source_location)
            filename = "alignments"
        logger.debug("Setting Alignments: (folder: '%s' filename: '%s')", folder, filename)

        if not self._plugin_is_file:
            return folder, filename, False

        full_path = os.path.join(folder, filename)
        for ext in (".json", ".fsa"):
            if os.path.splitext(filename)[-1].lower() in ext and os.path.exists(full_path):
                return folder, os.path.splitext(filename)[0], ext == ".json"
            full_file = f"{full_path}{ext}"
            if os.path.exists(full_file):
                return folder, filename, ext == ".json"

        logger.error("'File' has been selected for a Detect or Align plugin, but no alignments "
                     "file could be found. Check your paths.")
        sys.exit(1)

    def _load(self) -> dict[str, T.Any]:
        """Override the parent :func:`~lib.align.Alignments._load` to handle skip existing
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

        if (not self._skip_existing_frames
                and not self._skip_existing_faces
                and not self._plugin_is_file):
            logger.debug("No previous alignments file required. Returning empty dictionary")
            return data

        file_exists = self.have_alignments_file or self._import_json

        if not file_exists and (self._skip_existing_frames or self._skip_existing_faces):
            logger.warning("Skip Existing/Skip Faces selected, but no alignments file found!")
        if not file_exists:
            return data

        if self._import_json and self.have_alignments_file:
            logger.warning("Importing alignments from json, but alignments file exists: '%s'",
                           self._io.file)
            self.backup()
        if self._import_json:
            return data

        data = super()._load()
        return data

    def _import_from_json(self) -> None:
        """Import data from a JSON file when 'file' align/detect has been selected and a json file
        has been provided """
        if not self._import_json:
            return
        json_file = f"{os.path.splitext(self._io.file)[0]}.json"
        if self.data:
            logger.warning("Importing alignments from json file '%s', but data pre-exists in file "
                           "'%s'. Any matching frames will be overwritten.",
                           json_file, self._io.file)
        data = get_serializer("json").load(json_file)
        for k, v in data.items():
            faces = T.cast(list["AlignmentFileDict"], [{"x": f["detected"][0],
                                                        "y": f["detected"][1],
                                                        "w": f["detected"][2] - f["detected"][0],
                                                        "h": f["detected"][3] - f["detected"][1],
                                                        "landmarks_xy": np.array(f["landmarks_2d"],
                                                                                 dtype="float32"),
                                                        "mask": {},
                                                        "identity": {}}
                                                       for f in v])
            self._data[k] = {"faces": faces, "video_meta": {}}
        logger.info("Imported %s frames from '%s'", len(data), json_file)


class Images():
    """Handles the loading of frames from a folder of images or a video file for extract
    and convert processes.

    Parameters
    ----------
    arguments
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
        """``True`` if the input is a video file otherwise ``False``. """
        return self._is_video

    @property
    def input_images(self) -> str | list[str]:
        """Path to the video file if the input is a video otherwise list of image paths."""
        return self._input_images

    @property
    def images_found(self) -> int:
        """The number of frames that exist in the video file, or the folder of images."""
        return self._images_found

    def _count_images(self) -> int:
        """Get the number of Frames from a video file or folder of images.

        Returns
        -------
        The number of frames in the image source
        """
        if self._is_video:
            retval = int(count_frames(self._args.input_dir, fast=True))
        else:
            retval = len(self._input_images)
        return retval

    def _check_input_folder(self) -> bool:
        """Check whether the input is a folder or video.

        Returns
        -------
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
        """Return the list of images or path to video file that is to be processed.

        Returns
        -------
        Path to the video file if the input is a video otherwise list of image paths.
        """
        if self._is_video:
            input_images = self._args.input_dir
        else:
            input_images = get_image_paths(self._args.input_dir)

        return input_images

    def load(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Generator to load frames from a folder of images or from a video file.

        Yields
        ------
        filename
            The filename of the current frame
        image
            A single frame
        """
        iterator = self._load_video_frames if self._is_video else self._load_disk_frames
        for filename, image in iterator():
            yield filename, image

    def _load_disk_frames(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Generator to load frames from a folder of images.

        Yields
        ------
        filename
            The filename of the current frame
        image
            A single frame
        """
        logger.debug("Input is separate Frames. Loading images")
        for filename in self._input_images:
            image = read_image(filename, raise_error=False)
            if image is None:
                continue
            yield filename, image

    def _load_video_frames(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """Generator to load frames from a video file.

        Yields
        ------
        filename
            The filename of the current frame
        image
            A single frame
        """
        logger.debug("Input is video. Capturing frames")
        vid_name, ext = os.path.splitext(os.path.basename(self._args.input_dir))
        reader = imageio.get_reader(self._args.input_dir, "ffmpeg")  # type:ignore[arg-type]
        for i, frame in enumerate(T.cast(Iterator[np.ndarray], reader)):
            # Convert to BGR for cv2 compatibility
            frame = frame[:, :, ::-1]
            filename = f"{vid_name}_{i + 1:06d}{ext}"
            logger.trace("Loading video frame: '%s'", filename)  # type:ignore[attr-defined]
            yield filename, frame
        reader.close()

    def load_one_image(self, filename) -> np.ndarray:
        """Obtain a single image for the given filename.

        Parameters
        ----------
        filename
            The filename to return the image for

        Returns
        ------
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
        """Obtain a single frame from a video file.

        Parameters
        ----------
        frame_no
            The frame index for the required frame

        Returns
        ------
        The image for the requested frame index,
        """
        logger.trace("Loading video frame: %s", frame_no)  # type:ignore[attr-defined]
        reader = imageio.get_reader(self._args.input_dir, "ffmpeg")  # type:ignore[arg-type]
        reader.set_image_index(frame_no - 1)
        frame = reader.get_next_data()[:, :, ::-1]  # type:ignore[index]
        reader.close()
        return frame


__all__ = get_module_objects(__name__)
