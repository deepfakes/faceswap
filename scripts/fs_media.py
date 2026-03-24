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

import numpy as np

from lib.align import Alignments as AlignmentsBase
from lib.logger import parse_class_init
from lib.serializer import get_serializer
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
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
            faces: list[AlignmentFileDict] = []
            for face in v:
                if "detected" not in face:
                    lms = np.array(face["landmarks_2d"], dtype="float32")
                    assert len(lms) == 4, (
                        "Missing detection boxes are only valid for ROI 4 point landmarks")
                    # Just place the box corners in the same location as the ROI box
                    mins = np.rint(lms.min(axis=0)).astype(np.int32).tolist()
                    maxes = np.rint(lms.max(axis=0)).astype(np.int32).tolist()
                    face["detected"] = mins + maxes
                faces.append(T.cast("AlignmentFileDict", {
                    "x": face["detected"][0],
                    "y": face["detected"][1],
                    "w": face["detected"][2] - face["detected"][0],
                    "h": face["detected"][3] - face["detected"][1],
                    "landmarks_xy": np.array(face["landmarks_2d"], dtype="float32"),
                    "mask": {},
                    "identity": {}}))
            self._data[k] = {"faces": faces, "video_meta": {}}
        logger.info("Imported %s frames from '%s'", len(data), json_file)


__all__ = get_module_objects(__name__)
