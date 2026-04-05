#!/usr/bin/env python3
"""Handles loading of faces/frames from source locations and pairing with alignments
information"""
from __future__ import annotations

import logging
import os
import typing as T

import numpy as np
from tqdm import tqdm

from lib.align import alignments, DetectedFace
from lib.image import FacesLoader, ImagesLoader
from lib.utils import get_module_objects
from lib.infer.objects import FrameFaces

if T.TYPE_CHECKING:
    from lib.align.objects import FileAlignments, PNGAlignments, PNGHeader
logger = logging.getLogger(__name__)


class Loader:
    """Loader for reading source data from disk, and yielding the output paired with alignment
    information

    Parameters
    ----------
    location
        Full path to the source files location
    is_faces
        ``True`` if the source is a folder of faceswap extracted faces
    """
    def __init__(self, location: str, is_faces: bool) -> None:
        logger.debug("Initializing %s (location: %s, is_faces: %s)",
                     self.__class__.__name__, location, is_faces)

        self._is_faces = is_faces
        self._loader = FacesLoader(location) if is_faces else ImagesLoader(location)
        self._alignments: alignments.Alignments | None = None
        self._skip_count = 0

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def file_list(self) -> list[str]:
        """Full file list of source files to be loaded """
        return self._loader.file_list

    @property
    def is_video(self) -> bool:
        """``True`` if the source is a video file otherwise ``False`` """
        return self._loader.is_video

    @property
    def location(self) -> str:
        """Full path to the source folder/video file location """
        return self._loader.location

    @property
    def skip_count(self) -> int:
        """The number of faces/frames that have been skipped due to no match in alignments file"""
        return self._skip_count

    def add_alignments(self, alignments_object: alignments.Alignments | None) -> None:
        """Add the loaded alignments to :attr:`_alignments` for content matching

        Parameters
        ----------
        alignments_object
            The alignments file object or ``None`` if not provided
        """
        logger.debug("Adding alignments to loader: %s", alignments_object)
        self._alignments = alignments_object

    def _process_face(self,
                      filename: str,
                      image: np.ndarray,
                      metadata: PNGHeader) -> FrameFaces | None:
        """Process a single face when masking from face images

        Parameters
        ----------
        filename
            the filename currently being processed
        image
            The current face being processed
        metadata
            The source frame metadata from the PNG header

        Returns
        -------
        the extract media object for the processed face or ``None`` if alignment information
        could not be found
        """
        frame_name = metadata.source.source_filename
        face_index = metadata.source.face_index

        if self._alignments is None:  # mask from PNG header
            lookup_index = 0
            aligns: list[FileAlignments] | list[PNGAlignments] = [metadata.alignments]
        else:  # mask from Alignments file
            lookup_index = face_index
            aligns = self._alignments.get_faces_in_frame(frame_name)
            if not aligns or face_index > len(aligns) - 1:
                self._skip_count += 1
                logger.warning("Skipping Face not found in alignments file: '%s'", filename)
                return None

        alignment = aligns[lookup_index]
        retval = FrameFaces(filename, image, is_aligned=True, frame_metadata=metadata.source)
        retval.detected_faces = [DetectedFace().from_alignment(alignment)]
        return retval

    def _from_faces(self) -> T.Generator[FrameFaces, None, None]:
        """Load content from pre-aligned faces and pair with corresponding metadata

        Yields
        ------
        The extract media object for the processed face
        """
        for filename, image, metadata in tqdm(self._loader.load(), total=self._loader.count):
            if not metadata:
                self._skip_count += 1
                logger.warning("Non-Faceswap extracted face found. Image skipped: '%s'",
                               filename)
                continue

            retval = self._process_face(filename, image, metadata)
            if retval is None:
                continue

            yield retval

    def _from_frames(self) -> T.Generator[FrameFaces, None, None]:
        """Load content from frames and and pair with corresponding metadata

        Yields
        ------
        The extract media object for the processed face
        """
        assert self._alignments is not None
        for filename, image in tqdm(self._loader.load(), total=self._loader.count):
            frame = os.path.basename(filename)

            if not self._alignments.frame_exists(frame):
                self._skip_count += 1
                logger.warning("Skipping frame not in alignments file: '%s'", frame)
                continue

            if not self._alignments.frame_has_faces(frame):
                logger.debug("Skipping frame with no faces: '%s'", frame)
                continue

            faces_in_frame = self._alignments.get_faces_in_frame(frame)
            detected_faces = [DetectedFace().from_alignment(alignment)
                              for alignment in faces_in_frame]

            retval = FrameFaces(filename, image)
            retval.detected_faces = detected_faces

            yield retval

    def load(self) -> T.Generator[FrameFaces, None, None]:
        """Load content from source and pair with corresponding alignment data

        Yields
        ------
        The extract media object for the processed face
        """
        if self._is_faces:
            iterator = self._from_faces
        else:
            iterator = self._from_frames

        yield from iterator()

        if self._skip_count > 0:
            logger.warning("%s face(s) skipped due to not existing in the alignments file",
                           self._skip_count)


__all__ = get_module_objects(__name__)
