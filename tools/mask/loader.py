#!/usr/bin/env python3
""" Handles loading of faces/frames from source locations and pairing with alignments
information """
from __future__ import annotations

import logging
import os
import typing as T

import numpy as np
from tqdm import tqdm

from lib.align import DetectedFace, update_legacy_png_header
from lib.align.alignments import AlignmentFileDict
from lib.image import FacesLoader, ImagesLoader
from plugins.extract import ExtractMedia

if T.TYPE_CHECKING:
    from lib.align import Alignments
    from lib.align.alignments import PNGHeaderDict
logger = logging.getLogger(__name__)


class Loader:
    """ Loader for reading source data from disk, and yielding the output paired with alignment
    information

    Parameters
    ----------
    location: str
        Full path to the source files location
    is_faces: bool
        ``True`` if the source is a folder of faceswap extracted faces
    """
    def __init__(self, location: str, is_faces: bool) -> None:
        logger.debug("Initializing %s (location: %s, is_faces: %s)",
                     self.__class__.__name__, location, is_faces)

        self._is_faces = is_faces
        self._loader = FacesLoader(location) if is_faces else ImagesLoader(location)
        self._alignments: Alignments | None = None
        self._skip_count = 0

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def file_list(self) -> list[str]:
        """list[str]: Full file list of source files to be loaded """
        return self._loader.file_list

    @property
    def is_video(self) -> bool:
        """bool: ``True`` if the source is a video file otherwise ``False`` """
        return self._loader.is_video

    @property
    def location(self) -> str:
        """str: Full path to the source folder/video file location """
        return self._loader.location

    @property
    def skip_count(self) -> int:
        """int: The number of faces/frames that have been skipped due to no match in alignments
        file """
        return self._skip_count

    def add_alignments(self, alignments: Alignments | None) -> None:
        """ Add the loaded alignments to :attr:`_alignments` for content matching

        Parameters
        ----------
        alignments: :class:`~lib.align.Alignments` | None
            The alignments file object or ``None`` if not provided
        """
        logger.debug("Adding alignments to loader: %s", alignments)
        self._alignments = alignments

    @classmethod
    def _get_detected_face(cls, alignment: AlignmentFileDict) -> DetectedFace:
        """ Convert an alignment dict item to a detected_face object

        Parameters
        ----------
        alignment: :class:`lib.align.alignments.AlignmentFileDict`
            The alignment dict for a face

        Returns
        -------
        :class:`~lib.align.detected_face.DetectedFace`:
            The corresponding detected_face object for the alignment
        """
        detected_face = DetectedFace()
        detected_face.from_alignment(alignment)
        return detected_face

    def _process_face(self,
                      filename: str,
                      image: np.ndarray,
                      metadata: PNGHeaderDict) -> ExtractMedia | None:
        """ Process a single face when masking from face images

        Parameters
        ----------
        filename: str
            the filename currently being processed
        image: :class:`numpy.ndarray`
            The current face being processed
        metadata: dict
            The source frame metadata from the PNG header

        Returns
        -------
        :class:`plugins.pipeline.ExtractMedia` | None
            the extract media object for the processed face or ``None`` if alignment information
            could not be found
        """
        frame_name = metadata["source"]["source_filename"]
        face_index = metadata["source"]["face_index"]

        if self._alignments is None:  # mask from PNG header
            lookup_index = 0
            alignments = [T.cast(AlignmentFileDict, metadata["alignments"])]
        else:  # mask from Alignments file
            lookup_index = face_index
            alignments = self._alignments.get_faces_in_frame(frame_name)
            if not alignments or face_index > len(alignments) - 1:
                self._skip_count += 1
                logger.warning("Skipping Face not found in alignments file: '%s'", filename)
                return None

        alignment = alignments[lookup_index]
        detected_face = self._get_detected_face(alignment)

        retval = ExtractMedia(filename, image, detected_faces=[detected_face], is_aligned=True)
        retval.add_frame_metadata(metadata["source"])
        return retval

    def _from_faces(self) -> T.Generator[ExtractMedia, None, None]:
        """ Load content from pre-aligned faces and pair with corresponding metadata

        Yields
        ------
        :class:`plugins.pipeline.ExtractMedia`
            the extract media object for the processed face
        """
        log_once = False
        for filename, image, metadata in tqdm(self._loader.load(), total=self._loader.count):
            if not metadata:  # Legacy faces. Update the headers
                if self._alignments is None:
                    logger.error("Legacy faces have been discovered, but no alignments file "
                                 "provided. You must provide an alignments file for this face set")
                    break

                if not log_once:
                    logger.warning("Legacy faces discovered. These faces will be updated")
                    log_once = True

                metadata = update_legacy_png_header(filename, self._alignments)
                if not metadata:  # Face not found
                    self._skip_count += 1
                    logger.warning("Legacy face not found in alignments file. This face has not "
                                   "been updated: '%s'", filename)
                    continue

            if "source_frame_dims" not in metadata.get("source", {}):
                logger.error("The faces need to be re-extracted as at least some of them do not "
                             "contain information required to correctly generate masks.")
                logger.error("You can re-extract the face-set by using the Alignments Tool's "
                             "Extract job.")
                break

            retval = self._process_face(filename, image, metadata)
            if retval is None:
                continue

            yield retval

    def _from_frames(self) -> T.Generator[ExtractMedia, None, None]:
        """ Load content from frames and and pair with corresponding metadata

        Yields
        ------
        :class:`plugins.pipeline.ExtractMedia`
            the extract media object for the processed face
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
            detected_faces = [self._get_detected_face(alignment) for alignment in faces_in_frame]
            retval = ExtractMedia(filename, image, detected_faces=detected_faces)
            yield retval

    def load(self) -> T.Generator[ExtractMedia, None, None]:
        """ Load content from source and pair with corresponding alignment data

        Yields
        ------
        :class:`plugins.pipeline.ExtractMedia`
            the extract media object for the processed face
        """
        if self._is_faces:
            iterator = self._from_faces
        else:
            iterator = self._from_frames

        for media in iterator():
            yield media

        if self._skip_count > 0:
            logger.warning("%s face(s) skipped due to not existing in the alignments file",
                           self._skip_count)
