#!/usr/bin/env python3
""" Handles the generation of thumbnail jpgs for storing inside an alignments file/png header """
from __future__ import annotations

import logging
import typing as T

import numpy as np

from lib.logger import parse_class_init

if T.TYPE_CHECKING:
    from .alignments import Alignments

logger = logging.getLogger(__name__)


class Thumbnails():
    """ Thumbnail images stored in the alignments file.

    The thumbnails are stored as low resolution (64px), low quality jpg in the alignments file
    and are used for the Manual Alignments tool.

    Parameters
    ----------
    alignments: :class:'~lib.align.alignments.Alignments`
        The parent alignments class that these thumbs belong to
    """
    def __init__(self, alignments: Alignments) -> None:
        logger.debug(parse_class_init(locals()))
        self._alignments_dict = alignments.data
        self._frame_list = list(sorted(self._alignments_dict))
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def has_thumbnails(self) -> bool:
        """ bool: ``True`` if all faces in the alignments file contain thumbnail images
        otherwise ``False``. """
        retval = all(np.any(T.cast(np.ndarray, face.get("thumb")))
                     for frame in self._alignments_dict.values()
                     for face in frame["faces"])
        logger.trace(retval)  # type:ignore[attr-defined]
        return retval

    def get_thumbnail_by_index(self, frame_index: int, face_index: int) -> np.ndarray:
        """ Obtain a jpg thumbnail from the given frame index for the given face index

        Parameters
        ----------
        frame_index: int
            The frame index that contains the thumbnail
        face_index: int
            The face index within the frame to retrieve the thumbnail for

        Returns
        -------
        :class:`numpy.ndarray`
            The encoded jpg thumbnail
        """
        retval = self._alignments_dict[self._frame_list[frame_index]]["faces"][face_index]["thumb"]
        assert retval is not None
        logger.trace(  # type:ignore[attr-defined]
            "frame index: %s, face_index: %s, thumb shape: %s",
            frame_index, face_index, retval.shape)
        return retval

    def add_thumbnail(self, frame: str, face_index: int, thumb: np.ndarray) -> None:
        """ Add a thumbnail for the given face index for the given frame.

        Parameters
        ----------
        frame: str
            The name of the frame to add the thumbnail for
        face_index: int
            The face index within the given frame to add the thumbnail for
        thumb: :class:`numpy.ndarray`
            The encoded jpg thumbnail at 64px to add to the alignments file
        """
        logger.debug("frame: %s, face_index: %s, thumb shape: %s thumb dtype: %s",
                     frame, face_index, thumb.shape, thumb.dtype)
        self._alignments_dict[frame]["faces"][face_index]["thumb"] = thumb
