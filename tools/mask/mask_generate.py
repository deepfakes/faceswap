#!/usr/bin/env python3
"""Handles the generation of masks from faceswap for updating into an alignments file"""
from __future__ import annotations

import logging
import os
import typing as T

from lib.image import encode_image, ImagesSaver
from lib.logger import parse_class_init
from lib.multithreading import FSThread
from lib.utils import get_module_objects
from lib.infer import Mask

if T.TYPE_CHECKING:
    from lib import align
    from lib.align import DetectedFace
    from lib.infer.objects import FrameFaces
    from lib.infer.runner import ExtractRunner
    from lib.infer.handler import ExtractHandlerFace
    from . import loader


logger = logging.getLogger(__name__)


class MaskGenerator:
    """Uses faceswap's extract pipeline to generate masks and update them into the alignments file
    and/or extracted face PNG Headers

    Parameters
    ----------
    mask_type
        The mask type to generate
    update_all
        ``True`` to update all faces, ``False`` to only update faces missing masks
    input_is_faces
        ``True`` if the input are faceswap extracted faces otherwise ``False``
    loader
        The loader for loading source images/video from disk
    config_file
        Full path to a custom config file to load. ``None`` for default config
    """
    def __init__(self,
                 mask_type: str,
                 update_all: bool,
                 input_is_faces: bool,
                 loader: loader.Loader,
                 alignments: align.alignments.Alignments | None,
                 input_location: str,
                 config_file: str | None) -> None:
        logger.debug(parse_class_init(locals()))
        self._update_all = update_all
        self._is_faces = input_is_faces
        self._alignments = alignments

        self._extractor = T.cast("ExtractRunner[ExtractHandlerFace]",
                                 Mask(mask_type, config_file=config_file)())
        self._mask_type = self._extractor.handler.plugin.storage_name
        self._input_thread = self._set_loader_thread(loader)
        self._saver = ImagesSaver(input_location, as_bytes=True) if input_is_faces else None

        self._counts: dict[T.Literal["face", "update"], int] = {"face": 0, "update": 0}

        logger.debug("Initialized %s", self.__class__.__name__)

    def _needs_update(self, frame: str, idx: int, face: DetectedFace) -> bool:
        """Check if the mask for the current alignment needs updating for the requested mask_type

        Parameters
        ----------
        frame
            The frame name in the alignments file
        idx
            The index of the face for this frame in the alignments file
        face
            The detected face object to check

        Returns
        -------
        ``True`` if the mask needs to be updated otherwise ``False``
        """
        if self._update_all:
            return True

        retval = not face.mask or face.mask.get(self._mask_type, None) is None

        logger.trace("Needs updating: %s, '%s' - %s",  # type:ignore[attr-defined]
                     retval, frame, idx)
        return retval

    def _feed_extractor(self, loader: loader.Loader) -> None:
        """Process to feed the extractor from inside a thread

        Parameters
        ----------
        loader
            The loader for loading source images/video from disk
        """
        for media in loader.load():
            if self._input_thread.error_state.has_error:
                self._input_thread.error_state.re_raise()
            self._counts["face"] += len(media)

            if self._is_faces:
                assert media.frame_metadata is not None
                assert len(media) == 1
                needs_update = self._needs_update(media.frame_metadata["source_filename"],
                                                  media.frame_metadata["face_index"],
                                                  media.detected_faces[0])
            else:
                # To keep face indexes correct/cover off where only one face in an image is missing
                # a mask where there are multiple faces we process all faces again for any frames
                # which have missing masks.
                needs_update = any(self._needs_update(os.path.basename(media.filename),
                                                      idx,
                                                      detected_face)
                                   for idx, detected_face in enumerate(media.detected_faces))

            if not needs_update:
                logger.trace("No masks need updating in '%s'",  # type:ignore[attr-defined]
                             media.filename)
                continue

            logger.trace("Passing to extractor: '%s'", media.filename)  # type:ignore[attr-defined]
            self._extractor.put_media(media)

        logger.debug("Terminating loader thread")
        self._extractor.stop()

    def _set_loader_thread(self, loader: loader.Loader) -> FSThread:
        """Set the iterator to load FrameFaces objects into the mask extraction pipeline
        so we can just iterate through the output masks

        Parameters
        ----------
        loader
            The loader for loading source images/video from disk
        """
        logger.debug("Starting load thread: (loader: %s)", loader)
        in_thread = FSThread(self._feed_extractor, args=(loader, ))
        in_thread.start()
        logger.debug("Started load thread: %s", in_thread)
        return in_thread

    def _update_from_face(self, media: FrameFaces) -> None:
        """Update the alignments file and/or the extracted face

        Parameters
        ----------
        media
            The FrameFaces object with updated masks
        """
        assert len(media) == 1
        assert self._saver is not None
        assert media.frame_metadata is not None

        fname = media.frame_metadata["source_filename"]
        idx = media.frame_metadata["face_index"]
        face = media.detected_faces[0]

        if self._alignments is not None:
            logger.trace("Updating face %s in frame '%s'", idx, fname)  # type:ignore[attr-defined]
            self._alignments.update_face(fname, idx, face.to_alignment())

        logger.trace("Updating extracted face: '%s'", media.filename)  # type:ignore[attr-defined]
        meta: align.alignments.PNGHeaderDict = {"alignments": face.to_png_meta(),
                                                "source": media.frame_metadata}
        self._saver.save(os.path.basename(media.filename),
                         encode_image(media.image, ".png", metadata=meta))

    def _update_from_frame(self, media: FrameFaces) -> None:
        """Update the alignments file

        Parameters
        ----------
        media
            The FrameFaces object with updated masks
        """
        assert self._alignments is not None
        fname = os.path.basename(media.filename)
        logger.trace("Updating %s faces in frame '%s'",  # type:ignore[attr-defined]
                     len(media), fname)
        for idx, face in enumerate(media.detected_faces):
            self._alignments.update_face(fname, idx, face.to_alignment())

    def _finalize(self) -> None:
        """Close thread and save alignments on completion """
        logger.debug("Finalizing MaskGenerator")
        self._input_thread.join()

        if self._counts["update"] > 0 and self._alignments is not None:
            logger.debug("Saving alignments")
            self._alignments.backup()
            self._alignments.save()

        if self._saver is not None:
            logger.debug("Closing face saver")
            self._saver.close()

        if self._counts["update"] == 0:
            logger.warning("No masks were updated of the %s faces seen", self._counts["face"])
        else:
            logger.info("Updated masks for %s faces of %s",
                        self._counts["update"], self._counts["face"])

    def process(self) -> T.Generator[FrameFaces, None, None]:
        """Process the output from the extractor pipeline

        Yields
        ------
        The FrameFaces object with updated masks
        """
        for media in self._extractor:
            if self._input_thread.error_state.has_error:
                self._input_thread.error_state.re_raise()
            self._counts["update"] += len(media)

            if self._is_faces:
                self._update_from_face(media)
            else:
                self._update_from_frame(media)

            yield media

        self._finalize()
        logger.debug("Completed MaskGenerator process")


__all__ = get_module_objects(__name__)
