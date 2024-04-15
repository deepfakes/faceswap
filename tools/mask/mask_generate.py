#!/usr/bin/env python3
""" Handles the generation of masks from faceswap for upating into an alignments file """
from __future__ import annotations

import logging
import os
import typing as T

from lib.image import encode_image, ImagesSaver
from lib.multithreading import MultiThread
from plugins.extract import Extractor

if T.TYPE_CHECKING:
    from lib.align import Alignments, DetectedFace
    from lib.align.alignments import PNGHeaderDict
    from lib.queue_manager import EventQueue
    from plugins.extract import ExtractMedia
    from .loader import Loader


logger = logging.getLogger(__name__)


class MaskGenerator:
    """ Uses faceswap's extract pipeline to generate masks and update them into the alignments file
    and/or extracted face PNG Headers

    Parameters
    ----------
    mask_type: str
        The mask type to generate
    update_all: bool
        ``True`` to update all faces, ``False`` to only update faces missing masks
    input_is_faces: bool
        ``True`` if the input are faceswap extracted faces otherwise ``False``
    exclude_gpus: list[int]
        List of any GPU IDs that should be excluded
    loader: :class:`tools.mask.loader.Loader`
        The loader for loading source images/video from disk
    """
    def __init__(self,
                 mask_type: str,
                 update_all: bool,
                 input_is_faces: bool,
                 loader: Loader,
                 alignments: Alignments | None,
                 input_location: str,
                 exclude_gpus: list[int]) -> None:
        logger.debug("Initializing %s (mask_type: %s, update_all: %s, input_is_faces: %s, "
                     "loader: %s, alignments: %s, input_location: %s, exclude_gpus: %s)",
                     self.__class__.__name__, mask_type, update_all, input_is_faces, loader,
                     alignments, input_location, exclude_gpus)

        self._update_all = update_all
        self._is_faces = input_is_faces
        self._alignments = alignments

        self._extractor = self._get_extractor(mask_type, exclude_gpus)
        self._mask_type = self._set_correct_mask_type(mask_type)
        self._input_thread = self._set_loader_thread(loader)
        self._saver = ImagesSaver(input_location, as_bytes=True) if input_is_faces else None

        self._counts: dict[T.Literal["face", "update"], int] = {"face": 0, "update": 0}

        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_extractor(self, mask_type, exclude_gpus: list[int]) -> Extractor:
        """ Obtain a Mask extractor plugin and launch it

        Parameters
        ----------
        mask_type: str
            The mask type to generate
        exclude_gpus: list or ``None``
            A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
            ``None`` to not exclude any GPUs.

        Returns
        -------
        :class:`plugins.extract.pipeline.Extractor`:
            The launched Extractor
        """
        logger.debug("masker: %s", mask_type)
        extractor = Extractor(None, None, mask_type, exclude_gpus=exclude_gpus)
        extractor.launch()
        logger.debug(extractor)
        return extractor

    def _set_correct_mask_type(self, mask_type: str) -> str:
        """ Some masks have multiple variants that they can be saved as depending on config options

        Parameters
        ----------
        mask_type: str
            The mask type to generate

        Returns
        -------
        str
            The actual mask variant to update
        """
        if mask_type != "bisenet-fp":
            return mask_type

        # Hacky look up into masker to get the type of mask
        mask_plugin = self._extractor._mask[0]  # pylint:disable=protected-access
        assert mask_plugin is not None
        mtype = "head" if mask_plugin.config.get("include_hair", False) else "face"
        new_type = f"{mask_type}_{mtype}"
        logger.debug("Updating '%s' to '%s'", mask_type, new_type)
        return new_type

    def _needs_update(self, frame: str, idx: int, face: DetectedFace) -> bool:
        """ Check if the mask for the current alignment needs updating for the requested mask_type

        Parameters
        ----------
        frame: str
            The frame name in the alignments file
        idx: int
            The index of the face for this frame in the alignments file
        face: :class:`~lib.align.DetectedFace`
            The dected face object to check

        Returns
        -------
        bool:
            ``True`` if the mask needs to be updated otherwise ``False``
        """
        if self._update_all:
            return True

        retval = not face.mask or face.mask.get(self._mask_type, None) is None

        logger.trace("Needs updating: %s, '%s' - %s",  # type:ignore[attr-defined]
                     retval, frame, idx)
        return retval

    def _feed_extractor(self, loader: Loader, extract_queue: EventQueue) -> None:
        """ Process to feed the extractor from inside a thread

        Parameters
        ----------
        loader: class:`tools.mask.loader.Loader`
            The loader for loading source images/video from disk
        extract_queue: :class:`lib.queue_manager.EventQueue`
            The input queue to the extraction pipeline
        """
        for media in loader.load():
            self._counts["face"] += len(media.detected_faces)

            if self._is_faces:
                assert len(media.detected_faces) == 1
                needs_update = self._needs_update(media.frame_metadata["source_filename"],
                                                  media.frame_metadata["face_index"],
                                                  media.detected_faces[0])
            else:
                # To keep face indexes correct/cover off where only one face in an image is missing
                # a mask where there are multiple faces we process all faces again for any frames
                # which have missing masks.
                needs_update = any(self._needs_update(media.filename, idx, detected_face)
                                   for idx, detected_face in enumerate(media.detected_faces))

            if not needs_update:
                logger.trace("No masks need updating in '%s'",  # type:ignore[attr-defined]
                             media.filename)
                continue

            logger.trace("Passing to extractor: '%s'", media.filename)  # type:ignore[attr-defined]
            extract_queue.put(media)

        logger.debug("Terminating loader thread")
        extract_queue.put("EOF")

    def _set_loader_thread(self, loader: Loader) -> MultiThread:
        """ Set the iterator to load ExtractMedia objects into the mask extraction pipeline
        so we can just iterate through the output masks

        Parameters
        ----------
        loader: class:`tools.mask.loader.Loader`
            The loader for loading source images/video from disk
        """
        in_queue = self._extractor.input_queue
        logger.debug("Starting load thread: (loader: %s, queue: %s)", loader, in_queue)
        in_thread = MultiThread(self._feed_extractor, loader, in_queue, thread_count=1)
        in_thread.start()
        logger.debug("Started load thread: %s", in_thread)
        return in_thread

    def _update_from_face(self, media: ExtractMedia) -> None:
        """ Update the alignments file and/or the extracted face

        Parameters
        ----------
        media: :class:`~lib.extract.pipeline.ExtractMedia`
            The ExtractMedia object with updated masks
        """
        assert len(media.detected_faces) == 1
        assert self._saver is not None

        fname = media.frame_metadata["source_filename"]
        idx = media.frame_metadata["face_index"]
        face = media.detected_faces[0]

        if self._alignments is not None:
            logger.trace("Updating face %s in frame '%s'", idx, fname)  # type:ignore[attr-defined]
            self._alignments.update_face(fname, idx, face.to_alignment())

        logger.trace("Updating extracted face: '%s'", media.filename)  # type:ignore[attr-defined]
        meta: PNGHeaderDict = {"alignments": face.to_png_meta(), "source": media.frame_metadata}
        self._saver.save(media.filename, encode_image(media.image, ".png", metadata=meta))

    def _update_from_frame(self, media: ExtractMedia) -> None:
        """ Update the alignments file

        Parameters
        ----------
        media: :class:`~lib.extract.pipeline.ExtractMedia`
            The ExtractMedia object with updated masks
        """
        assert self._alignments is not None
        fname = os.path.basename(media.filename)
        logger.trace("Updating %s faces in frame '%s'",  # type:ignore[attr-defined]
                     len(media.detected_faces), fname)
        for idx, face in enumerate(media.detected_faces):
            self._alignments.update_face(fname, idx, face.to_alignment())

    def _finalize(self) -> None:
        """ Close thread and save alignments on completion """
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

    def process(self) -> T.Generator[ExtractMedia, None, None]:
        """ Process the output from the extractor pipeline

        Yields
        ------
        :class:`~lib.extract.pipeline.ExtractMedia`
            The ExtractMedia object with updated masks
        """
        for media in self._extractor.detected_faces():
            self._input_thread.check_and_raise_error()
            self._counts["update"] += len(media.detected_faces)

            if self._is_faces:
                self._update_from_face(media)
            else:
                self._update_from_frame(media)

            yield media

        self._finalize()
        logger.debug("Completed MaskGenerator process")
