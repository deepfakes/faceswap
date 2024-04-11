#!/usr/bin/env python3
""" Import mask processing for faceswap's mask tool """
from __future__ import annotations

import logging
import os
import re
import sys
import typing as T

import cv2
from tqdm import tqdm

from lib.align import AlignedFace
from lib.image import encode_image, ImagesSaver
from lib.utils import get_image_paths

if T.TYPE_CHECKING:
    import numpy as np
    from .loader import Loader
    from plugins.extract import ExtractMedia
    from lib.align import Alignments, DetectedFace
    from lib.align.alignments import PNGHeaderDict
    from lib.align.aligned_face import CenteringType

logger = logging.getLogger(__name__)


class Import:
    """ Import masks from disk into an Alignments file

    Parameters
    ----------
    import_path: str
        The path to the input images
    centering: Literal["face", "head", "legacy"]
        The centering to store the mask at
    storage_size: int
        The size to store the mask at
    input_is_faces: bool
        ``True`` if the input is aligned faces otherwise ``False``
    loader: :class:`~tools.mask.loader.Loader`
        The source file loader object
    alignments: :class:`~lib.align.alignments.Alignments` | None
        The alignments file object for the faces, if provided
    mask_type: str
        The mask type to update to
    """
    def __init__(self,
                 import_path: str,
                 centering: CenteringType,
                 storage_size: int,
                 input_is_faces: bool,
                 loader: Loader,
                 alignments: Alignments | None,
                 input_location: str,
                 mask_type: str) -> None:
        logger.debug("Initializing %s (import_path: %s, centering: %s, storage_size: %s, "
                     "input_is_faces: %s, loader: %s, alignments: %s, input_location: %s, "
                     "mask_type: %s)", self.__class__.__name__, import_path, centering,
                     storage_size, input_is_faces, loader, alignments, input_location, mask_type)

        self._validate_mask_type(mask_type)

        self._centering = centering
        self._size = storage_size
        self._is_faces = input_is_faces
        self._alignments = alignments
        self._re_frame_num = re.compile(r"\d+$")
        self._mapping = self._generate_mapping(import_path, loader)

        self._saver = ImagesSaver(input_location, as_bytes=True) if input_is_faces else None
        self._counts: dict[T.Literal["skip", "update"], int] = {"skip": 0, "update": 0}

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def skip_count(self) -> int:
        """ int: Number of masks that were skipped as they do not exist for given faces """
        return self._counts["skip"]

    @property
    def update_count(self) -> int:
        """ int: Number of masks that were skipped as they do not exist for given faces """
        return self._counts["update"]

    @classmethod
    def _validate_mask_type(cls, mask_type: str) -> None:
        """ Validate that the mask type is 'custom' to ensure user does not accidentally overwrite
        existing masks they may have editted

        Parameters
        ----------
        mask_type: str
            The mask type that has been selected
        """
        if mask_type == "custom":
            return

        logger.error("Masker 'custom' must be selected for importing masks")
        sys.exit(1)

    @classmethod
    def _get_file_list(cls, path: str) -> list[str]:
        """ Check the nask folder exists and obtain the list of images

        Parameters
        ----------
        path: str
            Full path to the location of mask images to be imported

        Returns
        -------
        list[str]
            list of full paths to all of the images in the mask folder
        """
        if not os.path.isdir(path):
            logger.error("Mask path: '%s' is not a folder", path)
            sys.exit(1)
        paths = get_image_paths(path)
        if not paths:
            logger.error("Mask path '%s' contains no images", path)
            sys.exit(1)
        return paths

    def _warn_extra_masks(self, file_list: list[str]) -> None:
        """ Generate a warning for each mask that exists that does not correspond to a match in the
        source input

        Parameters
        ----------
        file_list: list[str]
            List of mask files that could not be mapped to a source image
        """
        if not file_list:
            logger.debug("All masks exist in the source data")
            return

        for fname in file_list:
            logger.warning("Extra mask file found: '%s'", os.path.basename(fname))

        logger.warning("%s mask file(s) do not exist in the source data so will not be imported "
                       "(see above)", len(file_list))

    def _file_list_to_frame_number(self, file_list: list[str]) -> dict[int, str]:
        """ Extract frame numbers from mask file names and return as a dictionary

        Parameters
        ----------
        file_list: list[str]
            List of full paths to masks to extract frame number from

        Returns
        -------
        dict[int, str]
            Dictionary of frame numbers to filenames
        """
        retval: dict[int, str] = {}
        for filename in file_list:
            frame_num = self._re_frame_num.findall(os.path.splitext(os.path.basename(filename))[0])

            if not frame_num or len(frame_num) > 1:
                logger.error("Could not detect frame number from mask file '%s'. "
                             "Check your filenames", os.path.basename(filename))
                sys.exit(1)

            fnum = int(frame_num[0])

            if fnum in retval:
                logger.error("Frame number %s for mask file '%s' already exists from file: '%s'. "
                             "Check your filenames",
                             fnum, os.path.basename(filename), os.path.basename(retval[fnum]))
                sys.exit(1)

            retval[fnum] = filename

        logger.debug("Files: %s, frame_numbers: %s", len(file_list), len(retval))

        return retval

    def _map_video(self, file_list: list[str], source_files: list[str]) -> dict[str, str]:
        """ Generate the mapping between the source data and the masks to be imported for
        video sources

        Parameters
        ----------
        file_list: list[str]
            List of full paths to masks to be imported
        source_files: list[str]
            list of filenames withing the source file

        Returns
        -------
        dict[str, str]
            Source filenames mapped to full path location of mask to be imported
        """
        retval = {}
        unmapped = []
        mask_frames = self._file_list_to_frame_number(file_list)
        for filename in tqdm(source_files, desc="Mapping masks to input", leave=False):
            src_idx = int(os.path.splitext(filename)[0].rsplit("_", maxsplit=1)[-1])
            mapped = mask_frames.pop(src_idx, "")
            if not mapped:
                unmapped.append(filename)
                continue
            retval[os.path.basename(filename)] = mapped

        if len(unmapped) == len(source_files):
            logger.error("No masks map between the source data and the mask folder. "
                         "Check your filenames")
            sys.exit(1)

        self._warn_extra_masks(list(mask_frames.values()))
        logger.debug("Source: %s, Mask: %s, Mapped: %s",
                     len(source_files), len(file_list), len(retval))
        return retval

    def _map_images(self, file_list: list[str], source_files: list[str]) -> dict[str, str]:
        """ Generate the mapping between the source data and the masks to be imported for
        folder of image sources

        Parameters
        ----------
        file_list: list[str]
            List of full paths to masks to be imported
        source_files: list[str]
            list of filenames withing the source file

        Returns
        -------
        dict[str, str]
            Source filenames mapped to full path location of mask to be imported
        """
        mask_count = len(file_list)
        retval = {}
        unmapped = []
        for filename in tqdm(source_files, desc="Mapping masks to input", leave=False):
            fname = os.path.splitext(os.path.basename(filename))[0]
            mapped = next((f for f in file_list
                           if os.path.splitext(os.path.basename(f))[0] == fname), "")
            if not mapped:
                unmapped.append(filename)
                continue
            retval[os.path.basename(filename)] = file_list.pop(file_list.index(mapped))

        if len(unmapped) == len(source_files):
            logger.error("No masks map between the source data and the mask folder. "
                         "Check your filenames")
            sys.exit(1)

        self._warn_extra_masks(file_list)

        logger.debug("Source: %s, Mask: %s, Mapped: %s",
                     len(source_files), mask_count, len(retval))
        return retval

    def _generate_mapping(self, import_path: str, loader: Loader) -> dict[str, str]:
        """ Generate the mapping between the source data and the masks to be imported

        Parameters
        ----------
        import_path: str
            The path to the input images
        loader: :class:`~tools.mask.loader.Loader`
            The source file loader object

        Returns
        -------
        dict[str, str]
            Source filenames mapped to full path location of mask to be imported
        """
        file_list = self._get_file_list(import_path)
        if loader.is_video:
            retval = self._map_video(file_list, loader.file_list)
        else:
            retval = self._map_images(file_list, loader.file_list)

        return retval

    def _store_mask(self, face: DetectedFace, mask: np.ndarray) -> None:
        """ Store the mask to the given DetectedFace object

        Parameters
        ----------
        face: :class:`~lib.align.detected_face.DetectedFace`
            The detected face object to store the mask to
        mask: :class:`numpy.ndarray`
            The mask to store
        """
        aligned = AlignedFace(face.landmarks_xy,
                              mask[..., None] if self._is_faces else mask,
                              centering=self._centering,
                              size=self._size,
                              is_aligned=self._is_faces,
                              dtype="float32")
        assert aligned.face is not None
        face.add_mask(f"custom_{self._centering}",
                      aligned.face / 255.,
                      aligned.adjusted_matrix,
                      aligned.interpolators[1],
                      storage_size=self._size,
                      storage_centering=self._centering)

    def _store_mask_face(self, media: ExtractMedia, mask: np.ndarray) -> None:
        """ Store the mask when the input is aligned faceswap faces

        Parameters
        ----------
        media: :class:`~plugins.extract.extract_media.ExtractMedia`
            The extract media object containing the face(s) to import the mask for

        mask: :class:`numpy.ndarray`
            The mask loaded from disk
        """
        assert self._saver is not None
        assert len(media.detected_faces) == 1

        logger.trace("Adding mask for '%s'", media.filename)  # type:ignore[attr-defined]

        face = media.detected_faces[0]
        self._store_mask(face, mask)

        if self._alignments is not None:
            idx = media.frame_metadata["source_filename"]
            fname = media.frame_metadata["face_index"]
            logger.trace("Updating face %s in frame '%s'", idx, fname)  # type:ignore[attr-defined]
            self._alignments.update_face(idx,
                                         fname,
                                         face.to_alignment())

        logger.trace("Updating extracted face: '%s'", media.filename)  # type:ignore[attr-defined]
        meta: PNGHeaderDict = {"alignments": face.to_png_meta(), "source": media.frame_metadata}
        self._saver.save(media.filename, encode_image(media.image, ".png", metadata=meta))

    @classmethod
    def _resize_mask(cls, mask: np.ndarray, dims: tuple[int, int]) -> np.ndarray:
        """ Resize a mask to the given dimensions

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The mask to resize
        dims: tuple[int, int]
            The (height, width) target size

        Returns
        -------
        :class:`numpy.ndarray`
            The resized mask, or the original mask if no resizing required
        """
        if mask.shape[:2] == dims:
            return mask
        logger.trace("Resizing mask from %s to %s", mask.shape, dims)  # type:ignore[attr-defined]
        interp = cv2.INTER_AREA if mask.shape[0] > dims[0] else cv2.INTER_CUBIC

        mask = cv2.resize(mask, tuple(reversed(dims)), interpolation=interp)
        return mask

    def _store_mask_frame(self, media: ExtractMedia, mask: np.ndarray) -> None:
        """ Store the mask when the input is frames

        Parameters
        ----------
        media: :class:`~plugins.extract.extract_media.ExtractMedia`
            The extract media object containing the face(s) to import the mask for

        mask: :class:`numpy.ndarray`
            The mask loaded from disk
        """
        assert self._alignments is not None
        logger.trace("Adding %s mask(s) for '%s'",  # type:ignore[attr-defined]
                     len(media.detected_faces), media.filename)

        mask = self._resize_mask(mask, media.image_size)

        for idx, face in enumerate(media.detected_faces):
            self._store_mask(face, mask)
            self._alignments.update_face(os.path.basename(media.filename),
                                         idx,
                                         face.to_alignment())

    def import_mask(self, media: ExtractMedia) -> None:
        """ Import the mask for the given Extract Media object

        Parameters
        ----------
        media: :class:`~plugins.extract.extract_media.ExtractMedia`
            The extract media object containing the face(s) to import the mask for
        """
        mask_file = self._mapping.get(os.path.basename(media.filename))
        if not mask_file:
            self._counts["skip"] += 1
            logger.warning("No mask file found for: '%s'", os.path.basename(media.filename))
            return

        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        logger.trace("Loaded mask for frame '%s': %s",  # type:ignore[attr-defined]
                     os.path.basename(mask_file), mask.shape)

        self._counts["update"] += len(media.detected_faces)

        if self._is_faces:
            self._store_mask_face(media, mask)
        else:
            self._store_mask_frame(media, mask)
