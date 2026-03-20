#!/usr/bin/env python3
"""Output processing for faceswap's mask tool"""
from __future__ import annotations

import logging
import os
import sys
import typing as T
from argparse import Namespace

import cv2
import numpy as np
from tqdm import tqdm

from lib.align import AlignedFace
from lib.align.alignments import AlignmentDict

from lib.image import ImagesSaver, read_image_meta_batch
from lib.utils import get_folder, get_module_objects
from scripts.fs_media import Alignments as ExtractAlignments

if T.TYPE_CHECKING:
    from lib import align
    from lib.align import DetectedFace
    from lib.align.aligned_face import CenteringType

logger = logging.getLogger(__name__)


class Output:
    """Handles outputting of masks for preview/editing to disk

    Parameters
    ----------
    arguments
        The command line arguments that the mask tool was called with
    alignments
        The alignments file object (or ``None`` if not provided and input is faces)
    file_list
        Full file list for the loader. Used for extracting alignments from faces
    """
    def __init__(self, arguments: Namespace,
                 alignments: align.alignments.Alignments | None,
                 file_list: list[str]) -> None:
        logger.debug("Initializing %s (arguments: %s, alignments: %s, file_list: %s)",
                     self.__class__.__name__, arguments, alignments, len(file_list))

        self._blur_kernel: int = arguments.blur_kernel
        self._threshold: int = arguments.threshold
        self._type: T.Literal["combined", "masked", "mask"] = arguments.output_type
        self._full_frame: bool = arguments.full_frame
        self._mask_type = arguments.masker
        self._centering: CenteringType = arguments.centering

        self._input_is_faces = arguments.input_type == "faces"
        self._saver = self._set_saver(arguments.output, arguments.processing)
        self._alignments = self._get_alignments(alignments, file_list)

        self._full_frame_cache: dict[str, list[tuple[int, DetectedFace]]] = {}

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def should_save(self) -> bool:
        """``True`` if mask images should be output otherwise ``False``"""
        return self._saver is not None

    def _get_subfolder(self, output: str) -> str:
        """Obtain a subfolder within the output folder to save the output based on selected
        output options.

        Parameters
        ----------
        output
            Full path to the root output folder

        Returns
        -------
        The full path to where masks should be saved
        """
        out_type = "frame" if self._full_frame else "face"
        retval = os.path.join(output,
                              f"{self._mask_type}_{out_type}_{self._type}")
        logger.info("Saving masks to '%s'", retval)
        return retval

    def _set_saver(self, output: str | None, processing: str) -> ImagesSaver | None:
        """set the saver in a background thread

        Parameters
        ----------
        output
            Full path to the root output folder if provided
        processing
            The processing that has been selected

        Returns
        -------
        If output is requested, returns a :class:`lib.image.ImagesSaver` otherwise returns ``None``
        """
        if output is None or not output:
            if processing == "output":
                logger.error("Processing set as 'output' but no output folder provided.")
                sys.exit(0)
            logger.debug("No output provided. Not creating saver")
            return None
        output_dir = get_folder(self._get_subfolder(output), make_folder=True)
        retval = ImagesSaver(output_dir)
        logger.debug(retval)
        return retval

    def _get_alignments(self,
                        alignments: align.alignments.Alignments | None,
                        file_list: list[str]) -> align.alignments.Alignments | None:
        """Obtain the alignments file. If input is faces and full frame output is requested then
        the file needs to be generated from the input faces, if not provided

        Parameters
        ----------
        alignments
            The alignments file object (or ``None`` if not provided and input is faces)
        file_list
            Full paths to ihe mask tool input files

        Returns
        -------
        The alignments file if provided and/or is required otherwise ``None``
        """
        if alignments is not None or not self._full_frame:
            return alignments
        logger.debug("Generating alignments from faces")

        data = T.cast(dict[str, AlignmentDict], {})
        for _, meta in tqdm(read_image_meta_batch(file_list),
                            desc="Reading alignments from faces",
                            total=len(file_list),
                            leave=False):
            fname = meta["itxt"]["source"]["source_filename"]
            aln = meta["itxt"]["alignments"]
            data.setdefault(fname, {}).setdefault("faces",  # type:ignore[typeddict-item]
                                                  []).append(aln)

        retval = ExtractAlignments("/dummy/alignments.fsa", "", is_extract=True)
        retval.update_from_dict(data)
        return retval

    def _get_background_frame(self, detected_faces: list[DetectedFace], frame_dims: tuple[int, int]
                              ) -> np.ndarray:
        """Obtain the background image when final output is in full frame format. There will only
        ever be one background, even when there are multiple faces

        The output image will depend on the requested output type and whether the input is faces
        or frames

        Parameters
        ----------
        detected_faces
            Detected face objects for the output image
        frame_dims
            The size of the original frame

        Returns
        -------
        The full frame background image for applying masks to
        """
        if self._type == "mask":
            return np.zeros(frame_dims, dtype="uint8")

        if not self._input_is_faces:  # Frame is in the detected faces object
            assert detected_faces[0].image is not None
            return np.ascontiguousarray(detected_faces[0].image)

        # Outputting to frames, but input is faces. Apply the face patches to an empty canvas
        retval = np.zeros((*frame_dims, 3), dtype="uint8")
        for detected_face in detected_faces:
            assert detected_face.image is not None
            face = AlignedFace(detected_face.landmarks_xy,
                               image=detected_face.image,
                               centering="head",
                               size=detected_face.image.shape[0],
                               is_aligned=True)
            border = cv2.BORDER_TRANSPARENT if len(detected_faces) > 1 else cv2.BORDER_CONSTANT
            assert face.face is not None
            cv2.warpAffine(face.face,
                           face.adjusted_matrix,
                           tuple(reversed(frame_dims)),
                           retval,
                           flags=cv2.WARP_INVERSE_MAP | face.interpolators[1],
                           borderMode=border)
        return retval

    def _get_background_face(self,
                             detected_face: DetectedFace,
                             mask_centering: CenteringType,
                             mask_size: int) -> np.ndarray:
        """Obtain the background images when the output is faces

        The output image will depend on the requested output type and whether the input is faces
        or frames

        Parameters
        ----------
        detected_face
            Detected face object for the output image
        mask_centering
            The centering of the stored mask
        mask_size
            The pixel size of the stored mask

        Returns
        -------
        The face background image for applying masks to for each detected face object
        """
        if self._type == "mask":
            return np.zeros((mask_size, mask_size), dtype="uint8")

        assert detected_face.image is not None

        if self._input_is_faces:
            retval = AlignedFace(detected_face.landmarks_xy,
                                 image=detected_face.image,
                                 centering=mask_centering,
                                 size=mask_size,
                                 is_aligned=True).face
        else:
            detected_face.load_aligned(detected_face.image,
                                       size=mask_size,
                                       centering=mask_centering,
                                       force=True)
            retval = detected_face.aligned.face

        assert retval is not None
        return retval

    def _get_background(self,
                        detected_faces: list[DetectedFace],
                        frame_dims: tuple[int, int],
                        mask_centering: CenteringType,
                        mask_size: int) -> np.ndarray:
        """Obtain the background image that the final output will be placed on

        Parameters
        ----------
        detected_faces
            Detected face objects for the output image
        frame_dims
            The size of the original frame
        mask_centering
            The centering of the stored mask
        mask_size
            The pixel size of the stored mask

        Returns
        -------
        The background image for the mask output
        """
        if self._full_frame:
            retval = self._get_background_frame(detected_faces, frame_dims)
        else:
            assert len(detected_faces) == 1  # If outputting faces, we should only receive 1 face
            retval = self._get_background_face(detected_faces[0], mask_centering, mask_size)

        logger.trace("Background image (size: %s, dtype: %s)",  # type:ignore[attr-defined]
                     retval.shape, retval.dtype)
        return retval

    def _get_mask(self,
                  detected_faces: list[DetectedFace],
                  mask_type: str,
                  mask_dims: tuple[int, int]) -> np.ndarray:
        """Generate the mask to be applied to the final output frame

        Parameters
        ----------
        detected_faces
            Detected face objects to generate the masks from
        mask_type
            The mask-type to use
        mask_dims
            The size of the mask to output

        Returns
        -------
        The final mask to apply to the output image
        """
        retval = np.zeros(mask_dims, dtype="uint8")
        for face in detected_faces:
            mask_object = face.mask[mask_type]
            mask_object.set_blur_and_threshold(blur_kernel=self._blur_kernel,
                                               threshold=self._threshold)
            if self._full_frame:
                mask = mask_object.get_full_frame_mask(*reversed(mask_dims))
            else:
                mask = mask_object.mask[..., 0]
            np.maximum(retval, mask, out=retval)
        logger.trace("Final mask (shape: %s, dtype: %s)",  # type:ignore[attr-defined]
                     retval.shape, retval.dtype)
        return retval

    def _build_output_image(self, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Collate the mask and images for the final output image, depending on selected output
        type

        Parameters
        ----------
        background
            The image that the mask will be applied to
        mask
            The mask to output

        Returns
        -------
        The final output image
        """
        if self._type == "mask":
            return mask

        mask = mask[..., None]
        if self._type == "masked":
            return np.concatenate([background, mask], axis=-1)

        height, width = background.shape[:2]
        masked = (background.astype("float32") * mask.astype("float32") / 255.).astype("uint8")
        mask = np.tile(mask, 3)
        for img in (background, masked, mask):
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), (255, 255, 255), 1)
        axis = 0 if background.shape[0] < background.shape[1] else 1
        retval = np.concatenate((background, masked, mask), axis=axis)

        return retval

    def _create_image(self,
                      detected_faces: list[DetectedFace],
                      mask_type: str,
                      frame_dims: tuple[int, int] | None) -> np.ndarray:
        """Create a mask preview image for saving out to disk

        Parameters
        ----------
        detected_faces
            Detected face objects for the output image
        mask_type
            The mask_type to process
        frame_dims
            The size of the original frame, if input is faces otherwise ``None``

        Returns
        -------
        A preview image depending on the output type in one of the following forms:
          - Containing 3 sub images: The original face, the masked face and the mask
          - The mask only
          - The masked face
        """
        assert detected_faces[0].image is not None
        dims = T.cast(tuple[int, int],
                      frame_dims if self._input_is_faces else detected_faces[0].image.shape[:2])
        assert dims is not None and len(dims) == 2

        mask_centering = detected_faces[0].mask[mask_type].stored_centering
        mask_size = detected_faces[0].mask[mask_type].stored_size

        background = self._get_background(detected_faces, dims, mask_centering, mask_size)
        mask = self._get_mask(detected_faces,
                              mask_type,
                              dims if self._full_frame else (mask_size, mask_size))
        retval = self._build_output_image(background, mask)

        logger.trace("Output image (shape: %s, dtype: %s)",  # type:ignore[attr-defined]
                     retval.shape, retval.dtype)
        return retval

    def _handle_cache(self,
                      frame: str,
                      idx: int,
                      detected_face: DetectedFace) -> list[tuple[int, DetectedFace]]:
        """For full frame output, cache any faces until all detected faces have been seen. For
        face output, just return the detected_face object inside a list

        Parameters
        ----------
        frame
            The frame name in the alignments file
        idx
            The index of the face for this frame in the alignments file
        detected_face
            A detected_face object for a face

        Returns
        -------
        Face index and detected face objects to be processed for this output, if any
        """
        if not self._full_frame:
            return [(idx, detected_face)]

        assert self._alignments is not None
        faces_in_frame = self._alignments.count_faces_in_frame(frame)
        if faces_in_frame == 1:
            return [(idx, detected_face)]

        self._full_frame_cache.setdefault(frame, []).append((idx, detected_face))

        if len(self._full_frame_cache[frame]) != faces_in_frame:
            logger.trace("Caching face for frame '%s'", frame)  # type:ignore[attr-defined]
            return []

        retval = self._full_frame_cache.pop(frame)
        logger.trace("Processing '%s' from cache: %s", frame, retval)  # type:ignore[attr-defined]
        return retval

    def _get_mask_types(self,
                        frame: str,
                        detected_faces: list[tuple[int, DetectedFace]]) -> list[str]:
        """Get the mask type names for the select mask type. Remove any detected faces where
        the selected mask does not exist

        Parameters
        ----------
        frame
            The frame name in the alignments file
        idx
            The index of the face for this frame in the alignments file
        detected_face
            The face index and detected_face object for output

        Returns
        -------
        List of mask type names to be processed
        """
        if self._mask_type == "bisenet-fp":
            mask_types = [f"{self._mask_type}_{area}" for area in ("face", "head")]
        else:
            mask_types = [self._mask_type]

        if self._mask_type == "custom":
            mask_types.append(f"{self._mask_type}_{self._centering}")

        final_masks = set()
        for idx in reversed(range(len(detected_faces))):
            face_idx, detected_face = detected_faces[idx]
            if detected_face.mask is None or not any(mask in detected_face.mask
                                                     for mask in mask_types):
                logger.warning("Mask type '%s' does not exist for frame '%s' index %s. Skipping",
                               self._mask_type, frame, face_idx)
                del detected_faces[idx]
                continue
            final_masks.update([m for m in detected_face.mask if m in mask_types])

        retval = list(final_masks)
        logger.trace("Handling mask types: %s", retval)  # type:ignore[attr-defined]
        return retval

    def save(self,
             frame: str,
             idx: int,
             detected_face: DetectedFace,
             frame_dims: tuple[int, int] | None = None) -> None:
        """Build the mask preview image and save

        Parameters
        ----------
        frame
            The frame name in the alignments file
        idx
            The index of the face for this frame in the alignments file
        detected_face
            A detected_face object for a face
        frame_dims
            The size of the original frame, if input is faces otherwise ``None``. Default: ``None``
        """
        assert self._saver is not None

        faces = self._handle_cache(frame, idx, detected_face)
        if not faces:
            return

        mask_types = self._get_mask_types(frame, faces)
        if not faces or not mask_types:
            logger.debug("No valid faces/masks to process for '%s'", frame)
            return

        for mask_type in mask_types:
            detected_faces = [f[1] for f in faces if mask_type in f[1].mask]
            if not detected_face:
                logger.warning("No '%s' masks to output for '%s'", mask_type, frame)
                continue
            if len(detected_faces) != len(faces):
                logger.warning("Some '%s' masks are missing for '%s'", mask_type, frame)

            image = self._create_image(detected_faces, mask_type, frame_dims)
            filename = os.path.splitext(frame)[0]
            if len(mask_types) > 1:
                filename += f"_{mask_type}"
            if not self._full_frame:
                filename += f"_{idx}"
            filename = os.path.join(self._saver.location, f"{filename}.png")
            logger.trace("filename: '%s', image_shape: %s",  # type:ignore[attr-defined]
                         filename, image.shape)
            self._saver.save(filename, image)

    def close(self) -> None:
        """Shut down the image saver if it is open"""
        if self._saver is None:
            return
        logger.debug("Shutting down saver")
        self._saver.close()


__all__ = get_module_objects(__name__)
