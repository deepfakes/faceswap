#!/usr/bin/env python3
""" Face patch output writer for faceswap.py converter
    Extracts the swapped Face Patch from faceswap rather than the final composited frame along with
    the transformation matrix for re-inserting the face into the origial frame
"""
import json
import logging

import os
import cv2
import numpy as np

from lib.image import encode_image, png_read_meta, tiff_read_meta
from ._base import Output

logger = logging.getLogger(__name__)

# TODO change warp affine border type for single vs multi-faces


class Writer(Output):
    """ Face patch writer for outputting swapped face patches and transformation matrices

    Parameters
    ----------
    output_folder: str
        The full path to the output folder where the face patches should besaved
    patch_size: int
        The size of the face patch output from the model
    configfile: str, optional
        The full path to a custom configuration ini file. If ``None`` is passed
        then the file is loaded from the default location. Default: ``None``.
    """
    def __init__(self, output_folder: str, patch_size: int, **kwargs) -> None:
        logger.debug("patch_size: %s", patch_size)
        super().__init__(output_folder, **kwargs)
        self._patch_size = patch_size
        self._extension = {"png": ".png", "tiff": ".tif"}[self.config["format"]]
        self._separate_mask = self.config["separate_mask"]

        if self._extension == ".png" and self.config["bit_depth"] not in ("8", "16"):
            logger.warning("Patch Writer: Bit Depth '%s' is unsupported for format '%s'. "
                           "Updating to '16'", self.config["bit_depth"], self.config["format"])
            self.config["bit_depth"] = "16"
        self._dtype = {"8": np.uint8, "16": np.uint16, "32": np.float32}[self.config["bit_depth"]]
        self._multiplier = {"8": 255., "16": 65535., "32": 1.}[self.config["bit_depth"]]
        self._args = self._get_save_args()
        self._matrices: dict[str, dict[str, list[list[float]]]] = {}

    def _get_save_args(self) -> tuple[int, ...]:
        """ Obtain the save parameters for the file format.

        Returns
        -------
        tuple
            The OpenCV specific arguments for the selected file format
         """
        args: tuple[int, ...] = tuple()
        if self._extension == ".png" and self.config["png_compress_level"] > -1:
            args = (cv2.IMWRITE_PNG_COMPRESSION, self.config["png_compress_level"])
        if self._extension == ".tif" and self.config["bit_depth"] != "32":
            tiff_methods = {"none": 1, "lzw": 5, "deflate": 8}
            method = self.config["tiff_compression_method"]
            method = "none" if method is None else method
            args = (cv2.IMWRITE_TIFF_COMPRESSION, tiff_methods[method])
        logger.debug(args)
        return args

    def _get_new_filename(self, filename: str, face_index: int) -> str:
        """ Obtain the filename for the output file based on the frame's filename and the user
        selected naming options

        Parameters
        ----------
        filename: str
            The original frame's filename
        face_index: int
            The index of the face within the frame

        Returns
        -------
        str
            The new filename for naming the output face patch
        """
        face_idx = str(face_index).rjust(2, "0")
        fname, ext = os.path.splitext(filename)
        split_fname = fname.rsplit("_", 1)
        if split_fname[-1].isdigit():
            i_frame_no = (int(split_fname[-1]) +
                          (int(self.config["start_index"]) - 1) +
                          self.config["index_offset"])
            frame_no = f".{str(i_frame_no).rjust(self.config['number_padding'], '0')}"
        else:
            frame_no = ""

        retval = ""
        if self.config["include_filename"]:
            retval += f"{split_fname[0]}"
        if self.config["face_index_location"] == "before":
            retval = f"{retval}_{face_idx}"
        retval += frame_no
        if self.config["face_index_location"] == "after":
            retval = f"{retval}.{face_idx}"
        retval += ext
        logger.trace("source filename: '%s', output filename: '%s'",  # type:ignore[attr-defined]
                     filename, retval)
        return retval

    def write(self, filename: str, image: list[list[bytes]]) -> None:
        """ Write out the pre-encoded image to disk. If separate mask has been selected, write out
        the encoded mask to a sub-folder in the output directory.

        Parameters
        ----------
        filename: str
            The full path to write out the image to.
        image: list[list[bytes]]
            List of list of :class:`bytes` objects of containing all swapped faces from a frame to
            write out. The inner list will be of length 1 (mask included in the alpha channel) or
            length 2 (mask to write out separately)
        """
        logger.trace("Outputting: (filename: '%s')", filename)  # type:ignore[attr-defined]

        read_func = png_read_meta if self._extension == ".png" else tiff_read_meta
        for idx, face in enumerate(image):
            new_filename = self._get_new_filename(filename, idx)
            filenames = self.output_filename(new_filename, self._separate_mask)
            for fname, img in zip(filenames, face):
                try:
                    with open(fname, "wb") as outfile:
                        outfile.write(img)
                except Exception as err:  # pylint: disable=broad-except
                    logger.error("Failed to save image '%s'. Original Error: %s", filename, err)
                if not self.config["json_output"]:
                    continue
                mat = read_func(img)
                self._matrices[os.path.splitext(os.path.basename(fname))[0]] = mat

    @classmethod
    def _get_inverse_matrices(cls, matrices: np.ndarray) -> np.ndarray:
        """ Obtain the inverse matrices for the given matrices. If ``None`` is supplied return a
        dummy transformation matrix that performs no action

        Parameters
        ----------
        matrices : :class:`numpy.ndarray`
            The original transform matrices that the inverse needs to be calculated for

        Returns
        -------
        :class:`numpy.ndarray`
            The inverse transformation matrices
        """
        if not np.any(matrices):
            return np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]], dtype=np.float32)

        identity = np.array([[[0., 0., 1.]]], dtype=np.float32)
        mat = np.concatenate([matrices, np.repeat(identity, matrices.shape[0], axis=0)], axis=1)
        retval = np.linalg.inv(mat)
        logger.trace("matrix: %s, inverse: %s", mat, retval)  # type:ignore[attr-defined]
        return retval

    def _adjust_to_origin(self, matrices: np.ndarray, canvas_size: tuple[int, int]) -> None:
        """ Adjust the transformation matrix to use the correct target coordinates system. The
        matrix adjustment is done in place, so this does not return a value

        Parameters
        ----------
        matrices: :class:`numpy.ndarray`
            The transformation matrices to be adjusted
        canvas_size: tuple[int, int]
            The size of the canvas (height, width) that the transformation matrix applies to.
        """
        origin = self.config["origin"].split("-")
        if origin[0] == "bottom":
            matrices[..., 1, 1] *= -1.
            matrices[..., 1, 2] = canvas_size[1] - matrices[..., 1, 2] - 1.
        if origin[1] == "right":
            matrices[..., 0, 1] *= -1.
            matrices[..., 0, 2] = canvas_size[0] - matrices[..., 0, 2] - 1.

    def pre_encode(self, image: np.ndarray, **kwargs) -> list[list[bytes]]:
        """ Pre_encode the image in lib/convert.py threads as it is a LOT quicker.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            A 3 or 4 channel BGR swapped face batch as float32
        canvas_size: tuple[int, int]
            The size of the canvas (x, y) that the transformation matrix applies to.
        matrices: :class:`numpy.ndarray`, optional
            The transformation matrices for extracting the face patches from the original frame.
            Must be provided if an image is provided, otherwise ``None`` to insert a dummy matrix

        Returns
        -------
        list
            List of :class:`bytes` objects ready for writing. The list will be of length 1 with
            image bytes object as the only member unless separate mask has been requested, in which
            case it will be length 2 with the image in position 0 and mask in position 1
         """
        logger.trace("Pre-encoding image")  # type:ignore[attr-defined]
        retval = []
        canvas_size: tuple[int, int] = kwargs.get("canvas_size", (1, 1))
        matrices: np.ndarray = kwargs.get("matrices", np.array([]))

        if not np.any(image) and self.config["empty_frames"] == "blank":
            image = np.zeros((1, self._patch_size, self._patch_size, 4), dtype=np.float32)

        matrices = self._get_inverse_matrices(matrices)
        self._adjust_to_origin(matrices, canvas_size)
        patches = (image * self._multiplier).astype(self._dtype)

        for patch, matrix in zip(patches, matrices):
            this_face = []
            mat = json.dumps({"transform_matrix": matrix.tolist()},
                             ensure_ascii=True).encode("ascii")
            if self._separate_mask:
                mask = patch[..., -1]
                face = patch[..., :3]

                this_face.append(encode_image(mask,
                                              self._extension,
                                              encoding_args=self._args,
                                              metadata=mat))
            else:
                face = patch

            this_face.insert(0, encode_image(face,
                                             self._extension,
                                             encoding_args=self._args,
                                             metadata=mat))
            retval.append(this_face)
        return retval

    def close(self) -> None:
        """ Outputs json file if requested """
        if not self.config["json_output"]:
            return
        fname = os.path.join(self.output_folder, "matrices.json")
        with open(fname, "w", encoding="utf-8") as ofile:
            json.dump(self._matrices, ofile, indent=2)
        logger.info("Patch matrices written to: '%s'", fname)
