#!/usr/bin/env python3
"""Tools for working with aligned faces and aligned masks"""
from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np

from lib.utils import get_module_objects

from .constants import EXTRACT_RATIOS, LandmarkType, MAP_2D_68

if T.TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


if T.TYPE_CHECKING:
    from .constants import CenteringType


def get_adjusted_center(image_size: int,
                        source_offset: np.ndarray,
                        target_offset: np.ndarray,
                        source_centering: CenteringType,
                        y_offset: float) -> np.ndarray:
    """Obtain the correct center of a face extracted image to translate between two different
    extract centerings.

    Parameters
    ----------
    image_size
        The size of the image at the given :attr:`source_centering`
    source_offset
        The pose offset to translate a base extracted face to source centering
    target_offset
        The pose offset to translate a base extracted face to target centering
    source_centering
        The centering of the source image
    y_offset
        Amount to additionally offset the center of the image along the y-axis

    Returns
    -------
    The center point of the image at the given size for the target centering
    """
    source_size = image_size - (image_size * EXTRACT_RATIOS[source_centering])
    offset = target_offset - source_offset - [0., y_offset]
    offset *= source_size
    center = np.rint(offset + image_size / 2).astype("int32")
    logger.trace(  # type:ignore[attr-defined]
        "image_size: %s, source_offset: %s, target_offset: %s, source_centering: '%s', "
        "y_offset: %s, adjusted_offset: %s, center: %s",
        image_size, source_offset, target_offset, source_centering, y_offset, offset, center)
    return center


def get_centered_size(source_centering: CenteringType,
                      target_centering: CenteringType,
                      size: int,
                      coverage_ratio: float = 1.0) -> int:
    """Obtain the size of a cropped face from an aligned image.

    Given an image of a certain dimensions, returns the dimensions of the sub-crop within that
    image for the requested centering at the requested coverage ratio

    Notes
    -----
    `"legacy"` places the nose in the center of the image (the original method for aligning).
    `"face"` aligns for the nose to be in the center of the face (top to bottom) but the center
    of the skull for left to right. `"head"` places the center in the middle of the skull in 3D
    space.

    The ROI in relation to the source image is calculated by rounding the padding of one side
    to the nearest integer then applying this padding to the center of the crop, to ensure that
    any dimensions always have an even number of pixels.

    Parameters
    ----------
    source_centering
        The centering that the original image is aligned at
    target_centering
        The centering that the sub-crop size should be obtained for
    size
        The size of the source image to obtain the cropped size for
    coverage_ratio
        The coverage ratio to be applied to the target image. Default: `1.0`

    Returns
    -------
    The pixel size of a sub-crop image from a full head aligned image with the given coverage ratio
    """
    if source_centering == target_centering and coverage_ratio == 1.0:
        src_size: float | int = size
        retval = size
    else:
        src_size = size - (size * EXTRACT_RATIOS[source_centering])
        retval = 2 * int(np.rint((src_size / (1 - EXTRACT_RATIOS[target_centering])
                                 * coverage_ratio) / 2))
    logger.trace(  # type:ignore[attr-defined]
        "source_centering: %s, target_centering: %s, size: %s, coverage_ratio: %s, "
        "source_size: %s, crop_size: %s",
        source_centering, target_centering, size, coverage_ratio, src_size, retval)
    return retval


def get_matrix_scaling(matrix: np.ndarray) -> tuple[int, int]:
    """Given a matrix, return the cv2 Interpolation method and inverse interpolation method for
    applying the matrix on an image.

    Parameters
    ----------
    matrix
        The transform matrix to return the interpolator for

    Returns
    -------
    The interpolator and inverse interpolator for the given matrix. This will be (Cubic, Area) for
    an upscale matrix and (Area, Cubic) for a downscale matrix
    """
    x_scale = np.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[0, 1] * matrix[0, 1])
    if x_scale == 0:
        y_scale = 0.
    else:
        y_scale = (matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]) / x_scale
    avg_scale = (x_scale + y_scale) * 0.5
    if avg_scale >= 1.:
        interpolators = cv2.INTER_CUBIC, cv2.INTER_AREA
    else:
        interpolators = cv2.INTER_AREA, cv2.INTER_CUBIC
    logger.trace("interpolator: %s, inverse interpolator: %s",  # type:ignore[attr-defined]
                 interpolators[0], interpolators[1])
    return interpolators


def transform_image(image: np.ndarray,
                    matrix: np.ndarray,
                    size: int,
                    padding: int = 0) -> np.ndarray:
    """Perform transformation on an image, applying the given size and padding to the matrix.

    Parameters
    ----------
    image
        The image to transform
    matrix
        The transformation matrix to apply to the image
    size
        The final size of the transformed image
    padding
        The amount of padding to apply to the final image. Default: `0`

    Returns
    -------
    The transformed image
    """
    logger.trace("image shape: %s, matrix: %s, size: %s. padding: %s",  # type:ignore[attr-defined]
                 image.shape, matrix, size, padding)
    # transform the matrix for size and padding
    mat = matrix * (size - 2 * padding)
    mat[:, 2] += padding

    # transform image
    interpolators = get_matrix_scaling(mat)
    retval = cv2.warpAffine(image, mat, (size, size), flags=interpolators[0])
    logger.trace("transformed matrix: %s, final image shape: %s",  # type:ignore[attr-defined]
                 mat, image.shape)
    return retval


def batch_transform(matrices: npt.NDArray[np.float32],
                    points: npt.NDArray[np.float32],
                    in_place: bool = False) -> npt.NDArray[np.float32]:
    """Batch transform an array of (N, M, 2) points by the given (N, 3, 3) affine matrices

    Parameters
    ----------
    matrices
        The matrices to use to transform the points
    points
        The points to be transformed
    in_place
        ``True`` to directly transform the given points in place. ``False`` to return a new array

    Returns
    -------
    The transformed points
    """
    retval = points if in_place else np.empty_like(points)
    linear = matrices[:, :2, :2]
    translation = matrices[:, :2, 2]
    retval[:] = points @ linear.transpose(0, 2, 1) + translation[:, None, :]
    return retval


def batch_adjust_matrices(matrices: npt.NDArray[np.float32],
                          size: int,
                          padding: int,
                          reverse: bool = False) -> npt.NDArray[np.float32]:
    """Adjust a batch of normalized (0, 1) matrices to the given size and padding, or the reverse

    Parameters
    ----------
    matrices
        The (N, 3, 3) or (N, 2, 3) matrices to adjust
    size
        The size to adjust the matrices to
    padding
        The padding to apply to each side of the adjusted matrices
    reverse
        ``True`` to adjust normalized matrices to the given size. ``False`` to adjust the given
        sized matrices to normalized matrices. Default: ``False``

    Returns
    -------
    The adjusted matrices to the given size and padding if reverse is ``False`` or the normalized
    matrix if reverse is ``True``
    """
    retval = matrices.copy()
    scale = size - 2 * padding
    if reverse:
        retval[:, :2, 2] -= padding
        retval[:, :2] /= scale
    else:
        retval[:, :2] *= scale
        retval[:, :2, 2] += padding
    return retval


@T.overload
def batch_sub_crop(images: npt.NDArray[np.uint8],
                   offsets: npt.NDArray[np.int32],
                   out_size: int,
                   base_grid: tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]] | None = None
                   ) -> npt.NDArray[np.uint8]:
    ...


@T.overload
def batch_sub_crop(images: npt.NDArray[np.float32],
                   offsets: npt.NDArray[np.int32],
                   out_size: int,
                   base_grid: tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]] | None = None
                   ) -> npt.NDArray[np.float32]:
    ...


def batch_sub_crop(images: npt.NDArray[np.uint8 | np.float32],
                   offsets: npt.NDArray[np.int32],
                   out_size: int,
                   base_grid: tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]] | None = None
                   ) -> npt.NDArray[np.uint8 | np.float32]:
    """Obtain aligned sub-crops from larger aligned images

    Parameters
    ----------
    images
        The (N, H, W, C) full size extracted images
    offsets
        The (N, x, y) offsets to shift the sub-crops.
    out_size
        The output size of the sub-crop
    base_grid
        Pre-computed base mesh grid used to build crop indices. Should be a tuple (yy, xx) where
        each entry is a numpy array (int32) of shape (out_size, out_size) of row/column indices
        starting at 0, Providing this avoids rebuilding the meshgrid on every call.
        Default: ``None`` (calculate within the function)
    """
    batch_size, height, width, channels = images.shape

    if base_grid is None:
        yy, xx = np.meshgrid(np.arange(out_size, dtype="int32"),
                             np.arange(out_size, dtype="int32"),
                             indexing="ij")
    else:
        yy, xx = base_grid

    x_idx = xx[None] + offsets[:, 0, None, None]
    y_idx = yy[None] + offsets[:, 1, None, None]
    x_idx = np.clip(x_idx, 0, width - 1, out=x_idx)
    y_idx = np.clip(y_idx, 0, height - 1, out=y_idx)
    lin_idx = y_idx * width + x_idx

    flat = images.reshape(batch_size, height * width, channels)
    gathered = np.take_along_axis(flat,
                                  lin_idx.reshape(batch_size, -1)[..., None],
                                  axis=1)
    return gathered.reshape(batch_size, out_size, out_size, 3)


ImageDTypeT = T.TypeVar("ImageDTypeT", np.uint8, np.float32)


def batch_align(images: list[npt.NDArray[ImageDTypeT]],  # pylint:disable=too-many-locals
                image_ids: npt.NDArray[np.int32],
                matrices: npt.NDArray[np.float32],
                size: int,
                fast_upscale: bool = True) -> npt.NDArray[ImageDTypeT]:
    """Obtain a batch of aligned faces from the given images for the given matrices

    Parameters
    ----------
    images
        The full size images to obtain aligned faces from, either UINT8 or Float32 and 3 or 4
        channels. All images must be the same dtype and have the same number of channels
    image_ids
        The image id of each image in :attr:`image_ids` for each matrix in :attr:`matrices`
    matrices
        The adjustment matrices for taking the image patch from the frame for plugin input
    size
        The size of the returned aligned faces
    fast_upscale
        ``True`` to use cv2.INTER_LINEAR for upscale, ``False`` to use cv2.INTER_CUBIC.
        Default: ``True``

    Returns
    -------
    Batch of aligned face patches of the same dtype as the input images
    """
    channels = images[0].shape[-1]
    dtype = images[0].dtype
    assert all(i.shape[-1] == channels for i in images), (
        "All images must have the same number of channels")
    assert all(i.dtype == dtype for i in images), "All images must have the same dtype"
    assert np.any(matrices), "No matrices provided"
    mats = matrices[:, :2, :]  # Crop any Nx3x3 matrices to Nx2x3
    scales = np.hypot(matrices[..., 0, 0], matrices[..., 1, 0])  # Always same x/y scaling
    upscale = cv2.INTER_LINEAR if fast_upscale else cv2.INTER_CUBIC
    interpolations = np.where(scales > 1.0, cv2.INTER_LINEAR, upscale)

    dims: tuple[int, int] = (size, size)
    retval = np.zeros((len(image_ids), *dims, channels), dtype=dtype)

    for idx, (image_id, mat, interpolation) in enumerate(zip(image_ids, mats, interpolations)):
        cv2.warpAffine(images[image_id], mat, dims, dst=retval[idx], flags=interpolation)
    return retval


def batch_resize(images: npt.NDArray[ImageDTypeT], size: int, fast_upscale: bool = True
                 ) -> npt.NDArray[ImageDTypeT]:
    """Resize a batch of square images of the same dimensions to the given size

    Parameters
    ----------
    images
        The batch of square images to be resized
    size
        The required final size of the images
    fast_upscale
        ``True`` to use cv2.INTER_LINEAR for upscale, ``False`` to use cv2.INTER_CUBIC.
        Default: ``True``

    Returns
    -------
    The resized images
    """
    batch_size, height, width, channels = images.shape
    assert height == width, "Images must be square"
    if height == size:
        return images

    dims: tuple[int, int] = (size, size)
    retval = np.empty((batch_size, *dims, channels), dtype=images.dtype)
    upscale = cv2.INTER_LINEAR if fast_upscale else cv2.INTER_CUBIC
    interpolation = cv2.INTER_AREA if size < height else upscale
    for idx, img in enumerate(images):
        cv2.resize(img, dims, dst=retval[idx], interpolation=interpolation)
    return retval


def points_to_68(landmarks: npt.NDArray[np.float32],
                 landmark_type: LandmarkType | None = None) -> npt.NDArray[np.float32]:
    """Map the given non-68 point landmarks to 68 point landmarks

    Parameters
    ----------
    landmarks
        The non-68 point landmarks, either (N, P, 2) or (P, 2)
    landmark_type
        The type of landmarks that have been provided or ``None`` if to infer from the input
        landmarks. Default: ``None``

    Returns
    -------
    The (N, 68, 2) or (68, 2) mapped landmarks
    """
    is_batched = landmarks.ndim == 3
    if not is_batched:
        landmarks = landmarks[None]
    if landmark_type is None:
        landmark_type = LandmarkType.from_shape(landmarks.shape[1:])
    assert landmark_type in MAP_2D_68, f"{landmark_type} not supported"
    retval = landmarks[:, MAP_2D_68[landmark_type]]
    if is_batched:
        return retval
    return retval[0]


__all__ = get_module_objects(__name__)
