#!/usr/bin/env python3
""" Common multi-backend Keras utilities """
from __future__ import annotations
import typing as T

import numpy as np

import tensorflow.keras.backend as K  # pylint:disable=import-error

if T.TYPE_CHECKING:
    from tensorflow import Tensor


def frobenius_norm(matrix: Tensor,
                   axis: int = -1,
                   keep_dims: bool = True,
                   epsilon: float = 1e-15) -> Tensor:
    """ Frobenius normalization for Keras Tensor

    Parameters
    ----------
    matrix: Tensor
        The matrix to normalize
    axis: int, optional
        The axis to normalize. Default: `-1`
    keep_dims: bool, Optional
        Whether to retain the original matrix shape or not. Default:``True``
    epsilon: flot, optional
        Epsilon to apply to the normalization to preven NaN errors on zero values

    Returns
    -------
    Tensor
        The normalized output
    """
    return K.sqrt(K.sum(K.pow(matrix, 2), axis=axis, keepdims=keep_dims) + epsilon)


def replicate_pad(image: Tensor, padding: int) -> Tensor:
    """ Apply replication padding to an input batch of images. Expects 4D tensor in BHWC format.

    Notes
    -----
    At the time of writing Keras/Tensorflow does not have a native replication padding method.
    The implementation here is probably not the most efficient, but it is a pure keras method
    which should work on TF.

    Parameters
    ----------
    image: Tensor
        Image tensor to pad
    pad: int
        The amount of padding to apply to each side of the input image

    Returns
    -------
    Tensor
        The input image with replication padding applied
    """
    top_pad = K.tile(image[:, :1, ...], (1, padding, 1, 1))
    bottom_pad = K.tile(image[:, -1:, ...], (1, padding, 1, 1))
    pad_top_bottom = K.concatenate([top_pad, image, bottom_pad], axis=1)
    left_pad = K.tile(pad_top_bottom[..., :1, :], (1, 1, padding, 1))
    right_pad = K.tile(pad_top_bottom[..., -1:, :],  (1, 1, padding, 1))
    padded = K.concatenate([left_pad, pad_top_bottom, right_pad], axis=2)
    return padded


class ColorSpaceConvert():
    """ Transforms inputs between different color spaces on the GPU

    Notes
    -----
    The following color space transformations are implemented:
        - rgb to lab
        - rgb to xyz
        - srgb to _rgb
        - srgb to ycxcz
        - xyz to ycxcz
        - xyz to lab
        - xyz to rgb
        - ycxcz to rgb
        - ycxcz to xyz

    Parameters
    ----------
    from_space: str
        One of `"srgb"`, `"rgb"`, `"xyz"`
    to_space: str
        One of `"lab"`, `"rgb"`, `"ycxcz"`, `"xyz"`

    Raises
    ------
    ValueError
        If the requested color space conversion is not defined
    """
    def __init__(self, from_space: str, to_space: str) -> None:
        functions = {"rgb_lab": self._rgb_to_lab,
                     "rgb_xyz": self._rgb_to_xyz,
                     "srgb_rgb": self._srgb_to_rgb,
                     "srgb_ycxcz": self._srgb_to_ycxcz,
                     "xyz_ycxcz": self._xyz_to_ycxcz,
                     "xyz_lab": self._xyz_to_lab,
                     "xyz_to_rgb": self._xyz_to_rgb,
                     "ycxcz_rgb": self._ycxcz_to_rgb,
                     "ycxcz_xyz": self._ycxcz_to_xyz}
        func_name = f"{from_space.lower()}_{to_space.lower()}"
        if func_name not in functions:
            raise ValueError(f"The color transform {from_space} to {to_space} is not defined.")

        self._func = functions[func_name]
        self._ref_illuminant = K.constant(np.array([[[0.950428545, 1.000000000, 1.088900371]]]),
                                          dtype="float32")
        self._inv_ref_illuminant = 1. / self._ref_illuminant

        self._rgb_xyz_map = self._get_rgb_xyz_map()
        self._xyz_multipliers = K.constant([116, 500, 200], dtype="float32")

    @classmethod
    def _get_rgb_xyz_map(cls) -> tuple[Tensor, Tensor]:
        """ Obtain the mapping and inverse mapping for rgb to xyz color space conversion.

        Returns
        -------
        tuple
            The mapping and inverse Tensors for rgb to xyz color space conversion
        """
        mapping = np.array([[10135552 / 24577794,  8788810 / 24577794, 4435075 / 24577794],
                            [2613072 / 12288897, 8788810 / 12288897, 887015 / 12288897],
                            [1425312 / 73733382, 8788810 / 73733382, 70074185 / 73733382]])
        inverse = np.linalg.inv(mapping)
        return (K.constant(mapping, dtype="float32"), K.constant(inverse, dtype="float32"))

    def __call__(self, image: Tensor) -> Tensor:
        """ Call the colorspace conversion function.

        Parameters
        ----------
        image: Tensor
            The image tensor in the colorspace defined by :param:`from_space`

        Returns
        -------
        Tensor
            The image tensor in the colorspace defined by :param:`to_space`
        """
        return self._func(image)

    def _rgb_to_lab(self, image: Tensor) -> Tensor:
        """ RGB to LAB conversion.

        Parameters
        ----------
        image: Tensor
            The image tensor in RGB format

        Returns
        -------
        Tensor
            The image tensor in LAB format
        """
        converted = self._rgb_to_xyz(image)
        return self._xyz_to_lab(converted)

    def _rgb_xyz_rgb(self, image: Tensor, mapping: Tensor) -> Tensor:
        """ RGB to XYZ or XYZ to RGB conversion.

        Notes
        -----
        The conversion in both directions is the same, but the mappping matrix for XYZ to RGB is
        the inverse of RGB to XYZ.

        References
        ----------
        https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz

        Parameters
        ----------
        mapping: Tensor
            The mapping matrix to perform either the XYZ to RGB or RGB to XYZ color space
            conversion

        image: Tensor
            The image tensor in RGB format

        Returns
        -------
        Tensor
            The image tensor in XYZ format
        """
        dim = K.int_shape(image)
        image = K.permute_dimensions(image, (0, 3, 1, 2))
        image = K.reshape(image, (dim[0], dim[3], dim[1] * dim[2]))
        converted = K.permute_dimensions(K.dot(mapping, image), (1, 2, 0))
        return K.reshape(converted, dim)

    def _rgb_to_xyz(self, image: Tensor) -> Tensor:
        """ RGB to XYZ conversion.

        Parameters
        ----------
        image: Tensor
            The image tensor in RGB format

        Returns
        -------
        Tensor
            The image tensor in XYZ format
        """
        return self._rgb_xyz_rgb(image, self._rgb_xyz_map[0])

    @classmethod
    def _srgb_to_rgb(cls, image: Tensor) -> Tensor:
        """ SRGB to RGB conversion.

        Notes
        -----
        RGB Image is clipped to a small epsilon to stabalize training

        Parameters
        ----------
        image: Tensor
            The image tensor in SRGB format

        Returns
        -------
        Tensor
            The image tensor in RGB format
        """
        limit = 0.04045
        return K.switch(image > limit,
                        K.pow((K.clip(image, limit, None) + 0.055) / 1.055, 2.4),
                        image / 12.92)

    def _srgb_to_ycxcz(self, image: Tensor) -> Tensor:
        """ SRGB to YcXcZ conversion.

        Parameters
        ----------
        image: Tensor
            The image tensor in SRGB format

        Returns
        -------
        Tensor
            The image tensor in YcXcZ format
        """
        converted = self._srgb_to_rgb(image)
        converted = self._rgb_to_xyz(converted)
        return self._xyz_to_ycxcz(converted)

    def _xyz_to_lab(self, image: Tensor) -> Tensor:
        """ XYZ to LAB conversion.

        Parameters
        ----------
        image: Tensor
            The image tensor in XYZ format

        Returns
        -------
        Tensor
            The image tensor in LAB format
        """
        image = image * self._inv_ref_illuminant
        delta = 6 / 29
        delta_cube = delta ** 3
        factor = 1 / (3 * (delta ** 2))

        clamped_term = K.pow(K.clip(image, delta_cube, None), 1.0 / 3.0)
        div = factor * image + (4 / 29)

        image = K.switch(image > delta_cube, clamped_term, div)
        return K.concatenate([self._xyz_multipliers[0] * image[..., 1:2] - 16.,
                              self._xyz_multipliers[1:] * (image[..., :2] - image[..., 1:3])],
                             axis=-1)

    def _xyz_to_rgb(self, image: Tensor) -> Tensor:
        """ XYZ to YcXcZ conversion.

        Parameters
        ----------
        image: Tensor
            The image tensor in XYZ format

        Returns
        -------
        Tensor
            The image tensor in RGB format
        """
        return self._rgb_xyz_rgb(image, self._rgb_xyz_map[1])

    def _xyz_to_ycxcz(self, image: Tensor) -> Tensor:
        """ XYZ to YcXcZ conversion.

        Parameters
        ----------
        image: Tensor
            The image tensor in XYZ format

        Returns
        -------
        Tensor
            The image tensor in YcXcZ format
        """
        image = image * self._inv_ref_illuminant
        return K.concatenate([self._xyz_multipliers[0] * image[..., 1:2] - 16.,
                              self._xyz_multipliers[1:] * (image[..., :2] - image[..., 1:3])],
                             axis=-1)

    def _ycxcz_to_rgb(self, image: Tensor) -> Tensor:
        """ YcXcZ to RGB conversion.

        Parameters
        ----------
        image: Tensor
            The image tensor in YcXcZ format

        Returns
        -------
        Tensor
            The image tensor in RGB format
        """
        converted = self._ycxcz_to_xyz(image)
        return self._xyz_to_rgb(converted)

    def _ycxcz_to_xyz(self, image: Tensor) -> Tensor:
        """ YcXcZ to XYZ conversion.

        Parameters
        ----------
        image: Tensor
            The image tensor in YcXcZ format

        Returns
        -------
        Tensor
            The image tensor in XYZ format
        """
        ch_y = (image[..., 0:1] + 16.) / self._xyz_multipliers[0]
        return K.concatenate([ch_y + (image[..., 1:2] / self._xyz_multipliers[1]),
                              ch_y,
                              ch_y - (image[..., 2:3] / self._xyz_multipliers[2])],
                             axis=-1) * self._ref_illuminant
