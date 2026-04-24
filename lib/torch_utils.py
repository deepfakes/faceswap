#!/usr/bin/env python
"""Common multi-backend Torch utilities"""
from __future__ import annotations
import logging
import typing as T

import numpy as np

import torch
from torch import nn

from lib.logger import parse_class_init
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


def get_device(cpu: bool = False) -> torch.device:
    """Get the correctly configured device for running Torch

    Parameters
    ----------
    cpu
        ``True`` to force running on the CPU.

    Returns
    -------
    The device that torch should use
    """
    if cpu:
        logger.debug("CPU mode selected. Returning CPU device")
        return torch.device("cpu")

    if torch.cuda.is_available():
        logger.debug("Cuda available. Returning Cuda device")
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        logger.debug("MPS available. Returning MPS device context")
        return torch.device("mps")

    logger.debug("No backends available. Returning CPU device context")
    return torch.device("cpu")


class ColorSpaceConvert(nn.Module):
    """Transforms inputs between different color spaces on the GPU. Images expected in (N,C,H,W)
    order

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
    from_space
        One of "srgb", "rgb", "ycxcz", "xyz"
    to_space
        One of "lab", "rgb", "ycxcz", "xyz"

    Raises
    ------
    ValueError
        If the requested color space conversion is not defined
    """
    _ref_illuminant: torch.Tensor
    _inv_ref_illuminant: torch.Tensor
    _rgb_xyz_map: torch.Tensor

    def __init__(self, from_space: T.Literal["srgb", "rgb", "ycxcz", "xyz"],
                 to_space: T.Literal["lab", "rgb", "ycxcz", "xyz"]) -> None:
        functions = {"rgb_lab": self._rgb_to_lab,
                     "rgb_xyz": self._rgb_to_xyz,
                     "srgb_rgb": self._srgb_to_rgb,
                     "srgb_ycxcz": self._srgb_to_ycxcz,
                     "xyz_ycxcz": self._xyz_to_ycxcz,
                     "xyz_lab": self._xyz_to_lab,
                     "xyz_rgb": self._xyz_to_rgb,
                     "ycxcz_rgb": self._ycxcz_to_rgb,
                     "ycxcz_xyz": self._ycxcz_to_xyz}
        super().__init__()
        logger.debug(parse_class_init(locals()))
        func_name = f"{from_space.lower()}_{to_space.lower()}"
        if func_name not in functions:
            raise ValueError(f"The color transform {from_space} to {to_space} is not defined.")
        self._func = functions[func_name]

        ref_illuminant = np.array([[[0.950428545]], [[1.000000000]], [[1.088900371]]],
                                  dtype=np.float32)
        self.register_buffer("_ref_illuminant", torch.from_numpy(ref_illuminant).float())
        self.register_buffer("_inv_ref_illuminant", torch.from_numpy(1. / ref_illuminant).float())
        self.register_buffer("_rgb_xyz_map", self._get_rgb_xyz_map())

    @classmethod
    def _get_rgb_xyz_map(cls) -> torch.Tensor:
        """Obtain the mapping and inverse mapping for rgb to xyz color space conversion.

        Returns
        -------
        The mapping and inverse Tensors for rgb to xyz color space conversion
        """
        mapping = np.array([[10135552 / 24577794,  8788810 / 24577794, 4435075 / 24577794],
                            [2613072 / 12288897, 8788810 / 12288897, 887015 / 12288897],
                            [1425312 / 73733382, 8788810 / 73733382, 70074185 / 73733382]])
        inverse = np.linalg.inv(mapping)
        return torch.from_numpy(np.stack([mapping, inverse], axis=0)).float()

    def _rgb_to_lab(self, image: torch.Tensor) -> torch.Tensor:
        """RGB to LAB conversion.

        Parameters
        ----------
        image
            The image tensor in RGB format

        Returns
        -------
        The image tensor in LAB format
        """
        converted = self._rgb_to_xyz(image)
        return self._xyz_to_lab(converted)

    def _rgb_xyz_rgb(self, image: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
        """RGB to XYZ or XYZ to RGB conversion.

        Notes
        -----
        The conversion in both directions is the same, but the mapping matrix for XYZ to RGB is
        the inverse of RGB to XYZ.

        References
        ----------
        https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz

        Parameters
        ----------
        mapping
            The mapping matrix to perform either the XYZ to RGB or RGB to XYZ color space
            conversion

        image
            The image tensor in RGB format

        Returns
        -------
        The image tensor in XYZ format
        """
        dim = image.shape
        image = image.reshape(dim[0], dim[1], dim[2] * dim[3])
        converted = mapping @ image
        return converted.view(dim)

    def _rgb_to_xyz(self, image: torch.Tensor) -> torch.Tensor:
        """RGB to XYZ conversion.

        Parameters
        ----------
        image
            The image tensor in RGB format

        Returns
        -------
        The image tensor in XYZ format
        """
        return self._rgb_xyz_rgb(image, self._rgb_xyz_map[0])

    @classmethod
    def _srgb_to_rgb(cls, image: torch.Tensor) -> torch.Tensor:
        """SRGB to RGB conversion.

        Notes
        -----
        RGB Image is clipped to a small epsilon to stabilize training

        Parameters
        ----------
        image
            The image tensor in SRGB format

        Returns
        -------
        The image tensor in RGB format
        """
        limit = 0.04045
        return torch.where(image > limit,
                           ((torch.clamp(image, min=limit) + 0.055) / 1.055) ** 2.4,
                           image / 12.92)

    def _srgb_to_ycxcz(self, image: torch.Tensor) -> torch.Tensor:
        """SRGB to YcXcZ conversion.

        Parameters
        ----------
        image
            The image tensor in SRGB format

        Returns
        -------
        The image tensor in YcXcZ format
        """
        converted = self._srgb_to_rgb(image)
        converted = self._rgb_to_xyz(converted)
        return self._xyz_to_ycxcz(converted)

    def _xyz_to_lab(self, image: torch.Tensor) -> torch.Tensor:
        """XYZ to LAB conversion.

        Parameters
        ----------
        image
            The image tensor in XYZ format

        Returns
        -------
        The image tensor in LAB format
        """
        image = image * self._inv_ref_illuminant
        delta = 6 / 29
        delta_cube = delta ** 3
        factor = 1 / (3 * (delta ** 2))

        clamped_term = torch.clamp(image, min=delta_cube) ** (1.0 / 3.0)
        div = factor * image + (4 / 29)

        image = torch.where(image > delta_cube, clamped_term, div)
        return torch.cat([116 * image[:, 1:2] - 16.,
                          500 * (image[:, 0:1] - image[:, 1:2]),
                          200 * (image[:, 1:2] - image[:, 2:3])],
                         dim=1)

    def _xyz_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        """XYZ to YcXcZ conversion.

        Parameters
        ----------
        image
            The image tensor in XYZ format

        Returns
        -------
        The image tensor in RGB format
        """
        return self._rgb_xyz_rgb(image, self._rgb_xyz_map[1])

    def _xyz_to_ycxcz(self, image: torch.Tensor) -> torch.Tensor:
        """XYZ to YcXcZ conversion.

        Parameters
        ----------
        image
            The image tensor in XYZ format

        Returns
        -------
        The image tensor in YcXcZ format
        """
        image = image * self._inv_ref_illuminant
        return torch.cat([116 * image[:, 1:2] - 16.,
                          500 * (image[:, 0:1] - image[:, 1:2]),
                          200 * (image[:, 1:2] - image[:, 2:3])],
                         dim=1)

    def _ycxcz_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        """YcXcZ to RGB conversion.

        Parameters
        ----------
        image
            The image tensor in YcXcZ format

        Returns
        -------
        The image tensor in RGB format
        """
        converted = self._ycxcz_to_xyz(image)
        return self._xyz_to_rgb(converted)

    def _ycxcz_to_xyz(self, image: torch.Tensor) -> torch.Tensor:
        """YcXcZ to XYZ conversion.

        Parameters
        ----------
        image
            The image tensor in YcXcZ format

        Returns
        -------
        The image tensor in XYZ format
        """
        ch_y = (image[:, 0:1] + 16.) / 116
        return torch.cat([ch_y + (image[:, 1:2] / 500.),
                          ch_y,
                          ch_y - (image[:, 2:3] / 200.)],
                         dim=1) * self._ref_illuminant

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Call the color-space conversion function.

        Parameters
        ----------
        image
            The image tensor in the color-space defined by :attr:`from_space`

        Returns
        -------
        The image tensor in the color-space defined by :attr:`to_space`
        """
        return self._func(image)


__all__ = get_module_objects(__name__)
