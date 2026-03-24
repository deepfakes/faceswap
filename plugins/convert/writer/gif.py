#!/usr/bin/env python3
"""Animated GIF writer for faceswap.py converter"""
from __future__ import annotations

import logging
import os
import typing as T
from collections import deque

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree  # type:ignore[attr-defined]
from sklearn.cluster import MiniBatchKMeans

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from ._base import Output
from . import gif_defaults as cfg

if T.TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


class Writer(Output):
    """GIF output writer using PIL.


    Parameters
    ----------
    output_folder
        The folder to save the output gif to
    total_count
        The total number of frames to be converted
    frame_ranges
        List of tuples for starting and end values of each frame range to be converted or ``None``
        if all frames are to be converted
    kwargs
        Any additional standard :class:`plugins.convert.writer._base.Output` key word arguments.
    """
    def __init__(self,
                 output_folder: str,
                 total_count: int,
                 frame_ranges: list[tuple[int, int]] | None,
                 **kwargs) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(output_folder, **kwargs)
        self._frame_order: deque[int] = self._set_frame_order(total_count, frame_ranges)
        # Fix dims on 1st received frame
        self._dimensions = (0, 0)
        self._images: list[np.ndarray] = []
        self._palette: dict[int, int] = {}
        self._gif_file: str | None = None  # Set filename based on first file seen

    def _set_gif_filename(self, filename: str) -> None:
        """Set the full path to GIF output file to :attr:`_gif_file`

        The filename is the created from the source filename of the first input image received with
        `"_converted"` appended to the end and a .gif extension. If a file already exists with the
        given filename, then `"_1"` is appended to the end of the filename. This number iterates
        until a valid filename that does not exist is found.

        Parameters
        ----------
        filename
            The incoming frame filename.
        """

        logger.debug("[GIF] sample filename: '%s'", filename)
        filename = os.path.splitext(os.path.basename(filename))[0]
        snip = len(filename)
        for char in list(filename[::-1]):
            if not char.isdigit() and char not in ("_", "-"):
                break
            snip -= 1
        filename = filename[:snip]

        idx = 0
        while True:
            out_file = f"{filename}_converted{'' if idx == 0 else f'_{idx}'}.gif"
            retval = os.path.join(self.output_folder, out_file)
            if not os.path.exists(retval):
                break
            idx += 1

        self._gif_file = retval
        logger.info("[GIF] Outputting to: '%s'", self._gif_file)

    def _save_from_cache(self) -> None:
        """Writes any consecutive frames to the GIF container that are ready to be output
        from the cache."""
        while self._frame_order:
            if self._frame_order[0] not in self.cache:
                logger.trace(  # type: ignore[attr-defined]
                    "[GIF] Next frame not ready. Continuing")
                break
            save_no = self._frame_order.popleft()
            logger.trace("[GIF] Rendering from cache. Frame no: %s",  # type: ignore[attr-defined]
                         save_no)
            img = self.cache.pop(save_no)
            if img.size != self._dimensions:
                img = cv2.resize(img, self._dimensions)
            self._images.append(img)
        logger.trace("[GIF] Current cache size: %s", len(self.cache))  # type: ignore[attr-defined]

    def write(self, filename: str, image: npt.NDArray[np.uint8]) -> None:
        """Frames come from the pool in arbitrary order, so frames are cached for writing out
        in the correct order.

        Parameters
        ----------
        filename
            The incoming frame filename.
        image
            The converted image to be written
        """
        logger.trace(  # type: ignore[attr-defined]
            "[GIF] Received frame: (filename: '%s', shape: %s", filename, image.shape)
        dimensions = (image.shape[1], image.shape[0])
        if not self._gif_file:
            self._set_gif_filename(filename)
            self._dimensions = dimensions
        img = image[:, :, ::-1]
        self.cache_frame(filename, img)
        self._save_from_cache()

    def _build_palette(self, images: npt.NDArray[np.uint8]):
        """Obtain a color palette from the images to be saved

        Parameters
        ----------
        images
            The converted images batched into a single array
        """
        palette_size = int(cfg.palette_size())
        logger.info("[GIF] Generating palette of size %s...", palette_size)
        pixels = images.reshape(-1, 3)
        num_samples = 100000

        if pixels.shape[0] > num_samples:
            idx = np.random.choice(pixels.shape[0], num_samples, replace=False)
            pixels = pixels[idx]

        k_means = MiniBatchKMeans(n_clusters=palette_size, batch_size=4096)
        k_means.fit(pixels)

        palette = k_means.cluster_centers_.astype(np.uint8)
        return palette

    def _quantize_frame(self, mapped: np.ndarray, palette: bytes) -> Image.Image:
        """Quantize a frame and convert to PIL Image

        Parameters
        ----------
        mapped
            The mapped frame to quantize
        tree
            The K-Means tree to use for quantization
        palette
            The palette to apply to the frame

        Returns
        -------
        The quantized PIL image
        """
        img = Image.fromarray(mapped, mode='P')
        del mapped
        img.putpalette(palette)
        if cfg.dithering():
            meth = Image.FLOYDSTEINBERG  # type:ignore[attr-defined]  # pylint:disable=no-member
            img = img.convert("P", dither=meth)
        return img

    def _quantize_images(self) -> list[Image.Image]:
        """Quantize the images for writing to GIF

        Returns
        -------
        The list of quantized images
        """
        images = np.stack(self._images)
        im_shape = images.shape
        del self._images
        palette = self._build_palette(images)
        tree = cKDTree(palette)
        logger.info("[GIF] Mapping colors...")
        _, mapped_flat = tree.query(images.reshape(-1, 3))
        del images
        mapped = T.cast("npt.NDArray[np.uint8]",
                        mapped_flat.reshape(im_shape[:3]).astype(np.uint8))
        flat_palette = palette.flatten().tobytes()
        imgs = [self._quantize_frame(im, flat_palette) for im in mapped]
        return imgs

    def close(self) -> None:
        """Close the GIF writer on completion."""
        if not self._images:
            return
        assert self._gif_file is not None
        logger.info("[GIF] Creating GIF. Depending on the number of frames this may take a "
                    "while...")
        imgs = self._quantize_images()
        assert self._gif_file is not None
        logger.info("[GIF] Saving...")
        imgs[0].save(self._gif_file,
                     save_all=True,
                     append_images=imgs[1:],
                     duration=1000 / cfg.fps(),
                     loop=cfg.loop())


__all__ = get_module_objects(__name__)
