#!/usr/bin/env python3
""" Plugin to blend the edges of the face box that comes out of the Faceswap Model into the final
frame. """

import numpy as np

from lib.faces_detect import BlurMask
from ._base import Adjustment, logger


class Mask(Adjustment):
    """ Manipulations to perform on the edges of the box that is received from the Faceswap model.

    As the size of the box coming out of the model is identical for every face, the mask to be
    applied is just calculated once (at launch).

    Parameters
    ----------
    output_size: int
        The size of the output from the Faceswap model.
    **kwargs: dict, optional
        See the parent :class:`~plugins.convert.mask._base` for additional keyword arguments.
    """
    def __init__(self, output_size, **kwargs):
        super().__init__("none", output_size, **kwargs)
        self.mask = self._get_mask() if not self.skip else None

    def _get_mask(self):
        """ Create a mask to be used at the edges of the face box.

        The box for every face will be identical, so the mask is set just once on initialization.
        As gaussian blur technically blurs both sides of the mask, the mask ratio is reduced by
        half to give a more expected box.

        Returns
        -------
        :class:`numpy.ndarray`
            The mask to be used at the edges of the box output from the Faceswap model
        """
        logger.debug("Building box mask")
        mask_ratio = self.config["distance"] / 200
        facesize = self.dummy.shape[0]
        erode = slice(round(facesize * mask_ratio), -round(facesize * mask_ratio))
        mask = self.dummy[:, :, -1]
        mask[erode, erode] = 1.0

        mask = BlurMask(self.config["type"],
                        mask,
                        self.config["radius"],
                        is_ratio=True,
                        passes=self.config["passes"]).blurred
        logger.debug("Built box mask. Shape: %s", mask.shape)
        return mask

    def process(self, new_face):  # pylint:disable=arguments-differ
        """ Apply the box mask to the swapped face.

        Parameters
        ----------
        new_face: :class:`numpy.ndarray`
            The swapped face that has been output from the Faceswap model

        Returns
        -------
        :class:`numpy.ndarray`
            The input face is returned with the box mask added to the alpha channel if a blur type
            has been specified in the plugin configuration. If this configuration is set to
            ``None`` then the input face is returned with no mask applied.
        """
        if self.skip:
            logger.trace("Skipping blend box")
            return new_face

        logger.trace("Blending box")
        new_face = np.concatenate((new_face, self.mask), axis=-1)
        logger.trace("Blended box")
        return new_face
