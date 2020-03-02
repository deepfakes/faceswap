#!/usr/bin/env python3
""" Plugin to blend the edges of the face between the swap and the original face. """

import cv2
import numpy as np

from ._base import Adjustment, logger


class Mask(Adjustment):
    """ Manipulations to perform to the mask that is to be applied to the output of the Faceswap
    model.

    Parameters
    ----------
    mask_type: str
        The mask type to use for this plugin
    output_size: int
        The size of the output from the Faceswap model.
    coverage_ratio: float
        The coverage ratio that the Faceswap model was trained at.
    **kwargs: dict, optional
        See the parent :class:`~plugins.convert.mask._base` for additional keyword arguments.
    """
    def __init__(self, mask_type, output_size, coverage_ratio, **kwargs):
        super().__init__(mask_type, output_size, **kwargs)
        self._do_erode = self.config.get("erosion", 0) != 0
        self._coverage_ratio = coverage_ratio

    def process(self, detected_face, predicted_mask=None):  # pylint:disable=arguments-differ
        """ Obtain the requested mask type and perform any defined mask manipulations.

        Parameters
        ----------
        detected_face: :class:`lib.faces_detect.DetectedFace`
            The DetectedFace object as returned from :class:`scripts.convert.Predictor`.
        predicted_mask: :class:`numpy.ndarray`, optional
            The predicted mask as output from the Faceswap Model, if the model was trained
            with a mask, otherwise ``None``. Default: ``None``.

        Returns
        -------
        mask: :class:`numpy.ndarray`
            The mask with all requested manipulations applied
        raw_mask: :class:`numpy.ndarray`
            The mask with no erosion/dilation applied
        """
        mask = self._get_mask(detected_face, predicted_mask)
        raw_mask = mask.copy()
        if not self.skip and self._do_erode:
            mask = self._erode(mask)
        logger.trace("mask shape: %s, raw_mask shape: %s", mask.shape, raw_mask.shape)
        return mask, raw_mask

    def _get_mask(self, detected_face, predicted_mask):
        """ Return the requested mask with any requested blurring applied.

        Parameters
        ----------
        detected_face: :class:`lib.faces_detect.DetectedFace`
            The DetectedFace object as returned from :class:`scripts.convert.Predictor`.
        predicted_mask: :class:`numpy.ndarray`
            The predicted mask as output from the Faceswap Model if the model was trained
            with a mask, otherwise ``None``

        Returns
        -------
        :class:`numpy.ndarray`
            The mask sized to Faceswap model output with any requested blurring applied.
        """
        if self.mask_type == "none":
            # Return a dummy mask if not using a mask
            mask = np.ones_like(self.dummy[:, :, 1], dtype="float32")[..., None]
        elif self.mask_type == "predicted":
            mask = predicted_mask[..., None]
        else:
            mask = detected_face.mask[self.mask_type]
            mask.set_blur_and_threshold(blur_kernel=self.config["kernel_size"],
                                        blur_type=self.config["type"],
                                        blur_passes=self.config["passes"],
                                        threshold=self.config["threshold"])
            mask = self._crop_to_coverage(mask.mask)
            mask_size = mask.shape[0]
            face_size = self.dummy.shape[0]
            if mask_size != face_size:
                interp = cv2.INTER_CUBIC if mask_size < face_size else cv2.INTER_AREA
                mask = cv2.resize(mask,
                                  self.dummy.shape[:2],
                                  interpolation=interp)[..., None]
            mask = mask.astype("float32") / 255.0
        logger.trace(mask.shape)
        return mask

    def _crop_to_coverage(self, mask):
        """ Crop the mask to the correct dimensions based on coverage ratio.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The original mask to be cropped

        Returns
        -------
        :class:`numpy.ndarray`
            The cropped mask
        """
        if self._coverage_ratio == 1.0:
            return mask
        mask_size = mask.shape[0]
        padding = round((mask_size * (1 - self._coverage_ratio)) / 2)
        mask_slice = slice(padding, mask_size - padding)
        mask = mask[mask_slice, mask_slice, :]
        logger.trace("mask_size: %s, coverage: %s, padding: %s, final shape: %s",
                     mask_size, self._coverage_ratio, padding, mask.shape)
        return mask

    # MASK MANIPULATIONS
    def _erode(self, mask):
        """ Erode or dilate mask the mask based on configuration options.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The mask to be eroded or dilated

        Returns
        -------
        :class:`numpy.ndarray`
            The mask with erosion/dilation applied
        """
        kernel = self._get_erosion_kernel(mask)
        if self.config["erosion"] > 0:
            logger.trace("Eroding mask")
            mask = cv2.erode(mask, kernel, iterations=1)
        else:
            logger.trace("Dilating mask")
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def _get_erosion_kernel(self, mask):
        """ Get the erosion kernel.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The mask to be eroded or dilated

        Returns
        -------
        :class:`numpy.ndarray`
            The erosion kernel to be used for erosion/dilation
        """
        erosion_ratio = self.config["erosion"] / 100
        mask_radius = np.sqrt(np.sum(mask)) / 2
        kernel_size = max(1, int(abs(erosion_ratio * mask_radius)))
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        logger.trace("erosion_kernel shape: %s", erosion_kernel.shape)
        return erosion_kernel
