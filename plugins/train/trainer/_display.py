#!/usr/bin/env python3
""" Handles the creation of display images for preview window and timelapses """
from __future__ import annotations

import logging
import time
import typing as T
import os

import cv2
import numpy as np
import torch

from lib.image import hex_to_rgb
from lib.utils import get_folder, get_image_paths, get_module_objects
from plugins.train import train_config as cfg

if T.TYPE_CHECKING:
    from keras import KerasTensor
    from lib.training import Feeder
    from plugins.train.model._base import ModelBase

logger = logging.getLogger(__name__)


class Samples():
    """ Compile samples for display for preview and time-lapse

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    coverage_ratio: float
        Ratio of face to be cropped out of the training image.
    mask_opacity: int
        The opacity (as a percentage) to use for the mask overlay
    mask_color: str
        The hex RGB value to use the mask overlay

    Attributes
    ----------
    images: dict
        The :class:`numpy.ndarray` training images for generating previews on each side. The
        dictionary should contain 2 keys ("a" and "b") with the values being the training images
        for generating samples corresponding to each side.
    """
    def __init__(self,
                 model: ModelBase,
                 coverage_ratio: float,
                 mask_opacity: int,
                 mask_color: str) -> None:
        logger.debug("Initializing %s: model: '%s', coverage_ratio: %s, mask_opacity: %s, "
                     "mask_color: %s)",
                     self.__class__.__name__, model, coverage_ratio, mask_opacity, mask_color)
        self._model = model
        self._display_mask = cfg.Loss.learn_mask() or cfg.Loss.penalized_mask_loss()
        self.images: dict[T.Literal["a", "b"], list[np.ndarray]] = {}
        self._coverage_ratio = coverage_ratio
        self._mask_opacity = mask_opacity / 100.0
        self._mask_color = np.array(hex_to_rgb(mask_color))[..., 2::-1] / 255.
        logger.debug("Initialized %s", self.__class__.__name__)

    def toggle_mask_display(self) -> None:
        """ Toggle the mask overlay on or off depending on user input. """
        if not (cfg.Loss.learn_mask() or cfg.Loss.penalized_mask_loss()):
            return
        display_mask = not self._display_mask
        print("\x1b[2K", end="\r")  # Clear last line
        logger.info("Toggling mask display %s...", "on" if display_mask else "off")
        self._display_mask = display_mask

    def show_sample(self) -> np.ndarray:
        """ Compile a preview image.

        Returns
        -------
        :class:`numpy.ndarry`
            A compiled preview image ready for display or saving
        """
        logger.debug("Showing sample")
        feeds: dict[T.Literal["a", "b"], np.ndarray] = {}
        for idx, side in enumerate(T.get_args(T.Literal["a", "b"])):
            feed = self.images[side][0]
            input_shape = self._model.model.input_shape[idx][1:]
            if input_shape[0] / feed.shape[1] != 1.0:
                feeds[side] = self._resize_sample(side, feed, input_shape[0])
            else:
                feeds[side] = feed

        preds = self._get_predictions(feeds["a"], feeds["b"])
        return self._compile_preview(preds)

    @classmethod
    def _resize_sample(cls,
                       side: T.Literal["a", "b"],
                       sample: np.ndarray,
                       target_size: int) -> np.ndarray:
        """ Resize a given image to the target size.

        Parameters
        ----------
        side: str
            The side ("a" or "b") that the samples are being generated for
        sample: :class:`numpy.ndarray`
            The sample to be resized
        target_size: int
            The size that the sample should be resized to

        Returns
        -------
        :class:`numpy.ndarray`
            The sample resized to the target size
        """
        scale = target_size / sample.shape[1]
        if scale == 1.0:
            # cv2 complains if we don't do this :/
            return np.ascontiguousarray(sample)
        logger.debug("Resizing sample: (side: '%s', sample.shape: %s, target_size: %s, scale: %s)",
                     side, sample.shape, target_size, scale)
        interpn = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        retval = np.array([cv2.resize(img, (target_size, target_size), interpolation=interpn)
                           for img in sample])
        logger.debug("Resized sample: (side: '%s' shape: %s)", side, retval.shape)
        return retval

    def _filter_multiscale_output(self, standard: list[KerasTensor], swapped: list[KerasTensor]
                                  ) -> tuple[list[KerasTensor], list[KerasTensor]]:
        """ Only return the largest predictions if the model has multi-scaled output

        Parameters
        ----------
        standard: list[:class:`keras.KerasTensor`]
            The standard output from the model
        swapped: list[:class:`keras.KerasTensor`]
            The swapped output from the model

        Returns
        -------
        standard: list[:class:`keras.KerasTensor`]
            The standard output from the model, filtered to just the largest output
        swapped: list[:class:`keras.KerasTensor`]
            The swapped output from the model, filtered to just the largest output
        """
        sizes = T.cast(set[int], set(p.shape[1] for p in standard))
        if len(sizes) == 1:
            return standard, swapped
        logger.debug("Received outputs. standard: %s, swapped: %s",
                     [s.shape for s in standard], [s.shape for s in swapped])
        logger.debug("Stripping multi-scale outputs for sizes %s", sizes)
        standard = [s for s in standard if s.shape[1] == max(sizes)]
        swapped = [s for s in swapped if s.shape[1] == max(sizes)]
        logger.debug("Stripped outputs. standard: %s, swapped: %s",
                     [s.shape for s in standard], [s.shape for s in swapped])
        return standard, swapped

    def _collate_output(self, standard: list[torch.Tensor], swapped: list[torch.Tensor]
                        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """ Merge the mask onto the preview image's 4th channel if learn mask is selected.
        Return as numpy array

        Parameters
        ----------
        standard: list[:class:`torch.Tensor`]
            The standard output from the model
        swapped: list[:class:`torch.Tensor`]
            The swapped output from the model

        Returns
        -------
        standard: list[:class:`numpy.ndarray`]
            The standard output from the model, with mask merged
        swapped: list[:class:`numpy.ndarray`]
            The swapped output from the model, with mask merged
        """
        logger.debug("Received tensors. standard: %s, swapped: %s",
                     [s.shape for s in standard], [s.shape for s in swapped])

        # Pull down outputs
        nstandard = [p.cpu().detach().numpy() for p in standard]
        nswapped = [p.cpu().detach().numpy() for p in swapped]

        if cfg.Loss.learn_mask():  # Add mask to 4th channel of final output
            nstandard = [np.concatenate(nstandard[idx * 2: (idx * 2) + 2], axis=-1)
                         for idx in range(2)]
            nswapped = [np.concatenate(nswapped[idx * 2: (idx * 2) + 2], axis=-1)
                        for idx in range(2)]
        logger.debug("Collated output. standard: %s, swapped: %s",
                     [(s.shape, s.dtype) for s in nstandard],
                     [(s.shape, s.dtype) for s in nswapped])
        return nstandard, nswapped

    def _get_predictions(self, feed_a: np.ndarray, feed_b: np.ndarray
                         ) -> dict[T.Literal["a_a", "a_b", "b_b", "b_a"], np.ndarray]:
        """ Feed the samples to the model and return predictions

        Parameters
        ----------
        feed_a: :class:`numpy.ndarray`
            Feed images for the "a" side
        feed_a: :class:`numpy.ndarray`
            Feed images for the "b" side

        Returns
        -------
        list:
            List of :class:`numpy.ndarray` of predictions received from the model
        """
        logger.debug("Getting Predictions")
        preds: dict[T.Literal["a_a", "a_b", "b_b", "b_a"], np.ndarray] = {}

        with torch.inference_mode():
            standard = self._model.model([feed_a, feed_b])
            swapped = self._model.model([feed_b, feed_a])

        standard, swapped = self._filter_multiscale_output(standard, swapped)
        standard, swapped = self._collate_output(standard, swapped)

        preds["a_a"] = standard[0]
        preds["b_b"] = standard[1]
        preds["a_b"] = swapped[0]
        preds["b_a"] = swapped[1]

        logger.debug("Returning predictions: %s", {key: val.shape for key, val in preds.items()})
        return preds

    def _compile_preview(self, predictions: dict[T.Literal["a_a", "a_b", "b_b", "b_a"], np.ndarray]
                         ) -> np.ndarray:
        """ Compile predictions and images into the final preview image.

        Parameters
        ----------
        predictions: dict[Literal["a_a", "a_b", "b_b", "b_a"], np.ndarray
            The predictions from the model

        Returns
        -------
        :class:`numpy.ndarry`
            A compiled preview image ready for display or saving
        """
        figures: dict[T.Literal["a", "b"], np.ndarray] = {}
        headers: dict[T.Literal["a", "b"], np.ndarray] = {}

        for side, samples in self.images.items():
            other_side = "a" if side == "b" else "b"
            preds = [predictions[T.cast(T.Literal["a_a", "a_b", "b_b", "b_a"],
                                        f"{side}_{side}")],
                     predictions[T.cast(T.Literal["a_a", "a_b", "b_b", "b_a"],
                                        f"{other_side}_{side}")]]
            display = self._to_full_frame(side, samples, preds)
            headers[side] = self._get_headers(side, display[0].shape[1])
            figures[side] = np.stack([display[0], display[1], display[2], ], axis=1)
            if self.images[side][1].shape[0] % 2 == 1:
                figures[side] = np.concatenate([figures[side],
                                                np.expand_dims(figures[side][0], 0)])

        width = 4
        if width // 2 != 1:
            headers = self._duplicate_headers(headers, width // 2)

        header = np.concatenate([headers["a"], headers["b"]], axis=1)
        figure = np.concatenate([figures["a"], figures["b"]], axis=0)
        height = int(figure.shape[0] / width)
        figure = figure.reshape((width, height) + figure.shape[1:])
        figure = _stack_images(figure)
        figure = np.concatenate((header, figure), axis=0)

        logger.debug("Compiled sample")
        return np.clip(figure * 255, 0, 255).astype('uint8')

    def _to_full_frame(self,
                       side: T.Literal["a", "b"],
                       samples: list[np.ndarray],
                       predictions: list[np.ndarray]) -> list[np.ndarray]:
        """ Patch targets and prediction images into images of model output size.

        Parameters
        ----------
        side: {"a" or "b"}
            The side that these samples are for
        samples: list
            List of :class:`numpy.ndarray` of feed images and sample images
        predictions: list
            List of :class: `numpy.ndarray` of predictions from the model

        Returns
        -------
        list
            The images resized and collated for display in the preview frame
        """
        logger.debug("side: '%s', number of sample arrays: %s, prediction.shapes: %s)",
                     side, len(samples), [pred.shape for pred in predictions])
        faces, full = samples[:2]

        if self._model.color_order.lower() == "rgb":  # Switch color order for RGB model display
            full = full[..., ::-1]
            faces = faces[..., ::-1]
            predictions = [pred[..., 2::-1] for pred in predictions]

        full = self._process_full(side, full, predictions[0].shape[1], (0., 0., 1.0))
        images = [faces] + predictions

        if self._display_mask:
            images = self._compile_masked(images, samples[-1])
        elif cfg.Loss.learn_mask():
            # Remove masks when learn mask is selected but mask toggle is off
            images = [batch[..., :3] for batch in images]

        images = [self._overlay_foreground(full.copy(), image) for image in images]

        return images

    def _process_full(self,
                      side: T.Literal["a", "b"],
                      images: np.ndarray,
                      prediction_size: int,
                      color: tuple[float, float, float]) -> np.ndarray:
        """ Add a frame overlay to preview images indicating the region of interest.

        This applies the red border that appears in the preview images.

        Parameters
        ----------
        side: {"a" or "b"}
            The side that these samples are for
        images: :class:`numpy.ndarray`
            The input training images to to process
        prediction_size: int
            The size of the predicted output from the model
        color: tuple
            The (Blue, Green, Red) color to use for the frame

        Returns
        -------
        :class:`numpy,ndarray`
            The input training images, sized for output and annotated for coverage
        """
        logger.debug("full_size: %s, prediction_size: %s, color: %s",
                     images.shape[1], prediction_size, color)

        display_size = int((prediction_size / self._coverage_ratio // 2) * 2)
        images = self._resize_sample(side, images, display_size)  # Resize targets to display size
        padding = (display_size - prediction_size) // 2
        if padding == 0:
            logger.debug("Resized background. Shape: %s", images.shape)
            return images

        length = display_size // 4
        t_l, b_r = (padding - 1, display_size - padding)
        for img in images:
            cv2.rectangle(img, (t_l, t_l), (t_l + length, t_l + length), color, 1)
            cv2.rectangle(img, (b_r, t_l), (b_r - length, t_l + length), color, 1)
            cv2.rectangle(img, (b_r, b_r), (b_r - length, b_r - length), color, 1)
            cv2.rectangle(img, (t_l, b_r), (t_l + length, b_r - length), color, 1)
        logger.debug("Overlayed background. Shape: %s", images.shape)
        return images

    def _compile_masked(self, faces: list[np.ndarray], masks: np.ndarray) -> list[np.ndarray]:
        """ Add the mask to the faces for masked preview.

        Places an opaque red layer over areas of the face that are masked out.

        Parameters
        ----------
        faces: list
            The :class:`numpy.ndarray` sample faces and predictions that are to have the mask
            applied
        masks: :class:`numpy.ndarray`
            The masks that are to be applied to the faces

        Returns
        -------
        list
            List of :class:`numpy.ndarray` faces with the opaque mask layer applied
        """
        orig_masks = 1. - masks
        masks3: list[np.ndarray] | np.ndarray = []

        if faces[-1].shape[-1] == 4:  # Mask contained in alpha channel of predictions
            pred_masks = [1. - face[..., -1][..., None] for face in faces[-2:]]
            faces[-2:] = [face[..., :-1] for face in faces[-2:]]
            masks3 = [orig_masks, *pred_masks]
        else:
            masks3 = np.repeat(np.expand_dims(orig_masks, axis=0), 3, axis=0)

        retval: list[np.ndarray] = []
        overlays3 = np.ones_like(faces) * self._mask_color
        for previews, overlays, compiled_masks in zip(faces, overlays3, masks3):
            compiled_masks *= self._mask_opacity
            overlays *= compiled_masks
            previews *= (1. - compiled_masks)
            retval.append(previews + overlays)
        logger.debug("masked shapes: %s", [faces.shape for faces in retval])
        return retval

    @classmethod
    def _overlay_foreground(cls, backgrounds: np.ndarray, foregrounds: np.ndarray) -> np.ndarray:
        """ Overlay the preview images into the center of the background images

        Parameters
        ----------
        backgrounds: :class:`numpy.ndarray`
            Background images for placing the preview images onto
        backgrounds: :class:`numpy.ndarray`
            Preview images for placing onto the background images

        Returns
        -------
        :class:`numpy.ndarray`
            The preview images compiled into the full frame size for each preview
        """
        offset = (backgrounds.shape[1] - foregrounds.shape[1]) // 2
        for foreground, background in zip(foregrounds, backgrounds):
            background[offset:offset + foreground.shape[0],
                       offset:offset + foreground.shape[1], :3] = foreground
        logger.debug("Overlayed foreground. Shape: %s", backgrounds.shape)
        return backgrounds

    @classmethod
    def _get_headers(cls, side: T.Literal["a", "b"], width: int) -> np.ndarray:
        """ Set header row for the final preview frame

        Parameters
        ----------
        side: {"a" or "b"}
            The side that the headers should be generated for
        width: int
            The width of each column in the preview frame

        Returns
        -------
        :class:`numpy.ndarray`
            The column headings for the given side
        """
        logger.debug("side: '%s', width: %s",
                     side, width)
        titles = ("Original", "Swap") if side == "a" else ("Swap", "Original")
        height = int(width / 4.5)
        total_width = width * 3
        logger.debug("height: %s, total_width: %s", height, total_width)
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = [f"{titles[0]} ({side.upper()})",
                 f"{titles[0]} > {titles[0]}",
                 f"{titles[0]} > {titles[1]}"]
        scaling = (width / 144) * 0.45
        text_sizes = [cv2.getTextSize(texts[idx], font, scaling, 1)[0]
                      for idx in range(len(texts))]
        text_y = int((height + text_sizes[0][1]) / 2)
        text_x = [int((width - text_sizes[idx][0]) / 2) + width * idx
                  for idx in range(len(texts))]
        logger.debug("texts: %s, text_sizes: %s, text_x: %s, text_y: %s",
                     texts, text_sizes, text_x, text_y)
        header_box = np.ones((height, total_width, 3), np.float32)
        for idx, text in enumerate(texts):
            cv2.putText(header_box,
                        text,
                        (text_x[idx], text_y),
                        font,
                        scaling,
                        (0, 0, 0),
                        1,
                        lineType=cv2.LINE_AA)
        logger.debug("header_box.shape: %s", header_box.shape)
        return header_box

    @classmethod
    def _duplicate_headers(cls,
                           headers: dict[T.Literal["a", "b"], np.ndarray],
                           columns: int) -> dict[T.Literal["a", "b"], np.ndarray]:
        """ Duplicate headers for the number of columns displayed for each side.

        Parameters
        ----------
        headers: dict
            The headers to be duplicated for each side
        columns: int
            The number of columns that the header needs to be duplicated for

        Returns
        -------
        :class:dict
            The original headers duplicated by the number of columns for each side
        """
        for side, header in headers.items():
            duped = tuple(header for _ in range(columns))
            headers[side] = np.concatenate(duped, axis=1)
            logger.debug("side: %s header.shape: %s", side, header.shape)
        return headers


class Timelapse():
    """ Create a time-lapse preview image.

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    coverage_ratio: float
        Ratio of face to be cropped out of the training image.
    image_count: int
        The number of preview images to be displayed in the time-lapse
    mask_opacity: int
        The opacity (as a percentage) to use for the mask overlay
    mask_color: str
        The hex RGB value to use the mask overlay
    feeder: :class:`~lib.training.generator.Feeder`
        The feeder for generating the time-lapse images.
    image_paths: dict
        The full paths to the training images for each side of the model
    """
    def __init__(self,
                 model: ModelBase,
                 coverage_ratio: float,
                 image_count: int,
                 mask_opacity: int,
                 mask_color: str,
                 feeder: Feeder,
                 image_paths: dict[T.Literal["a", "b"], list[str]]) -> None:
        logger.debug("Initializing %s: model: %s, coverage_ratio: %s, image_count: %s, "
                     "mask_opacity: %s, mask_color: %s, feeder: %s, image_paths: %s)",
                     self.__class__.__name__, model, coverage_ratio, image_count, mask_opacity,
                     mask_color, feeder, len(image_paths))
        self._num_images = image_count
        self._samples = Samples(model, coverage_ratio, mask_opacity, mask_color)
        self._model = model
        self._feeder = feeder
        self._image_paths = image_paths
        self._output_file = ""
        logger.debug("Initialized %s", self.__class__.__name__)

    def _setup(self, input_a: str, input_b: str, output: str) -> None:
        """ Setup the time-lapse folder locations and the time-lapse feed.

        Parameters
        ----------
        input_a: str
            The full path to the time-lapse input folder containing faces for the "a" side
        input_b: str
            The full path to the time-lapse input folder containing faces for the "b" side
        output: str, optional
            The full path to the time-lapse output folder. If ``None`` is provided this will
            default to the model folder
        """
        logger.debug("Setting up time-lapse")
        if not output:
            output = get_folder(os.path.join(str(self._model.io.model_dir),
                                             f"{self._model.name}_timelapse"))
        self._output_file = output
        logger.debug("Time-lapse output set to '%s'", self._output_file)

        # Rewrite paths to pull from the training images so mask and face data can be accessed
        images: dict[T.Literal["a", "b"], list[str]] = {}
        for side, input_ in zip(T.get_args(T.Literal["a", "b"]), (input_a, input_b)):
            training_path = os.path.dirname(self._image_paths[side][0])
            images[side] = [os.path.join(training_path, os.path.basename(pth))
                            for pth in get_image_paths(input_)]

        batchsize = min(len(images["a"]),
                        len(images["b"]),
                        self._num_images)
        self._feeder.set_timelapse_feed(images, batchsize)
        logger.debug("Set up time-lapse")

    def output_timelapse(self, timelapse_kwargs: dict[T.Literal["input_a",
                                                                "input_b",
                                                                "output"], str]) -> None:
        """ Generate the time-lapse samples and output the created time-lapse to the specified
        output folder.

        Parameters
        ----------
        timelapse_kwargs: dict:
            The keyword arguments for setting up the time-lapse. All values should be full paths
            the keys being `input_a`, `input_b`, `output`
        """
        logger.debug("Ouputting time-lapse")
        if not self._output_file:
            self._setup(**T.cast(dict[str, str], timelapse_kwargs))

        logger.debug("Getting time-lapse samples")
        self._samples.images = self._feeder.generate_preview(is_timelapse=True)
        logger.debug("Got time-lapse samples: %s",
                     {side: len(images) for side, images in self._samples.images.items()})

        image = self._samples.show_sample()
        if image is None:
            return
        filename = os.path.join(self._output_file, str(int(time.time())) + ".jpg")

        cv2.imwrite(filename, image)
        logger.debug("Created time-lapse: '%s'", filename)


def _stack_images(images: np.ndarray) -> np.ndarray:
    """ Stack images evenly for preview.

    Parameters
    ----------
    images: :class:`numpy.ndarray`
        The preview images to be stacked

    Returns
    -------
    :class:`numpy.ndarray`
        The stacked preview images
    """
    logger.debug("Stack images")

    def get_transpose_axes(num):
        if num % 2 == 0:
            logger.debug("Even number of images to stack")
            y_axes = list(range(1, num - 1, 2))
            x_axes = list(range(0, num - 1, 2))
        else:
            logger.debug("Odd number of images to stack")
            y_axes = list(range(0, num - 1, 2))
            x_axes = list(range(1, num - 1, 2))
        return y_axes, x_axes, [num - 1]

    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    logger.debug("Stacked images")
    return np.transpose(images, axes=np.concatenate(new_axes)).reshape(new_shape)


__all__ = get_module_objects(__name__)
