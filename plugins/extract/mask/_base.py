#!/usr/bin/env python3
""" Base class for Face Masker plugins
    Plugins should inherit from this class

    See the override methods for which methods are required.

    The plugin will receive a dict containing:
    {"filename": <filename of source frame>,
     "image": <source image>,
     "detected_faces": <list of bounding box dicts from lib/plugins/extract/detect/_base>}

    For each source item, the plugin must pass a dict to finalize containing:
    {"filename": <filename of source frame>,
     "image": <four channel source image>,
     "detected_faces": <list of bounding box dicts from lib/plugins/extract/detect/_base>
    """

import logging
import os
import traceback
import cv2
import numpy as np
import keras

from io import StringIO
from lib.faces_detect import DetectedFace
from lib.aligner import Extract
from plugins.extract._base import Extractor, logger

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name


class Masker(Extractor):
    """ Aligner plugin _base Object

    All Aligner plugins must inherit from this class

    Parameters
    ----------
    git_model_id: int
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information
    model_filename: str
        The name of the model file to be loaded
    normalize_method: {`None`, 'clahe', 'hist', 'mean'}, optional
        Normalize the images fed to the aligner. Default: ``None``

    Other Parameters
    ----------------
    configfile: str, optional
        Path to a custom configuration ``ini`` file. Default: Use system configfile

    See Also
    --------
    plugins.extract.align : Aligner plugins
    plugins.extract._base : Parent class for all extraction plugins
    plugins.extract.detect._base : Detector parent class for extraction plugins.
    plugins.extract.align._base : Aligner parent class for extraction plugins.
    """

    def __init__(self, git_model_id=None, model_filename=None,
                 configfile=None, input_size=256, output_size=256, coverage_ratio=1.):
        logger.debug("Initializing %s: (configfile: %s, input_size: %s, "
                     "output_size: %s, coverage_ratio: %s)",
                     self.__class__.__name__, configfile, input_size, output_size, coverage_ratio)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile)
        self.input_size = input_size
        self.output_size = output_size
        self.coverage_ratio = coverage_ratio
        self.extract = Extract()

        self._plugin_type = "mask"
        self._faces_per_filename = dict()  # Tracking for recompiling face batches
        self._rollover = []  # Items that are rolled over from the previous batch in get_batch
        self._output_faces = []
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_batch(self, queue):
        """ Get items for inputting into the aligner from the queue in batches

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        To ensure consistent batchsizes for aligner the items are split into separate items for
        each :class:`lib.faces_detect.DetectedFace` object.

        Remember to put ``'EOF'`` to the out queue after processing
        the final batch

        Outputs items in the following format. All lists are of length
        :attr:`~plugins.extract._base.Extractor.batchsize`:

        >>> {'filename': [<filenames of source frames>],
        >>>  'image': [<source images>],
        >>>  'detected_faces': [[<lib.faces_detect.DetectedFace objects]]}

        Parameters
        ----------
        queue : queue.Queue()
            The ``queue`` that the plugin will be fed from.

        Returns
        -------
        exhausted, bool
            ``True`` if queue is exhausted, ``False`` if not
        batch, dict
            A dictionary of lists of :attr:`~plugins.extract._base.Extractor.batchsize`:
        """
        exhausted = False
        batch = dict()
        idx = 0
        while idx < self.batchsize:
            item = self._collect_item(queue)
            if item == "EOF":
                logger.trace("EOF received")
                exhausted = True
                break
            # Put frames with no faces into the out queue to keep TQDM consistent
            if not item["detected_faces"]:
                self._queues["out"].put(item)
                continue
            for f_idx, face in enumerate(item["detected_faces"]):
                batch.setdefault("detected_faces", []).append(face)
                batch.setdefault("filename", []).append(item["filename"])
                batch.setdefault("image", []).append(item["image"])
                idx += 1
                if idx == self.batchsize:
                    frame_faces = len(item["detected_faces"])
                    if f_idx + 1 != frame_faces:
                        self._rollover = {k: v[f_idx + 1:] if k == "detected_faces" else v
                                          for k, v in item.items()}
                        logger.trace("Rolled over %s faces of %s to next batch for '%s'",
                                     len(self._rollover["detected_faces"]),
                                     frame_faces, item["filename"])
                    break
        if batch:
            logger.trace("Returning batch: %s", {k: v.shape if isinstance(v, np.ndarray) else v
                                                 for k, v in batch.items()})
        else:
            logger.trace(item)
        return exhausted, batch

    def _collect_item(self, queue):
        """ Collect the item from the _rollover dict or from the queue
            Add face count per frame to self._faces_per_filename for joining
            batches back up in finalize """
        if self._rollover:
            logger.trace("Getting from _rollover: (filename: `%s`, faces: %s)",
                         self._rollover["filename"], len(self._rollover["detected_faces"]))
            item = self._rollover
            self._rollover = dict()
        else:
            item = self._get_item(queue)
            if item != "EOF":
                logger.trace("Getting from queue: (filename: %s, faces: %s)",
                             item["filename"], len(item["detected_faces"]))
                self._faces_per_filename[item["filename"]] = len(item["detected_faces"])
        return item

    def _predict(self, batch):
        """ Just return the aligner's predict function """
        return self.predict(batch)

    def finalize(self, batch):
        """ Finalize the output from Aligner

        This should be called as the final task of each `plugin`.

        It strips unneeded items from the :attr:`batch` ``dict`` and pairs the detected faces back
        up with their original frame before yielding each frame.

        Outputs items in the format:

        >>> {'image': [<original frame>],
        >>>  'filename': [<frame filename>),
        >>>  'detected_faces': [<lib.faces_detect.DetectedFace objects>]}

        Parameters
        ----------
        batch : dict
            The final ``dict`` from the `plugin` process. It must contain the `keys`:
            ``detected_faces``, ``landmarks``, ``filename``, ``image``

        Yields
        ------
        dict
            A ``dict`` for each frame containing the ``image``, ``filename`` and list of
            :class:`lib.faces_detect.DetectedFace` objects.

        """
        self._remove_invalid_keys(batch, ("detected_faces", "filename", "image"))
        logger.trace("Item out: %s", {key: val
                                      for key, val in batch.items()
                                      if key != "image"})
        for filename, image, face in zip(batch["filename"],
                                         batch["image"],
                                         batch["detected_faces"]):
            self._output_faces.append(face)
            if len(self._output_faces) != self._faces_per_filename[filename]:
                continue
            retval = dict(filename=filename, image=image, detected_faces=self._output_faces)

            self._output_faces = []
            logger.trace("Yielding: (filename: '%s', image: %s, detected_faces: %s)",
                         retval["filename"], retval["image"].shape, len(retval["detected_faces"]))
            yield retval

    # <<< PROTECTED ACCESS METHODS >>> #
    @staticmethod
    def _resize(image, target_size):
        """ resize input and output of mask models appropriately """
        height, width, channels = image.shape
        image_size = max(height, width)
        scale = target_size / image_size
        if scale == 1.:
            return image
        method = cv2.INTER_CUBIC if scale > 1. else cv2.INTER_AREA  # pylint: disable=no-member
        resized = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=method)
        resized = resized if channels > 1 else resized[..., None]
        return resized

    @staticmethod
    def postprocessing(mask):
        """ Post-processing of Nirkin style segmentation masks """
        # Select_largest_segment
        if pop_small_segments:
            results = cv2.connectedComponentsWithStats(mask,  # pylint: disable=no-member
                                                       4,
                                                       cv2.CV_32S)  # pylint: disable=no-member
            _, labels, stats, _ = results
            segments_ranked_by_area = np.argsort(stats[:, -1])[::-1]
            mask[labels != segments_ranked_by_area[0, 0]] = 0.

        # Smooth contours
        if smooth_contours:
            iters = 2
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,  # pylint: disable=no-member
                                               (5, 5))
            cv2.morphologyEx(mask, cv2.MORPH_OPEN,  # pylint: disable=no-member
                             kernel, iterations=iters)
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  # pylint: disable=no-member
                             kernel, iterations=iters)
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  # pylint: disable=no-member
                             kernel, iterations=iters)
            cv2.morphologyEx(mask, cv2.MORPH_OPEN,  # pylint: disable=no-member
                             kernel, iterations=iters)

        # Fill holes
        if fill_holes:
            not_holes = mask.copy()
            not_holes = np.pad(not_holes, ((2, 2), (2, 2), (0, 0)), 'constant')
            cv2.floodFill(not_holes, None, (0, 0), 255)  # pylint: disable=no-member
            holes = cv2.bitwise_not(not_holes)[2:-2, 2:-2]  # pylint: disable=no-member
            mask = cv2.bitwise_or(mask, holes)  # pylint: disable=no-member
            mask = np.expand_dims(mask, axis=-1)
        return mask
