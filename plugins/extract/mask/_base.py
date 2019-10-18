#!/usr/bin/env python3
""" Base class for Face Masker plugins

Plugins should inherit from this class

See the override methods for which methods are required.

The plugin will receive a dict containing:

>>> {"filename": <filename of source frame>,
>>>  "image": <source image>,
>>>  "detected_faces": <list of bounding box dicts from lib/plugins/extract/detect/_base>}

For each source item, the plugin must pass a dict to finalize containing:

>>> {"filename": <filename of source frame>,
>>>  "image": <four channel source image>,
>>>  "detected_faces": <list of bounding box dicts from lib/plugins/extract/detect/_base>}
"""

import cv2
import numpy as np

from plugins.extract._base import Extractor, logger


class Masker(Extractor):  # pylint:disable=abstract-method
    """ Masker plugin _base Object

    All Masker plugins must inherit from this class

    Parameters
    ----------
    git_model_id: int
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information
    model_filename: str
        The name of the model file to be loaded

    Other Parameters
    ----------------
    configfile: str, optional
        Path to a custom configuration ``ini`` file. Default: Use system configfile

    Attributes
    ----------
    blur_kernel, int
        The size of the kernel for applying gaussian blur to the output of the mask

    See Also
    --------
    plugins.extract.align : Aligner plugins
    plugins.extract._base : Parent class for all extraction plugins
    plugins.extract.detect._base : Detector parent class for extraction plugins.
    plugins.extract.align._base : Aligner parent class for extraction plugins.
    """

    def __init__(self, git_model_id=None, model_filename=None, configfile=None):
        logger.debug("Initializing %s: (configfile: %s, )", self.__class__.__name__, configfile)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile)
        self.input_size = 256  # Override for model specific input_size
        self.blur_kernel = 5  # Override for model specific blur_kernel size
        self.coverage_ratio = 1.0  # Override for model specific coverage_ratio

        self._plugin_type = "mask"
        self._storage_name = self.__module__.split(".")[-1].replace("_", "-")
        self._storage_size = 128  # Size to store masks at. Leave this at default
        self._faces_per_filename = dict()  # Tracking for recompiling face batches
        self._rollover = []  # Items that are rolled over from the previous batch in get_batch
        self._output_faces = []
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_batch(self, queue):
        """ Get items for inputting into the masker from the queue in batches

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        To ensure consistent batch sizes for masker the items are split into separate items for
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
                face.image = self._convert_color(item["image"])
                face.load_feed_face(face.image,
                                    size=self.input_size,
                                    coverage_ratio=1.0,
                                    dtype="float32")
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
        """ Just return the masker's predict function """
        return self.predict(batch)

    def finalize(self, batch):
        """ Finalize the output from Masker

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
            ``detected_faces``, ``filename``, ``image``

        Yields
        ------
        dict
            A ``dict`` for each frame containing the ``image``, ``filename`` and list of
            :class:`lib.faces_detect.DetectedFace` objects.

        """
        # TODO Migrate these settings to retrieval rather than storage
        # if self.blur_kernel is not None:
        #    predicted = np.array([cv2.GaussianBlur(mask, (self.blur_kernel, self.blur_kernel), 0)
        #                          for mask in batch["prediction"]])
        # else:
        #    predicted = batch["prediction"]
        # predicted[predicted < 0.04] = 0.0
        # predicted[predicted > 0.96] = 1.0
        for mask, face in zip(batch["prediction"], batch["detected_faces"]):
            face.add_mask(self._storage_name,
                          mask,
                          face.feed_matrix,
                          (face.image.shape[1], face.image.shape[0]),
                          face.feed_interpolators[1],
                          storage_size=self._storage_size)
            face.feed = dict()

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
