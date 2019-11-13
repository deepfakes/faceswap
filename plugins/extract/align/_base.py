#!/usr/bin/env python3
""" Base class for Face Aligner plugins

All Aligner Plugins should inherit from this class.
See the override methods for which methods are required.

The plugin will receive a dict containing:

>>> {"filename": [<filename of source frame>],
>>>  "image": [<source image>],
>>>  "detected_faces": [<list of DetectedFace objects]}

For each source item, the plugin must pass a dict to finalize containing:

>>> {"filename": [<filename of source frame>],
>>>  "image": [<source image>],
>>>  "landmarks": [list of 68 point face landmarks]
>>>  "detected_faces": [<list of DetectedFace objects>]}
"""


import cv2
import numpy as np

from plugins.extract._base import Extractor, logger


class Aligner(Extractor):
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
    plugins.extract.mask._base : Masker parent class for extraction plugins.
    """

    def __init__(self, git_model_id=None, model_filename=None,
                 configfile=None, normalize_method=None):
        logger.debug("Initializing %s: (normalize_method: %s)", self.__class__.__name__,
                     normalize_method)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile)
        self.normalize_method = normalize_method

        self._plugin_type = "align"
        self._faces_per_filename = dict()  # Tracking for recompiling face batches
        self._rollover = []  # Items that are rolled over from the previous batch in get_batch
        self._output_faces = []
        logger.debug("Initialized %s", self.__class__.__name__)

    # << QUEUE METHODS >>> #
    def get_batch(self, queue):
        """ Get items for inputting into the aligner from the queue in batches

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        To ensure consistent batch sizes for aligner the items are split into separate items for
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

    # <<< FINALIZE METHODS >>> #
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

        for face, landmarks in zip(batch["detected_faces"], batch["landmarks"]):
            if not isinstance(landmarks, np.ndarray):
                landmarks = np.array(landmarks)
            face.landmarks_xy = np.rint(landmarks).astype("int32")
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

    # <<< PROTECTED METHODS >>> #
    # <<< PREDICT WRAPPER >>> #
    def _predict(self, batch):
        """ Just return the aligner's predict function """
        return self.predict(batch)

    # <<< FACE NORMALIZATION METHODS >>> #
    def _normalize_faces(self, faces):
        """ Normalizes the face for feeding into model

        The normalization method is dictated by the command line argument:
            -nh (--normalization)
        """
        if self.normalize_method is None:
            return faces
        logger.trace("Normalizing faces")
        meth = getattr(self, "_normalize_{}".format(self.normalize_method.lower()))
        faces = [meth(face) for face in faces]
        logger.trace("Normalized faces")
        return faces

    @staticmethod
    def _normalize_mean(face):
        """ Normalize Face to the Mean """
        face = face / 255.0
        for chan in range(3):
            layer = face[:, :, chan]
            layer = (layer - layer.min()) / (layer.max() - layer.min())
            face[:, :, chan] = layer
        return face * 255.0

    @staticmethod
    def _normalize_hist(face):
        """ Equalize the RGB histogram channels """
        for chan in range(3):
            face[:, :, chan] = cv2.equalizeHist(face[:, :, chan])
        return face

    @staticmethod
    def _normalize_clahe(face):
        """ Perform Contrast Limited Adaptive Histogram Equalization """
        clahe = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(4, 4))
        for chan in range(3):
            face[:, :, chan] = clahe.apply(face[:, :, chan])
        return face
