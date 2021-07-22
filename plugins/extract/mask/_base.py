#!/usr/bin/env python3
""" Base class for Face Masker plugins

Plugins should inherit from this class

See the override methods for which methods are required.

The plugin will receive a :class:`~plugins.extract.pipeline.ExtractMedia` object.

For each source item, the plugin must pass a dict to finalize containing:

>>> {"filename": <filename of source frame>,
>>>  "detected_faces": <list of bounding box dicts from lib/plugins/extract/detect/_base>}
"""

import cv2
import numpy as np

from tensorflow.python.framework import errors_impl as tf_errors

from lib.align import AlignedFace, transform_image
from lib.utils import get_backend, FaceswapError
from plugins.extract._base import Extractor, ExtractMedia, logger


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
    image_is_aligned: bool, optional
        Indicates that the passed in image is an aligned face rather than a frame.
        Default: ``False``

    Other Parameters
    ----------------
    configfile: str, optional
        Path to a custom configuration ``ini`` file. Default: Use system configfile

    See Also
    --------
    plugins.extract.pipeline : The extraction pipeline for calling plugins
    plugins.extract.align : Aligner plugins
    plugins.extract._base : Parent class for all extraction plugins
    plugins.extract.detect._base : Detector parent class for extraction plugins.
    plugins.extract.align._base : Aligner parent class for extraction plugins.
    """

    def __init__(self, git_model_id=None, model_filename=None, configfile=None,
                 instance=0, image_is_aligned=False, **kwargs):
        logger.debug("Initializing %s: (configfile: %s, )", self.__class__.__name__, configfile)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile,
                         instance=instance,
                         **kwargs)
        self.input_size = 256  # Override for model specific input_size
        self.coverage_ratio = 1.0  # Override for model specific coverage_ratio

        self._plugin_type = "mask"
        self._image_is_aligned = image_is_aligned
        self._storage_name = self.__module__.split(".")[-1].replace("_", "-")
        self._storage_centering = "face"  # Centering to store the mask at
        self._storage_size = 128  # Size to store masks at. Leave this at default
        self._faces_per_filename = dict()  # Tracking for recompiling face batches
        self._rollover = None  # Items that are rolled over from the previous batch in get_batch
        self._output_faces = []
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_batch(self, queue):
        """ Get items for inputting into the masker from the queue in batches

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        Items are received as :class:`~plugins.extract.pipeline.ExtractMedia` objects and converted
        to ``dict`` for internal processing.

        To ensure consistent batch sizes for masker the items are split into separate items for
        each :class:`~lib.align.DetectedFace` object.

        Remember to put ``'EOF'`` to the out queue after processing
        the final batch

        Outputs items in the following format. All lists are of length
        :attr:`~plugins.extract._base.Extractor.batchsize`:

        >>> {'filename': [<filenames of source frames>],
        >>>  'detected_faces': [[<lib.align.DetectedFace objects]]}

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
            if not item.detected_faces:
                self._queues["out"].put(item)
                continue
            for f_idx, face in enumerate(item.detected_faces):

                image = item.get_image_copy(self.color_format)
                roi = np.ones((*item.image_size[:2], 1), dtype="float32")

                if not self._image_is_aligned:
                    # Add the ROI mask to image so we can get the ROI mask with a single warp
                    image = np.concatenate([image, roi], axis=-1)

                feed_face = AlignedFace(face.landmarks_xy,
                                        image=image,
                                        centering=self._storage_centering,
                                        size=self.input_size,
                                        coverage_ratio=self.coverage_ratio,
                                        dtype="float32",
                                        is_aligned=self._image_is_aligned)

                if not self._image_is_aligned:
                    # Split roi mask from feed face alpha channel
                    roi_mask = feed_face.face[..., 3]
                    feed_face._face = feed_face.face[..., :3]  # pylint:disable=protected-access
                else:
                    # We have to do the warp here as AlignedFace did not perform it
                    roi_mask = transform_image(roi,
                                               feed_face.matrix,
                                               feed_face.size,
                                               padding=feed_face.padding)

                batch.setdefault("roi_masks", []).append(roi_mask)
                batch.setdefault("detected_faces", []).append(face)
                batch.setdefault("feed_faces", []).append(feed_face)
                batch.setdefault("filename", []).append(item.filename)
                idx += 1
                if idx == self.batchsize:
                    frame_faces = len(item.detected_faces)
                    if f_idx + 1 != frame_faces:
                        self._rollover = ExtractMedia(
                            item.filename,
                            item.image,
                            detected_faces=item.detected_faces[f_idx + 1:])
                        logger.trace("Rolled over %s faces of %s to next batch for '%s'",
                                     len(self._rollover.detected_faces), frame_faces,
                                     item.filename)
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
        if self._rollover is not None:
            logger.trace("Getting from _rollover: (filename: `%s`, faces: %s)",
                         self._rollover.filename, len(self._rollover.detected_faces))
            item = self._rollover
            self._rollover = None
        else:
            item = self._get_item(queue)
            if item != "EOF":
                logger.trace("Getting from queue: (filename: %s, faces: %s)",
                             item.filename, len(item.detected_faces))
                self._faces_per_filename[item.filename] = len(item.detected_faces)
        return item

    def _predict(self, batch):
        """ Just return the masker's predict function """
        try:
            return self.predict(batch)
        except tf_errors.ResourceExhaustedError as err:
            msg = ("You do not have enough GPU memory available to run detection at the "
                   "selected batch size. You can try a number of things:"
                   "\n1) Close any other application that is using your GPU (web browsers are "
                   "particularly bad for this)."
                   "\n2) Lower the batchsize (the amount of images fed into the model) by "
                   "editing the plugin settings (GUI: Settings > Configure extract settings, "
                   "CLI: Edit the file faceswap/config/extract.ini)."
                   "\n3) Enable 'Single Process' mode.")
            raise FaceswapError(msg) from err
        except Exception as err:
            if get_backend() == "amd":
                # pylint:disable=import-outside-toplevel
                from lib.plaidml_utils import is_plaidml_error
                if (is_plaidml_error(err) and (
                        "CL_MEM_OBJECT_ALLOCATION_FAILURE" in str(err).upper() or
                        "enough memory for the current schedule" in str(err).lower())):
                    msg = ("You do not have enough GPU memory available to run detection at "
                           "the selected batch size. You can try a number of things:"
                           "\n1) Close any other application that is using your GPU (web "
                           "browsers are particularly bad for this)."
                           "\n2) Lower the batchsize (the amount of images fed into the "
                           "model) by editing the plugin settings (GUI: Settings > Configure "
                           "extract settings, CLI: Edit the file "
                           "faceswap/config/extract.ini).")
                    raise FaceswapError(msg) from err
            raise

    def finalize(self, batch):
        """ Finalize the output from Masker

        This should be called as the final task of each `plugin`.

        Pairs the detected faces back up with their original frame before yielding each frame.

        Parameters
        ----------
        batch : dict
            The final ``dict`` from the `plugin` process. It must contain the `keys`:
            ``detected_faces``, ``filename``, ``feed_faces``, ``roi_masks``

        Yields
        ------
        :class:`~plugins.extract.pipeline.ExtractMedia`
            The :attr:`DetectedFaces` list will be populated for this class with the bounding
            boxes, landmarks and masks for the detected faces found in the frame.
        """
        for mask, face, feed_face, roi_mask in zip(batch["prediction"],
                                                   batch["detected_faces"],
                                                   batch["feed_faces"],
                                                   batch["roi_masks"]):
            self._crop_out_of_bounds(mask, roi_mask)
            face.add_mask(self._storage_name,
                          mask,
                          feed_face.adjusted_matrix,
                          feed_face.interpolators[1],
                          storage_size=self._storage_size,
                          storage_centering=self._storage_centering)
        del batch["feed_faces"]

        logger.trace("Item out: %s", {key: val.shape if isinstance(val, np.ndarray) else val
                                      for key, val in batch.items()})
        for filename, face in zip(batch["filename"], batch["detected_faces"]):
            self._output_faces.append(face)
            if len(self._output_faces) != self._faces_per_filename[filename]:
                continue

            output = self._extract_media.pop(filename)
            output.add_detected_faces(self._output_faces)
            self._output_faces = []
            logger.trace("Yielding: (filename: '%s', image: %s, detected_faces: %s)",
                         output.filename, output.image_shape, len(output.detected_faces))
            yield output

    # <<< PROTECTED ACCESS METHODS >>> #
    @classmethod
    def _resize(cls, image, target_size):
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

    @classmethod
    def _crop_out_of_bounds(cls, mask, roi_mask):
        """ Un-mask any area of the predicted mask that falls outside of the original frame.

        Parameters
        ----------
        masks: :class:`numpy.ndarray`
            The predicted masks from the plugin
        roi_mask: :class:`numpy.ndarray`
            The roi mask. In frame is white, out of frame is black
        """
        if np.all(roi_mask):
            return  # The whole of the face is within the frame
        roi_mask = roi_mask[..., None] if mask.ndim == 3 else roi_mask
        mask *= roi_mask
