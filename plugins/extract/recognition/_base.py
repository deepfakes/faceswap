#!/usr/bin/env python3
""" Base class for Face Recognition plugins

All Recognition Plugins should inherit from this class.
See the override methods for which methods are required.

The plugin will receive a :class:`~plugins.extract.pipeline.ExtractMedia` object.

For each source frame, the plugin must pass a dict to finalize containing:

>>> {'filename': <filename of source frame>,
>>>  'detected_faces': <list of DetectedFace objects containing bounding box points}}

To get a :class:`~lib.align.DetectedFace` object use the function:

>>> face = self.to_detected_face(<face left>, <face top>, <face right>, <face bottom>)
"""
import logging

from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from tensorflow.python.framework import errors_impl as tf_errors  # pylint:disable=no-name-in-module  # noqa

from lib.align import AlignedFace
from lib.utils import FaceswapError, get_backend
from plugins.extract._base import BatchType, Extractor, ExtractorBatch
from plugins.extract.pipeline import ExtractMedia

if TYPE_CHECKING:
    from queue import Queue
    from lib.align import DetectedFace
    from lib.align.aligned_face import CenteringType

logger = logging.getLogger(__name__)


@dataclass
class RecogBatch(ExtractorBatch):
    """ Dataclass for holding items flowing through the aligner.

    Inherits from :class:`~plugins.extract._base.ExtractorBatch`
    """
    detected_faces: List["DetectedFace"] = field(default_factory=list)
    feed_faces: List[AlignedFace] = field(default_factory=list)


class Identity(Extractor):  # pylint:disable=abstract-method
    """ Face Recognition Object

    Parent class for all Recognition plugins

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
    plugins.extract.detect : Detector plugins
    plugins.extract._base : Parent class for all extraction plugins
    plugins.extract.align._base : Aligner parent class for extraction plugins.
    plugins.extract.mask._base : Masker parent class for extraction plugins.
    """

    def __init__(self,
                 git_model_id: Optional[int] = None,
                 model_filename: Optional[str] = None,
                 configfile: Optional[str] = None,
                 instance: int = 0,
                 image_is_aligned=False,
                 **kwargs):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile,
                         instance=instance,
                         **kwargs)
        self.input_size = 256  # Override for model specific input_size
        self.centering: "CenteringType" = "legacy"  # Override for model specific centering
        self.coverage_ratio = 1.0  # Override for model specific coverage_ratio

        self._plugin_type = "recognition"
        self._image_is_aligned = image_is_aligned
        logger.debug("Initialized _base %s", self.__class__.__name__)

    def get_batch(self, queue: "Queue") -> Tuple[bool, RecogBatch]:
        """ Get items for inputting into the recognition from the queue in batches

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        Items are received as :class:`~plugins.extract.pipeline.ExtractMedia` objects and converted
        to :class:`RecogBatch` for internal processing.

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
        batch, :class:`~plugins.extract._base.ExtractorBatch`
            The batch object for the current batch
        """
        exhausted = False
        batch = RecogBatch()
        idx = 0
        while idx < self.batchsize:
            item = self.rollover_collector(queue)
            if item == "EOF":
                logger.trace("EOF received")  # type: ignore
                exhausted = True
                break
            # Put frames with no faces into the out queue to keep TQDM consistent
            if not item.detected_faces:
                self._queues["out"].put(item)
                continue
            for f_idx, face in enumerate(item.detected_faces):

                image = item.get_image_copy(self.color_format)
                feed_face = AlignedFace(face.landmarks_xy,
                                        image=image,
                                        centering=self.centering,
                                        size=self.input_size,
                                        coverage_ratio=self.coverage_ratio,
                                        dtype="float32",
                                        is_aligned=self._image_is_aligned)

                batch.detected_faces.append(face)
                batch.feed_faces.append(feed_face)
                batch.filename.append(item.filename)
                idx += 1
                if idx == self.batchsize:
                    frame_faces = len(item.detected_faces)
                    if f_idx + 1 != frame_faces:
                        self._rollover = ExtractMedia(
                            item.filename,
                            item.image,
                            detected_faces=item.detected_faces[f_idx + 1:])
                        logger.trace("Rolled over %s faces of %s to next batch "  # type:ignore
                                     "for '%s'", len(self._rollover.detected_faces), frame_faces,
                                     item.filename)
                    break
        if batch:
            logger.trace("Returning batch: %s",  # type:ignore
                         {k: len(v) if isinstance(v, (list, np.ndarray)) else v
                          for k, v in batch.__dict__.items()})
        else:
            logger.trace(item)  # type:ignore
        return exhausted, batch

    def _predict(self, batch: BatchType) -> RecogBatch:
        """ Just return the recognition's predict function """
        assert isinstance(batch, RecogBatch)
        try:
            # slightly hacky workaround to deal with landmarks based masks:
            batch.prediction = self.predict(batch.feed)
            return batch
        except tf_errors.ResourceExhaustedError as err:
            msg = ("You do not have enough GPU memory available to run recognition at the "
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

    def finalize(self, batch: BatchType) -> Generator[ExtractMedia, None, None]:
        """ Finalize the output from Masker

        This should be called as the final task of each `plugin`.

        Pairs the detected faces back up with their original frame before yielding each frame.

        Parameters
        ----------
        batch : :class:`RecogBatch`
            The final batch item from the `plugin` process.

        Yields
        ------
        :class:`~plugins.extract.pipeline.ExtractMedia`
            The :attr:`DetectedFaces` list will be populated for this class with the bounding
            boxes, landmarks and masks for the detected faces found in the frame.
        """
        assert isinstance(batch, RecogBatch)
        assert isinstance(self.name, str)
        for identity, face in zip(batch.prediction, batch.detected_faces):
            face.add_identity(self.name.lower(), identity)
        del batch.feed

        logger.trace("Item out: %s",  # type: ignore
                     {key: val.shape if isinstance(val, np.ndarray) else val
                                      for key, val in batch.__dict__.items()})
        for filename, face in zip(batch.filename, batch.detected_faces):
            self._output_faces.append(face)
            if len(self._output_faces) != self._faces_per_filename[filename]:
                continue

            output = self._extract_media.pop(filename)
            output.add_detected_faces(self._output_faces)
            self._output_faces = []
            logger.trace("Yielding: (filename: '%s', image: %s, "  # type:ignore
                         "detected_faces: %s)", output.filename, output.image_shape,
                         len(output.detected_faces))
            yield output
