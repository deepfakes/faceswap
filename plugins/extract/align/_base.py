#!/usr/bin/env python3
""" Base class for Face Aligner plugins

All Aligner Plugins should inherit from this class.
See the override methods for which methods are required.

The plugin will receive a :class:`~plugins.extract.pipeline.ExtractMedia` object.

For each source item, the plugin must pass a dict to finalize containing:

>>> {"filename": [<filename of source frame>],
>>>  "landmarks": [list of 68 point face landmarks]
>>>  "detected_faces": [<list of DetectedFace objects>]}
"""
import sys

from dataclasses import dataclass, field
from typing import Any, cast, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING, Union

import cv2
import numpy as np

from tensorflow.python.framework import errors_impl as tf_errors  # pylint:disable=no-name-in-module # noqa

from lib.align import AlignedFace, DetectedFace
from lib.utils import get_backend, FaceswapError
from plugins.extract._base import Extractor, logger, ExtractMedia

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

if TYPE_CHECKING:
    from queue import Queue


@dataclass
class AlignerBatch:
    """ Dataclass for holding items flowing through the aligner.

    Parameters
    ----------
    image: list
        List of :class:`numpy.ndarray` containing the original frame
    detected_faces: list
        List of :class:`~lib.align.DetectedFace` objects
    filename: list
        List of original frame filenames for the batch
    feed: list
        List of feed images to feed the aligner net for each re-feed increment
    prediction: list
        List of predictions. Direct output from the aligner net
    landmarks: list
        List of 68 point :class:`numpy.ndarray` landmark points returned from the aligner
    data: dict
        Any aligner specific data required during the processing phase. List of dictionaries for
        holding data on each sub-batch if re-feed > 1
    """
    image: List[np.ndarray] = field(default_factory=list)
    detected_faces: List[DetectedFace] = field(default_factory=list)
    filename: List[str] = field(default_factory=list)
    feed: List[np.ndarray] = field(default_factory=list)
    prediction: np.ndarray = np.empty([])
    landmarks: np.ndarray = np.empty([])
    data: List[Dict[str, Any]] = field(default_factory=list)


class Aligner(Extractor):  # pylint:disable=abstract-method
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
    re_feed: int
        The number of times to re-feed a slightly adjusted bounding box into the aligner.
        Default: `0`

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
    plugins.extract.mask._base : Masker parent class for extraction plugins.
    """

    def __init__(self,
                 git_model_id: Optional[int] = None,
                 model_filename: Optional[str] = None,
                 configfile: Optional[str] = None,
                 instance: int = 0,
                 normalize_method: Optional[Literal["none", "clahe", "hist", "mean"]] = None,
                 re_feed: int = 0, **kwargs) -> None:
        logger.debug("Initializing %s: (normalize_method: %s, re_feed: %s)",
                     self.__class__.__name__, normalize_method, re_feed)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile,
                         instance=instance,
                         **kwargs)
        self._normalize_method: Optional[Literal["clahe", "hist", "mean"]] = None
        self._re_feed = re_feed
        self.set_normalize_method(normalize_method)

        self._plugin_type = "align"
        self._faces_per_filename: Dict[str, int] = {}  # Tracking for recompiling batches
        self._rollover: Optional[ExtractMedia] = None  # batch rollover items
        self._output_faces: List[DetectedFace] = []
        self._filter = AlignedFilter(min_scale=self.config["aligner_min_scale"],
                                     max_scale=self.config["aligner_max_scale"],
                                     distance=self.config["aligner_distance"],
                                     save_output=self.config["save_filtered"])
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_normalize_method(self,
                             method: Optional[Literal["none", "clahe", "hist", "mean"]]) -> None:
        """ Set the normalization method for feeding faces into the aligner.

        Parameters
        ----------
        method: {"none", "clahe", "hist", "mean"}
            The normalization method to apply to faces prior to feeding into the model
        """
        method = None if method is None or method.lower() == "none" else method
        self._normalize_method = cast(Optional[Literal["clahe", "hist", "mean"]], method)

    # << QUEUE METHODS >>> #
    def get_batch(self, queue: "Queue") -> Tuple[bool, AlignerBatch]:
        """ Get items for inputting into the aligner from the queue in batches

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        Items are received as :class:`~plugins.extract.pipeline.ExtractMedia` objects and converted
        to ``dict`` for internal processing.

        To ensure consistent batch sizes for aligner the items are split into separate items for
        each :class:`~lib.align.DetectedFace` object.

        Remember to put ``'EOF'`` to the out queue after processing
        the final batch

        Outputs items in the following format. All lists are of length
        :attr:`~plugins.extract._base.Extractor.batchsize`:

        >>> {'filename': [<filenames of source frames>],
        >>>  'image': [<source images>],
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
        batch = AlignerBatch()
        idx = 0
        while idx < self.batchsize:
            item = self._collect_item(queue)
            if item == "EOF":
                logger.trace("EOF received")  # type:ignore
                exhausted = True
                break
            # Put frames with no faces into the out queue to keep TQDM consistent
            if not item.detected_faces:
                self._queues["out"].put(item)
                continue

            converted_image = item.get_image_copy(self.color_format)
            for f_idx, face in enumerate(item.detected_faces):
                batch.image.append(converted_image)
                batch.detected_faces.append(face)
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
        if batch.filename:
            logger.trace("Returning batch: %s", {k: len(v)  # type:ignore
                                                 if isinstance(v, list) else v
                                                 for k, v in batch.__dict__.items()})
        else:
            logger.debug(item)  # type:ignore

        # TODO Move to end of process not beginning
        if exhausted:
            self._filter.output_counts()

        return exhausted, batch

    def _collect_item(self, queue: "Queue") -> Union[Literal["EOF"], ExtractMedia]:
        """ Collect the item from the :attr:`_rollover` dict or from the queue. Add face count per
        frame to self._faces_per_filename for joining batches back up in finalize

        Parameters
        ----------
        queue: :class:`queue.Queue`
            The input queue to the aligner. Should contain
            :class:`~plugins.extract.pipeline.ExtractMedia` objects

        Returns
        -------
        :class:`~plugins.extract.pipeline.ExtractMedia` or EOF
            The next extract media object, or EOF if pipe has ended
        """
        if self._rollover is not None:
            logger.trace("Getting from _rollover: (filename: `%s`, faces: %s)",  # type:ignore
                         self._rollover.filename, len(self._rollover.detected_faces))
            item = self._rollover
            self._rollover = None
        else:
            item = self._get_item(queue)
            if item != "EOF":
                logger.trace("Getting from queue: (filename: %s, faces: %s)",  # type:ignore
                             item.filename, len(item.detected_faces))
                self._faces_per_filename[item.filename] = len(item.detected_faces)
        return item

    # <<< FINALIZE METHODS >>> #
    def finalize(self, batch: AlignerBatch) -> Generator[ExtractMedia, None, None]:
        """ Finalize the output from Aligner

        This should be called as the final task of each `plugin`.

        Pairs the detected faces back up with their original frame before yielding each frame.

        Parameters
        ----------
        batch : :class:`AlignerBatch`
            The final batch item from the `plugin` process.

        Yields
        ------
        :class:`~plugins.extract.pipeline.ExtractMedia`
            The :attr:`DetectedFaces` list will be populated for this class with the bounding boxes
            and landmarks for the detected faces found in the frame.
        """

        for face, landmarks in zip(batch.detected_faces, batch.landmarks):
            if not isinstance(landmarks, np.ndarray):
                landmarks = np.array(landmarks)
            face._landmarks_xy = landmarks

        logger.trace("Item out: %s", {key: val.shape  # type:ignore
                                      if isinstance(val, np.ndarray) else val
                                      for key, val in batch.__dict__.items()})

        for frame, filename, face in zip(batch.image, batch.filename, batch.detected_faces):
            self._output_faces.append(face)
            if len(self._output_faces) != self._faces_per_filename[filename]:
                continue

            self._output_faces, folders = self._filter(self._output_faces, min(frame.shape[:2]))

            output = self._extract_media.pop(filename)
            output.add_detected_faces(self._output_faces)
            output.add_sub_folders(folders)
            self._output_faces = []

            logger.trace("Final Output: (filename: '%s', image shape: %s, "  # type:ignore
                         "detected_faces: %s, item: %s)",
                         output.filename, output.image_shape, output.detected_faces, output)
            yield output

    # <<< PROTECTED METHODS >>> #

    # << PROCESS_INPUT WRAPPER >>
    def _process_input(self, batch: AlignerBatch) -> AlignerBatch:
        """ Process the input to the aligner model multiple times based on the user selected
        `re-feed` command line option. This adjusts the bounding box for the face to be fed
        into the model by a random amount within 0.05 pixels of the detected face's shortest axis.

        References
        ----------
        https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            Contains the batch that is currently being passed through the plugin process

        Returns
        -------
        :class:`AlignerBatch`
            The batch with input processed
        """
        original_boxes = np.array([(face.left, face.top, face.width, face.height)
                                   for face in batch.detected_faces])
        adjusted_boxes = self._get_adjusted_boxes(original_boxes)

        # Put in random re-feed data to the bounding boxes
        for bounding_boxes in adjusted_boxes:
            for face, box in zip(batch.detected_faces, bounding_boxes):
                face.left, face.top, face.width, face.height = box

            self.process_input(batch)

        # Place the original bounding box back to detected face objects
        for face, box in zip(batch.detected_faces, original_boxes):
            face.left, face.top, face.width, face.height = box

        return batch

    def _get_adjusted_boxes(self, original_boxes: np.ndarray) -> np.ndarray:
        """ Obtain an array of adjusted bounding boxes based on the number of re-feed iterations
        that have been selected and the minimum dimension of the original bounding box.

        Parameters
        ----------
        original_boxes: :class:`numpy.ndarray`
            The original ('x', 'y', 'w', 'h') detected face boxes corresponding to the incoming
            detected face objects

        Returns
        -------
        :class:`numpy.ndarray`
            The original boxes (in position 0) and the randomly adjusted bounding boxes
        """
        if self._re_feed == 0:
            return original_boxes[None, ...]
        beta = 0.05
        max_shift = np.min(original_boxes[..., 2:], axis=1) * beta
        rands = np.random.rand(self._re_feed, *original_boxes.shape) * 2 - 1
        new_boxes = np.rint(original_boxes + (rands * max_shift[None, :, None])).astype("int32")
        retval = np.concatenate((original_boxes[None, ...], new_boxes))
        logger.trace(retval)  # type:ignore
        return retval

    # <<< PREDICT WRAPPER >>> #
    def _predict(self, batch: AlignerBatch) -> AlignerBatch:
        """ Just return the aligner's predict function

        Parameters
        ----------
        batch: :class:`AlignerBatch`
            The current batch to find alignments for

        Returns
        -------
        :class:`AlignerBatch`
            The batch item with the :attr:`prediction` populated

        Raises
        ------
        FaceswapError
            If GPU resources are exhausted
        """
        try:
            batch.prediction = np.array([self.predict(feed) for feed in batch.feed])
            return batch
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

    def _process_output(self, batch: AlignerBatch) -> AlignerBatch:
        """ Process the output from the aligner model multiple times based on the user selected
        `re-feed amount` configuration option, then average the results for final prediction.

        Parameters
        ----------
        batch : :class:`AlignerBatch`
            Contains the batch that is currently being passed through the plugin process

        Returns
        -------
        :class:`AlignerBatch`
            The batch item with :attr:`landmarks` populated
        """
        landmarks = []
        for idx in range(self._re_feed + 1):
            # Create a pseudo object that only populates the data, feed and prediction slots with
            # the current re-feed iteration
            subbatch = AlignerBatch(image=batch.image,
                                    detected_faces=batch.detected_faces,
                                    filename=batch.filename,
                                    feed=[batch.feed[idx]],
                                    prediction=batch.prediction[idx],
                                    data=[batch.data[idx]])
            self.process_output(subbatch)
            landmarks.append(subbatch.landmarks)
        batch.landmarks = np.average(landmarks, axis=0)
        return batch

    # <<< FACE NORMALIZATION METHODS >>> #
    def _normalize_faces(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        """ Normalizes the face for feeding into model
        The normalization method is dictated by the normalization command line argument

        Parameters
        ----------
        faces: :class:`numpy.ndarray`
            The faces to normalize

        Returns
        -------
        :class:`numpy.ndarray`
            The normalized faces
        """
        if self._normalize_method is None:
            return faces
        logger.trace("Normalizing faces")  # type:ignore
        meth = getattr(self, f"_normalize_{self._normalize_method.lower()}")
        faces = [meth(face) for face in faces]
        logger.trace("Normalized faces")  # type:ignore
        return faces

    @classmethod
    def _normalize_mean(cls, face: np.ndarray) -> np.ndarray:
        """ Normalize Face to the Mean

        Parameters
        ----------
        faces: :class:`numpy.ndarray`
            The faces to normalize

        Returns
        -------
        :class:`numpy.ndarray`
            The normalized faces
        """
        face = face / 255.0
        for chan in range(3):
            layer = face[:, :, chan]
            layer = (layer - layer.min()) / (layer.max() - layer.min())
            face[:, :, chan] = layer
        return face * 255.0

    @classmethod
    def _normalize_hist(cls, face: np.ndarray) -> np.ndarray:
        """ Equalize the RGB histogram channels

        Parameters
        ----------
        faces: :class:`numpy.ndarray`
            The faces to normalize

        Returns
        -------
        :class:`numpy.ndarray`
            The normalized faces
        """
        for chan in range(3):
            face[:, :, chan] = cv2.equalizeHist(face[:, :, chan])
        return face

    @classmethod
    def _normalize_clahe(cls, face: np.ndarray) -> np.ndarray:
        """ Perform Contrast Limited Adaptive Histogram Equalization

        Parameters
        ----------
        faces: :class:`numpy.ndarray`
            The faces to normalize

        Returns
        -------
        :class:`numpy.ndarray`
            The normalized faces
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        for chan in range(3):
            face[:, :, chan] = clahe.apply(face[:, :, chan])
        return face


class AlignedFilter():
    """ Applies filters on the output of the aligner

    Parameters
    ----------
    min_scale: float
        Filters out faces that have been aligned at below this value as a multiplier of the
        minimum frame dimension. Set to ``0`` for off.
    max_scale: float
        Filters out faces that have been aligned at above this value as a multiplier of the
        minimum frame dimension. Set to ``0`` for off.
    distance: float
        Filters out faces that are further than this distance from an "average" face. Set to
        ``0`` for off.
    save_output: bool
        ``True`` if the filtered faces should be kept as they are being saved. ``False`` if they
        should be deleted
    """
    def __init__(self,
                 min_scale: float,
                 max_scale: float,
                 distance: float,
                 save_output: bool) -> None:
        logger.debug("Initializing %s: (min_scale: %s, max_scale: %s, distance: %s, "
                     "save_output: %s)", self.__class__.__name__, min_scale, max_scale, distance,
                     save_output)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._distance = distance / 100.
        self._save_output = save_output
        self._active = max_scale > 0.0 or min_scale > 0.0 or distance > 0.0
        self._counts: Dict[str, int] = dict(min_scale=0, max_scale=0, distance=0)
        logger.debug("Initialized %s: ", self.__class__.__name__)

    def __call__(self, faces: List[DetectedFace], minimum_dimension: int
                 ) -> Tuple[List[DetectedFace], List[Optional[str]]]:
        """ Apply the filter to the incoming batch

        Parameters
        ----------
        batch: list
            List of detected face objects to filter out on size
        minimum_dimension: int
            The minimum (height, width) of the original frame

        Returns
        -------
        detected_faces: list
            The filtered list of detected face objects, if saving filtered faces has not been
            selected or the full list of detected faces
        sub_folders: list
            List of ``Nones`` if saving filtered faces has not been selected or list of ``Nones``
            and sub folder names corresponding the filtered face location
        """
        sub_folders: List[Optional[str]] = [None for _ in range(len(faces))]
        if not self._active:
            return faces, sub_folders

        max_size = minimum_dimension * self._max_scale
        min_size = minimum_dimension * self._min_scale
        retval: List[DetectedFace] = []
        for idx, face in enumerate(faces):
            test = AlignedFace(landmarks=face.landmarks_xy, centering="face")
            if self._min_scale > 0.0 or self._max_scale > 0.0:
                roi = test.original_roi
                size = ((roi[1][0] - roi[0][0]) ** 2 + (roi[1][1] - roi[0][1]) ** 2) ** 0.5
                if self._min_scale > 0.0 and size < min_size:
                    self._counts["min_scale"] += 1
                    if self._save_output:
                        retval.append(face)
                        sub_folders[idx] = "_align_filt_min_scale"
                    continue
                if self._max_scale > 0.0 and size > max_size:
                    self._counts["max_scale"] += 1
                    if self._save_output:
                        retval.append(face)
                        sub_folders[idx] = "_align_filt_max_scale"
                    continue
            if 0.0 < self._distance < test.average_distance:
                self._counts["distance"] += 1
                if self._save_output:
                    retval.append(face)
                    sub_folders[idx] = "_align_filt_distance"
                continue
            retval.append(face)
        return retval, sub_folders

    def output_counts(self):
        """ Output the counts of filtered items """
        if not self._active:
            return
        counts = [f"{key} ({getattr(self, f'_{key}'):.2f}): {count}"
                  for key, count in self._counts.items()
                  if count > 0]
        if counts:
            logger.info("Aligner filtered: [%s)", ", ".join(counts))
