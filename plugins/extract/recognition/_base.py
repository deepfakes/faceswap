#!/usr/bin/env python3
""" Base class for Face Recognition plugins

All Recognition Plugins should inherit from this class.
See the override methods for which methods are required.

The plugin will receive a :class:`~plugins.extract.extract_media.ExtractMedia` object.

For each source frame, the plugin must pass a dict to finalize containing:

>>> {'filename': <filename of source frame>,
>>>  'detected_faces': <list of DetectedFace objects containing bounding box points}}

To get a :class:`~lib.align.DetectedFace` object use the function:

>>> face = self.to_detected_face(<face left>, <face top>, <face right>, <face bottom>)
"""
from __future__ import annotations
import logging
import typing as T

from dataclasses import dataclass, field

import numpy as np
from tensorflow.python.framework import errors_impl as tf_errors  # pylint:disable=no-name-in-module  # noqa

from lib.align import AlignedFace, DetectedFace, LandmarkType
from lib.image import read_image_meta
from lib.utils import FaceswapError
from plugins.extract import ExtractMedia
from plugins.extract._base import BatchType, ExtractorBatch, Extractor

if T.TYPE_CHECKING:
    from collections.abc import Generator
    from queue import Queue
    from lib.align.aligned_face import CenteringType

logger = logging.getLogger(__name__)


@dataclass
class RecogBatch(ExtractorBatch):
    """ Dataclass for holding items flowing through the aligner.

    Inherits from :class:`~plugins.extract._base.ExtractorBatch`
    """
    detected_faces: list[DetectedFace] = field(default_factory=list)
    feed_faces: list[AlignedFace] = field(default_factory=list)


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

    _logged_lm_count_once = False

    def __init__(self,
                 git_model_id: int | None = None,
                 model_filename: str | None = None,
                 configfile: str | None = None,
                 instance: int = 0,
                 **kwargs):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile,
                         instance=instance,
                         **kwargs)
        self.input_size = 256  # Override for model specific input_size
        self.centering: CenteringType = "legacy"  # Override for model specific centering
        self.coverage_ratio = 1.0  # Override for model specific coverage_ratio

        self._plugin_type = "recognition"
        self._filter = IdentityFilter(self.config["save_filtered"])
        logger.debug("Initialized _base %s", self.__class__.__name__)

    def _get_detected_from_aligned(self, item: ExtractMedia) -> None:
        """ Obtain detected face objects for when loading in aligned faces and a detected face
        object does not exist

        Parameters
        ----------
        item: :class:`~plugins.extract.extract_media.ExtractMedia`
            The extract media to populate the detected face for
         """
        detected_face = DetectedFace()
        meta = read_image_meta(item.filename).get("itxt", {}).get("alignments")
        if meta:
            detected_face.from_png_meta(meta)
        item.add_detected_faces([detected_face])
        self._faces_per_filename[item.filename] += 1  # Track this added face
        logger.debug("Obtained detected face: (filename: %s, detected_face: %s)",
                     item.filename, item.detected_faces)

    def _maybe_log_warning(self, face: AlignedFace) -> None:
        """ Log a warning, once, if we do not have full facial landmarks

        Parameters
        ----------
        face: :class:`~lib.align.aligned_face.AlignedFace`
            The aligned face object to test the landmark type for
        """
        if face.landmark_type != LandmarkType.LM_2D_4 or self._logged_lm_count_once:
            return
        logger.warning("Extracted faces do not contain facial landmark data. '%s' "
                       "identity data is likely to be sub-standard.", self.name)
        self._logged_lm_count_once = True

    def get_batch(self, queue: Queue) -> tuple[bool, RecogBatch]:
        """ Get items for inputting into the recognition from the queue in batches

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        Items are received as :class:`~plugins.extract.extract_media.ExtractMedia` objects and
        converted to :class:`RecogBatch` for internal processing.

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
            if not item.is_aligned and not item.detected_faces:
                self._queues["out"].put(item)
                continue
            if item.is_aligned and not item.detected_faces:
                self._get_detected_from_aligned(item)

            for f_idx, face in enumerate(item.detected_faces):

                image = item.get_image_copy(self.color_format)
                feed_face = AlignedFace(face.landmarks_xy,
                                        image=image,
                                        centering=self.centering,
                                        size=self.input_size,
                                        coverage_ratio=self.coverage_ratio,
                                        dtype="float32",
                                        is_aligned=item.is_aligned)

                self._maybe_log_warning(feed_face)

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
                            detected_faces=item.detected_faces[f_idx + 1:],
                            is_aligned=item.is_aligned)
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

        # TODO Move to end of process not beginning
        if exhausted:
            self._filter.output_counts()

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
        :class:`~plugins.extract.extract_media.ExtractMedia`
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
            self._output_faces = self._filter(self._output_faces, output.sub_folders)

            output.add_detected_faces(self._output_faces)
            self._output_faces = []
            logger.trace("Yielding: (filename: '%s', image: %s, "  # type:ignore
                         "detected_faces: %s)", output.filename, output.image_shape,
                         len(output.detected_faces))
            yield output

    def add_identity_filters(self,
                             filters: np.ndarray,
                             nfilters: np.ndarray,
                             threshold: float) -> None:
        """ Add identity encodings to filter by identity in the recognition plugin

        Parameters
        ----------
        filters: :class:`numpy.ndarray`
            The array of filter embeddings to use
        nfilters: :class:`numpy.ndarray`
            The array of nfilter embeddings to use
        threshold: float
            The threshold for a positive filter match
        """
        logger.debug("Adding identity filters")
        self._filter.add_filters(filters, nfilters, threshold)
        logger.debug("Added identity filters")


class IdentityFilter():
    """ Applies filters on the output of the recognition plugin

    Parameters
    ----------
    save_output: bool
        ``True`` if the filtered faces should be kept as they are being saved. ``False`` if they
        should be deleted
    """
    def __init__(self, save_output: bool) -> None:
        logger.debug("Initializing %s: (save_output: %s)", self.__class__.__name__, save_output)
        self._save_output = save_output
        self._filter: np.ndarray | None = None
        self._nfilter: np.ndarray | None = None
        self._threshold = 0.0
        self._filter_enabled: bool = False
        self._nfilter_enabled: bool = False
        self._active: bool = False
        self._counts = 0
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_filters(self, filters: np.ndarray, nfilters: np.ndarray, threshold) -> None:
        """ Add identity encodings to the filter and set whether each filter is enabled

        Parameters
        ----------
        filters: :class:`numpy.ndarray`
            The array of filter embeddings to use
        nfilters: :class:`numpy.ndarray`
            The array of nfilter embeddings to use
        threshold: float
            The threshold for a positive filter match
        """
        logger.debug("Adding filters: %s, nfilters: %s, threshold: %s",
                     filters.shape, nfilters.shape, threshold)
        self._filter = filters
        self._nfilter = nfilters
        self._threshold = threshold
        self._filter_enabled = bool(np.any(self._filter))
        self._nfilter_enabled = bool(np.any(self._nfilter))
        self._active = self._filter_enabled or self._nfilter_enabled
        logger.debug("filter active: %s, nfilter active: %s, all active: %s",
                     self._filter_enabled, self._nfilter_enabled, self._active)

    @classmethod
    def _find_cosine_similiarity(cls,
                                 source_identities: np.ndarray,
                                 test_identity: np.ndarray) -> np.ndarray:
        """ Find the cosine similarity between a source face identity and a test face identity

        Parameters
        ---------
        source_identities: :class:`numpy.ndarray`
            The identity encoding for the source face identities
        test_identity: :class:`numpy.ndarray`
            The identity encoding for the face identity to test against the sources

        Returns
        -------
        :class:`numpy.ndarray`:
            The cosine similarity between a face identity and the source identities
        """
        s_norm = np.linalg.norm(source_identities, axis=1)
        i_norm = np.linalg.norm(test_identity)
        retval = source_identities @ test_identity / (s_norm * i_norm)
        return retval

    def _get_matches(self,
                     filter_type: T.Literal["filter", "nfilter"],
                     identities: np.ndarray) -> np.ndarray:
        """ Obtain the average and minimum distances for each face against the source identities
        to test against

        Parameters
        ----------
        filter_type ["filter", "nfilter"]
            The filter type to use for calculating the distance
        identities: :class:`numpy.ndarray`
            The identity encodings for the current face(s) being checked

        Returns
        -------
        :class:`numpy.ndarray`
            Boolean array. ``True`` if identity should be filtered otherwise ``False``
        """
        encodings = self._filter if filter_type == "filter" else self._nfilter
        assert encodings is not None
        distances = np.array([self._find_cosine_similiarity(encodings, identity)
                              for identity in identities])
        is_match = np.any(distances >= self._threshold, axis=-1)
        # Invert for filter (set the `True` match to `False` for should filter)
        retval = np.invert(is_match) if filter_type == "filter" else is_match
        logger.trace("filter_type: %s, distances shape: %s, is_match: %s, ",  # type: ignore
                     "retval: %s", filter_type, distances.shape, is_match, retval)
        return retval

    def _filter_faces(self,
                      faces: list[DetectedFace],
                      sub_folders: list[str | None],
                      should_filter: list[bool]) -> list[DetectedFace]:
        """ Filter the detected faces, either removing filtered faces from the list of detected
        faces or setting the output subfolder to `"_identity_filt"` for any filtered faces if
        saving output is enabled.

        Parameters
        ----------
        faces: list
            List of detected face objects to filter out on size
        sub_folders: list
            List of subfolder locations for any faces that have already been filtered when
            config option `save_filtered` has been enabled.
        should_filter: list
            List of 'bool' corresponding to face that have not already been marked for filtering.
            ``True`` indicates face should be filtered, ``False`` indicates face should be kept

        Returns
        -------
        detected_faces: list
            The filtered list of detected face objects, if saving filtered faces has not been
            selected or the full list of detected faces
        """
        retval: list[DetectedFace] = []
        self._counts += sum(should_filter)
        for idx, face in enumerate(faces):
            fldr = sub_folders[idx]
            if fldr is not None:
                # Saving to sub folder is selected and face is already filtered
                # so this face was excluded from identity check
                retval.append(face)
                continue
            to_filter = should_filter.pop(0)
            if not to_filter or self._save_output:
                # Keep the face if not marked as filtered or we are to output to a subfolder
                retval.append(face)
            if to_filter and self._save_output:
                sub_folders[idx] = "_identity_filt"

        return retval

    def __call__(self,
                 faces: list[DetectedFace],
                 sub_folders: list[str | None]) -> list[DetectedFace]:
        """ Call the identity filter function

        Parameters
        ----------
        faces: list
            List of detected face objects to filter out on size
        sub_folders: list
            List of subfolder locations for any faces that have already been filtered when
            config option `save_filtered` has been enabled.

        Returns
        -------
        detected_faces: list
            The filtered list of detected face objects, if saving filtered faces has not been
            selected or the full list of detected faces
        """
        if not self._active:
            return faces

        identities = np.array([face.identity["vggface2"] for face, fldr in zip(faces, sub_folders)
                               if fldr is None])
        logger.trace("face_count: %s, already_filtered: %s, identity_shape: %s",  # type: ignore
                     len(faces), sum(x is not None for x in sub_folders), identities.shape)

        if not np.any(identities):
            logger.trace("All faces already filtered: %s", sub_folders)  # type: ignore
            return faces

        should_filter: list[np.ndarray] = []
        for f_type in T.get_args(T.Literal["filter", "nfilter"]):
            if not getattr(self, f"_{f_type}_enabled"):
                continue
            should_filter.append(self._get_matches(f_type, identities))

        # If any of the filter or nfilter evaluate to 'should filter' then filter out face
        final_filter: list[bool] = np.array(should_filter).max(axis=0).tolist()
        logger.trace("should_filter: %s, final_filter: %s",  # type: ignore
                     should_filter, final_filter)
        return self._filter_faces(faces, sub_folders, final_filter)

    def output_counts(self):
        """ Output the counts of filtered items """
        if not self._active or not self._counts:
            return
        logger.info("Identity filtered (%s): %s", self._threshold, self._counts)
