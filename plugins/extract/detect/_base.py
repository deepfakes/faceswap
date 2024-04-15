#!/usr/bin/env python3
""" Base class for Face Detector plugins

All Detector Plugins should inherit from this class.
See the override methods for which methods are required.

The plugin will receive a :class:`~plugins.extract.extract_media.ExtractMedia` object.

For each source frame, the plugin must pass a dict to finalize containing:

>>> {'filename': <filename of source frame>,
>>>  'detected_faces': <list of DetectedFace objects containing bounding box points}}

To get a :class:`~lib.align.DetectedFace` object use the function:

>>> face = self._to_detected_face(<face left>, <face top>, <face right>, <face bottom>)
"""
from __future__ import annotations
import logging
import typing as T

from dataclasses import dataclass, field

import cv2
import numpy as np

from tensorflow.python.framework import errors_impl as tf_errors  # pylint:disable=no-name-in-module # noqa

from lib.align import DetectedFace
from lib.utils import FaceswapError

from plugins.extract._base import BatchType, Extractor, ExtractorBatch
from plugins.extract import ExtractMedia

if T.TYPE_CHECKING:
    from collections.abc import Generator
    from queue import Queue

logger = logging.getLogger(__name__)


@dataclass
class DetectorBatch(ExtractorBatch):
    """ Dataclass for holding items flowing through the aligner.

    Inherits from :class:`~plugins.extract._base.ExtractorBatch`

    Parameters
    ----------
    rotation_matrix: :class:`numpy.ndarray`
        The rotation matrix for any requested rotations
    scale: float
        The scaling factor to take the input image back to original size
    pad: tuple
        The amount of padding to apply to the image to feed the network
    initial_feed: :class:`numpy.ndarray`
        Used to hold the initial :attr:`feed` when rotate images is enabled
    """
    detected_faces: list[list["DetectedFace"]] = field(default_factory=list)
    rotation_matrix: list[np.ndarray] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    pad: list[tuple[int, int]] = field(default_factory=list)
    initial_feed: np.ndarray = np.array([])

    def __repr__(self):
        """ Prettier repr for debug printing """
        retval = super().__repr__()
        retval += (f", rotation_matrix={self.rotation_matrix}, "
                   f"scale={self.scale}, "
                   f"pad={self.pad}, "
                   f"initial_feed=({self.initial_feed.shape}, {self.initial_feed.dtype})")
        return retval


class Detector(Extractor):  # pylint:disable=abstract-method
    """ Detector Object

    Parent class for all Detector plugins

    Parameters
    ----------
    git_model_id: int
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information
    model_filename: str
        The name of the model file to be loaded
    rotation: str, optional
        Pass in a single number to use increments of that size up to 360, or pass in a ``list`` of
        ``ints`` to enumerate exactly what angles to check. Can also pass in ``'on'`` to increment
        at 90 degree intervals. Default: ``None``
    min_size: int, optional
        Filters out faces detected below this size. Length, in pixels across the diagonal of the
        bounding box. Set to ``0`` for off. Default: ``0``

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
                 git_model_id: int | None = None,
                 model_filename: str | list[str] | None = None,
                 configfile: str | None = None,
                 instance: int = 0,
                 rotation: str | None = None,
                 min_size: int = 0,
                 **kwargs) -> None:
        logger.debug("Initializing %s: (rotation: %s, min_size: %s)", self.__class__.__name__,
                     rotation, min_size)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile,
                         instance=instance,
                         **kwargs)
        self.rotation = self._get_rotation_angles(rotation)
        self.min_size = min_size

        self._plugin_type = "detect"

        logger.debug("Initialized _base %s", self.__class__.__name__)

    # <<< QUEUE METHODS >>> #
    def get_batch(self, queue: Queue) -> tuple[bool, DetectorBatch]:
        """ Get items for inputting to the detector plugin in batches

        Items are received as :class:`~plugins.extract.extract_media.ExtractMedia` objects and
        converted to ``dict`` for internal processing.

        Items are returned from the ``queue`` in batches of
        :attr:`~plugins.extract._base.Extractor.batchsize`

        Remember to put ``'EOF'`` to the out queue after processing
        the final batch

        Outputs items in the following format. All lists are of length
        :attr:`~plugins.extract._base.Extractor.batchsize`:

        >>> {'filename': [<filenames of source frames>],
        >>>  'image': <numpy.ndarray of images standardized for prediction>,
        >>>  'scale': [<scaling factors for each image>],
        >>>  'pad': [<padding for each image>],
        >>>  'detected_faces': [[<lib.align.DetectedFace objects]]}

        Parameters
        ----------
        queue : queue.Queue()
            The ``queue`` that the batch will be fed from. This will be a queue that loads
            images.

        Returns
        -------
        exhausted, bool
            ``True`` if queue is exhausted, ``False`` if not.
        batch, :class:`~plugins.extract._base.ExtractorBatch`
            The batch object for the current batch
        """
        exhausted = False
        batch = DetectorBatch()
        for _ in range(self.batchsize):
            item = self._get_item(queue)
            if item == "EOF":
                exhausted = True
                break
            assert isinstance(item, ExtractMedia)
            # Put items that are already aligned into the out queue
            if item.is_aligned:
                self._queues["out"].put(item)
                continue
            batch.filename.append(item.filename)
            image, scale, pad = self._compile_detection_image(item)
            batch.image.append(image)
            batch.scale.append(scale)
            batch.pad.append(pad)

        if batch.filename:
            logger.trace("Returning batch: %s",  # type: ignore
                         {k: len(v) if isinstance(v, (list, np.ndarray)) else v
                          for k, v in batch.__dict__.items()})
        else:
            logger.trace(item)  # type:ignore[attr-defined]

        if not exhausted and not batch.filename:
            # This occurs when face filter is fed aligned faces.
            # Need to re-run until EOF is hit
            return self.get_batch(queue)

        return exhausted, batch

    # <<< FINALIZE METHODS>>> #
    def finalize(self, batch: BatchType) -> Generator[ExtractMedia, None, None]:
        """ Finalize the output from Detector

        This should be called as the final task of each ``plugin``.

        Parameters
        ----------
        batch : :class:`~plugins.extract._base.ExtractorBatch`
            The batch object for the current batch

        Yields
        ------
        :class:`~plugins.extract.extract_media.ExtractMedia`
            The :attr:`DetectedFaces` list will be populated for this class with the bounding boxes
            for the detected faces found in the frame.
        """
        assert isinstance(batch, DetectorBatch)
        logger.trace("Item out: %s",  # type:ignore[attr-defined]
                     {k: len(v) if isinstance(v, (list, np.ndarray)) else v
                      for k, v in batch.__dict__.items()})

        batch_faces = [[self._to_detected_face(face[0], face[1], face[2], face[3])
                        for face in faces]
                       for faces in batch.prediction]
        # Rotations
        if any(m.any() for m in batch.rotation_matrix) and any(batch_faces):
            batch_faces = [[self._rotate_face(face, rotmat) if rotmat.any() else face
                            for face in faces]
                           for faces, rotmat in zip(batch_faces, batch.rotation_matrix)]

        # Remove zero sized faces
        batch_faces = self._remove_zero_sized_faces(batch_faces)

        # Scale back out to original frame
        batch.detected_faces = [[self._to_detected_face((face.left - pad[0]) / scale,
                                                        (face.top - pad[1]) / scale,
                                                        (face.right - pad[0]) / scale,
                                                        (face.bottom - pad[1]) / scale)
                                 for face in faces
                                 if face.left is not None and face.top is not None]
                                for scale, pad, faces in zip(batch.scale,
                                                             batch.pad,
                                                             batch_faces)]

        if self.min_size > 0 and batch.detected_faces:
            batch.detected_faces = self._filter_small_faces(batch.detected_faces)

        for idx, filename in enumerate(batch.filename):
            output = self._extract_media.pop(filename)
            output.add_detected_faces(batch.detected_faces[idx])

            logger.trace("final output: (filename: '%s', "  # type:ignore[attr-defined]
                         "image shape: %s, detected_faces: %s, item: %s",
                         output.filename, output.image_shape, output.detected_faces, output)
            yield output

    @staticmethod
    def _to_detected_face(left: float, top: float, right: float, bottom: float) -> DetectedFace:
        """ Convert a bounding box to a detected face object

        Parameters
        ----------
        left: float
            The left point of the detection bounding box
        top: float
            The top point of the detection bounding box
        right: float
            The right point of the detection bounding box
        bottom: float
            The bottom point of the detection bounding box

        Returns
        -------
        class:`~lib.align.DetectedFace`
            The detected face object for the given bounding box
        """
        return DetectedFace(left=int(round(left)),
                            width=int(round(right - left)),
                            top=int(round(top)),
                            height=int(round(bottom - top)))

    # <<< PROTECTED ACCESS METHODS >>> #
    # <<< PREDICT WRAPPER >>> #
    def _predict(self, batch: BatchType) -> DetectorBatch:
        """ Wrap models predict function in rotations """
        assert isinstance(batch, DetectorBatch)
        batch.rotation_matrix = [np.array([]) for _ in range(len(batch.feed))]
        found_faces: list[np.ndarray] = [np.array([]) for _ in range(len(batch.feed))]
        for angle in self.rotation:
            # Rotate the batch and insert placeholders for already found faces
            self._rotate_batch(batch, angle)
            try:
                pred = self.predict(batch.feed)
                if angle == 0:
                    batch.prediction = pred
                else:
                    try:
                        batch.prediction = np.array([b if b.any() else p
                                                    for b, p in zip(batch.prediction, pred)])
                    except ValueError as err:
                        # If batches are different sizes after rotation Numpy will error, so we
                        # need to explicitly set the dtype to 'object' rather than let it infer
                        # numpy error:
                        # ValueError: setting an array element with a sequence. The requested array
                        # has an inhomogeneous shape after 1 dimensions. The detected shape was
                        # (8,) + inhomogeneous part
                        if "inhomogeneous" in str(err):
                            batch.prediction = np.array([b if b.any() else p
                                                         for b, p in zip(batch.prediction, pred)],
                                                        dtype="object")
                            logger.trace(  # type:ignore[attr-defined]
                                "Mismatched array sizes, setting dtype to object: %s",
                                [p.shape for p in batch.prediction])
                        else:
                            raise

                logger.trace("angle: %s, filenames: %s, "  # type:ignore[attr-defined]
                             "prediction: %s",
                             angle, batch.filename, pred)
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

            if angle != 0 and any(face.any() for face in batch.prediction):
                logger.verbose("found face(s) by rotating image %s "  # type:ignore[attr-defined]
                               "degrees",
                               angle)

            found_faces = T.cast(list[np.ndarray], ([face if not found.any() else found
                                                     for face, found in zip(batch.prediction,
                                                                            found_faces)]))
            if all(face.any() for face in found_faces):
                logger.trace("Faces found for all images")  # type:ignore[attr-defined]
                break

        batch.prediction = np.array(found_faces, dtype="object")
        logger.trace("detect_prediction output: (filenames: %s, "  # type:ignore[attr-defined]
                     "prediction: %s, rotmat: %s)",
                     batch.filename, batch.prediction, batch.rotation_matrix)
        return batch

    # <<< DETECTION IMAGE COMPILATION METHODS >>> #
    def _compile_detection_image(self, item: ExtractMedia
                                 ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """ Compile the detection image for feeding into the model

        Parameters
        ----------
        item: :class:`~plugins.extract.extract_media.ExtractMedia`
            The input item from the pipeline

        Returns
        -------
        image: :class:`numpy.ndarray`
            The original image formatted for detection
        scale: float
            The scaling factor for the image
        pad: int
            The amount of padding applied to the image
        """
        image = item.get_image_copy(self.color_format)
        scale = self._set_scale(item.image_size)
        pad = self._set_padding(item.image_size, scale)

        image = self._scale_image(image, item.image_size, scale)
        image = self._pad_image(image)
        logger.trace("compiled: (images shape: %s, "  # type:ignore[attr-defined]
                     "scale: %s, pad: %s)",
                     image.shape, scale, pad)
        return image, scale, pad

    def _set_scale(self, image_size: tuple[int, int]) -> float:
        """ Set the scale factor for incoming image

        Parameters
        ----------
        image_size: tuple
            The (height, width) of the original image

        Returns
        -------
        float
            The scaling factor from original image size to model input size
        """
        scale = self.input_size / max(image_size)
        logger.trace("Detector scale: %s", scale)  # type:ignore[attr-defined]
        return scale

    def _set_padding(self, image_size: tuple[int, int], scale: float) -> tuple[int, int]:
        """ Set the image padding for non-square images

        Parameters
        ----------
        image_size: tuple
            The (height, width) of the original image
        scale: float
            The scaling factor from original image size to model input size

        Returns
        -------
        tuple
            The amount of padding to apply to the x and y axes
        """
        pad_left = int(self.input_size - int(image_size[1] * scale)) // 2
        pad_top = int(self.input_size - int(image_size[0] * scale)) // 2
        return pad_left, pad_top

    @staticmethod
    def _scale_image(image: np.ndarray, image_size: tuple[int, int], scale: float) -> np.ndarray:
        """ Scale the image and optional pad to given size

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The image to be scalued
        image_size: tuple
            The image (height, width)
        scale: float
            The scaling factor to apply to the image

        Returns
        -------
        :class:`numpy.ndarray`
            The scaled image
        """
        interpln = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        if scale != 1.0:
            dims = (int(image_size[1] * scale), int(image_size[0] * scale))
            logger.trace("Resizing detection image from %s to %s. "  # type:ignore[attr-defined]
                         "Scale=%s",
                         "x".join(str(i) for i in reversed(image_size)),
                         "x".join(str(i) for i in dims), scale)
            image = cv2.resize(image, dims, interpolation=interpln)
        logger.trace("Resized image shape: %s", image.shape)  # type:ignore[attr-defined]
        return image

    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        """ Pad a resized image to input size

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The image to have padding applied

        Returns
        -------
        :class:`numpy.ndarray`
            The image with padding applied
        """
        height, width = image.shape[:2]
        if width < self.input_size or height < self.input_size:
            pad_l = (self.input_size - width) // 2
            pad_r = (self.input_size - width) - pad_l
            pad_t = (self.input_size - height) // 2
            pad_b = (self.input_size - height) - pad_t
            image = cv2.copyMakeBorder(image,
                                       pad_t,
                                       pad_b,
                                       pad_l,
                                       pad_r,
                                       cv2.BORDER_CONSTANT)
        logger.trace("Padded image shape: %s", image.shape)  # type:ignore[attr-defined]
        return image

    # <<< FINALIZE METHODS >>> #
    def _remove_zero_sized_faces(self, batch_faces: list[list[DetectedFace]]
                                 ) -> list[list[DetectedFace]]:
        """ Remove items from batch_faces where detected face is of zero size or face falls
        entirely outside of image

        Parameters
        ----------
        batch_faces: list
            List of detected face objects

        Returns
        -------
        list
            List of detected face objects with filtered out faces removed
        """
        logger.trace("Input sizes: %s", [len(face) for face in batch_faces])  # type: ignore
        retval = [[face
                   for face in faces
                   if face.right > 0 and face.left is not None and face.left < self.input_size
                   and face.bottom > 0 and face.top is not None and face.top < self.input_size]
                  for faces in batch_faces]
        logger.trace("Output sizes: %s", [len(face) for face in retval])  # type: ignore
        return retval

    def _filter_small_faces(self, detected_faces: list[list[DetectedFace]]
                            ) -> list[list[DetectedFace]]:
        """ Filter out any faces smaller than the min size threshold

        Parameters
        ----------
        detected_faces: list
            List of detected face objects

        Returns
        -------
        list
            List of detected face objects with filtered out faces removed
        """
        retval = []
        for faces in detected_faces:
            this_image = []
            for face in faces:
                assert face.width is not None and face.height is not None
                face_size = (face.width ** 2 + face.height ** 2) ** 0.5
                if face_size < self.min_size:
                    logger.debug("Removing detected face: (face_size: %s, min_size: %s",
                                 face_size, self.min_size)
                    continue
                this_image.append(face)
            retval.append(this_image)
        return retval

    # <<< IMAGE ROTATION METHODS >>> #
    @staticmethod
    def _get_rotation_angles(rotation: str | None) -> list[int]:
        """ Set the rotation angles.

        Parameters
        ----------
        str
            List of requested rotation angles

        Returns
        -------
        list
            The complete list of rotation angles to apply
        """
        rotation_angles = [0]

        if not rotation:
            logger.debug("Not setting rotation angles")
            return rotation_angles

        passed_angles = [int(angle)
                         for angle in rotation.split(",")
                         if int(angle) != 0]
        if len(passed_angles) == 1:
            rotation_step_size = passed_angles[0]
            rotation_angles.extend(range(rotation_step_size,
                                         360,
                                         rotation_step_size))
        elif len(passed_angles) > 1:
            rotation_angles.extend(passed_angles)

        logger.debug("Rotation Angles: %s", rotation_angles)
        return rotation_angles

    def _rotate_batch(self, batch: DetectorBatch, angle: int) -> None:
        """ Rotate images in a batch by given angle

            if any faces have already been detected for a batch, store the existing rotation
            matrix and replace the feed image with a placeholder

            Parameters
            ----------
            batch: :class:`DetectorBatch`
                The batch to apply rotation to
            angle: int
                The amount of degrees to rotate the image by
            """
        if angle == 0:
            # Set the initial batch so we always rotate from zero
            batch.initial_feed = batch.feed.copy()
            return

        feeds: list[np.ndarray] = []
        rotmats: list[np.ndarray] = []
        for img, faces, rotmat in zip(batch.initial_feed,
                                      batch.prediction,
                                      batch.rotation_matrix):
            if faces.any():
                image = np.zeros_like(img)
                matrix = rotmat
            else:
                image, matrix = self._rotate_image_by_angle(img, angle)
            feeds.append(image)
            rotmats.append(matrix)
        batch.feed = np.array(feeds, dtype="float32")
        batch.rotation_matrix = rotmats

    @staticmethod
    def _rotate_face(face: DetectedFace, rotation_matrix: np.ndarray) -> DetectedFace:
        """ Rotates the detection bounding box around the given rotation matrix.

        Parameters
        ----------
        face: :class:`DetectedFace`
            A :class:`DetectedFace` containing the `x`, `w`, `y`, `h` detection bounding box
            points.
        rotation_matrix: numpy.ndarray
            The rotation matrix to rotate the given object by.

        Returns
        -------
        :class:`DetectedFace`
            The same class with the detection bounding box points rotated by the given matrix.
        """
        logger.trace("Rotating face: (face: %s, rotation_matrix: %s)",  # type: ignore
                     face, rotation_matrix)
        bounding_box = [[face.left, face.top],
                        [face.right, face.top],
                        [face.right, face.bottom],
                        [face.left, face.bottom]]
        rotation_matrix = cv2.invertAffineTransform(rotation_matrix)

        points = np.array(bounding_box, "int32")
        points = np.expand_dims(points, axis=0)
        transformed = cv2.transform(points, rotation_matrix).astype("int32")
        rotated = transformed.squeeze()

        # Bounding box should follow x, y planes, so get min/max for non-90 degree rotations
        pt_x = min(pnt[0] for pnt in rotated)
        pt_y = min(pnt[1] for pnt in rotated)
        pt_x1 = max(pnt[0] for pnt in rotated)
        pt_y1 = max(pnt[1] for pnt in rotated)
        width = pt_x1 - pt_x
        height = pt_y1 - pt_y

        face.left = int(pt_x)
        face.top = int(pt_y)
        face.width = int(width)
        face.height = int(height)
        return face

    def _rotate_image_by_angle(self,
                               image: np.ndarray,
                               angle: int) -> tuple[np.ndarray, np.ndarray]:
        """ Rotate an image by a given angle.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The image to be rotated
        angle: int
            The angle, in degrees, to rotate the image by

        Returns
        -------
        image: :class:`numpy.ndarray`
            The rotated image
        rotation_matrix: :class:`numpy.ndarray`
            The rotation matrix used to rotate the image

        Reference
        ---------
        https://stackoverflow.com/questions/22041699
        """

        logger.trace("Rotating image: (image: %s, angle: %s)",  # type:ignore[attr-defined]
                     image.shape, angle)
        channels_first = image.shape[0] <= 4
        if channels_first:
            image = np.moveaxis(image, 0, 2)

        height, width = image.shape[:2]
        image_center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, -1.*angle, 1.)
        rotation_matrix[0, 2] += self.input_size / 2 - image_center[0]
        rotation_matrix[1, 2] += self.input_size / 2 - image_center[1]
        logger.trace("Rotated image: (rotation_matrix: %s",  # type:ignore[attr-defined]
                     rotation_matrix)
        image = cv2.warpAffine(image, rotation_matrix, (self.input_size, self.input_size))
        if channels_first:
            image = np.moveaxis(image, 2, 0)

        return image, rotation_matrix
