#!/usr/bin/env python3
""" Base class for Face Detector plugins

All Detector Plugins should inherit from this class.
See the override methods for which methods are required.

The plugin will receive a :class:`~plugins.extract.pipeline.ExtractMedia` object.

For each source frame, the plugin must pass a dict to finalize containing:

>>> {'filename': <filename of source frame>,
>>>  'detected_faces': <list of DetectedFace objects containing bounding box points}}

To get a :class:`~lib.faces_detect.DetectedFace` object use the function:

>>> face = self.to_detected_face(<face left>, <face top>, <face right>, <face bottom>)

"""
import cv2
import numpy as np

from lib.faces_detect import DetectedFace
from plugins.extract._base import Extractor, logger


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

    def __init__(self, git_model_id=None, model_filename=None,
                 configfile=None, instance=0, rotation=None, min_size=0):
        logger.debug("Initializing %s: (rotation: %s, min_size: %s)", self.__class__.__name__,
                     rotation, min_size)
        super().__init__(git_model_id,
                         model_filename,
                         configfile=configfile,
                         instance=instance)
        self.rotation = self._get_rotation_angles(rotation)
        self.min_size = min_size

        self._plugin_type = "detect"

        logger.debug("Initialized _base %s", self.__class__.__name__)

    # <<< QUEUE METHODS >>> #
    def get_batch(self, queue):
        """ Get items for inputting to the detector plugin in batches

        Items are received as :class:`~plugins.extract.pipeline.ExtractMedia` objects and converted
        to ``dict`` for internal processing.

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
        >>>  'detected_faces': [[<lib.faces_detect.DetectedFace objects]]}

        Parameters
        ----------
        queue : queue.Queue()
            The ``queue`` that the batch will be fed from. This will be a queue that loads
            images.

        Returns
        -------
        exhausted, bool
            ``True`` if queue is exhausted, ``False`` if not.
        batch, dict
            A dictionary of lists of :attr:`~plugins.extract._base.Extractor.batchsize`.
        """
        exhausted = False
        batch = dict()
        for _ in range(self.batchsize):
            item = self._get_item(queue)
            if item == "EOF":
                exhausted = True
                break
            batch.setdefault("filename", []).append(item.filename)
            image, scale, pad = self._compile_detection_image(item)
            batch.setdefault("image", []).append(image)
            batch.setdefault("scale", []).append(scale)
            batch.setdefault("pad", []).append(pad)

        if batch:
            batch["image"] = np.array(batch["image"], dtype="float32")
            logger.trace("Returning batch: %s", {k: v.shape if isinstance(v, np.ndarray) else v
                                                 for k, v in batch.items()})
        else:
            logger.trace(item)
        return exhausted, batch

    # <<< FINALIZE METHODS>>> #
    def finalize(self, batch):
        """ Finalize the output from Detector

        This should be called as the final task of each ``plugin``.

        Parameters
        ----------
        batch : dict
            The final ``dict`` from the `plugin` process. It must contain the keys  ``filename``,
            ``faces``

        Yields
        ------
        :class:`~plugins.extract.pipeline.ExtractMedia`
            The :attr:`DetectedFaces` list will be populated for this class with the bounding boxes
            for the detected faces found in the frame.
        """
        if not isinstance(batch, dict):
            logger.trace("Item out: %s", batch)
            return batch

        logger.trace("Item out: %s", {k: v.shape if isinstance(v, np.ndarray) else v
                                      for k, v in batch.items()})

        batch_faces = [[self.to_detected_face(face[0], face[1], face[2], face[3])
                        for face in faces]
                       for faces in batch["prediction"]]
        # Rotations
        if any(m.any() for m in batch["rotmat"]) and any(batch_faces):
            batch_faces = [[self._rotate_face(face, rotmat) if rotmat.any() else face
                            for face in faces]
                           for faces, rotmat in zip(batch_faces, batch["rotmat"])]

        # Remove zero sized faces
        batch_faces = self._remove_zero_sized_faces(batch_faces)

        # Scale back out to original frame
        batch["detected_faces"] = [[self.to_detected_face((face.left - pad[0]) / scale,
                                                          (face.top - pad[1]) / scale,
                                                          (face.right - pad[0]) / scale,
                                                          (face.bottom - pad[1]) / scale)
                                    for face in faces]
                                   for scale, pad, faces in zip(batch["scale"],
                                                                batch["pad"],
                                                                batch_faces)]

        if self.min_size > 0 and batch.get("detected_faces", None):
            batch["detected_faces"] = self._filter_small_faces(batch["detected_faces"])

        batch = self._dict_lists_to_list_dicts(batch)
        for item in batch:
            output = self._extract_media.pop(item["filename"])
            output.add_detected_faces(item["detected_faces"])
            logger.trace("final output: (filename: '%s', image shape: %s, detected_faces: %s, "
                         "item: %s", output.filename, output.image_shape, output.detected_faces,
                         output)
            yield output

    @staticmethod
    def to_detected_face(left, top, right, bottom):
        """ Return a :class:`~lib.faces_detect.DetectedFace` object for the bounding box """
        return DetectedFace(x=int(round(left)),
                            w=int(round(right - left)),
                            y=int(round(top)),
                            h=int(round(bottom - top)))

    # <<< PROTECTED ACCESS METHODS >>> #
    # <<< PREDICT WRAPPER >>> #
    def _predict(self, batch):
        """ Wrap models predict function in rotations """
        batch["rotmat"] = [np.array([]) for _ in range(len(batch["feed"]))]
        found_faces = [np.array([]) for _ in range(len(batch["feed"]))]
        for angle in self.rotation:
            # Rotate the batch and insert placeholders for already found faces
            self._rotate_batch(batch, angle)
            batch = self.predict(batch)

            if angle != 0 and any([face.any() for face in batch["prediction"]]):
                logger.verbose("found face(s) by rotating image %s degrees", angle)

            found_faces = [face if not found.any() else found
                           for face, found in zip(batch["prediction"], found_faces)]

            if all([face.any() for face in found_faces]):
                logger.trace("Faces found for all images")
                break

        batch["prediction"] = found_faces
        logger.trace("detect_prediction output: (filenames: %s, prediction: %s, rotmat: %s)",
                     batch["filename"], batch["prediction"], batch["rotmat"])
        return batch

    # <<< DETECTION IMAGE COMPILATION METHODS >>> #
    def _compile_detection_image(self, item):
        """ Compile the detection image for feeding into the model

        Parameters
        ----------
        item: :class:`plugins.extract.pipeline.ExtractMedia`
            The input item from the pipeline
        """
        image = item.get_image_copy(self.color_format)
        scale = self._set_scale(item.image_size)
        pad = self._set_padding(item.image_size, scale)

        image = self._scale_image(image, item.image_size, scale)
        image = self._pad_image(image)
        logger.trace("compiled: (images shape: %s, scale: %s, pad: %s)", image.shape, scale, pad)
        return image, scale, pad

    def _set_scale(self, image_size):
        """ Set the scale factor for incoming image """
        scale = self.input_size / max(image_size)
        logger.trace("Detector scale: %s", scale)
        return scale

    def _set_padding(self, image_size, scale):
        """ Set the image padding for non-square images """
        pad_left = int(self.input_size - int(image_size[1] * scale)) // 2
        pad_top = int(self.input_size - int(image_size[0] * scale)) // 2
        return pad_left, pad_top

    @staticmethod
    def _scale_image(image, image_size, scale):
        """ Scale the image and optional pad to given size """
        interpln = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        if scale != 1.0:
            dims = (int(image_size[1] * scale), int(image_size[0] * scale))
            logger.trace("Resizing detection image from %s to %s. Scale=%s",
                         "x".join(str(i) for i in reversed(image_size)),
                         "x".join(str(i) for i in dims), scale)
            image = cv2.resize(image, dims, interpolation=interpln)
        logger.trace("Resized image shape: %s", image.shape)
        return image

    def _pad_image(self, image):
        """ Pad a resized image to input size """
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
        logger.trace("Padded image shape: %s", image.shape)
        return image

    # <<< FINALIZE METHODS >>> #
    def _remove_zero_sized_faces(self, batch_faces):
        """ Remove items from batch_faces where detected face is of zero size
            or face falls entirely outside of image """
        logger.trace("Input sizes: %s", [len(face) for face in batch_faces])
        retval = [[face
                   for face in faces
                   if face.right > 0 and face.left < self.input_size
                   and face.bottom > 0 and face.top < self.input_size]
                  for faces in batch_faces]
        logger.trace("Output sizes: %s", [len(face) for face in retval])
        return retval

    def _filter_small_faces(self, detected_faces):
        """ Filter out any faces smaller than the min size threshold """
        retval = []
        for faces in detected_faces:
            this_image = []
            for face in faces:
                face_size = (face.w ** 2 + face.h ** 2) ** 0.5
                if face_size < self.min_size:
                    logger.debug("Removing detected face: (face_size: %s, min_size: %s",
                                 face_size, self.min_size)
                    continue
                this_image.append(face)
            retval.append(this_image)
        return retval

    # <<< IMAGE ROTATION METHODS >>> #
    @staticmethod
    def _get_rotation_angles(rotation):
        """ Set the rotation angles. Includes backwards compatibility for the
            'on' and 'off' options:
                - 'on' - increment 90 degrees
                - 'off' - disable
                - 0 is prepended to the list, as whatever happens, we want to
                  scan the image in it's upright state """
        rotation_angles = [0]

        if not rotation or rotation.lower() == "off":
            logger.debug("Not setting rotation angles")
            return rotation_angles

        if rotation.lower() == "on":
            rotation_angles.extend(range(90, 360, 90))
        else:
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

    def _rotate_batch(self, batch, angle):
        """ Rotate images in a batch by given angle
            if any faces have already been detected for a batch, store the existing rotation
            matrix and replace the feed image with a placeholder """
        if angle == 0:
            # Set the initial batch so we always rotate from zero
            batch["initial_feed"] = batch["feed"].copy()
            return

        retval = dict()
        for img, faces, rotmat in zip(batch["initial_feed"], batch["prediction"], batch["rotmat"]):
            if faces.any():
                image = np.zeros_like(img)
                matrix = rotmat
            else:
                image, matrix = self._rotate_image_by_angle(img, angle)
            retval.setdefault("feed", []).append(image)
            retval.setdefault("rotmat", []).append(matrix)
        batch["feed"] = np.array(retval["feed"], dtype="float32")
        batch["rotmat"] = retval["rotmat"]

    @staticmethod
    def _rotate_face(face, rotation_matrix):
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
        logger.trace("Rotating face: (face: %s, rotation_matrix: %s)", face, rotation_matrix)
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
        pt_x = min([pnt[0] for pnt in rotated])
        pt_y = min([pnt[1] for pnt in rotated])
        pt_x1 = max([pnt[0] for pnt in rotated])
        pt_y1 = max([pnt[1] for pnt in rotated])
        width = pt_x1 - pt_x
        height = pt_y1 - pt_y

        face.x = int(pt_x)
        face.y = int(pt_y)
        face.w = int(width)
        face.h = int(height)
        return face

    def _rotate_image_by_angle(self, image, angle):
        """ Rotate an image by a given angle.
            From: https://stackoverflow.com/questions/22041699 """

        logger.trace("Rotating image: (image: %s, angle: %s)", image.shape, angle)
        channels_first = image.shape[0] <= 4
        if channels_first:
            image = np.moveaxis(image, 0, 2)

        height, width = image.shape[:2]
        image_center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, -1.*angle, 1.)
        rotation_matrix[0, 2] += self.input_size / 2 - image_center[0]
        rotation_matrix[1, 2] += self.input_size / 2 - image_center[1]
        logger.trace("Rotated image: (rotation_matrix: %s", rotation_matrix)
        image = cv2.warpAffine(image, rotation_matrix, (self.input_size, self.input_size))
        if channels_first:
            image = np.moveaxis(image, 2, 0)

        return image, rotation_matrix
