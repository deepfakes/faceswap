#!/usr/bin/env python3
""" Media objects for the manual adjustments tool """
import logging

import cv2
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TKFace():
    """ An object that holds a single :class:`tkinter.PhotoImage` face, ready for placement in the
    :class:`~tools.manual.faceviewer.frames.FacesViewer` canvas, along with the face's associated
    mesh annotation coordinates.

    Parameters
    ----------
    size: int, optional
        The pixel size of the face image. Default: `128`
    face: :class:`numpy.ndarray` or ``None``, optional
        The face, sized correctly, to create a :class:`tkinter.PhotoImage` from. Pass ``None`` if
        an empty photo image should be created. Default: ``None``
    mask: :class:`numpy.ndarray` or ``None``, optional
        The mask to be applied to the face image. Pass ``None`` if no mask is to be used.
        Default ``None``
    """
    def __init__(self, face, size=128, mask=None):
        logger.trace("Initializing %s: (face: %s, size: %s, mask: %s)",
                     self.__class__.__name__,
                     face if face is None else face.shape,
                     size,
                     mask if mask is None else mask.shape)
        self._size = size
        if face.ndim == 2 and face.shape[1] == 1:
            self._face = self._image_from_jpg(face)
        else:
            self._face = face[..., 2::-1]
        self._photo = ImageTk.PhotoImage(self._generate_tk_face_data(mask))

        logger.trace("Initialized %s", self.__class__.__name__)

    # << PUBLIC PROPERTIES >> #
    @property
    def photo(self):
        """ :class:`tkinter.PhotoImage`: The face in a format that can be placed on the
        :class:`~tools.manual.manual.FaceViewer` canvas. """
        return self._photo

    # << PUBLIC METHODS >> #
    def update(self, face, mask):
        """ Update the :attr:`face`, :attr:`mesh_points` and attr:`mesh_is_poly` objects with the
        given information.

        Parameters
        ----------
        face: :class:`numpy.ndarray`
            The face, sized correctly, to be updated in :attr:`tk_faces`
        mask: :class:`numpy.ndarray` or ``None``
            The mask to be applied to the face image. Pass ``None`` if no mask is to be used
        """
        self._face = face[..., 2::-1]
        self._photo.paste(self._generate_tk_face_data(mask))

    def update_mask(self, mask):
        """ Update the mask in the 4th channel of :attr:`face` to the given mask.

        Parameters
        ----------
        mask: :class:`numpy.ndarray` or ``None``
            The mask to be applied to the face image. Pass ``None`` if no mask is to be used
        """
        self._photo.paste(self._generate_tk_face_data(mask))

    # << PRIVATE METHODS >> #
    def _image_from_jpg(self, face):
        """ Convert an encoded jpg into 3 channel BGR image.

        Parameters
        ----------
        face: :class:`numpy.ndarray`
            The encoded jpg as a two dimension numpy array

        Returns
        -------
        :class:`numpy.ndarray`
            The decoded jpg as a 3 channel BGR image
        """
        face = cv2.imdecode(face, cv2.IMREAD_UNCHANGED)
        interp = cv2.INTER_CUBIC if face.shape[0] < self._size else cv2.INTER_AREA
        if face.shape[0] != self._size:
            face = cv2.resize(face, (self._size, self._size), interpolation=interp)
        return face[..., 2::-1]

    def _generate_tk_face_data(self, mask):
        """ Create the :class:`tkinter.PhotoImage` face for the given face image.

        Parameters
        ----------
        mask: :class:`numpy.ndarray` or ``None``
            The mask to add to the image. ``None`` if a mask is not being used

        Returns
        -------
        :class:`tkinter.PhotoImage`
            The face formatted for the :class:`~tools.manual.manual.FaceViewer` canvas.
        """
        mask = np.ones(self._face.shape[:2], dtype="uint8") * 255 if mask is None else mask
        if mask.shape[0] != self._size:
            mask = cv2.resize(mask, self._face.shape[:2], interpolation=cv2.INTER_AREA)
        img = np.concatenate((self._face, mask[..., None]), axis=-1)
        return Image.fromarray(img)
