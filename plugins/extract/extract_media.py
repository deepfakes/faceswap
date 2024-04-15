#!/usr/bin/env python3
""" Object for holding and manipulating media passing through a faceswap extraction pipeline """
from __future__ import annotations
import logging
import typing as T

import cv2

from lib.logger import parse_class_init

if T.TYPE_CHECKING:
    import numpy as np
    from lib.align.alignments import PNGHeaderSourceDict
    from lib.align.detected_face import DetectedFace

logger = logging.getLogger(__name__)


class ExtractMedia:
    """ An object that passes through the :class:`~plugins.extract.pipeline.Extractor` pipeline.

    Parameters
    ----------
    filename: str
        The base name of the original frame's filename
    image: :class:`numpy.ndarray`
        The original frame or a faceswap aligned face image
    detected_faces: list, optional
        A list of :class:`~lib.align.DetectedFace` objects. Detected faces can be added
        later with :func:`add_detected_faces`. Setting ``None`` will default to an empty list.
        Default: ``None``
    is_aligned: bool, optional
        ``True`` if the :attr:`image` is an aligned faceswap image otherwise ``False``. Used for
        face filtering with vggface2. Aligned faceswap images will automatically skip detection,
        alignment and masking. Default: ``False``
    """

    def __init__(self,
                 filename: str,
                 image: np.ndarray,
                 detected_faces: list[DetectedFace] | None = None,
                 is_aligned: bool = False) -> None:
        logger.trace(parse_class_init(locals()))  # type:ignore[attr-defined]
        self._filename = filename
        self._image: np.ndarray | None = image
        self._image_shape = T.cast(tuple[int, int, int], image.shape)
        self._detected_faces: list[DetectedFace] = ([] if detected_faces is None
                                                    else detected_faces)
        self._is_aligned = is_aligned
        self._frame_metadata: PNGHeaderSourceDict | None = None
        self._sub_folders: list[str | None] = []

    @property
    def filename(self) -> str:
        """ str: The base name of the :attr:`image` filename. """
        return self._filename

    @property
    def image(self) -> np.ndarray:
        """ :class:`numpy.ndarray`: The source frame for this object. """
        assert self._image is not None
        return self._image

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """ tuple: The shape of the stored :attr:`image`. """
        return self._image_shape

    @property
    def image_size(self) -> tuple[int, int]:
        """ tuple: The (`height`, `width`) of the stored :attr:`image`. """
        return self._image_shape[:2]

    @property
    def detected_faces(self) -> list[DetectedFace]:
        """list: A list of :class:`~lib.align.DetectedFace` objects in the :attr:`image`. """
        return self._detected_faces

    @property
    def is_aligned(self) -> bool:
        """ bool. ``True`` if :attr:`image` is an aligned faceswap image otherwise ``False`` """
        return self._is_aligned

    @property
    def frame_metadata(self) -> PNGHeaderSourceDict:
        """ dict: The frame metadata that has been added from an aligned image. This property
        should only be called after :func:`add_frame_metadata` has been called when processing
        an aligned face. For all other instances an assertion error will be raised.

        Raises
        ------
        AssertionError
            If frame metadata has not been populated from an aligned image
        """
        assert self._frame_metadata is not None
        return self._frame_metadata

    @property
    def sub_folders(self) -> list[str | None]:
        """ list: The sub_folders that the faces should be output to. Used when binning filter
        output is enabled. The list corresponds to the list of detected faces
        """
        return self._sub_folders

    def get_image_copy(self, color_format: T.Literal["BGR", "RGB", "GRAY"]) -> np.ndarray:
        """ Get a copy of the image in the requested color format.

        Parameters
        ----------
        color_format: ['BGR', 'RGB', 'GRAY']
            The requested color format of :attr:`image`

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in the requested :attr:`color_format`
        """
        logger.trace("Requested color format '%s' for frame '%s'",  # type:ignore[attr-defined]
                     color_format, self._filename)
        image = getattr(self, f"_image_as_{color_format.lower()}")()
        return image

    def add_detected_faces(self, faces: list[DetectedFace]) -> None:
        """ Add detected faces to the object. Called at the end of each extraction phase.

        Parameters
        ----------
        faces: list
            A list of :class:`~lib.align.DetectedFace` objects
        """
        logger.trace("Adding detected faces for filename: '%s'. "  # type:ignore[attr-defined]
                     "(faces: %s, lrtb: %s)", self._filename, faces,
                     [(face.left, face.right, face.top, face.bottom) for face in faces])
        self._detected_faces = faces

    def add_sub_folders(self, folders: list[str | None]) -> None:
        """ Add detected faces to the object. Called at the end of each extraction phase.

        Parameters
        ----------
        folders: list
            A list of str sub folder names or ``None`` if no sub folder is required. Should
            correspond to the detected faces list
        """
        logger.trace("Adding sub folders for filename: '%s'. "  # type:ignore[attr-defined]
                     "(folders: %s)", self._filename, folders,)
        self._sub_folders = folders

    def remove_image(self) -> None:
        """ Delete the image and reset :attr:`image` to ``None``.

        Required for multi-phase extraction to avoid the frames stacking RAM.
        """
        logger.trace("Removing image for filename: '%s'",  # type:ignore[attr-defined]
                     self._filename)
        del self._image
        self._image = None

    def set_image(self, image: np.ndarray) -> None:
        """ Add the image back into :attr:`image`

        Required for multi-phase extraction adds the image back to this object.

        Parameters
        ----------
        image: :class:`numpy.ndarry`
            The original frame to be re-applied to for this :attr:`filename`
        """
        logger.trace("Reapplying image: (filename: `%s`, "  # type:ignore[attr-defined]
                     "image shape: %s)", self._filename, image.shape)
        self._image = image

    def add_frame_metadata(self, metadata: PNGHeaderSourceDict) -> None:
        """ Add the source frame metadata from an aligned PNG's header data.

        metadata: dict
            The contents of the 'source' field in the PNG header
        """
        logger.trace("Adding PNG Source data for '%s': %s",  # type:ignore[attr-defined]
                     self._filename, metadata)
        dims = T.cast(tuple[int, int], metadata["source_frame_dims"])
        self._image_shape = (*dims, 3)
        self._frame_metadata = metadata

    def _image_as_bgr(self) -> np.ndarray:
        """ Get a copy of the source frame in BGR format.

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in BGR color format """
        return self.image[..., :3].copy()

    def _image_as_rgb(self) -> np.ndarray:
        """ Get a copy of the source frame in RGB format.

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in RGB color format """
        return self.image[..., 2::-1].copy()

    def _image_as_gray(self) -> np.ndarray:
        """ Get a copy of the source frame in gray-scale format.

        Returns
        -------
        :class:`numpy.ndarray`:
            A copy of :attr:`image` in gray-scale color format """
        return cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
