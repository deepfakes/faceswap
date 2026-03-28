#!/usr/bin/env python3
"""Manages the widgets that hold the top 'viewer' area of the preview tool"""
from __future__ import annotations
import logging
import os
import tkinter as tk
import typing as T

from tkinter import ttk
from dataclasses import dataclass, field, InitVar

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.align import transform_image
from lib.align.aligned_face import CenteringType
from lib.logger import parse_class_init
from lib.utils import get_module_objects
from scripts.convert import ConvertItem


if T.TYPE_CHECKING:
    import numpy.typing as npt
    from .preview import Preview

logger = logging.getLogger(__name__)


@dataclass
class _Faces:
    """Dataclass for holding faces

    Parameters
    ----------
    size
        The size of each individual face sample in pixels
    num_faces
        The number of faces to be displayed in the preview window
    """
    num_faces: InitVar[int]
    size: InitVar[int]

    filenames: list[str] = field(default_factory=list)
    matrix: npt.NDArray[np.float32] = field(init=False)
    src: npt.NDArray[np.uint8] = field(init=False)
    dst: npt.NDArray[np.uint8] = field(init=False)

    def __post_init__(self, num_faces: int, size: int) -> None:
        """Initialize the matrices based on input sizes"""
        self.matrix = np.empty((num_faces, 2, 3), dtype=np.float32)
        self.src = np.empty((num_faces, size, size, 3), dtype=np.uint8)
        self.dst = np.empty((num_faces, size, size, 3), dtype=np.uint8)


class FacesDisplay():  # pylint:disable=too-many-instance-attributes
    """Compiles the 2 rows of sample faces (original and swapped) into a single image

    Parameters
    ----------
    app
        The main tkinter Preview app
    size
        The size of each individual face sample in pixels
    padding
        The amount of extra padding to apply to the outside of the face
    num_faces
        The number of faces to be displayed in the preview window
    """
    def __init__(self, app: Preview, size: int, padding: int, num_faces: int) -> None:
        logger.debug(parse_class_init(locals()))
        self._size = size
        self._display_dims = (1, 1)
        self._app = app
        self._padding = padding
        self._num_faces = num_faces

        self._faces = _Faces(num_faces=num_faces, size=size)
        self._centering: CenteringType | None = None
        self._y_offset = 0.0
        self._faces_source: np.ndarray = np.array([])
        self._faces_dest: np.ndarray = np.array([])
        self._tk_image: ImageTk.PhotoImage | None = None

        # Set from Samples
        self.update_source: bool = False
        """Flag to indicate that the source images for the preview have been updated, so the
        preview should be recompiled."""
        self.source: list[ConvertItem] = []  # Source images, filenames + detected faces
        """The list of :class:`numpy.ndarray` source preview images for top row of display"""
        # Set from Patch
        self.destination: list[np.ndarray] = []  # Swapped + patched images
        """The list of :class:`numpy.ndarray` swapped and patched preview images for bottom row of
        display"""

        logger.trace("Initialized %s", self.__class__.__name__)  # type: ignore

    @property
    def tk_image(self) -> ImageTk.PhotoImage | None:
        """The compiled preview display in tkinter display format"""
        return self._tk_image

    @property
    def _total_columns(self) -> int:
        """The total number of images that are being displayed"""
        return len(self.source)

    def set_centering_offset(self, centering: CenteringType, y_offset: float) -> None:
        """The centering and y-offset that the model uses is not known at initialization time.
        Set :attr:`_centering` and y_offset when the model has been loaded.

        Parameters
        ----------
        centering
            The centering that the model was trained on
        """
        self._centering = centering
        self._y_offset = y_offset

    def set_display_dimensions(self, dimensions: tuple[int, int]) -> None:
        """Adjust the size of the frame that will hold the preview samples.

        Parameters
        ----------
        dimensions
            The (`width`, `height`) of the frame that holds the preview
        """
        self._display_dims = dimensions

    def update_tk_image(self) -> None:
        """Build the full preview images and compile :attr:`tk_image` for display."""
        logger.trace("Updating tk image")  # type: ignore
        self._build_faces_image()
        img = np.vstack((self._faces_source, self._faces_dest))
        size = self._get_scale_size(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(size, Image.Resampling.BICUBIC)
        self._tk_image = ImageTk.PhotoImage(pil_img)
        logger.trace("Updated tk image")  # type: ignore

    def _get_scale_size(self, image: np.ndarray) -> tuple[int, int]:
        """Get the size that the full preview image should be resized to fit in the
        display window.

        Parameters
        ----------
        image
            The full sized compiled preview image

        Returns
        -------
        The (`width`, `height`) that the display image should be sized to fit in the display window
        """
        frame_ratio = float(self._display_dims[0]) / float(self._display_dims[1])
        img_ratio = float(image.shape[1]) / float(image.shape[0])

        if frame_ratio <= img_ratio:
            scale = self._display_dims[0] / float(image.shape[1])
            size = (self._display_dims[0], max(1, int(image.shape[0] * scale)))
        else:
            scale = self._display_dims[1] / float(image.shape[0])
            size = (max(1, int(image.shape[1] * scale)), self._display_dims[1])
        logger.trace("scale: %s, size: %s", scale, size)  # type: ignore
        return size

    def _build_faces_image(self) -> None:
        """Compile the source and destination rows of the preview image."""
        logger.trace("Building Faces Image")  # type: ignore
        update_all = self.update_source
        self._faces_from_frames()
        if update_all:
            header = self._header_text()
            source = np.hstack([self._draw_rect(face) for face in self._faces.src])
            self._faces_source = np.vstack((header, source))
        self._faces_dest = np.hstack([self._draw_rect(face) for face in self._faces.dst])
        logger.debug("source row shape: %s, swapped row shape: %s",
                     self._faces_dest.shape, self._faces_source.shape)

    def _faces_from_frames(self) -> None:
        """Extract the preview faces from the source frames and apply the requisite padding."""
        logger.debug("Extracting faces from frames: Number images: %s", len(self.source))
        if self.update_source:
            self._crop_source_faces()
        self._crop_destination_faces()
        logger.debug("Extracted faces from frames: %s",
                     {k: len(v) for k, v in self._faces.__dict__.items()})

    def _crop_source_faces(self) -> None:
        """Extract the source faces from the source frames, along with their filenames and the
        transformation matrix used to extract the faces."""
        logger.debug("Updating source faces")
        self._faces = _Faces(num_faces=self._num_faces, size=self._size)  # Init new class
        for i, item in enumerate(self.source):
            detected_face = item.inbound.detected_faces[0]
            src_img = item.inbound.image
            detected_face.load_aligned(src_img,
                                       size=self._size,
                                       centering=T.cast(CenteringType, self._centering))
            matrix = detected_face.aligned.matrix
            if self._y_offset:
                matrix = matrix.copy()
                matrix[1, 2] += self._y_offset
            self._faces.filenames.append(os.path.splitext(item.inbound.filename)[0])
            self._faces.matrix[i] = matrix
            self._faces.src[i] = transform_image(src_img, matrix, self._size, self._padding)
        self.update_source = False
        logger.debug("Updated source faces")

    def _crop_destination_faces(self) -> None:
        """Extract the swapped faces from the swapped frames using the source face destination
        matrices."""
        logger.debug("Updating destination faces")
        destination = self.destination if self.destination else [np.ones_like(src.inbound.image)
                                                                 for src in self.source]
        for i, image in enumerate(destination):
            self._faces.dst[i] = transform_image(image,
                                                 self._faces.matrix[i],
                                                 self._size,
                                                 self._padding)
        logger.debug("Updated destination faces")

    def _header_text(self) -> np.ndarray:
        """Create the header text displaying the frame name for each preview column.

        Returns
        -------
        The header row of the preview image containing the frame names for each column
        """
        font_scale = self._size / 640
        height = self._size // 8
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Get size of placed text for positioning
        text_sizes = [cv2.getTextSize(self._faces.filenames[idx],
                                      font,
                                      font_scale,
                                      1)[0]
                      for idx in range(self._total_columns)]
        # Get X and Y co-ordinates for each text item
        text_y = int((height + text_sizes[0][1]) / 2)
        text_x = [int((self._size - text_sizes[idx][0]) / 2) + self._size * idx
                  for idx in range(self._total_columns)]
        logger.debug("filenames: %s, text_sizes: %s, text_x: %s, text_y: %s",
                     self._faces.filenames, text_sizes, text_x, text_y)
        header_box = np.ones((height, self._size * self._total_columns, 3), np.uint8) * 255
        for idx, text in enumerate(self._faces.filenames):
            cv2.putText(header_box,
                        text,
                        (text_x[idx], text_y),
                        font,
                        font_scale,
                        (0, 0, 0),
                        1,
                        lineType=cv2.LINE_AA)
        logger.debug("header_box.shape: %s", header_box.shape)
        return header_box

    def _draw_rect(self, image: np.ndarray) -> np.ndarray:
        """Place a white border around a given image.

        Parameters
        ----------
        image
            The image to place a border on to

        Returns
        -------
        The given image with a border drawn around the outside
        """
        cv2.rectangle(image, (0, 0), (self._size - 1, self._size - 1), (255, 255, 255), 1)
        image = np.clip(image, 0.0, 255.0)
        return image.astype("uint8")


class ImagesCanvas(ttk.Frame):  # pylint:disable=too-many-ancestors
    """tkinter Canvas that holds the preview images.

    Parameters
    ----------
    app
        The main tkinter Preview app
    parent
        The parent tkinter object that holds the canvas
    """
    def __init__(self, app: Preview, parent: ttk.PanedWindow) -> None:
        logger.debug("Initializing %s: (app: %s, parent: %s)",
                     self.__class__.__name__, app, parent)
        super().__init__(parent)
        self.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

        self._display: FacesDisplay = parent.preview_display  # type: ignore
        self._canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self._canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._display_canvas = self._canvas.create_image(0, 0,
                                                         image=self._display.tk_image,
                                                         anchor=tk.NW)
        self.bind("<Configure>", self._resize)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _resize(self, event: tk.Event) -> None:
        """Resize the image to fit the frame, maintaining aspect ratio."""
        logger.debug("Resizing preview image")
        frame_size = (event.width, event.height)
        self._display.set_display_dimensions(frame_size)
        self.reload()

    def reload(self) -> None:
        """Update the images in the canvas and redraw."""
        logger.debug("Reloading preview image")
        self._display.update_tk_image()
        self._canvas.itemconfig(self._display_canvas, image=self._display.tk_image)
        logger.debug("Reloaded preview image")


__all__ = get_module_objects(__name__)
