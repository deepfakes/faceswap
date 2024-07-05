#!/usr/bin/env python3
""" Holds global tkinter variables and information pertaining to the entire Manual tool """
from __future__ import annotations

import logging
import os
import sys
import tkinter as tk

from dataclasses import dataclass, field

import cv2
import numpy as np

from lib.gui.utils import get_config
from lib.logger import parse_class_init
from lib.utils import VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)


@dataclass
class CurrentFrame:
    """ Dataclass for holding information about the currently displayed frame """
    image: np.ndarray = field(default_factory=lambda: np.zeros(1))
    """:class:`numpy.ndarry`: The currently displayed frame in original dimensions """
    scale: float = 1.0
    """float: The scaling factor to use to resize the image to the display window """
    interpolation: int = cv2.INTER_AREA
    """int: The opencv interpolator ID to use for resizing the image to the display window """
    display_dims: tuple[int, int] = (0, 0)
    """tuple[int, int]`: The size of the currently displayed frame, in the display window """
    filename: str = ""
    """str: The filename of the currently displayed frame """

    def __repr__(self) -> str:
        """ Clean string representation showing numpy arrays as shape and dtype

        Returns
        -------
        str
            Loggable representation of the dataclass
        """
        properties = [f"{k}={(v.shape, v.dtype) if isinstance(v, np.ndarray) else v}"
                      for k, v in self.__dict__.items()]
        return f"{self.__class__.__name__} ({', '.join(properties)}"


@dataclass
class TKVars:
    """ Holds the global TK Variables """
    frame_index: tk.IntVar
    """:class:`tkinter.IntVar`: The absolute frame index of the currently displayed frame"""
    transport_index: tk.IntVar
    """:class:`tkinter.IntVar`: The transport index of the currently displayed frame when filters
    have been applied """
    face_index: tk.IntVar
    """:class:`tkinter.IntVar`: The face index of the currently selected face"""
    filter_distance: tk.IntVar
    """:class:`tkinter.IntVar`: The amount to filter by distance"""

    update: tk.BooleanVar
    """:class:`tkinter.BooleanVar`: Whether an update has been performed """
    update_active_viewport: tk.BooleanVar
    """:class:`tkinter.BooleanVar`: Whether the viewport needs updating """
    is_zoomed: tk.BooleanVar
    """:class:`tkinter.BooleanVar`: Whether the main window is zoomed in to a face or out to a
    full frame"""

    filter_mode: tk.StringVar
    """:class:`tkinter.StringVar`: The currently selected filter mode """
    faces_size: tk.StringVar
    """:class:`tkinter.StringVar`: The pixel size of faces in the viewport """

    def __repr__(self) -> str:
        """ Clean string representation showing variable type as well as their value

        Returns
        -------
        str
            Loggable representation of the dataclass
        """
        properties = [f"{k}={v.__class__.__name__}({v.get()})" for k, v in self.__dict__.items()]
        return f"{self.__class__.__name__} ({', '.join(properties)}"


class TkGlobals():
    """ Holds Tkinter Variables and other frame information that need to be accessible from all
    areas of the GUI.

    Parameters
    ----------
    input_location: str
        The location of the input folder of frames or video file
    """
    def __init__(self, input_location: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._tk_vars = self._get_tk_vars()

        self._is_video = self._check_input(input_location)
        self._frame_count = 0  # set by FrameLoader
        self._frame_display_dims = (int(round(896 * get_config().scaling_factor)),
                                    int(round(504 * get_config().scaling_factor)))
        self._current_frame = CurrentFrame()
        logger.debug("Initialized %s", self.__class__.__name__)

    @classmethod
    def _get_tk_vars(cls) -> TKVars:
        """ Create and initialize the tkinter variables.

        Returns
        -------
        :class:`TKVars`
            The global tkinter variables
        """
        retval = TKVars(frame_index=tk.IntVar(value=0),
                        transport_index=tk.IntVar(value=0),
                        face_index=tk.IntVar(value=0),
                        filter_distance=tk.IntVar(value=10),
                        update=tk.BooleanVar(value=False),
                        update_active_viewport=tk.BooleanVar(value=False),
                        is_zoomed=tk.BooleanVar(value=False),
                        filter_mode=tk.StringVar(),
                        faces_size=tk.StringVar())
        logger.debug(retval)
        return retval

    @property
    def current_frame(self) -> CurrentFrame:
        """ :class:`CurrentFrame`: The currently displayed frame in the frame viewer with it's
        meta information. """
        return self._current_frame

    @property
    def frame_count(self) -> int:
        """ int: The total number of frames for the input location """
        return self._frame_count

    @property
    def frame_display_dims(self) -> tuple[int, int]:
        """ tuple: The (`width`, `height`) of the video display frame in pixels. """
        return self._frame_display_dims

    @property
    def is_video(self) -> bool:
        """ bool: ``True`` if the input is a video file, ``False`` if it is a folder of images. """
        return self._is_video

    # TK Variables that need to be exposed
    @property
    def var_full_update(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`: Flag to indicate that whole GUI should be refreshed """
        return self._tk_vars.update

    @property
    def var_transport_index(self) -> tk.IntVar:
        """ :class:`tkinter.IntVar`: The current index of the display frame's transport slider. """
        return self._tk_vars.transport_index

    @property
    def var_frame_index(self) -> tk.IntVar:
        """ :class:`tkinter.IntVar`: The current absolute frame index of the currently
        displayed frame. """
        return self._tk_vars.frame_index

    @property
    def var_filter_distance(self) -> tk.IntVar:
        """ :class:`tkinter.IntVar`: The variable holding the currently selected threshold
        distance for misaligned filter mode. """
        return self._tk_vars.filter_distance

    @property
    def var_filter_mode(self) -> tk.StringVar:
        """ :class:`tkinter.StringVar`: The variable holding the currently selected navigation
        filter mode. """
        return self._tk_vars.filter_mode

    @property
    def var_faces_size(self) -> tk.StringVar:
        """ :class:`tkinter..IntVar`: The variable holding the currently selected Faces Viewer
        thumbnail size. """
        return self._tk_vars.faces_size

    @property
    def var_update_active_viewport(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`: Boolean Variable that is traced by the viewport's active
        frame to update. """
        return self._tk_vars.update_active_viewport

    # Raw values returned from TK Variables
    @property
    def face_index(self) -> int:
        """ int: The currently displayed face index when in zoomed mode. """
        return self._tk_vars.face_index.get()

    @property
    def frame_index(self) -> int:
        """ int: The currently displayed frame index. NB This returns -1 if there are no frames
        that meet the currently selected filter criteria. """
        return self._tk_vars.frame_index.get()

    @property
    def is_zoomed(self) -> bool:
        """ bool: ``True`` if the frame viewer is zoomed into a face, ``False`` if the frame viewer
        is displaying a full frame. """
        return self._tk_vars.is_zoomed.get()

    @staticmethod
    def _check_input(frames_location: str) -> bool:
        """ Check whether the input is a video

        Parameters
        ----------
        frames_location: str
            The input location for video or images

        Returns
        -------
        bool: 'True' if input is a video 'False' if it is a folder.
        """
        if os.path.isdir(frames_location):
            retval = False
        elif os.path.splitext(frames_location)[1].lower() in VIDEO_EXTENSIONS:
            retval = True
        else:
            logger.error("The input location '%s' is not valid", frames_location)
            sys.exit(1)
        logger.debug("Input '%s' is_video: %s", frames_location, retval)
        return retval

    def set_face_index(self, index: int) -> None:
        """ Set the currently selected face index

        Parameters
        ----------
        index: int
            The currently selected face index
        """
        logger.trace("Setting face index from %s to %s",  # type:ignore[attr-defined]
                     self.face_index, index)
        self._tk_vars.face_index.set(index)

    def set_frame_count(self, count: int) -> None:
        """ Set the count of total number of frames to :attr:`frame_count` when the
        :class:`FramesLoader` has completed loading.

        Parameters
        ----------
        count: int
            The number of frames that exist for this session
        """
        logger.debug("Setting frame_count to : %s", count)
        self._frame_count = count

    def set_current_frame(self, image: np.ndarray, filename: str) -> None:
        """ Set the frame and meta information for the currently displayed frame. Populates the
        attribute :attr:`current_frame`

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The image used to display in the Frame Viewer
        filename: str
            The filename of the current frame
        """
        scale = min(self.frame_display_dims[0] / image.shape[1],
                    self.frame_display_dims[1] / image.shape[0])
        self._current_frame.image = image
        self._current_frame.filename = filename
        self._current_frame.scale = scale
        self._current_frame.interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        self._current_frame.display_dims = (int(round(image.shape[1] * scale)),
                                            int(round(image.shape[0] * scale)))
        logger.trace(self._current_frame)  # type:ignore[attr-defined]

    def set_frame_display_dims(self, width: int, height: int) -> None:
        """ Set the size, in pixels, of the video frame display window and resize the displayed
        frame.

        Used on a frame resize callback, sets the :attr:frame_display_dims`.

        Parameters
        ----------
        width: int
            The width of the frame holding the video canvas in pixels
        height: int
            The height of the frame holding the video canvas in pixels
        """
        self._frame_display_dims = (int(width), int(height))
        image = self._current_frame.image
        scale = min(self.frame_display_dims[0] / image.shape[1],
                    self.frame_display_dims[1] / image.shape[0])
        self._current_frame.scale = scale
        self._current_frame.interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        self._current_frame.display_dims = (int(round(image.shape[1] * scale)),
                                            int(round(image.shape[0] * scale)))
        logger.trace(self._current_frame)  # type:ignore[attr-defined]

    def set_zoomed(self, state: bool) -> None:
        """ Set the current zoom state

        Parameters
        ----------
        state: bool
            ``True`` for zoomed ``False`` for full frame
        """
        logger.trace("Setting zoom state from %s to %s",  # type:ignore[attr-defined]
                     self.is_zoomed, state)
        self._tk_vars.is_zoomed.set(state)
