#!/usr/bin/env python3
""" Handles Navigation and Background Image for the Frame Viewer section of the manual
tool GUI. """

import logging
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from lib.align import AlignedFace

logger = logging.getLogger(__name__)


class Navigation():
    """ Handles playback and frame navigation for the Frame Viewer Window.

    Parameters
    ----------
    display_frame: :class:`DisplayFrame`
        The parent frame viewer window
    """
    def __init__(self, display_frame):
        logger.debug("Initializing %s", self.__class__.__name__)
        self._display_frame = display_frame
        self._globals = display_frame._globals
        self._det_faces = display_frame._det_faces
        self._nav = display_frame._nav
        self._tk_is_playing = tk.BooleanVar()
        self._tk_is_playing.set(False)
        self._det_faces.tk_face_count_changed.trace("w", self._update_total_frame_count)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _current_nav_frame_count(self):
        """ int: The current frame count for the transport slider """
        return self._nav["scale"].cget("to") + 1

    def nav_scale_callback(self, *args, reset_progress=True):  # pylint:disable=unused-argument
        """ Adjust transport slider scale for different filters. Hide or display optional filter
        controls.
        """
        self._display_frame.pack_threshold_slider()
        if reset_progress:
            self.stop_playback()
        frame_count = self._det_faces.filter.count
        if self._current_nav_frame_count == frame_count:
            logger.trace("Filtered count has not changed. Returning")
        if self._globals.var_filter_mode.get() == "Misaligned Faces":
            self._det_faces.tk_face_count_changed.set(True)
        self._update_total_frame_count()
        if reset_progress:
            self._globals.var_transport_index.set(0)

    def _update_total_frame_count(self, *args):  # pylint:disable=unused-argument
        """ Update the displayed number of total frames that meet the current filter criteria.

        Parameters
        ----------
        args: tuple
            Required for tkinter trace callback but unused
        """
        frame_count = self._det_faces.filter.count
        if self._current_nav_frame_count == frame_count:
            logger.trace("Filtered count has not changed. Returning")
            return
        max_frame = max(0, frame_count - 1)
        logger.debug("Filtered frame count has changed. Updating from %s to %s",
                     self._current_nav_frame_count, frame_count)
        self._nav["scale"].config(to=max_frame)
        self._nav["label"].config(text=f"/{max_frame}")
        state = "disabled" if max_frame == 0 else "normal"
        self._nav["entry"].config(state=state)

    @property
    def tk_is_playing(self):
        """ :class:`tkinter.BooleanVar`: Whether the stream is currently playing. """
        return self._tk_is_playing

    def handle_play_button(self):
        """ Handle the play button.

        Switches the :attr:`tk_is_playing` variable.
        """
        is_playing = self.tk_is_playing.get()
        self.tk_is_playing.set(not is_playing)

    def stop_playback(self):
        """ Stop play back if playing """
        if self.tk_is_playing.get():
            logger.trace("Stopping playback")
            self.tk_is_playing.set(False)

    def increment_frame(self, frame_count=None, is_playing=False):
        """ Update The frame navigation position to the next frame based on filter. """
        if not is_playing:
            self.stop_playback()
        position = self._get_safe_frame_index()
        face_count_change = not self._det_faces.filter.frame_meets_criteria
        if face_count_change:
            position -= 1
        frame_count = self._det_faces.filter.count if frame_count is None else frame_count
        if not face_count_change and (frame_count == 0 or position == frame_count - 1):
            logger.debug("End of Stream. Not incrementing")
            self.stop_playback()
            return
        self._globals.var_transport_index.set(min(position + 1, max(0, frame_count - 1)))

    def decrement_frame(self):
        """ Update The frame navigation position to the previous frame based on filter. """
        self.stop_playback()
        position = self._get_safe_frame_index()
        face_count_change = not self._det_faces.filter.frame_meets_criteria
        if not face_count_change and (self._det_faces.filter.count == 0 or position == 0):
            logger.debug("End of Stream. Not decrementing")
            return
        self._globals.var_transport_index.set(min(max(0, self._det_faces.filter.count - 1),
                                                  max(0, position - 1)))

    def _get_safe_frame_index(self):
        """ Obtain the current frame position from the var_transport_index variable in
        a safe manner (i.e. handle for non-numeric)

        Returns
        -------
        int
            The current transport frame index
        """
        try:
            retval = self._globals.var_transport_index.get()
        except tk.TclError as err:
            if "expected floating-point" not in str(err):
                raise
            val = str(err).rsplit(" ", maxsplit=1)[-1].replace("\"", "")
            retval = "".join(ch for ch in val if ch.isdigit())
            retval = 0 if not retval else int(retval)
            self._globals.var_transport_index.set(retval)
        return retval

    def goto_first_frame(self):
        """ Go to the first frame that meets the filter criteria. """
        self.stop_playback()
        position = self._globals.var_transport_index.get()
        if position == 0:
            return
        self._globals.var_transport_index.set(0)

    def goto_last_frame(self):
        """ Go to the last frame that meets the filter criteria. """
        self.stop_playback()
        position = self._globals.var_transport_index.get()
        frame_count = self._det_faces.filter.count
        if position == frame_count - 1:
            return
        self._globals.var_transport_index.set(frame_count - 1)


class BackgroundImage():
    """ The background image of the canvas """
    def __init__(self, canvas):
        self._canvas = canvas
        self._globals = canvas._globals
        self._det_faces = canvas._det_faces
        placeholder = np.ones((*reversed(self._globals.frame_display_dims), 3), dtype="uint8")
        self._tk_frame = ImageTk.PhotoImage(Image.fromarray(placeholder))
        self._tk_face = ImageTk.PhotoImage(Image.fromarray(placeholder))
        self._image = self._canvas.create_image(self._globals.frame_display_dims[0] / 2,
                                                self._globals.frame_display_dims[1] / 2,
                                                image=self._tk_frame,
                                                anchor=tk.CENTER,
                                                tags="main_image")
        self._zoomed_centering = "face"

    @property
    def _current_view_mode(self):
        """ str: `frame` if global zoom mode variable is set to ``False`` other wise `face`. """
        retval = "face" if self._globals.is_zoomed else "frame"
        logger.trace(retval)
        return retval

    def refresh(self, view_mode):
        """ Update the displayed frame.

        Parameters
        ----------
        view_mode: ["frame", "face"]
            The currently active editor's selected view mode.
        """
        self._switch_image(view_mode)
        logger.trace("Updating background frame")
        getattr(self, f"_update_tk_{self._current_view_mode}")()

    def _switch_image(self, view_mode):
        """ Switch the image between the full frame image and the zoomed face image.

        Parameters
        ----------
        view_mode: ["frame", "face"]
            The currently active editor's selected view mode.
        """
        if view_mode == self._current_view_mode and (
                self._canvas.active_editor.zoomed_centering == self._zoomed_centering):
            return
        self._zoomed_centering = self._canvas.active_editor.zoomed_centering
        logger.trace("Switching background image from '%s' to '%s'",
                     self._current_view_mode, view_mode)
        img = getattr(self, f"_tk_{view_mode}")
        self._canvas.itemconfig(self._image, image=img)
        self._globals.set_zoomed(view_mode == "face")
        self._globals.set_face_index(0)

    def _update_tk_face(self):
        """ Update the currently zoomed face. """
        face = self._get_zoomed_face()
        padding = self._get_padding((min(self._globals.frame_display_dims),
                                     min(self._globals.frame_display_dims)))
        face = cv2.copyMakeBorder(face, *padding, cv2.BORDER_CONSTANT)
        if self._tk_frame.height() != face.shape[0]:
            self._resize_frame()

        logger.trace("final shape: %s", face.shape)
        self._tk_face.paste(Image.fromarray(face))

    def _get_zoomed_face(self):
        """ Get the zoomed face or a blank image if no faces are available.

        Returns
        -------
        :class:`numpy.ndarray`
            The face sized to the shortest dimensions of the face viewer
        """
        frame_idx = self._globals.frame_index
        face_idx = self._globals.face_index
        faces_in_frame = self._det_faces.face_count_per_index[frame_idx]
        size = min(self._globals.frame_display_dims)

        if face_idx + 1 > faces_in_frame:
            logger.debug("Resetting face index to 0 for more faces in frame than current index: ("
                         "faces_in_frame: %s, zoomed_face_index: %s", faces_in_frame, face_idx)
            self._globals.set_face_index(0)

        if faces_in_frame == 0:
            face = np.ones((size, size, 3), dtype="uint8")
        else:
            det_face = self._det_faces.current_faces[frame_idx][face_idx]
            face = AlignedFace(det_face.landmarks_xy,
                               image=self._globals.current_frame.image,
                               centering=self._zoomed_centering,
                               size=size).face
        logger.trace("face shape: %s", face.shape)
        return face[..., 2::-1]

    def _update_tk_frame(self):
        """ Place the currently held frame into :attr:`_tk_frame`. """
        img = cv2.resize(self._globals.current_frame.image,
                         self._globals.current_frame.display_dims,
                         interpolation=self._globals.current_frame.interpolation)[..., 2::-1]
        padding = self._get_padding(img.shape[:2])
        if any(padding):
            img = cv2.copyMakeBorder(img, *padding, cv2.BORDER_CONSTANT)
        logger.trace("final shape: %s", img.shape)

        if self._tk_frame.height() != img.shape[0]:
            self._resize_frame()

        self._tk_frame.paste(Image.fromarray(img))

    def _get_padding(self, size):
        """ Obtain the Left, Top, Right, Bottom padding required to place the square face or frame
        in to the Photo Image

        Returns
        -------
        tuple
            The (Left, Top, Right, Bottom) padding to apply to the face image in pixels
        """
        pad_lt = ((self._globals.frame_display_dims[1] - size[0]) // 2,
                  (self._globals.frame_display_dims[0] - size[1]) // 2)
        padding = (pad_lt[0],
                   self._globals.frame_display_dims[1] - size[0] - pad_lt[0],
                   pad_lt[1],
                   self._globals.frame_display_dims[0] - size[1] - pad_lt[1])
        logger.debug("Frame dimensions: %s, size: %s, padding: %s",
                     self._globals.frame_display_dims, size, padding)
        return padding

    def _resize_frame(self):
        """ Resize the :attr:`_tk_frame`, attr:`_tk_face` photo images, update the canvas to
        offset the image correctly.
        """
        logger.trace("Resizing video frame on resize event: %s", self._globals.frame_display_dims)
        placeholder = np.ones((*reversed(self._globals.frame_display_dims), 3), dtype="uint8")
        self._tk_frame = ImageTk.PhotoImage(Image.fromarray(placeholder))
        self._tk_face = ImageTk.PhotoImage(Image.fromarray(placeholder))
        self._canvas.coords(self._image,
                            self._globals.frame_display_dims[0] / 2,
                            self._globals.frame_display_dims[1] / 2)
        img = self._tk_face if self._current_view_mode == "face" else self._tk_frame
        self._canvas.itemconfig(self._image, image=img)
