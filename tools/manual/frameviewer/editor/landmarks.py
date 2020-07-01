#!/usr/bin/env python3
""" Landmarks Editor and Landmarks Mesh viewer for the manual adjustments tool """
import numpy as np

from ._base import Editor, logger


class Landmarks(Editor):
    """ The Landmarks Editor.

    Adjust individual landmark points and re-generate Extract Box.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The canvas that holds the image and annotations
    detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
        The _detected_faces data for this manual session
    """
    def __init__(self, canvas, detected_faces):
        control_text = ("Landmark Point Editor\nEdit the individual landmark points.\n\n"
                        " - Click and drag individual landmark points to relocate.")
        super().__init__(canvas, detected_faces, control_text)

    def _add_actions(self):
        """ Add the optional action buttons to the viewer. Current actions are Draw, Erase
        and Zoom. """
        self._add_action("magnify", "zoom", "Magnify/Demagnify the View", group=None, hotkey="M")
        self._actions["magnify"]["tk_var"].trace("w", lambda *e: self._globals.tk_update.set(True))

    def update_annotation(self):
        """ Get the latest Landmarks points and update. """
        for face_idx, face in enumerate(self._face_iterator):
            face_index = self._globals.face_index if self._globals.is_zoomed else face_idx
            if self._globals.is_zoomed:
                landmarks = face.aligned_landmarks + self._zoomed_roi[:2]
                # Hide all landmarks and only display selected
                self._canvas.itemconfig("lm_dsp", state="hidden")
                self._canvas.itemconfig("lm_dsp_face_{}".format(face_index), state="normal")
            else:
                landmarks = self._scale_to_display(face.landmarks_xy)
            for lm_idx, landmark in enumerate(landmarks):
                self._display_landmark(landmark, face_index, lm_idx)
                self._label_landmark(landmark, face_index, lm_idx)
                self._grab_landmark(landmark, face_index, lm_idx)
        logger.trace("Updated landmark annotations")

    def _display_landmark(self, bounding_box, face_index, landmark_index):
        """ Add an individual landmark display annotation to the canvas.

        Parameters
        ----------
        bounding_box: :class:`numpy.ndarray`
            The (left, top), (right, bottom) (x, y) coordinates of the oval bounding box for this
            landmark
        face_index: int
            The index of the face within the current frame
        landmark_index: int
            The index point of this landmark
        """
        radius = 1
        color = self._control_color
        bbox = (bounding_box[0] - radius, bounding_box[1] - radius,
                bounding_box[0] + radius, bounding_box[1] + radius)
        key = "lm_dsp_{}".format(landmark_index)
        kwargs = dict(outline=color, fill=color, width=radius)
        self._object_tracker(key, "oval", face_index, bbox, kwargs)

    def _label_landmark(self, bounding_box, face_index, landmark_index):
        """ Add a text label for a landmark to the canvas.

        Parameters
        ----------
        bounding_box: :class:`numpy.ndarray`
            The (left, top), (right, bottom) (x, y) coordinates of the oval bounding box for this
            landmark
        face_index: int
            The index of the face within the current frame
        landmark_index: int
            The index point of this landmark
        """
        if not self._is_active:
            return
        top_left = np.array(bounding_box[:2]) - 16
        # NB The text must be visible to be able to get the bounding box, so set to hidden
        # after the bounding box has been retrieved

        keys = ["lm_lbl_{}".format(landmark_index), "lm_lbl_bg_{}".format(landmark_index)]
        text_kwargs = dict(fill="black", font=("Default", 10), text=str(landmark_index + 1))
        bg_kwargs = dict(fill="#ffffea", outline="black")

        text_id = self._object_tracker(keys[0], "text", face_index, top_left, text_kwargs)
        bbox = self._canvas.bbox(text_id)
        bbox = [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2]
        bg_id = self._object_tracker(keys[1], "rectangle", face_index, bbox, bg_kwargs)
        self._canvas.tag_lower(bg_id, text_id)
        self._canvas.itemconfig(text_id, state="hidden")
        self._canvas.itemconfig(bg_id, state="hidden")

    def _grab_landmark(self, bounding_box, face_index, landmark_index):
        """ Add an individual landmark grab anchor to the canvas.

        Parameters
        ----------
        bounding_box: :class:`numpy.ndarray`
            The (left, top), (right, bottom) (x, y) coordinates of the oval bounding box for this
            landmark
        face_index: int
            The index of the face within the current frame
        landmark_index: int
            The index point of this landmark
        """
        if not self._is_active:
            return
        radius = 6
        bbox = (bounding_box[0] - radius, bounding_box[1] - radius,
                bounding_box[0] + radius, bounding_box[1] + radius)
        key = "lm_grb_{}".format(landmark_index)
        kwargs = dict(outline="",
                      fill="",
                      width=radius,
                      activeoutline="black",
                      activefill="white")
        self._object_tracker(key, "oval", face_index, bbox, kwargs)

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """ Set the cursor action.

        Update :attr:`_mouse_location` with the current cursor position and display appropriate
        icon.

        Checks whether the mouse is over a landmark grab anchor and pops the
        grab icon and the landmark label.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The current tkinter mouse event
        """
        self._hide_labels()
        objs = self._canvas.find_withtag("lm_grb_face_{}".format(self._globals.face_index)
                                         if self._globals.is_zoomed else "lm_grb")
        item_ids = set(self._canvas.find_withtag("current")).intersection(objs)
        if not item_ids:
            self._canvas.config(cursor="")
            self._mouse_location = None
            return
        item_id = list(item_ids)[0]
        tags = self._canvas.gettags(item_id)
        face_idx = int(next(tag for tag in tags if tag.startswith("face_")).split("_")[-1])

        lm_idx = int(next(tag for tag in tags if tag.startswith("lm_grb_")).split("_")[-1])
        obj_idx = (face_idx, lm_idx)
        self._canvas.config(cursor="fleur")
        for prefix in ("lm_lbl_", "lm_lbl_bg_"):
            tag = "{}{}_face_{}".format(prefix, lm_idx, face_idx)
            logger.trace("Displaying: %s tag: %s", self._canvas.type(tag), tag)
            self._canvas.itemconfig(tag, state="normal")
        self._mouse_location = obj_idx

    def _hide_labels(self):
        """ Clear all landmark text labels from display """
        self._canvas.itemconfig("lm_lbl", state="hidden")
        self._canvas.itemconfig("lm_lbl_bg", state="hidden")

    # Mouse actions
    def _drag_start(self, event):
        """ The action to perform when the user starts clicking and dragging the mouse.

        The underlying Detected Face's landmark is updated for the point being edited.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if self._mouse_location is None:
            self._drag_data = dict()
            self._drag_callback = None
        else:
            self._drag_data["current_location"] = (event.x, event.y)
            self._drag_callback = self._move

    def _move(self, event):
        """ Moves the selected landmark point box and updates the underlying landmark on a point
        drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        face_idx, lm_idx = self._mouse_location
        shift_x = event.x - self._drag_data["current_location"][0]
        shift_y = event.y - self._drag_data["current_location"][1]

        if self._globals.is_zoomed:
            scaled_shift = np.array((shift_x, shift_y))
        else:
            scaled_shift = self.scale_from_display(np.array((shift_x, shift_y)), do_offset=False)
        self._det_faces.update.landmark(self._globals.frame_index,
                                        face_idx,
                                        lm_idx,
                                        *scaled_shift,
                                        self._globals.is_zoomed)
        self._drag_data["current_location"] = (event.x, event.y)


class Mesh(Editor):
    """ The Landmarks Mesh Display.

    There are no editing options for Mesh editor. It is purely aesthetic and updated when other
    editors are used.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The canvas that holds the image and annotations
    detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
        The _detected_faces data for this manual session
    """
    def __init__(self, canvas, detected_faces):
        self._landmark_mapping = dict(mouth_inner=(60, 68),
                                      mouth_outer=(48, 60),
                                      right_eyebrow=(17, 22),
                                      left_eyebrow=(22, 27),
                                      right_eye=(36, 42),
                                      left_eye=(42, 48),
                                      nose=(27, 36),
                                      jaw=(0, 17),
                                      chin=(8, 11))
        super().__init__(canvas, detected_faces, None)

    def update_annotation(self):
        """ Get the latest Landmarks and update the mesh."""
        key = "mesh"
        color = self._control_color
        for face_idx, face in enumerate(self._face_iterator):
            face_index = self._globals.face_index if self._globals.is_zoomed else face_idx
            if self._globals.is_zoomed:
                landmarks = face.aligned_landmarks + self._zoomed_roi[:2]
                # Hide all meshes and only display selected
                self._canvas.itemconfig("Mesh", state="hidden")
                self._canvas.itemconfig("Mesh_face_{}".format(face_index), state="normal")
            else:
                landmarks = self._scale_to_display(face.landmarks_xy)
            logger.trace("Drawing Landmarks Mesh: (landmarks: %s, color: %s)", landmarks, color)
            for idx, (segment, val) in enumerate(self._landmark_mapping.items()):
                key = "mesh_{}".format(idx)
                pts = landmarks[val[0]:val[1]].flatten()
                if segment in ("right_eye", "left_eye", "mouth_inner", "mouth_outer"):
                    kwargs = dict(fill="", outline=color, width=1)
                    self._object_tracker(key, "polygon", face_index, pts, kwargs)
                else:
                    self._object_tracker(key, "line", face_index, pts, dict(fill=color, width=1))
        # Place mesh as bottom annotation
        self._canvas.tag_raise(self.__class__.__name__, "main_image")
