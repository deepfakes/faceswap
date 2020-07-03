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
        self._selection_box = canvas.create_rectangle(0, 0, 0, 0,
                                                      dash=(2, 4),
                                                      state="hidden",
                                                      outline="gray",
                                                      fill="blue",
                                                      stipple="gray12")
        super().__init__(canvas, detected_faces, control_text)
        # Clear selection box on an editor or frame change
        self._canvas._tk_action_var.trace("w", self._clear_selection)
        self._globals.tk_frame_index.trace("w", self._clear_selection)

    @property
    def _edit_mode(self):
        """ str: The currently selected edit mode based on optional action button.
        One of "point" or "select" """
        action = [name for name, option in self._actions.items()
                  if option["group"] == "mode" and option["tk_var"].get()]
        return "point" if not action else action[0]

    def _add_actions(self):
        """ Add the optional action buttons to the viewer. Current actions are Point, Select
        and Zoom. """
        self._add_action("magnify", "zoom", "Magnify/Demagnify the View", group=None, hotkey="M")
        self._add_action("point", "point", "Individual Point Editor", group="mode", hotkey="P")
        self._add_action("select", "selection", "Selected Points Editor",
                         group="mode", hotkey="O")
        self._actions["magnify"]["tk_var"].trace("w", self._toggle_zoom)
        self._actions["select"]["tk_var"].trace("w", self._toggle_grab_points)

    # CALLBACKS
    def _toggle_zoom(self, *args):  # pylint:disable=unused-argument
        """ Clear any selections when switching mode and perform an update.

        Parameters
        ----------
        args: tuple
            tkinter callback arguments. Required but unused.
        """
        if self._edit_mode == "select":
            self._reset_selection()
        self._globals.tk_update.set(True)

    def _toggle_grab_points(self, *args):  # pylint:disable=unused-argument
        """ Toggle the individual landmark grabbers off and on depending on edit mode.

        Parameters
        ----------
        args: tuple
            tkinter callback arguments. Required but unused.
        """
        state = "hidden" if self._actions["select"]["tk_var"].get() else "normal"
        self._canvas.itemconfig("lm_grb", state=state)

    def _clear_selection(self, *args):  # pylint:disable=unused-argument
        """ Callback to clear any active selections on an editor or frame change.

        Parameters
        ----------
        args: tuple
            tkinter callback arguments. Required but unused.
        """
        if self._edit_mode == "select" and self._drag_data:
            logger.debug("Resetting active selection")
            self._reset_selection()

    def update_annotation(self):
        """ Get the latest Landmarks points and update. """
        zoomed_offset = self._zoomed_roi[:2]
        for face_idx, face in enumerate(self._face_iterator):
            face_index = self._globals.face_index if self._globals.is_zoomed else face_idx
            if self._globals.is_zoomed:
                landmarks = face.aligned_landmarks + zoomed_offset
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

        Launch the cursor update action for the currently selected edit mode.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The current tkinter mouse event
        """
        if self._edit_mode == "point":
            self._update_cursor_point_mode()
        else:
            self._update_cursor_select_mode(event)

    def _update_cursor_point_mode(self):
        """ Update the cursor when in individual landmark point editor mode.

        Update :attr:`_mouse_location` with the current cursor position and display appropriate
        icon.

        Checks whether the mouse is over a landmark grab anchor and pops the
        grab icon and the landmark label.
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

    def _update_cursor_select_mode(self, event):
        """ Update the mouse cursor when in select mode.

        Standard cursor returned when creating a new selection box. Move cursor returned when over
        an existing selection box

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The current tkinter mouse event
        """
        if not self._drag_data:  # New selection box, standard cursor
            self._canvas.config(cursor="")
            return
        bbox = self._canvas.coords(self._selection_box)
        if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
            self._canvas.config(cursor="fleur")
        else:
            self._canvas.config(cursor="")

    # Mouse actions
    def _drag_start(self, event):
        """ The action to perform when the user starts clicking and dragging the mouse.

        The underlying Detected Face's landmark is updated for the point being edited.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if self._edit_mode == "select":
            self._drag_start_select(event)
        elif self._mouse_location is None:
            self._drag_data = dict()
            self._drag_callback = None
        else:
            self._drag_data["start_location"] = (event.x, event.y)
            self._drag_callback = self._move

    def _drag_start_select(self, event):
        """ The actions to perform when the user starts clicking and dragging the mouse when in
        multiple landmark selection edit mode.

        Can be called to either set a new selection or to move an existing selection.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        bbox = self._canvas.coords(self._selection_box)
        if not self._drag_data:
            self._drag_data["start_location"] = (event.x, event.y)
            self._drag_callback = self._select
        elif self._drag_data and bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
            self._drag_data["start_location"] = (event.x, event.y)
            self._drag_callback = self._move_selection
        else:
            self._reset_selection(event)

    def _drag_stop(self, event):  # pylint: disable=unused-argument
        """ In select mode, call the select mode callback.

        In point mode: trigger a viewport thumbnail update on click + drag release

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Required but unused.
        """
        if self._edit_mode == "select":
            self._drag_stop_selected()
        elif self._mouse_location is not None:
            self._det_faces.update.post_edit_trigger(self._globals.frame_index,
                                                     self._mouse_location[0])

    def _drag_stop_selected(self):
        """ Action to perform when mouse drag is stopped in selected points editor mode.

        If no drag data, or no selected points, then clear selection box.

        If there is already a selection, update the viewport thumbnail

        If this is a new selection, then track the selected objects
        """
        if not self._drag_data or not self._drag_data.get("selected", False):
            logger.debug("No selected data. Clearing. drag_data: %s", self._drag_data)
            self._reset_selection()
            return

        if "face_index" in self._drag_data:
            self._det_faces.update.post_edit_trigger(self._globals.frame_index,
                                                     self._drag_data["face_index"])
            return

        self._canvas.itemconfig(self._selection_box, stipple="", fill="")
        face_idx = set()
        landmark_indices = []

        for item_id in self._drag_data["selected"]:
            tags = self._canvas.gettags(item_id)
            face_idx.add(next(int(tag.split("_")[-1])
                              for tag in tags if tag.startswith("face_")))
            landmark_indices.append(next(int(tag.split("_")[-1])
                                         for tag in tags
                                         if tag.startswith("lm_dsp_") and "face" not in tag))
            self._canvas.addtag_withtag("lm_selected", item_id)

        if len(face_idx) != 1:
            logger.info("More than 1 face in selection. Aborting. Face indices: %s", face_idx)
            self._reset_selection()
            return

        self._drag_data["face_index"] = face_idx.pop()
        self._drag_data["landmarks"] = landmark_indices
        self._canvas.itemconfig("lm_selected", outline="#ffff00")
        self._snap_selection_to_points()

    def _snap_selection_to_points(self):
        """ Snap the selection box to the selected points.

        As the landmarks are calculated and redrawn, the selection box can drift. This is
        particularly true in zoomed mode. The selection box is therefore redrawn to bind just
        outside of the selected points.
        """
        all_coords = np.array([self._canvas.coords(item_id)
                               for item_id in self._drag_data["selected"]])
        mins = np.min(all_coords, axis=0)
        maxes = np.max(all_coords, axis=0)
        box_coords = [np.min(mins[[0, 2]] - 5),
                      np.min(mins[[1, 3]] - 5),
                      np.max(maxes[[0, 2]] + 5),
                      np.max(maxes[[1, 3]]) + 5]
        self._canvas.coords(self._selection_box, *box_coords)

    def _move(self, event):
        """ Moves the selected landmark point box and updates the underlying landmark on a point
        drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        face_idx, lm_idx = self._mouse_location
        shift_x = event.x - self._drag_data["start_location"][0]
        shift_y = event.y - self._drag_data["start_location"][1]

        if self._globals.is_zoomed:
            scaled_shift = np.array((shift_x, shift_y))
        else:
            scaled_shift = self.scale_from_display(np.array((shift_x, shift_y)), do_offset=False)
        self._det_faces.update.landmark(self._globals.frame_index,
                                        face_idx,
                                        lm_idx,
                                        *scaled_shift,
                                        self._globals.is_zoomed)
        self._drag_data["start_location"] = (event.x, event.y)

    def _select(self, event):
        """ Create a selection box on mouse drag event when in "select" mode

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if self._canvas.itemcget(self._selection_box, "state") == "hidden":
            self._canvas.itemconfig(self._selection_box, state="normal")
        coords = (*self._drag_data["start_location"], event.x, event.y)
        self._canvas.coords(self._selection_box, *coords)
        enclosed = set(self._canvas.find_enclosed(*coords))
        landmarks = set(self._canvas.find_withtag("lm_dsp"))
        self._drag_data["selected"] = list(enclosed.intersection(landmarks))

    def _move_selection(self, event):
        """ Move a selection box and the landmarks contained when in "select" mode and a selection
        box has been drawn. """
        shift_x = event.x - self._drag_data["start_location"][0]
        shift_y = event.y - self._drag_data["start_location"][1]
        if self._globals.is_zoomed:
            scaled_shift = np.array((shift_x, shift_y))
        else:
            scaled_shift = self.scale_from_display(np.array((shift_x, shift_y)), do_offset=False)
        self._canvas.move(self._selection_box, shift_x, shift_y)

        self._det_faces.update.landmark(self._globals.frame_index,
                                        self._drag_data["face_index"],
                                        self._drag_data["landmarks"],
                                        *scaled_shift,
                                        self._globals.is_zoomed)
        self._snap_selection_to_points()
        self._drag_data["start_location"] = (event.x, event.y)

    def _reset_selection(self, event=None):
        """ Reset the selection box and the selected landmark annotations. """
        self._canvas.itemconfig("lm_selected", outline=self._control_color)
        self._canvas.dtag("lm_selected")
        self._canvas.itemconfig(self._selection_box, stipple="gray12", fill="blue", state="hidden")
        self._canvas.coords(self._selection_box, 0, 0, 0, 0)
        self._drag_data = dict()
        if event is not None:
            self._drag_start(event)


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
        zoomed_offset = self._zoomed_roi[:2]
        for face_idx, face in enumerate(self._face_iterator):
            face_index = self._globals.face_index if self._globals.is_zoomed else face_idx
            if self._globals.is_zoomed:
                landmarks = face.aligned_landmarks + zoomed_offset
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
