#!/usr/bin/env python3
""" Landmarks Editor and Landmarks Mesh viewer for the manual adjustments tool """
import gettext
import numpy as np

from lib.align import AlignedFace, LANDMARK_PARTS, LandmarkType
from ._base import Editor, logger

# LOCALES
_LANG = gettext.translation("tools.manual", localedir="locales", fallback=True)
_ = _LANG.gettext


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
        control_text = _("Landmark Point Editor\nEdit the individual landmark points.\n\n"
                         " - Click and drag individual points to relocate.\n"
                         " - Draw a box to select multiple points to relocate.")
        self._selection_box = canvas.create_rectangle(0, 0, 0, 0,
                                                      dash=(2, 4),
                                                      state="hidden",
                                                      outline="gray",
                                                      fill="blue",
                                                      stipple="gray12")
        super().__init__(canvas, detected_faces, control_text)
        # Clear selection box on an editor or frame change
        self._canvas._tk_action_var.trace("w", lambda *e: self._reset_selection())
        self._globals.var_frame_index.trace_add("write", lambda *e: self._reset_selection())

    def _add_actions(self):
        """ Add the optional action buttons to the viewer. Current actions are Point, Select
        and Zoom. """
        self._add_action("magnify", "zoom", _("Magnify/Demagnify the View"),
                         group=None, hotkey="M")
        self._actions["magnify"]["tk_var"].trace("w", self._toggle_zoom)

    # CALLBACKS
    def _toggle_zoom(self, *args):  # pylint:disable=unused-argument
        """ Clear any selections when switching mode and perform an update.

        Parameters
        ----------
        args: tuple
            tkinter callback arguments. Required but unused.
        """
        self._reset_selection()
        self._globals.var_full_update.set(True)

    def _reset_selection(self, event=None):  # pylint:disable=unused-argument
        """ Reset the selection box and the selected landmark annotations. """
        self._canvas.itemconfig("lm_selected", outline=self._control_color)
        self._canvas.dtag("lm_selected")
        self._canvas.itemconfig(self._selection_box,
                                stipple="gray12",
                                fill="blue",
                                outline="gray",
                                state="hidden")
        self._canvas.coords(self._selection_box, 0, 0, 0, 0)
        self._drag_data = {}
        if event is not None:
            self._drag_start(event)

    def update_annotation(self):
        """ Get the latest Landmarks points and update. """
        zoomed_offset = self._zoomed_roi[:2]
        for face_idx, face in enumerate(self._face_iterator):
            face_index = self._globals.face_index if self._globals.is_zoomed else face_idx
            if self._globals.is_zoomed:
                aligned = AlignedFace(face.landmarks_xy,
                                      centering="face",
                                      size=min(self._globals.frame_display_dims))
                landmarks = aligned.landmarks + zoomed_offset
                # Hide all landmarks and only display selected
                self._canvas.itemconfig("lm_dsp", state="hidden")
                self._canvas.itemconfig(f"lm_dsp_face_{face_index}", state="normal")
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
        key = f"lm_dsp_{landmark_index}"
        kwargs = {"outline": color, "fill": color, "width": radius}
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
        top_left = np.array(bounding_box[:2]) - 20
        # NB The text must be visible to be able to get the bounding box, so set to hidden
        # after the bounding box has been retrieved

        keys = [f"lm_lbl_{landmark_index}", f"lm_lbl_bg_{landmark_index}"]
        text_kwargs = {"fill": "black", "font": ("Default", 10), "text": str(landmark_index + 1)}
        bg_kwargs = {"fill": "#ffffea", "outline": "black"}

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
        radius = 7
        bbox = (bounding_box[0] - radius, bounding_box[1] - radius,
                bounding_box[0] + radius, bounding_box[1] + radius)
        key = f"lm_grb_{landmark_index}"
        kwargs = {"outline": "",
                  "fill": "",
                  "width": 1,
                  "dash": (2, 4)}
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
        self._hide_labels()
        if self._drag_data:
            self._update_cursor_select_mode(event)
        else:
            objs = self._canvas.find_withtag(f"lm_grb_face_{self._globals.face_index}"
                                             if self._globals.is_zoomed else "lm_grb")
            item_ids = set(self._canvas.find_overlapping(event.x - 6,
                                                         event.y - 6,
                                                         event.x + 6,
                                                         event.y + 6)).intersection(objs)
            bboxes = [self._canvas.bbox(idx) for idx in item_ids]
            item_id = next((idx for idx, bbox in zip(item_ids, bboxes)
                            if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]),
                           None)
            if item_id:
                self._update_cursor_point_mode(item_id)
            else:
                self._canvas.config(cursor="")
                self._mouse_location = None
                return

    def _hide_labels(self):
        """ Clear all landmark text labels from display """
        self._canvas.itemconfig("lm_lbl", state="hidden")
        self._canvas.itemconfig("lm_lbl_bg", state="hidden")
        self._canvas.itemconfig("lm_grb", fill="", outline="")

    def _update_cursor_point_mode(self, item_id):
        """ Update the cursor when the mouse is over an individual landmark's grab anchor. Displays
        the landmark label for the landmark under the cursor. Updates :attr:`_mouse_location` with
        the current cursor position.

        Parameters
        ----------
        item_id: int
            The tkinter canvas object id for the landmark point that the cursor is over
        """
        self._canvas.itemconfig(item_id, outline="yellow")
        tags = self._canvas.gettags(item_id)
        face_idx = int(next(tag for tag in tags if tag.startswith("face_")).split("_")[-1])
        lm_idx = int(next(tag for tag in tags if tag.startswith("lm_grb_")).split("_")[-1])
        obj_idx = (face_idx, lm_idx)

        self._canvas.config(cursor="none")
        for prefix in ("lm_lbl_", "lm_lbl_bg_"):
            tag = f"{prefix}{lm_idx}_face_{face_idx}"
            logger.trace("Displaying: %s tag: %s", self._canvas.type(tag), tag)
            self._canvas.itemconfig(tag, state="normal")
        self._mouse_location = obj_idx

    def _update_cursor_select_mode(self, event):
        """ Update the mouse cursor when in select mode.

        Standard cursor returned when creating a new selection box. Move cursor returned when over
        an existing selection box

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The current tkinter mouse event
        """
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
        sel_box = self._canvas.coords(self._selection_box)
        if self._mouse_location is not None:  # Point edit mode
            self._drag_data["start_location"] = (event.x, event.y)
            self._drag_callback = self._move_point
        elif not self._drag_data:  # Initial point selection box
            self._drag_data["start_location"] = (event.x, event.y)
            self._drag_callback = self._select
        elif sel_box[0] <= event.x <= sel_box[2] and sel_box[1] <= event.y <= sel_box[3]:
            # Move point selection box
            self._drag_data["start_location"] = (event.x, event.y)
            self._drag_callback = self._move_selection
        else:  # Reset
            self._drag_data = {}
            self._drag_callback = None
            self._reset_selection(event)

    def _drag_stop(self, event):  # pylint:disable=unused-argument
        """ In select mode, call the select mode callback.

        In point mode: trigger a viewport thumbnail update on click + drag release

        If there is drag data, and there are selected points in the drag data then
        trigger the selected points stop code.

        Otherwise reset the selection box and return

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Required but unused.
        """
        if self._mouse_location is not None:  # Point edit mode
            self._det_faces.update.post_edit_trigger(self._globals.frame_index,
                                                     self._mouse_location[0])
            self._mouse_location = None
            self._drag_data = {}
        elif self._drag_data and self._drag_data.get("selected", False):
            self._drag_stop_selected()
        else:
            logger.debug("No selected data. Clearing. drag_data: %s", self._drag_data)
            self._reset_selection()

    def _drag_stop_selected(self):
        """ Action to perform when mouse drag is stopped in selected points editor mode.

        If there is already a selection, update the viewport thumbnail

        If this is a new selection, then obtain the selected points and track
        """
        if "face_index" in self._drag_data:  # Selected data has been moved
            self._det_faces.update.post_edit_trigger(self._globals.frame_index,
                                                     self._drag_data["face_index"])
            return

        # This is a new selection
        face_idx = set()
        landmark_indices = []

        for item_id in self._canvas.find_withtag("lm_selected"):
            tags = self._canvas.gettags(item_id)
            face_idx.add(next(int(tag.split("_")[-1])
                              for tag in tags if tag.startswith("face_")))
            landmark_indices.append(next(int(tag.split("_")[-1])
                                         for tag in tags
                                         if tag.startswith("lm_dsp_") and "face" not in tag))
        if len(face_idx) != 1:
            logger.trace("Not exactly 1 face in selection. Aborting. Face indices: %s", face_idx)
            self._reset_selection()
            return

        self._drag_data["face_index"] = face_idx.pop()
        self._drag_data["landmarks"] = landmark_indices
        self._canvas.itemconfig(self._selection_box, stipple="", fill="", outline="#ffff00")
        self._snap_selection_to_points()

    def _snap_selection_to_points(self):
        """ Snap the selection box to the selected points.

        As the landmarks are calculated and redrawn, the selection box can drift. This is
        particularly true in zoomed mode. The selection box is therefore redrawn to bind just
        outside of the selected points.
        """
        all_coords = np.array([self._canvas.coords(item_id)
                               for item_id in self._canvas.find_withtag("lm_selected")])
        mins = np.min(all_coords, axis=0)
        maxes = np.max(all_coords, axis=0)
        box_coords = [np.min(mins[[0, 2]] - 5),
                      np.min(mins[[1, 3]] - 5),
                      np.max(maxes[[0, 2]] + 5),
                      np.max(maxes[[1, 3]]) + 5]
        self._canvas.coords(self._selection_box, *box_coords)

    def _move_point(self, event):
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

        for item_id in list(enclosed.intersection(landmarks)):
            self._canvas.addtag_withtag("lm_selected", item_id)
        self._canvas.itemconfig("lm_selected", outline="#ffff00")
        self._drag_data["selected"] = True

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
        super().__init__(canvas, detected_faces, None)

    def update_annotation(self):
        """ Get the latest Landmarks and update the mesh."""
        key = "mesh"
        color = self._control_color
        zoomed_offset = self._zoomed_roi[:2]
        for face_idx, face in enumerate(self._face_iterator):
            face_index = self._globals.face_index if self._globals.is_zoomed else face_idx
            if self._globals.is_zoomed:
                aligned = AlignedFace(face.landmarks_xy,
                                      centering="face",
                                      size=min(self._globals.frame_display_dims))
                landmarks = aligned.landmarks + zoomed_offset
                landmark_mapping = LANDMARK_PARTS[aligned.landmark_type]
                # Hide all meshes and only display selected
                self._canvas.itemconfig("Mesh", state="hidden")
                self._canvas.itemconfig(f"Mesh_face_{face_index}", state="normal")
            else:
                landmarks = self._scale_to_display(face.landmarks_xy)
                landmark_mapping = LANDMARK_PARTS[LandmarkType.from_shape(landmarks.shape)]
            logger.trace("Drawing Landmarks Mesh: (landmarks: %s, color: %s)", landmarks, color)
            for idx, (start, end, fill) in enumerate(landmark_mapping.values()):
                key = f"mesh_{idx}"
                pts = landmarks[start:end].flatten()
                if fill:
                    kwargs = {"fill": "", "outline": color, "width": 1}
                    asset = "polygon"
                else:
                    kwargs = {"fill": color, "width": 1}
                    asset = "line"
                self._object_tracker(key, asset, face_index, pts, kwargs)
        # Place mesh as bottom annotation
        self._canvas.tag_raise(self.__class__.__name__, "main_image")
