#!/usr/bin/env python3
""" Landmarks Editor and Landmarks Mesh viewer for the manual adjustments tool """
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from ._base import ControlPanelOption, Editor, logger


class Landmarks(Editor):
    """ The Landmarks Editor. """
    def __init__(self, canvas, alignments, frames):
        self._zoomed_face = None
        self._zoomed_face_index = None
        control_text = ("Landmark Point Editor\nEdit the individual landmark points.\n\n"
                        " - Click and drag individual landmark points to relocate.")
        super().__init__(canvas, alignments, frames, control_text)

    @property
    def _edit_mode(self):
        """ str: The currently selected edit mode based on optional action button.
        One of "draw" or "zoom" """
        action = [name for name, option in self._actions.items() if option["tk_var"].get()]
        return "move" if not action else action[0]

    def _add_actions(self):
        self._add_action("drag", "move", "Drag individual Landmarks", hotkey="D")
        self._add_action("zoom", "zoom", "Zoom in or out of the selected face", hotkey="Z")
        self._add_edit_mode_callback()

    def _add_edit_mode_callback(self):
        """ Add a callback to change the top most object to be extract box (in zoom mode) or
        landmark grab points (in drag) mode) for mouse tracking """
        for key, option in self._actions.items():
            tk_var = option["tk_var"]
            tk_var.trace("w", lambda *e, k=key, v=tk_var: self._raise_object(k, v))

    def _raise_object(self, key, tk_var):
        """ Raise the extract box to the top (if in zoom mode) otherwise raise the landmark
        grab points """
        if not tk_var.get():
            logger.debug("Action %s is not active. Returning", key)
            return
        objects = (self._get_extract_boxes() if key == "zoom"
                   else self._flatten_list(self._objects.get("lm_grab", [])))
        if not objects:
            logger.debug("Objects for %s not yet created. Returning", key)
            return
        logger.debug("Raising objects for '%s': %s", key, objects)
        for item_id in objects:
            self._canvas.tag_raise(item_id)

    def _get_extract_boxes(self):
        """ Get flattened list of extract box ids """
        return self._flatten_list(self._canvas.editors["extractbox"].objects.get("extractbox", []))

    def _add_controls(self):
        self._add_control(ControlPanelOption("Mesh",
                                             bool,
                                             group="Display",
                                             default=False,
                                             helptext="Show the Mesh annotations"))

    def update_annotation(self):
        """ Draw the Landmarks and set the objects to :attr:`_object`"""
        for face_idx, face in enumerate(self._alignments.current_faces):
            if self._is_zoomed:
                landmarks = face.aligned_landmarks + self._zoomed_roi[:2]
            else:
                landmarks = self._scale_to_display(face.landmarks_xy)
            for lm_idx, landmark in enumerate(landmarks):
                self._display_landmark(landmark, face_idx, lm_idx)
                self._grab_landmark(landmark, face_idx, lm_idx)
                self._label_landmark(landmark, face_idx, lm_idx)
        if self._is_zoomed:
            self._zoom_face(update_only=True)
        logger.trace("Updated landmark annotations: %s", self._objects)

    def _display_landmark(self, bounding_box, face_index, landmark_index):
        """ Add a display landmark to the canvas.

        Parameters
        ----------
        box: :class:`numpy.ndarray`
            The (left, top), (right, bottom) (x, y) coordinates of the oval bounding box
        object_index: int
            The index of the this item in :attr:`_objects`
        """
        key = "lm_display"
        if not self._should_display:
            self._hide_annotation(key)
            return
        radius = 1
        color = self._control_color
        bbox = (bounding_box[0] - radius, bounding_box[1] - radius,
                bounding_box[0] + radius, bounding_box[1] + radius)
        kwargs = dict(outline=color, fill=color, width=radius)
        self._object_tracker(key, "oval", face_index, landmark_index, bbox, kwargs)

    def _grab_landmark(self, bounding_box, face_index, landmark_index):
        """ Add a grab landmark to the canvas.

        Parameters
        ----------
        box: :class:`numpy.ndarray`
            The (left, top), (right, bottom) (x, y) coordinates of the oval bounding box
        object_index: int
            The index of the this item in :attr:`_objects`
        """
        key = "lm_grab"
        if not self._is_active:
            self._hide_annotation(key)
            return
        radius = 6
        activeoutline_color = "black" if self._is_active else ""
        activefill_color = "white" if self._is_active else ""
        bbox = (bounding_box[0] - radius, bounding_box[1] - radius,
                bounding_box[0] + radius, bounding_box[1] + radius)
        kwargs = dict(outline="",
                      fill="",
                      width=radius,
                      activeoutline=activeoutline_color,
                      activefill=activefill_color)
        self._object_tracker(key, "oval", face_index, landmark_index, bbox, kwargs)
        # Bring the grabbers above the extract box
        self._canvas.tag_raise(self._objects[key][face_index][landmark_index])

    def _label_landmark(self, bounding_box, face_index, landmark_index):
        """ Add a text label for a landmark to the canvas.

        Parameters
        ----------
        box: :class:`numpy.ndarray`
            The (left, top), (right, bottom) (x, y) coordinates of the oval bounding box
        object_index: int
            The index of the this item in :attr:`_objects`
        landmark_index
            The index of the landmark being annotated
        """
        keys = ["lm_label", "lm_label_bg"]
        if not self._is_active:
            for key in keys:
                self._hide_annotation(key)
            return
        top_left = np.array(bounding_box[:2]) - 16
        # NB The text must be visible to be able to get the bounding box, so set to hidden
        # after the bounding box has been retrieved

        text_kwargs = dict(fill="black", font=("Default", 10), text=str(landmark_index + 1))
        self._object_tracker(keys[0], "text", face_index, landmark_index, top_left, text_kwargs)

        bbox = self._canvas.bbox(self._objects[keys[0]][face_index][landmark_index])
        bbox = [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2]
        self._object_tracker(keys[1],
                             "rectangle",
                             face_index,
                             landmark_index,
                             bbox,
                             dict(fill="#ffffea", outline="black"))
        self._canvas.lower(self._objects[keys[1]][face_index][landmark_index],
                           self._objects[keys[0]][face_index][landmark_index])
        self._canvas.itemconfig(self._objects[keys[0]][face_index][landmark_index], state="hidden")
        self._canvas.itemconfig(self._objects[keys[1]][face_index][landmark_index], state="hidden")

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """ Update the cursors for hovering over extract boxes and update
        :attr:`_mouse_location`. """
        self._hide_labels()
        objs = (self._get_extract_boxes() if self._edit_mode == "zoom"
                else self._flatten_list(self._objects.get("lm_grab", [])))
        item_ids = set(self._canvas.find_withtag("current")).intersection(objs)
        if not item_ids:
            self._canvas.config(cursor="")
            self._mouse_location = None
            return
        item_id = list(item_ids)[0]

        obj_idx = objs.index(list(item_ids)[0])
        if self._edit_mode == "zoom":
            # Lazy lookup, but flattened list will have same face_index
            # as standard object list for extract boxes
            obj_idx = objs.index(item_id)
            self._canvas.config(cursor="sizing")
        else:
            obj_idx = [(face_idx, face.index(item_id))
                       for face_idx, face in enumerate(self._objects["lm_grab"])
                       if item_id in face][0]
            self._canvas.config(cursor="fleur")
            for label in [self._objects["lm_label"][obj_idx[0]][obj_idx[1]],
                          self._objects["lm_label_bg"][obj_idx[0]][obj_idx[1]]]:
                logger.trace("Displaying: %s id: %s", self._canvas.type(label), label)
                self._canvas.itemconfig(label, state="normal")
        self._mouse_location = obj_idx

    def _hide_labels(self):
        """ Clear all landmark text labels from display """
        lbl_items = self._flatten_list(self._objects["lm_label"] + self._objects["lm_label_bg"])
        labels = [idx for idx in lbl_items
                  if self._canvas.itemcget(idx, "state") == "normal"]
        if not labels:
            return
        logger.trace("Clearing labels")
        for item_id in labels:
            logger.trace("hiding: %s id: %s", self._canvas.type(item_id), item_id)
            self._canvas.itemconfig(item_id, state="hidden")
        logger.trace("Cleared labels")

    # Mouse actions
    def _drag_start(self, event):
        """ The action to perform when the user starts clicking and dragging the mouse.

        Collect information about the landmark being clicked on and add to :attr:`_drag_data`

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if self._mouse_location is None:
            self._drag_data = dict()
            self._drag_callback = None
        elif self._edit_mode == "zoom":
            self._drag_data = dict()
            self._drag_callback = None
            self._zoomed_face_index = self._mouse_location
            self._zoom_face()
        else:
            self._drag_data["current_location"] = (event.x, event.y)
            self._drag_callback = self._move

    def _zoom_face(self, update_only=False):
        """ Zoom in on the selected face.

        Parameters
        ----------
        update_only: bool, optional
            `` True`` if the zoomed image is being updated by a landmark edit, ``False`` if the
            zoom action button has been pressed. Default: ``False``
        """
        face_index = self._zoomed_face_index
        if not update_only:
            self._canvas.toggle_image_display()
        coords = (self._frames.display_dims[0] / 2, self._frames.display_dims[1] / 2)
        if self._is_zoomed:
            face = self._alignments.get_aligned_face_at_index(face_index)[..., 2::-1]
            display = ImageTk.PhotoImage(Image.fromarray(face))
            self._zoomed_face = display
            kwargs = dict(image=self._zoomed_face, anchor=tk.CENTER)
        else:
            kwargs = dict(state="hidden")
            self._zoomed_face_index = None
        self._object_tracker("zoom", "image", face_index, 0, coords, kwargs)
        self._canvas.tag_lower(self._objects["zoom"][face_index][0])
        self._frames.tk_update.set(True)
        if not update_only:
            for obj in self._get_extract_boxes():
                self._canvas.tag_raise(obj)

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
        objects = [self._objects[key][face_idx][lm_idx] for key in self._objects
                   if key != "zoom"]
        for obj in objects:
            logger.trace("Moving: %s id: %s", self._canvas.type(obj), obj)
            logger.trace(self._canvas.itemcget(obj, "state"))
            self._canvas.move(obj, shift_x, shift_y)
        if self._is_zoomed:
            scaled_shift = np.array((shift_x, shift_y))
        else:
            scaled_shift = self.scale_from_display(np.array((shift_x, shift_y)), do_offset=False)
        self._alignments.shift_landmark(face_idx, lm_idx, *scaled_shift, self._is_zoomed)
        self._drag_data["current_location"] = (event.x, event.y)


class Mesh(Editor):
    """ The Landmarks Mesh Display. """
    def __init__(self, canvas, alignments, frames):
        self._landmark_mapping = dict(mouth=(48, 68),
                                      right_eyebrow=(17, 22),
                                      left_eyebrow=(22, 27),
                                      right_eye=(36, 42),
                                      left_eye=(42, 48),
                                      nose=(27, 36),
                                      jaw=(0, 17),
                                      chin=(8, 11))
        super().__init__(canvas, alignments, frames, None)

    def update_annotation(self):
        """ Draw the Landmarks Mesh and set the objects to :attr:`_object`"""
        key = self.__class__.__name__.lower()
        if not self._should_display:
            self._hide_annotation(key)
            return
        color = self._control_color
        for face_idx, face in enumerate(self._alignments.current_faces):
            if self._is_zoomed:
                landmarks = face.aligned_landmarks + self._zoomed_roi[:2]
            else:
                landmarks = self._scale_to_display(face.landmarks_xy)
            logger.trace("Drawing Landmarks Mesh: (landmarks: %s, color: %s)", landmarks, color)
            for idx, (segment, val) in enumerate(self._landmark_mapping.items()):
                pts = landmarks[val[0]:val[1]].flatten()
                if segment in ("right_eye", "left_eye", "mouth"):
                    kwargs = dict(fill="", outline=color, width=1)
                    self._object_tracker(key, "polygon", face_idx, idx, pts, kwargs)
                else:
                    self._object_tracker(key,
                                         "line",
                                         face_idx,
                                         idx,
                                         pts,
                                         dict(fill=color, width=1))
