#!/usr/bin/env python3
""" Extract Box Editor for the manual adjustments tool """

import platform

import numpy as np

from lib.gui.custom_widgets import RightClickMenu
from ._base import Editor, logger


class ExtractBox(Editor):
    """ The Extract Box Editor.

    Adjust the calculated Extract Box to shift all of the 68 point landmarks in place.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The canvas that holds the image and annotations
    detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
        The _detected_faces data for this manual session
    """
    def __init__(self, canvas, detected_faces):
        self._right_click_menu = RightClickMenu(["Delete Face"],
                                                [self._delete_current_face],
                                                ["Del"])
        control_text = ("Extract Box Editor\nMove the extract box that has been generated by the "
                        "aligner.\n\n"
                        " - Click and drag the bounding box to relocate the landmarks without "
                        "recalculating them.")
        key_bindings = {"<Delete>": self._delete_current_face}
        super().__init__(canvas, detected_faces,
                         control_text=control_text, key_bindings=key_bindings)

    def update_annotation(self):
        """ Draw the latest Extract Boxes around the faces. """
        color = self._control_color
        for idx, face in enumerate(self._face_iterator):
            logger.trace("Drawing Extract Box: (idx: %s, roi: %s)", idx, face.original_roi)
            if self._globals.is_zoomed:
                box = np.array((self._zoomed_roi[0], self._zoomed_roi[1],
                                self._zoomed_roi[2], self._zoomed_roi[1],
                                self._zoomed_roi[2], self._zoomed_roi[3],
                                self._zoomed_roi[0], self._zoomed_roi[3]))
            else:
                face.load_aligned(None)
                box = self._scale_to_display(face.original_roi).flatten()
            top_left = box[:2] - 10
            kwargs = dict(fill=color, font=("Default", 20, "bold"), text=str(idx))
            self._object_tracker("eb_text", "text", idx, top_left, kwargs)
            kwargs = dict(fill="", outline=color, width=1)
            self._object_tracker("eb_box", "polygon", idx, box, kwargs)
        logger.trace("Updated extract box annotations")

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """ Update the cursor when it is hovering over an extract box and update
        :attr:`_mouse_location` with the current cursor position.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The current tkinter mouse event
        """
        extract_boxes = set(self._canvas.find_withtag("eb_box"))
        item_ids = set(self._canvas.find_withtag("current")).intersection(extract_boxes)
        if not item_ids:
            self._canvas.config(cursor="")
            self._mouse_location = None
        else:
            item_id = list(item_ids)[0]
            self._canvas.config(cursor="fleur")
            self._mouse_location = next(int(tag.split("_")[-1])
                                        for tag in self._canvas.gettags(item_id)
                                        if tag.startswith("face_"))

    # Mouse click actions
    def set_mouse_click_actions(self):
        """ Add context menu to OS specific right click action. """
        super().set_mouse_click_actions()
        self._canvas.bind("<Button-2>" if platform.system() == "Darwin" else "<Button-3>",
                          self._context_menu)

    def _drag_start(self, event):
        """ The action to perform when the user starts clicking and dragging the mouse.

        Moves the extract box and the underlying 68 point landmarks to the dragged to
        location.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if self._mouse_location is None:
            self._drag_data = dict()
            return
        self._drag_data["current_location"] = (event.x, event.y)
        self._drag_callback = self._move

    def _move(self, event):
        """ Updates the underlying detected faces landmarks based on mouse dragging delta,
        which moves the Extract box on a drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if not self._drag_data:
            return
        shift_x = event.x - self._drag_data["current_location"][0]
        shift_y = event.y - self._drag_data["current_location"][1]
        scaled_shift = self.scale_from_display(np.array((shift_x, shift_y)), do_offset=False)
        self._det_faces.update.landmarks(self._globals.frame_index,
                                         self._mouse_location,
                                         *scaled_shift)
        self._drag_data["current_location"] = (event.x, event.y)

    def _context_menu(self, event):
        """ Create a right click context menu to delete the alignment that is being
        hovered over. """
        if self._mouse_location is None:
            return
        self._right_click_menu.popup(event)

    def _delete_current_face(self, *args):  # pylint:disable=unused-argument
        """ Called by the right click delete event. Deletes the face that the mouse is currently
        over.

        Parameters
        ----------
        args: tuple (unused)
            The event parameter is passed in by the hot key binding, so args is required
        """
        if self._mouse_location is None:
            return
        self._det_faces.update.delete(self._globals.frame_index, self._mouse_location)
