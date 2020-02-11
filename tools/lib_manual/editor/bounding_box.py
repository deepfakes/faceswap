#!/usr/bin/env python3
""" Bounding Box Editor for the manual adjustments tool """

from functools import partial

import numpy as np

from ._base import ControlPanelOption, Editor, RightClickMenu, logger


class BoundingBox(Editor):
    """ The Bounding Box Editor. """
    def __init__(self, canvas, alignments, frames):
        self._right_click_menu = RightClickMenu(["Delete Face"],
                                                [self._delete_current_face],
                                                ["Del"])
        control_text = ("Bounding Box Editor\nEdit the bounding box being fed into the aligner "
                        "and recalculate landmarks.\n\n"
                        " - Grab the corner anchors to resize the bounding box.\n"
                        " - Click and drag the bounding box to relocate.\n"
                        " - Click in empty space to create a new bounding box")
        super().__init__(canvas, alignments, frames, control_text)
        self._bind_hotkeys()

    @property
    def _coords_layout(self):
        """ tuple: The layout order of tkinter canvas bounding box points """
        return ("left", "top", "right", "bottom")

    @property
    def _corner_order(self):
        """ dict: The position index of bounding box corners """
        return {0: ("top", "left"),
                1: ("top", "right"),
                2: ("bottom", "right"),
                3: ("bottom", "left")}

    @property
    def _bounding_boxes(self):
        """ list: List of (`Left`, `Top`, `Right`, `Bottom`) tuples for each displayed face's
        bounding box. """
        return [self._canvas.coords(rect) for rect in self._objects.get("boundingbox", [])]

    def _add_controls(self):
        for dsp in ("Extract Box", "Landmarks", "Mesh"):
            self._add_control(ControlPanelOption(dsp,
                                                 bool,
                                                 group="Display",
                                                 default=dsp not in ("Extract Box", "Landmarks"),
                                                 helptext="Show the {} annotations".format(dsp)))

    def _bind_hotkeys(self):
        """ Add keyboard shortcuts.

        We bind to root because the canvas does not get focus, so keyboard shortcuts won't do
        anything

        * Delete - Delete the currently hovered over face
        """
        self._canvas.winfo_toplevel().bind("<Delete>", self._delete_current_face)

    def update_annotation(self):
        """ Draw the bounding box around faces and set the object to :attr:`_object`"""
        if self._drag_data:
            logger.trace("Object being edited. Not updating annotation")
            return

        if not self._should_display:
            self._hide_annotation()
            return

        key = self.__class__.__name__.lower()
        color = self._control_color
        for idx, face in enumerate(self._alignments.current_faces):
            box = np.array([(face.left, face.top), (face.right, face.bottom)])
            box = self._scale_to_display(box).astype("int32").flatten()
            kwargs = dict(outline=color, width=1)
            self._object_tracker(key, "rectangle", idx, box, kwargs)
            self._update_anchor_annotation(idx, box, color)
        logger.trace("Updated bounding box annotations: %s", self._objects[key])

    def _update_anchor_annotation(self, face_index, bounding_box, color):
        """ Update the anchor annotations for each corner of the bounding box.

        The anchors only display when the bounding box editor is active.

        Parameters
        ----------
        face_index: int
            The index of the face being annotated
        bounding_box: :class:`numpy.ndarray`
            The scaled bounding box to get the corner anchors for
        color: str
            The hex color of the bounding box line
        """
        keys = ["anchor_display", "anchor_grab"]
        if not self._is_active:
            for key in keys:
                self._hide_annotation(key)
            return
        fill_color = "gray"
        activefill_color = "white" if self._is_active else ""
        anchor_points = self._get_anchor_points(self._corners_from_coords(bounding_box))
        for idx, (anc_dsp, anc_grb) in enumerate(zip(*anchor_points)):
            obj_idx = (face_index * 4) + idx
            dsp_kwargs = dict(outline=color, fill=fill_color, width=1)
            self._object_tracker(keys[0], "oval", obj_idx, anc_dsp, dsp_kwargs)
            grb_kwargs = dict(outline="", fill="", width=1, activefill=activefill_color)
            self._object_tracker(keys[1], "oval", obj_idx, anc_grb, grb_kwargs)
        logger.trace("Updated bounding box anchor annotations: %s", {key: self._objects[key]
                                                                     for key in keys})

    @staticmethod
    def _corners_from_coords(bounding_box):
        """ Retrieve the (x, y) co-ordinates of each corner from a bounding box.

        Parameters
        bounding_box: :class:`numpy.ndarray`, list or tuple
            The (left, top), (right, bottom) (x, y) coordinates of the bounding box

        Returns
        -------
        The (`top-left`, `top-right`, `bottom-right`, `bottom-left`) (x, y) coordinates of the
        bounding box
        """
        return ((bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[1]),
                (bounding_box[2], bounding_box[3]), (bounding_box[0], bounding_box[3]))

    @staticmethod
    def _get_anchor_points(bounding_box):
        """ Retrieve the (x, y) co-ordinates for each of the 4 corners of a bounding box's anchors
        for both the displayed anchors and the anchor grab locations.

        Parameters
        ----------
        bounding_box: tuple
            The (`top-left`, `top-right`, `bottom-right`, `bottom-left`) (x, y) coordinates of the
            bounding box

        Returns
            display_anchors: tuple
                The (`top`, `left`, `bottom`, `right`) co-ordinates for each circle at each point
                of the bounding box corners, sized for display
            grab_anchors: tuple
                The (`top`, `left`, `bottom`, `right`) co-ordinates for each circle at each point
                of the bounding box corners, at a larger size for grabbing with a mouse
        """
        radius = 4
        grab_radius = radius * 2
        display_anchors = tuple((cnr[0] - radius, cnr[1] - radius,
                                 cnr[0] + radius, cnr[1] + radius)
                                for cnr in bounding_box)
        grab_anchors = tuple((cnr[0] - grab_radius, cnr[1] - grab_radius,
                              cnr[0] + grab_radius, cnr[1] + grab_radius)
                             for cnr in bounding_box)
        return display_anchors, grab_anchors

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """ Update the cursors for hovering over bounding boxes or bounding box corner anchors and
        update :attr:`_mouse_location`. """
        if self._check_cursor_anchors(event):
            return
        if self._check_cursor_bounding_box(event):
            return
        if self._check_cursor_image(event):
            return

        self._canvas.config(cursor="")
        self._mouse_location = None

    def _check_cursor_anchors(self, event):  # pylint:disable=unused-argument
        """ Check whether the cursor is over an anchor.

        If it is, set the appropriate cursor type and set :attr:`_mouse_location` to:
            ("anchor", (`face index`, `anchor index`)

        Parameters
        ----------
        event: :class:`tkinter.Event`, unused
            The tkinter mouse event, required for callback but unused

        Returns
        -------
        bool
            ``True`` if cursor is over an anchor point otherwise ``False``
        """
        anchors = self._objects["anchor_grab"]
        item_ids = set(self._canvas.find_withtag("current")).intersection(anchors)
        if not item_ids:
            return False
        corners = 4
        obj_idx = anchors.index(list(item_ids)[0])
        corner_idx = obj_idx % corners
        self._canvas.config(cursor="{}_{}_corner".format(*self._corner_order[corner_idx]))
        self._mouse_location = ("anchor", "_".join((str(obj_idx // corners), str(corner_idx))))
        return True

    def _check_cursor_bounding_box(self, event):
        """ Check whether the cursor is over a bounding box.

        If it is, set the appropriate cursor type and set :attr:`_mouse_location` to:
        ("box", `face index`)

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event

        Returns
        -------
        bool
            ``True`` if cursor is over a bounding box otherwise ``False``

        Notes
        -----
        We can't use tags on unfilled rectangles as the interior of the rectangle is not tagged.
        """
        bounding_coords = self._bounding_boxes
        for face_idx, bbox in enumerate(bounding_coords):
            if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                self._canvas.config(cursor="fleur")
                self._mouse_location = ("box", str(face_idx))
                return True
        return False

    def _check_cursor_image(self, event):
        """ Check whether the cursor is over the image.

        If it is, set the appropriate cursor type and set :attr:`_mouse_location` to:
            ("image", )

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event

        Returns
        -------
        bool
            ``True`` if cursor is over a bounding box otherwise ``False``
        """
        display_dims = self._frames.current_meta_data["display_dims"]
        if (self._canvas.offset[0] <= event.x <= display_dims[0] + self._canvas.offset[0] and
                self._canvas.offset[1] <= event.y <= display_dims[1] + self._canvas.offset[1]):
            self._canvas.config(cursor="plus")
            self._mouse_location = ("image", )
            return True
        return False

    # Mouse Actions
    def set_mouse_click_actions(self):
        """ Add right click context menu to default mouse click bindings """
        super().set_mouse_click_actions()
        self._canvas.bind(self._right_click_button, self._context_menu)

    def _drag_start(self, event):
        """ The action to perform when the user starts clicking and dragging the mouse.

        Collect information about the object being clicked on and add to :attr:`_drag_data`

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if self._mouse_location is None:
            self._drag_data = dict()
            self._drag_callback = None
            return
        if self._mouse_location[0] == "anchor":
            corner_idx = int(self._mouse_location[1].split("_")[-1])
            self._drag_data["corner"] = self._corner_order[corner_idx]
            self._drag_callback = self._resize
        elif self._mouse_location[0] == "box":
            self._drag_data["current_location"] = (event.x, event.y)
            self._drag_callback = self._move
        elif self._mouse_location[0] == "image":
            self._create_new_bounding_box(event)
            # Refresh cursor and _mouse_location for new bounding box and reset _drag_start
            self._update_cursor(event)
            self._drag_start(event)

    def _create_new_bounding_box(self, event):
        """ Create a new bounding box when user clicks on image, outside of existing boxes.

        The bounding box is created as a square located around the click location, with dimensions
        1 quarter the size of the frame's shortest side

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event
        """
        size = min(self._frames.current_meta_data["display_dims"]) // 8
        box = (event.x - size, event.y - size, event.x + size, event.y + size)
        logger.debug("Creating new bounding box: %s ", box)
        self._alignments.add_face(*self._coords_to_bounding_box(box))

    def _resize(self, event):
        """ Resizes a bounding box on an anchor drag event

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        face_idx = int(self._mouse_location[1].split("_")[0])
        rect = self._objects["boundingbox"][face_idx]
        box = self._canvas.coords(rect)
        # Switch top/bottom and left/right and set partial so indices match and we don't
        # need branching logic for min/max.
        limits = (partial(min, box[2] - 20),
                  partial(min, box[3] - 20),
                  partial(max, box[0] + 20),
                  partial(max, box[1] + 20))
        rect_xy_indices = [self._coords_layout.index(pnt) for pnt in self._drag_data["corner"]]
        box[rect_xy_indices[1]] = limits[rect_xy_indices[1]](event.x)
        box[rect_xy_indices[0]] = limits[rect_xy_indices[0]](event.y)
        self._canvas.coords(rect, *box)
        corners = self._corners_from_coords(box)
        base_idx = face_idx * len(corners)
        for idx, (anc_dsp, anc_grb) in enumerate(zip(*self._get_anchor_points(corners))):
            obj_idx = base_idx + idx
            self._canvas.coords(self._objects["anchor_display"][obj_idx], *anc_dsp)
            self._canvas.coords(self._objects["anchor_grab"][obj_idx], *anc_grb)
        self._alignments.set_current_bounding_box(face_idx, *self._coords_to_bounding_box(box))

    def _move(self, event):
        """ Moves the bounding box on a bounding box drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        face_idx = int(self._mouse_location[1])
        shift_x = event.x - self._drag_data["current_location"][0]
        shift_y = event.y - self._drag_data["current_location"][1]
        rect = self._objects["boundingbox"][face_idx]
        objects = [rect]
        corner_count = 4
        base_idx = face_idx * corner_count
        for idx in range(corner_count):
            obj_idx = base_idx + idx
            objects.append(self._objects["anchor_display"][obj_idx])
            objects.append(self._objects["anchor_grab"][obj_idx])

        for obj in objects:
            self._canvas.move(obj, shift_x, shift_y)
        coords = self._canvas.coords(rect)
        self._alignments.set_current_bounding_box(face_idx, *self._coords_to_bounding_box(coords))
        self._drag_data["current_location"] = (event.x, event.y)

    def _coords_to_bounding_box(self, coords):
        """ Converts tkinter coordinates to :class:`lib.faces_detect.DetectedFace` bounding
        box format, scaled up and offset for feeding the model.

        Returns
        -------
        tuple
            The (`x`, `width`, `y`, `height`) integer points of the bounding box.
        """
        coords = self.scale_from_display(
            np.array(coords).reshape((2, 2))).flatten().astype("int32")
        return (coords[0], coords[2] - coords[0], coords[1], coords[3] - coords[1])

    def _context_menu(self, event):
        """ Create a right click context menu to delete the alignment that is being
        hovered over. """
        if self._mouse_location is None or self._mouse_location[0] != "box":
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
        if self._mouse_location is None or self._mouse_location[0] != "box":
            return
        self._alignments.delete_face_at_index(int(self._mouse_location[1]))
