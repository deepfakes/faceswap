#!/usr/bin/env python3
""" Editor objects for the manual adjustments tool """

import logging
import platform
import tkinter as tk

from functools import partial

import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# pylint:disable=too-many-lines


class Editor():
    """ Parent Class for Object Editors.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The canvas that holds the image and annotations
    alignments: :class:`AlignmentsData`
        The alignments data for this manual session
    frames: :class:`FrameNavigation`
        The frames navigator for this manual session
    """
    def __init__(self, canvas, alignments, frames):
        logger.debug("Initializing %s: (canvas: '%s', alignments: %s, frames: %s)",
                     self.__class__.__name__, canvas, alignments, frames)
        self._canvas = canvas
        self._alignments = alignments
        self._frames = frames
        self._colors = dict(red="#ff0000",
                            green="#00ff00",
                            blue="#0000ff",
                            cyan="#00ffff",
                            yellow="#ffff00",
                            magenta="#ff00ff")
        self._objects = []
        self._mouse_location = None
        self._drag_data = dict()
        self._drag_callback = None
        self._right_click_button = "<Button-2>" if platform.system() == "Darwin" else "<Button-3>"
        self.update_annotation()
        self.bind_mouse_motion()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _is_active(self):
        """ bool: ``True`` if this editor is currently active otherwise ``False``.

        Notes
        -----
        When initializing, the active_editor parameter will not be set in the parent,
        so return ``False`` in this instance
        """
        return hasattr(self._canvas, "active_editor") and self._canvas.active_editor == self

    @property
    def _active_editor(self):
        """ str: The name of the currently active editor """
        return self._canvas.selected_action

    def update_annotation(self):
        """ Update the display annotations for the current objects.

        Override for specific editors.
        """
        logger.trace("Default annotations. Not storing Objects")
        self._clear_annotation()
        self._objects = []

    def _clear_annotation(self):
        """ Removes all currently drawn annotations for the current :class:`Editor`. """
        logger.trace("clearing annotation")
        for faces in self._objects:
            for obj in faces:
                logger.trace("Deleting object: %s (id: %s)", self._canvas.type(obj), obj)
                self._canvas.delete(obj)

    # << MOUSE CALLBACKS >>
    # Mouse cursor display
    def bind_mouse_motion(self):
        """ Binds the mouse motion to the current editor's mouse <Motion> event.

        Called on initialization and active editor update.
        """
        self._canvas.bind("<Motion>", self._update_cursor)

    def _update_cursor(self, event):  # pylint: disable=unused-argument
        """ The mouse cursor display as bound to the mouses <Motion> event..

        The default is to always return a standard cursor, so this method should be overridden for
        editor specific cursor update.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Unused for default tracking, but available for specific editor
            tracking.
        """
        self._canvas.config(cursor="")

    # Mouse click and drag actions
    def set_mouse_click_actions(self):
        """ Add the bindings for left mouse button click and drag actions.

        This binds the mouse to the :func:`_drag_start`, :func:`_drag` and :func:`_drag_stop`
        methods.

        By default these methods do nothing (except for :func:`_drag_stop` which resets
        :attr:`_drag_data`.

        This bindings should be added for all editors. To add additional bindings,
        `super().set_mouse_click_actions` should be called prior to adding them..
        """
        logger.debug("Setting mouse bindings")
        self._canvas.bind("<ButtonPress-1>", self._drag_start)
        self._canvas.bind("<ButtonRelease-1>", self._drag_stop)
        self._canvas.bind("<B1-Motion>", self._drag)

    def _drag_start(self, event):  # pylint:disable=unused-argument
        """ The action to perform when the user starts clicking and dragging the mouse.

        The default does nothing except reset the attr:`drag_data` and attr:`drag_callback`.
        Override for Editor specific click and drag start actions.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Unused but for default action, but available for editor
            specific actions
        """
        self._drag_data = dict()
        self._drag_callback = None

    def _drag(self, event):
        """ The default callback for the drag part of a mouse click and drag action.

        :attr:`_drag_callback` should be set in :func:`self._drag_start`. This callback will then
        be executed on a mouse drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if self._drag_callback is None:
            return
        self._drag_callback(event)

    def _drag_stop(self, event):  # pylint:disable=unused-argument
        """ The action to perform when the user stops clicking and dragging the mouse.

        Default is to set :attr:`_drag_data` to `dict`. Override for Editor specific stop actions.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Unused but required
        """
        self._drag_data = dict()

    def _get_objects_from_identifier(self, identifier):
        """ Retrieve all of the objects that match the given identifier suffix

        Parameters
        ----------
        identifier: str
            The identifier suffix to search tags for

        Returns
        -------
        Tuple
            The object ids of objects containing the identifier suffix
        """
        return tuple(obj for obj in self._canvas.find_all()
                     if any(tag for tag in self._canvas.gettags(obj) if tag.endswith(identifier)))

    def _scale_to_display(self, points):
        """ Scale and offset the given points to the current display scale and offset values.

        Parameters
        ----------
        points: :class:`numpy.ndarray`
            Array of x, y co-ordinates to adjust

        Returns
        -------
        :class:`numpy.ndarray`
            The adjusted x, y co-ordinates for display purposes
        """
        retval = (points * self._frames.current_scale) + self._canvas.offset
        logger.trace("Original points: %s, scaled points: %s", points, retval)
        return retval

    def scale_from_display(self, points, do_offset=True):
        """ Scale and offset the given points from the current display to the correct original
        values.

        Parameters
        ----------
        points: :class:`numpy.ndarray`
            Array of x, y co-ordinates to adjust
        offset: bool, optional
            ``True`` if the offset should be calculated otherwise ``False``. Default: ``True``

        Returns
        -------
        :class:`numpy.ndarray`
            The adjusted x, y co-ordinates to the original frame location
        """
        offset = self._canvas.offset if do_offset else (0, 0)
        retval = (points - offset) / self._frames.current_scale
        logger.trace("Original points: %s, scaled points: %s", points, retval)
        return retval


class BoundingBox(Editor):
    """ The Bounding Box Editor. """
    def __init__(self, canvas, alignments, frames):
        self._right_click_menu = RightClickMenu(["Delete Face"],
                                                [self._delete_current_face],
                                                ["Del"])
        super().__init__(canvas, alignments, frames)
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
    def _anchors(self):
        """ list: List of bounding box anchors for the corners of each face's bounding box. """
        return [[self._canvas.coords(obj) for obj in face[1:]]
                for face in self._objects]

    @property
    def _corner_points(self):
        """ list: List of bounding box tuples for each face's bounding box """
        return [((self._canvas.coords(obj[0])[0], self._canvas.coords(obj[0])[1]),
                 (self._canvas.coords(obj[0])[0], self._canvas.coords(obj[0])[3]),
                 (self._canvas.coords(obj[0])[2], self._canvas.coords(obj[0])[1]),
                 (self._canvas.coords(obj[0])[2], self._canvas.coords(obj[0])[3]))
                for obj in self._objects]

    @property
    def _bounding_boxes(self):
        """ list: List of (`Left`, `Top`, `Right`, `Bottom`) tuples for each displayed face's
        bounding box. """
        return [self._canvas.coords(face[0]) for face in self._objects]

    def _bind_hotkeys(self):
        """ Add keyboard shortcuts.

        We bind to root because the canvas does not get focus, so keyboard shortcuts won't do
        anything

        * Delete - Delete the currently hovered over face
        """
        self._canvas.winfo_toplevel().bind("<Delete>", self._delete_current_face)

    def _bbox_objects_for_face(self, index):
        """ Return the bounding box object with the anchor objects for the given face index.

        Parameters
        ----------
        index: int
            The face index to return the bounding box objects for

        Returns
        -------
        list
            A list of bounding box object and bounding box anchor objects. Bounding box is in
            position 0, anchors in positions 1 to 4.
        """
        retval = self._objects[index]
        logger.trace("objects: %s, index: %s, selected object: %s", self._objects, index, retval)
        return retval

    def update_annotation(self):
        """ Draw the bounding box around faces and set the object to :attr:`_object`"""
        if self._drag_data:
            logger.trace("Object being edited. Not updating annotation")
            return
        self._clear_annotation()
        if not self._is_active and self._active_editor != "view":
            self._objects = []
            return
        color = self._colors["blue"]
        thickness = 1
        faces = []
        for idx, face in enumerate(self._alignments.current_faces):
            bbox = []
            box = np.array([(face.left, face.top), (face.right, face.bottom)])
            box = self._scale_to_display(box).astype("int32").flatten()
            bbox.append(self._canvas.create_rectangle(*box, outline=color, width=thickness))
            bbox.extend(self._update_anchor_annotation(idx, box, thickness, color))
            faces.append(bbox)
        logger.trace("Updated annotations: %s", faces)
        self._objects = faces

    def _update_anchor_annotation(self, face_index, bounding_box, thickness, color):
        """ Update the anchor annotations for each corner of the bounding box.

        The anchors only display when the bounding box editor is active.

        Parameters
        ----------
        face_index: int
            The index of the face being annotated
        bounding_box: :class:`numpy.ndarray`
            The scaled bounding box to get the corner anchors for
        thickness: int
            The line thickness of the bounding box
        color: str
            The hex color of the bounding box line
        """
        bbox = []
        color = color if self._is_active else ""
        fill_color = "gray" if self._is_active else ""
        activefill_color = "white" if self._is_active else ""
        corners = self._corners_from_coords(bounding_box)
        for idx, (anc_dsp, anc_grb) in enumerate(zip(*self._get_anchor_points(corners))):
            bbox.append(self._canvas.create_oval(*anc_dsp,
                                                 outline=color,
                                                 fill=fill_color,
                                                 width=thickness))
            bbox.append(self._canvas.create_oval(*anc_grb,
                                                 outline="",
                                                 fill="",
                                                 width=thickness,
                                                 activefill=activefill_color,
                                                 tags="anchor_{}_{}".format(face_index, idx)))
        return bbox

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
        item_ids = self._canvas.find_withtag("current")
        if not item_ids:
            return False

        item_id = item_ids[0]
        tags = [name for name in self._canvas.gettags(item_id) if name.startswith("anchor")]
        if not tags:
            return False

        tag = tags[0]
        face_idx, corner_idx = (int(idx) for idx in tag.split("_")[1:])
        self._canvas.config(cursor="{}_{}_corner".format(*self._corner_order[corner_idx]))
        self._mouse_location = ("anchor", (face_idx, corner_idx))
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
        for face_idx in range(len(self._bounding_boxes)):
            bbox = self._bounding_boxes[face_idx]
            if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                self._canvas.config(cursor="fleur")
                self._mouse_location = ("box", face_idx)
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
            return
        if self._mouse_location[0] == "anchor":
            self._drag_data["face_index"] = self._mouse_location[1][0]
            self._drag_data["corner"] = self._corner_order[self._mouse_location[1][1]]
            self._drag_data["objects"] = self._bbox_objects_for_face(self._mouse_location[1][0])
            self._drag_callback = self._resize
        elif self._mouse_location[0] == "box":
            self._drag_data["face_index"] = self._mouse_location[1]
            self._drag_data["objects"] = self._bbox_objects_for_face(self._mouse_location[1])
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
        rect = self._drag_data["objects"][0]
        box = list(self._canvas.coords(rect))
        # Switch top/bottom and left/right and set partial so indices match and we don't
        # need branching logic for min/max.
        limits = (partial(min, box[2] - 20),
                  partial(min, box[3] - 20),
                  partial(max, box[0] + 20),
                  partial(max, box[1] + 20))
        rect_xy_indices = [self._coords_layout.index(pnt)
                           for pnt in self._drag_data["corner"]]
        box[rect_xy_indices[1]] = limits[rect_xy_indices[1]](event.x)
        box[rect_xy_indices[0]] = limits[rect_xy_indices[0]](event.y)
        self._canvas.coords(rect, *box)
        corners = self._corners_from_coords(box)
        for idx, (anc_dsp, anc_grb) in enumerate(zip(*self._get_anchor_points(corners))):
            self._canvas.coords(self._drag_data["objects"][(idx * 2) + 1], *anc_dsp)
            self._canvas.coords(self._drag_data["objects"][(idx * 2) + 2], *anc_grb)
        self._alignments.set_current_bounding_box(self._drag_data["face_index"],
                                                  *self._coords_to_bounding_box(box))

    def _move(self, event):
        """ Moves the bounding box on a bounding box drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.

        """
        shift_x = event.x - self._drag_data["current_location"][0]
        shift_y = event.y - self._drag_data["current_location"][1]
        selected_objects = self._drag_data["objects"]
        for obj in selected_objects:
            self._canvas.move(obj, shift_x, shift_y)
        box = self._canvas.coords(selected_objects[0])
        self._alignments.set_current_bounding_box(self._drag_data["face_index"],
                                                  *self._coords_to_bounding_box(box))
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
        self._alignments.delete_face_at_index(self._mouse_location[1])


class ExtractBox(Editor):
    """ The Extract Box Editor. """
    def __init__(self, canvas, alignments, frames):
        self._right_click_menu = RightClickMenu(["Delete Face"],
                                                [self._delete_current_face],
                                                ["Del"])
        super().__init__(canvas, alignments, frames)
        self._bind_hotkeys()

    def _bind_hotkeys(self):
        """ Add keyboard shortcuts.

        We bind to root because the canvas does not get focus, so keyboard shortcuts won't do
        anything

        * Delete - Delete the currently hovered over face
        """
        self._canvas.winfo_toplevel().bind("<Delete>", self._delete_current_face)

    def update_annotation(self):
        """ Draw the Extract Box around faces and set the object to :attr:`_object`"""
        self._clear_annotation()
        if not self._is_active and self._active_editor != "view":
            self._objects = []
            return
        color = self._colors["green"]
        thickness = 1
        faces = []
        # TODO FIX THIS TEST
        #  if not all(face.original_roi for face in self._alignments.current_faces):
        #      return extract_box
        for idx, face in enumerate(self._alignments.current_faces):
            extract_box = []
            logger.trace("Drawing Extract Box: (idx: %s, roi: %s)", idx, face.original_roi)
            box = self._scale_to_display(face.original_roi).flatten()
            top_left = box[:2] - 10
            extract_box.append(self._canvas.create_text(*top_left,
                                                        fill=color,
                                                        font=("Default", 20, "bold"),
                                                        text=str(idx)))
            extract_box.append(self._canvas.create_polygon(*box,
                                                           fill="",
                                                           outline=color,
                                                           width=thickness,
                                                           tags="extract_box_{}".format(idx)))
            faces.append(extract_box)
        logger.trace("Updated annotations: %s", faces)
        self._objects = faces

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """ Update the cursors for hovering over extract boxes and update
        :attr:`_mouse_location`. """
        item_ids = self._canvas.find_withtag("current")
        if not item_ids:
            self._canvas.config(cursor="")
            self._mouse_location = None
            return
        item_id = item_ids[0]
        tags = [name for name in self._canvas.gettags(item_id) if name.startswith("extract_box")]
        if not tags:
            self._canvas.config(cursor="")
            self._mouse_location = None
            return
        tag = tags[0]
        self._canvas.config(cursor="fleur")
        self._mouse_location = int(tag.split("_")[-1])

    # Mouse click actions
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
            return
        self._drag_data["face_index"] = self._mouse_location
        self._drag_data["objects"] = self._objects[self._mouse_location]
        self._drag_data["current_location"] = (event.x, event.y)
        self._drag_callback = self._move

    def _move(self, event):
        """ Moves the Extract box and the underlying landmarks on an extract box drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.

        """
        shift_x = event.x - self._drag_data["current_location"][0]
        shift_y = event.y - self._drag_data["current_location"][1]
        selected_objects = self._drag_data["objects"]
        for obj in selected_objects:
            self._canvas.move(obj, shift_x, shift_y)
        scaled_shift = self.scale_from_display(np.array((shift_x, shift_y)), do_offset=False)
        self._alignments.shift_landmarks(self._drag_data["face_index"], *scaled_shift)
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
        self._alignments.delete_face_at_index(self._mouse_location)


class Landmarks(Editor):
    """ The Landmarks Editor. """
    def __init__(self, canvas, alignments, frames):
        self._landmark_mapping = dict(mouth=(48, 68),
                                      right_eyebrow=(17, 22),
                                      left_eyebrow=(22, 27),
                                      right_eye=(36, 42),
                                      left_eye=(42, 48),
                                      nose=(27, 36),
                                      jaw=(0, 17),
                                      chin=(8, 11))
        super().__init__(canvas, alignments, frames)

    def update_annotation(self):
        """ Draw the Landmarks and the Face Mesh set the objects to :attr:`_object`"""
        self._clear_annotation()
        landmarks = self._update_landmarks()
        mesh = self._update_mesh()
        self._objects = landmarks + mesh

    def _update_landmarks(self):
        """ Draw the facial landmarks """
        faces = []
        for face_idx, face in enumerate(self._alignments.current_faces):
            landmarks = []
            for lm_idx, landmark in enumerate(face.landmarks_xy):
                box = self._scale_to_display(landmark).astype("int32")
                landmarks.append(self._display_landmark(box, face_idx, lm_idx))
                landmarks.append(self._grab_landmark(box, face_idx, lm_idx))
                landmarks.extend(self._label_landmark(box, face_idx, lm_idx))
            faces.append(landmarks)
        logger.trace("Updated landmark annotations: %s", faces)
        return faces

    def _display_landmark(self, bounding_box, face_index, landmark_index):
        """ Add a display landmark to the canvas.

        Parameters
        ----------
        box: :class:`numpy.ndarray`
            The (left, top), (right, bottom) (x, y) coordinates of the oval bounding box
        face_index: int
            The index of the face being annotated
        landmark_index
            The index of the landmark being annotated

        Returns
        int
            The id of the canvas oval object
        """
        radius = 1
        color = self._colors["red"]
        bbox = (bounding_box[0] - radius, bounding_box[1] - radius,
                bounding_box[0] + radius, bounding_box[1] + radius)
        return self._canvas.create_oval(*bbox,
                                        outline=color,
                                        fill=color,
                                        width=radius,
                                        tags="dsp_{}_{}".format(face_index, landmark_index))

    def _grab_landmark(self, bounding_box, face_index, landmark_index):
        """ Add a grab landmark to the canvas.

        Parameters
        ----------
        box: :class:`numpy.ndarray`
            The (left, top), (right, bottom) (x, y) coordinates of the oval bounding box
        face_index: int
            The index of the face being annotated
        landmark_index
            The index of the landmark being annotated

        Returns
        int
            The id of the canvas oval object
        """
        radius = 6
        activeoutline_color = "black" if self._is_active else ""
        activefill_color = "white" if self._is_active else ""
        bbox = (bounding_box[0] - radius, bounding_box[1] - radius,
                bounding_box[0] + radius, bounding_box[1] + radius)
        return self._canvas.create_oval(*bbox,
                                        outline="",
                                        fill="",
                                        width=radius,
                                        activeoutline=activeoutline_color,
                                        activefill=activefill_color,
                                        tags="grb_{}_{}".format(face_index, landmark_index))

    def _label_landmark(self, bounding_box, face_index, landmark_index):
        """ Add a text label for a landmark to the canvas.

        Parameters
        ----------
        box: :class:`numpy.ndarray`
            The (left, top), (right, bottom) (x, y) coordinates of the oval bounding box
        face_index: int
            The index of the face being annotated
        landmark_index
            The index of the landmark being annotated

        Returns
        tuple
            The ids of the canvas text object and the background canvas rectangle object
        """
        top_left = np.array(bounding_box[:2]) - 16
        # NB The fill must be visible to be able to get the bounding box, so set fill to empty
        # after the bounding box has been retrieved
        text = self._canvas.create_text(*top_left,
                                        fill="black",
                                        font=("Default", 10),
                                        text=str(landmark_index + 1),
                                        tags=("label",
                                              "lbltxt_{}_{}".format(face_index, landmark_index)))
        bbox = self._canvas.bbox(text)
        bbox = [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2]
        bgr = self._canvas.create_rectangle(bbox,
                                            fill="",
                                            outline="",
                                            tags=("label",
                                                  "lblbg_{}_{}".format(face_index,
                                                                       landmark_index)))
        self._canvas.lower(bgr, text)
        self._canvas.itemconfig(text, fill="")
        return text, bgr

    def _update_mesh(self):
        """ Draw the facial landmarks """
        color = "" if self._is_active or self._active_editor == "mask" else self._colors["cyan"]
        thickness = 1
        faces = []
        for face in self._alignments.current_faces:
            mesh = []
            landmarks = face.landmarks_xy
            logger.trace("Drawing Landmarks Mesh: (landmarks: %s, color: %s, thickness: %s)",
                         landmarks, color, thickness)
            for key, val in self._landmark_mapping.items():
                pts = self._scale_to_display(landmarks[val[0]:val[1]]).astype("int32").flatten()
                if key in ("right_eye", "left_eye", "mouth"):
                    mesh.append(self._canvas.create_polygon(*pts,
                                                            fill="",
                                                            outline=color,
                                                            width=thickness))
                else:
                    mesh.append(self._canvas.create_line(*pts, fill=color, width=thickness))
            faces.append(mesh)
        logger.trace("Updated mesh annotations: %s", faces)
        return faces

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """ Update the cursors for hovering over extract boxes and update
        :attr:`_mouse_location`. """
        self._clear_labels()
        item_ids = self._canvas.find_withtag("current")
        if not item_ids:
            self._canvas.config(cursor="")
            self._mouse_location = None
            return
        item_id = item_ids[0]
        tags = [tag for tag in self._canvas.gettags(item_id) if tag.startswith("grb")]
        if not tags:
            self._canvas.config(cursor="")
            self._mouse_location = None
            return
        identifier = tags[0][tags[0].find("_") + 1:]
        self._canvas.config(cursor="fleur")
        txt_label = self._canvas.find_withtag("lbltxt_{}".format(identifier))
        txt_bg = self._canvas.find_withtag("lblbg_{}".format(identifier))
        self._canvas.itemconfig(txt_label, fill="black")
        self._canvas.itemconfig(txt_bg, fill="#ffffea", outline="black")
        self._mouse_location = identifier

    def _clear_labels(self):
        """ Clear all landmark text labels from display """
        labels = self._canvas.find_withtag("label")
        for item_id in labels:
            self._canvas.itemconfig(item_id, fill="")
            if self._canvas.type(item_id) == "rectangle":
                self._canvas.itemconfig(item_id, outline="")

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
            return
        self._drag_data["current_location"] = (event.x, event.y)
        self._drag_callback = self._move

    def _move(self, event):
        """ Moves the Extract box and the underlying landmarks on an extract box drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        shift_x = event.x - self._drag_data["current_location"][0]
        shift_y = event.y - self._drag_data["current_location"][1]
        objects = self._get_objects_from_identifier(self._mouse_location)
        for obj in objects:
            self._canvas.move(obj, shift_x, shift_y)
        scaled_shift = self.scale_from_display(np.array((shift_x, shift_y)), do_offset=False)
        self._alignments.shift_landmark(*[int(idx) for idx in self._mouse_location.split("_")],
                                        *scaled_shift)
        self._drag_data["current_location"] = (event.x, event.y)


class RightClickMenu(tk.Menu):  # pylint: disable=too-many-ancestors
    """ A Pop up menu that can be bound to a right click mouse event to bring up a context menu

    Parameters
    ----------
    labels: list
        A list of label titles that will appear in the right click menu
    actions: list
        A list of python functions that are called when the corresponding label is clicked on
    hotkeys: list, optional
        The hotkeys corresponding to the labels. If using hotkeys, then there must be an entry in
        the list for every label even if they don't all use hotkeys. Labels without a hotkey can be
        an empty string or ``None``. Passing ``None`` instead of a list means that no actions will
        be given hotkeys. NB: The hotkey is not bound by this class, that needs to be done in code.
        Giving hotkeys here means that they will be displayed in the menu though. Default: ``None``
    """
    def __init__(self, labels, actions, hotkeys=None):
        logger.debug("Initializing %s: (labels: %s, actions: %s)", self.__class__.__name__, labels,
                     actions)
        super().__init__(tearoff=0)
        self._labels = labels
        self._actions = actions
        self._hotkeys = hotkeys
        self._create_menu()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _create_menu(self):
        """ Create the menu based on :attr:`_labels` and :attr:`_actions`. """
        for idx, (label, action) in enumerate(zip(self._labels, self._actions)):
            kwargs = dict(label=label, command=action)
            if isinstance(self._hotkeys, (list, tuple)) and self._hotkeys[idx]:
                kwargs["accelerator"] = self._hotkeys[idx]
            self.add_command(**kwargs)

    def popup(self, event):
        """ Pop up the right click menu.

        Parameters
        ----------
        event: class:`tkinter.Event`
            The tkinter mouse event calling this popup
        """
        self.tk_popup(event.x_root + 15, event.y_root + 5, 0)
