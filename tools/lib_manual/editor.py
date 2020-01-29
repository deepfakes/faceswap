#!/usr/bin/env python3
""" Editor objects for the manual adjustments tool """

import logging
import platform
import tkinter as tk

from functools import partial

import cv2
import numpy as np

from lib.gui.control_helper import ControlPanelOption

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# pylint:disable=too-many-lines


# Tracking for all control panel variables, for access from all editors
_CONTROL_VARS = dict()
_ANNOTATION_FORMAT = dict()

# TODO Hide annotations for additional faces
# TODO Landmarks, Color outline and fill


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
    control_text: str
        The text that is to be displayed at the top of the Editor's control panel.
    """
    def __init__(self, canvas, alignments, frames, control_text=""):
        logger.debug("Initializing %s: (canvas: '%s', alignments: %s, frames: %s, "
                     "control_text: %s)", self.__class__.__name__, canvas, alignments, frames,
                     control_text)
        self._canvas = canvas
        self._alignments = alignments
        self._frames = frames

        self._current_color = dict()
        self._controls = dict(header=control_text, controls=[])
        self._add_controls()
        self._add_annotation_format_controls()

        self._objects = dict()
        self._mouse_location = None
        self._drag_data = dict()
        self._drag_callback = None
        self._right_click_button = "<Button-2>" if platform.system() == "Darwin" else "<Button-3>"
        self.bind_mouse_motion()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _colors(self):
        """ dict: Available colors for annotations """
        return dict(black="#000000",
                    red="#ff0000",
                    green="#00ff00",
                    blue="#0000ff",
                    cyan="#00ffff",
                    yellow="#ffff00",
                    magenta="#ff00ff",
                    white="#ffffff")

    @property
    def _default_colors(self):
        """ dict: The default colors for each annotation """
        return {"Bounding Box": "blue",
                "Extract Box": "green",
                "Landmarks": "magenta",
                "Mask": "red",
                "Mesh": "cyan"}

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

    @property
    def controls(self):
        """ dict: The control panel options and header text for the current editor """
        return self._controls

    @property
    def _control_color(self):
        """ str: The hex color code set in the control panel for the current editor. """
        annotation = self.__class__.__name__.lower()
        return self._colors[_ANNOTATION_FORMAT[annotation]["color"].get()]

    @property
    def _should_display(self):
        """ bool: Whether the control panel option for the current editor is set to display
        or not. """
        annotation = self.__class__.__name__.lower()
        should_display = _CONTROL_VARS.get(self._active_editor,
                                           dict()).get("display", dict()).get(annotation, None)
        return self._is_active or (should_display is not None and should_display.get())

    def update_annotation(self):
        """ Update the display annotations for the current objects.

        Override for specific editors.
        """
        logger.trace("Default annotations. Not storing Objects")
        self._hide_annotation()

    def _hide_annotation(self, key=None):
        """ Hide annotations for this editor.

        Parameters
        ----------
        key: str, optional
            The object key from :attr:`_objects` to hide the annotations for. If ``None`` then
            all annotations are hidden for this editor. Default: ``None``
        """
        objects = self._objects.values() if key is None else [self._objects.get(key, [])]
        for objs in objects:
            for item_id in objs:
                if self._canvas.itemcget(item_id, "state") == "hidden":
                    continue
                logger.trace("Hiding: %s, id: %s", self._canvas.type(item_id), item_id)
                self._canvas.itemconfig(item_id, state="hidden")

    def _create_or_update(self, key, object_type, object_index, coordinates, object_kwargs):
        """ Create an annotation object and add it to :attr:`_objects` or update an existing
        annotation if it has already been created.

        Parameters
        ----------
        key: str
            The key for this annotation in :attr:`_objects`
        object_type: str
            This can be any string that is a natural extension to :class:`tkinter.Canvas.create_`
        object_index: int
            The object_index for this item for the list returned from :attr:`_objects`[`key`]
        coordinates: tuple or list
            The bounding box coordinates for this object
        object_kwargs: dict
            The keyword arguments for this object
        """
        object_color_key = "fill" if object_type in ("text", "line") else "outline"
        tracking_id = "_".join((key, str(object_index)))
        if key not in self._objects or len(self._objects[key]) - 1 < object_index:
            logger.trace("Adding object: (key: '%s', object_type: '%s', object_index: %s, "
                         "coordinates: %s, object_kwargs: %s)",
                         key, object_type, object_index, coordinates, object_kwargs)
            obj = getattr(self._canvas, "create_{}".format(object_type))
            self._objects.setdefault(key, []).append(obj(*coordinates, **object_kwargs))
            self._current_color[tracking_id] = object_kwargs[object_color_key]
        else:
            obj = self._objects[key][object_index]
            update_kwargs = dict()
            if object_kwargs.get("state", "normal") != "hidden":
                logger.trace("Setting object state to normal: %s id: %s",
                             self._canvas.type(obj), obj)
                update_kwargs["state"] = "normal"
            if object_kwargs[object_color_key] != self._current_color[tracking_id]:
                new_color = object_kwargs[object_color_key]
                update_kwargs[object_color_key] = new_color
                self._current_color[tracking_id] = new_color
            if update_kwargs:
                self._canvas.itemconfig(self._objects[key][object_index], **update_kwargs)
            self._canvas.coords(self._objects[key][object_index], *coordinates)
            logger.trace("Updating object: (key: '%s', object_type: %s, object_index:%s, "
                         "coordinates: %s)", key, object_type, object_index, coordinates)

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

    # << CONTROL PANEL OPTIONS >>
    def _add_controls(self):
        """ Add the controls for this editor's control panel.

        The default does nothing. Override for editor specific controls
        """
        self._controls = self._controls

    def _add_control(self, option, global_control=False):
        """ Add a control panel control to :attr:`_controls` and add a trace to the variable
        to update display.

        Parameters
        ----------
        option: :class:`lib.gui.control_helper.ControlPanelOption'
            The control panel option to add to this editor's control
        global_control: bool, optional
            Whether the given control is a global control (i.e. annotation formatting).
            Default: ``False``
        """
        self._controls["controls"].append(option)
        if global_control:
            return

        editor_key = self.__class__.__name__.lower()
        group_key = option.group.replace(" ", "").lower()
        group_key = "none" if group_key == "_master" else group_key
        annotation_key = option.title.replace(" ", "").lower()
        _CONTROL_VARS.setdefault(editor_key,
                                 dict()).setdefault(group_key,
                                                    dict())[annotation_key] = option.tk_var

    def _add_annotation_format_controls(self):
        """ Add the annotation display (color/size) controls.

        These should be universal and available for all editors.
        """
        editors = ("Bounding Box", "Extract Box", "Landmarks", "Mask", "Mesh")
        if not _ANNOTATION_FORMAT:
            opacity = ControlPanelOption("Mask Opacity",
                                         int,
                                         group="Color",
                                         min_max=(0, 100),
                                         default=20,
                                         rounding=1,
                                         helptext="Set the mask opacity")
            for editor in editors:
                logger.debug("Adding to global format controls: '%s'", editor)
                colors = ControlPanelOption(editor,
                                            str,
                                            group="Color",
                                            choices=sorted(self._colors),
                                            default=self._default_colors[editor],
                                            is_radio=False,
                                            state="readonly",
                                            helptext="Set the annotation color")
                annotation_key = editor.replace(" ", "").lower()
                _ANNOTATION_FORMAT.setdefault(annotation_key, dict())["color"] = colors
                _ANNOTATION_FORMAT[annotation_key]["mask_opacity"] = opacity

        for editor in editors:
            annotation_key = editor.replace(" ", "").lower()
            for group, ctl in _ANNOTATION_FORMAT[annotation_key].items():
                logger.debug("Adding global format control to editor: (editor:'%s', group: '%s')",
                             editor, group)
                self._add_control(ctl, global_control=True)


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
        thickness = 1
        for idx, face in enumerate(self._alignments.current_faces):
            box = np.array([(face.left, face.top), (face.right, face.bottom)])
            box = self._scale_to_display(box).astype("int32").flatten()
            kwargs = dict(outline=color, width=thickness)
            self._create_or_update(key, "rectangle", idx, box, kwargs)
            self._update_anchor_annotation(idx, box, thickness, color)
        logger.trace("Updated bounding box annotations: %s", self._objects[key])

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
            dsp_kwargs = dict(outline=color, fill=fill_color, width=thickness)
            self._create_or_update(keys[0], "oval", obj_idx, anc_dsp, dsp_kwargs)
            grb_kwargs = dict(outline="", fill="", width=thickness, activefill=activefill_color)
            self._create_or_update(keys[1], "oval", obj_idx, anc_grb, grb_kwargs)
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
        self._alignments.delete_face_at_index(self._mouse_location[1])


class ExtractBox(Editor):
    """ The Extract Box Editor. """
    def __init__(self, canvas, alignments, frames):
        self._right_click_menu = RightClickMenu(["Delete Face"],
                                                [self._delete_current_face],
                                                ["Del"])
        control_text = ("Extract Box Editor\nMove the extract box that has been generated by the "
                        "aligner.\n\n"
                        " - Click and drag the bounding box to relocate the landmarks without "
                        "recalculating them.")
        super().__init__(canvas, alignments, frames, control_text)
        self._bind_hotkeys()

    def _add_controls(self):
        for dsp in ("Landmarks", "Mesh"):
            self._add_control(ControlPanelOption(dsp,
                                                 bool,
                                                 group="Display",
                                                 default=dsp != "Landmarks",
                                                 helptext="Show the {} annotations".format(dsp)))

    def _bind_hotkeys(self):
        """ Add keyboard shortcuts.

        We bind to root because the canvas does not get focus, so keyboard shortcuts won't do
        anything

        * Delete - Delete the currently hovered over face
        """
        self._canvas.winfo_toplevel().bind("<Delete>", self._delete_current_face)

    def update_annotation(self):
        """ Draw the Extract Box around faces and set the object to :attr:`_object`"""
        if not self._should_display:
            self._hide_annotation()
            return
        keys = ("text", "extractbox")
        color = self._control_color
        thickness = 1
        # TODO FIX THIS TEST
        #  if not all(face.original_roi for face in self._alignments.current_faces):
        #      return extract_box
        for idx, face in enumerate(self._alignments.current_faces):
            logger.trace("Drawing Extract Box: (idx: %s, roi: %s)", idx, face.original_roi)
            box = self._scale_to_display(face.original_roi).flatten()
            top_left = box[:2] - 10
            kwargs = dict(fill=color, font=("Default", 20, "bold"), text=str(idx))
            self._create_or_update(keys[0], "text", idx, top_left, kwargs)
            kwargs = dict(fill="", outline=color, width=thickness)
            self._create_or_update(keys[1], "polygon", idx, box, kwargs)
            self._canvas.tag_raise(self._objects[keys[1]][idx])

        logger.trace("Updated extract box annotations: %s", {key: self._objects[key]
                                                             for key in keys})

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """ Update the cursors for hovering over extract boxes and update
        :attr:`_mouse_location`. """
        extract_boxes = self._objects["extractbox"]
        item_ids = set(self._canvas.find_withtag("current")).intersection(extract_boxes)
        if not item_ids:
            self._canvas.config(cursor="")
            self._mouse_location = None
        else:
            self._canvas.config(cursor="fleur")
            self._mouse_location = extract_boxes.index(list(item_ids)[0])

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
        self._drag_data["current_location"] = (event.x, event.y)
        self._drag_callback = self._move

    def _move(self, event):
        """ Moves the Extract box and the underlying landmarks on an extract box drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.

        """
        if not self._drag_data:
            # TODO This should never have no data, but sometimes it does. It doesn't appear
            # to interfere with the GUI beyond spitting out errors.
            return
        shift_x = event.x - self._drag_data["current_location"][0]
        shift_y = event.y - self._drag_data["current_location"][1]
        for obj in self._objects.values():
            self._canvas.move(obj[self._mouse_location], shift_x, shift_y)
        scaled_shift = self.scale_from_display(np.array((shift_x, shift_y)), do_offset=False)
        self._alignments.shift_landmarks(self._mouse_location, *scaled_shift)
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
        self._landmark_count = None
        control_text = ("Landmark Point Editor\nEdit the individual landmark points.\n\n"
                        " - Click and drag individual landmark points to relocate.")
        super().__init__(canvas, alignments, frames, control_text)

    def _add_controls(self):
        for dsp in ("Extract Box", "Landmarks", "Mesh"):
            self._add_control(ControlPanelOption(dsp,
                                                 bool,
                                                 group="Display",
                                                 default=dsp == "Landmarks",
                                                 helptext="Show the {} annotations".format(dsp)))

    def update_annotation(self):
        """ Draw the Landmarks and set the objects to :attr:`_object`"""
        for face_idx, face in enumerate(self._alignments.current_faces):
            if self._landmark_count is None:
                self._landmark_count = len(face.landmarks_xy)
            for lm_idx, landmark in enumerate(face.landmarks_xy):
                box = self._scale_to_display(landmark).astype("int32")
                obj_idx = (face_idx * self._landmark_count) + lm_idx
                self._display_landmark(box, obj_idx)
                self._grab_landmark(box, obj_idx)
                self._label_landmark(box, obj_idx, lm_idx)
        logger.trace("Updated landmark annotations: %s", self._objects)

    def _display_landmark(self, bounding_box, object_index):
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
        self._create_or_update(key, "oval", object_index, bbox, kwargs)

    def _grab_landmark(self, bounding_box, object_index):
        """ Add a grab landmark to the canvas.

        Parameters
        ----------
        box: :class:`numpy.ndarray`
            The (left, top), (right, bottom) (x, y) coordinates of the oval bounding box
        object_index: int
            The index of the this item in :attr:`_objects`
        """
        key = "grab"
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
        self._create_or_update(key, "oval", object_index, bbox, kwargs)

    def _label_landmark(self, bounding_box, object_index, landmark_index):
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
        keys = ["label", "label_background"]
        if not self._is_active:
            for key in keys:
                self._hide_annotation(key)
            return
        top_left = np.array(bounding_box[:2]) - 16
        # NB The text must be visible to be able to get the bounding box, so set to hidden
        # after the bounding box has been retrieved

        text_kwargs = dict(fill="black", font=("Default", 10), text=str(landmark_index + 1))
        self._create_or_update(keys[0], "text", object_index, top_left, text_kwargs)

        bbox = self._canvas.bbox(self._objects[keys[0]][object_index])
        bbox = [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2]
        background_kwargs = dict(fill="#ffffea", outline="black")
        self._create_or_update(keys[1], "rectangle", object_index, bbox, background_kwargs)
        self._canvas.lower(self._objects[keys[1]][object_index],
                           self._objects[keys[0]][object_index])
        self._canvas.itemconfig(self._objects[keys[0]][object_index], state="hidden")
        self._canvas.itemconfig(self._objects[keys[1]][object_index], state="hidden")

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """ Update the cursors for hovering over extract boxes and update
        :attr:`_mouse_location`. """
        self._hide_labels()
        item_ids = set(self._canvas.find_withtag("current")).intersection(self._objects["grab"])
        if not item_ids:
            self._canvas.config(cursor="")
            self._mouse_location = None
            return
        obj_idx = self._objects["grab"].index(list(item_ids)[0])
        self._canvas.config(cursor="fleur")
        for label in [self._objects["label"][obj_idx], self._objects["label_background"][obj_idx]]:
            logger.trace("Displaying: %s id: %s", self._canvas.type(label), label)
            self._canvas.itemconfig(label, state="normal")
        self._mouse_location = obj_idx

    def _hide_labels(self):
        """ Clear all landmark text labels from display """
        labels = [idx for key, val in self._objects.items() for idx in val
                  if key.startswith("label") and self._canvas.itemcget(idx, "state") == "normal"]
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
            return
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
        shift_x = event.x - self._drag_data["current_location"][0]
        shift_y = event.y - self._drag_data["current_location"][1]
        objects = [self._objects[key][self._mouse_location] for key in self._objects
                   if key != "mesh"]
        for obj in objects:
            logger.trace("Moving: %s id: %s", self._canvas.type(obj), obj)
            logger.trace(self._canvas.itemcget(obj, "state"))
            self._canvas.move(obj, shift_x, shift_y)
        scaled_shift = self.scale_from_display(np.array((shift_x, shift_y)), do_offset=False)
        self._alignments.shift_landmark(self._mouse_location // self._landmark_count,
                                        self._mouse_location % self._landmark_count,
                                        *scaled_shift)
        self._drag_data["current_location"] = (event.x, event.y)


class Mask(Editor):
    """ The mask Editor """
    def __init__(self, canvas, alignments, frames):
        control_text = ("Mask Editor\nEdit the mask."
                        "\n - NB: For Landmark based masks (e.g. components/extended) it is "
                        "better to make sure the landmarks are correct rather than editing the "
                        "mask directly. Any change to the landmarks after editing the mask will "
                        "override your manual edits.")
        super().__init__(canvas, alignments, frames, control_text)

    @property
    def _opacity(self):
        """ float: The mask opacity setting from the control panel from 0.0 - 1.0. """
        annotation = self.__class__.__name__.lower()
        return _ANNOTATION_FORMAT[annotation]["mask_opacity"].get() / 100.0

    def _add_controls(self):
        masks = sorted(msk.title() for msk in list(self._alignments.available_masks) + ["None"])
        self._add_control(ControlPanelOption("Mask type",
                                             str,
                                             group="Display",
                                             choices=masks,
                                             default=masks[0],
                                             is_radio=True,
                                             helptext="Select which mask to edit"))

    def update_annotation(self):
        """ Draw the Landmarks and set the objects to :attr:`_object`"""
        if not self._should_display:
            self._frames.set_current_default_frame()
            self._canvas.refresh_display_image()
            return
        key = self.__class__.__name__.lower()
        mask_type = _CONTROL_VARS[key]["display"]["masktype"].get().lower()
        background = self._frames.current_frame[..., 2::-1].copy()
        combined = background
        color = self._control_color[1:]
        opacity = self._opacity
        rgb_color = np.array(tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))) / 255.0
        for face in self._alignments.current_faces:
            mask = face.mask.get(mask_type, None)
            if mask is None:
                continue
            mask = mask.get_full_frame_mask(*reversed(self._frames.current_frame_dims))[..., None]
            mask = np.tile(np.rint(mask / 255.0), 3)
            mask[np.where((mask == [1., 1., 1.]).all(axis=2))] = rgb_color
            combined = cv2.addWeighted(background, 1.0, (mask * 255.0).astype("uint8"), opacity, 0)

        display = cv2.resize(combined,
                             self._frames.current_meta_data["display_dims"],
                             interpolation=self._frames.current_meta_data["interpolation"])
        self._frames.set_annotated_frame(display)
        self._canvas.refresh_display_image()
        logger.trace("Updated mask annotation")


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
        thickness = 1
        for face_idx, face in enumerate(self._alignments.current_faces):
            landmarks = face.landmarks_xy
            base_idx = (face_idx * len(self._landmark_mapping))
            logger.trace("Drawing Landmarks Mesh: (landmarks: %s, color: %s, thickness: %s)",
                         landmarks, color, thickness)
            for idx, (segment, val) in enumerate(self._landmark_mapping.items()):
                obj_idx = base_idx + idx
                pts = self._scale_to_display(landmarks[val[0]:val[1]]).astype("int32").flatten()
                if segment in ("right_eye", "left_eye", "mouth"):
                    kwargs = dict(fill="", outline=color, width=thickness)
                    self._create_or_update(key, "polygon", obj_idx, pts, kwargs)
                else:
                    kwargs = dict(fill=color, width=thickness)
                    self._create_or_update(key, "line", obj_idx, pts, kwargs)


class View(Editor):
    """ The view Editor """
    def __init__(self, canvas, alignments, frames):
        control_text = "Viewer\nPreview the frame's annotations."
        super().__init__(canvas, alignments, frames, control_text)

    def _add_controls(self):
        for dsp in ("Bounding Box", "Extract Box", "Landmarks", "Mask", "Mesh"):
            self._add_control(ControlPanelOption(dsp,
                                                 bool,
                                                 group="Display",
                                                 default=dsp != "Mask",
                                                 helptext="Show the {} annotations".format(dsp)))


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
