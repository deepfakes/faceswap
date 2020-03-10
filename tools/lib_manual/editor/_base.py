#!/usr/bin/env python3
""" Editor objects for the manual adjustments tool """

import logging
import tkinter as tk

from collections import OrderedDict

import numpy as np

from lib.gui.control_helper import ControlPanelOption

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
    def __init__(self, canvas, alignments, frames, control_text="", key_bindings=None):
        logger.debug("Initializing %s: (canvas: '%s', alignments: %s, frames: %s, "
                     "control_text: %s)", self.__class__.__name__, canvas, alignments, frames,
                     control_text)
        self._canvas = canvas
        self._alignments = alignments
        self._frames = frames

        self._current_color = dict()
        self._actions = OrderedDict()
        self._controls = dict(header=control_text, controls=[])
        self._add_key_bindings(key_bindings)

        self._add_actions()
        self._add_controls()
        self._add_annotation_format_controls()

        self._zoomed_roi = self._get_zoomed_roi()
        self._objects = dict()
        self._mouse_location = None
        self._drag_data = dict()
        self._drag_callback = None
        self.bind_mouse_motion()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _default_colors(self):
        """ dict: The default colors for each annotation """
        return {"BoundingBox": "blue",
                "ExtractBox": "green",
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
    def _is_zoomed(self):
        """ bool: ``True`` if a face is currently zoomed in, ``False`` if the full frame is
        displayed """
        return self._canvas.image_is_hidden

    @property
    def _active_editor(self):
        """ str: The name of the currently active editor """
        return self._canvas.selected_action

    @property
    def _control_vars(self):
        """ dict: The tk control panel variables for the currently selected editor. """
        return self._canvas.control_tk_vars.get(self.__class__.__name__, dict())

    @property
    def _annotation_formats(self):
        return self._canvas.annotation_formats

    @property
    def actions(self):
        """ list: The optional action buttons for the actions frame in the GUI for the
        current editor """
        return self._actions

    @property
    def controls(self):
        """ dict: The control panel options and header text for the current editor """
        return self._controls

    @property
    def objects(self):
        """ dict: The objects currently stored for this editor """
        return self._objects

    @property
    def _control_color(self):
        """ str: The hex color code set in the control panel for the current editor. """
        annotation = self.__class__.__name__
        return self._canvas.colors[self._annotation_formats[annotation]["color"].get()]

    @property
    def _zoomed_dims(self):
        """ tuple: The (`width`, `height`) of the zoomed ROI """
        return (self._zoomed_roi[2] - self._zoomed_roi[0],
                self._zoomed_roi[3] - self._zoomed_roi[1])

    def _add_key_bindings(self, key_bindings):
        if key_bindings is None:
            return
        for key, method in key_bindings.items():
            self._canvas.key_bindings.setdefault(key, dict())["bound_to"] = None
            self._canvas.key_bindings[key][self.__class__.__name__] = method

    def _get_zoomed_roi(self):
        """ Get the Region of Interest for when the face is zoomed.

        The ROI is dictated by display frame size, so it will be constant for every face

        Returns
        -------
        :class:`numpy.ndarray`
            The (`left`, `top`, `right`, `bottom`) roi of the zoomed face in the display frame
        """
        half_size = min(self._frames.display_dims) / 2
        left = self._frames.display_dims[0] / 2 - half_size
        top = 0
        right = self._frames.display_dims[0] / 2 + half_size
        bottom = self._frames.display_dims[1]
        retval = np.rint(np.array((left, top, right, bottom))).astype("int32")
        logger.debug("Zoomed ROI: %s", retval)
        return retval

    def update_annotation(self):  # pylint:disable=no-self-use
        """ Update the display annotations for the current objects.

        Override for specific editors.
        """
        logger.trace("Default annotations. Not storing Objects")

    def hide_annotation(self):
        """ Hide annotations for this editor. """
        for item_id in self._flatten_list(list(self._objects.values())):
            if self._canvas.itemcget(item_id, "state") != "hidden":
                logger.debug("Hiding: %s, id: %s", self._canvas.type(item_id), item_id)
                self._canvas.itemconfig(item_id, state="hidden")

    def hide_additional_annotations(self):
        """ Hide any excess face annotations """
        current_face_count = len(self._alignments.current_faces)
        hide_objects = [self._flatten_list(face[-(len(face) - current_face_count):])
                        for face in self._objects.values()
                        if len(face) > current_face_count]
        if not hide_objects:
            return
        for item_id in self._flatten_list(hide_objects):
            if self._canvas.itemcget(item_id, "state") == "hidden":
                continue
            logger.trace("Hiding annotation %s for type: %s", item_id, self._canvas.type(item_id))
            self._canvas.itemconfig(item_id, state="hidden")

    def _object_tracker(self, key, object_type, face_index,
                        object_index, coordinates, object_kwargs):
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
        object_color_keys = self._get_object_color_keys(key, object_type)
        tracking_id = "_".join((key, str(face_index), str(object_index)))

        if (key not in self._objects
                or len(self.objects[key]) - 1 < face_index
                or len(self._objects[key][face_index]) - 1 < object_index):
            self._add_new_object(key, object_type, face_index, coordinates, object_kwargs)
            update_color = bool(object_color_keys)
        else:
            update_color = self._update_existing_object(
                self._objects[key][face_index][object_index],
                coordinates,
                object_kwargs,
                tracking_id,
                object_color_keys)
        if update_color:
            self._current_color[tracking_id] = object_kwargs[object_color_keys[0]]

    @staticmethod
    def _get_object_color_keys(key, object_type):
        """ The canvas object's parameter that needs to be adjusted for color varies based on
        the type of object that is being used. Returns the correct parameter based on object.

        Parameters
        ----------
        key: str
            The key for this annotation in :attr:`_objects`
        object_type: str
            This can be any string that is a natural extension to :class:`tkinter.Canvas.create_`

        Returns
        -------
        list:
            The list of keyword arguments for this objects color parameter(s) or an empty list
            if it is not relevant for this object
        """
        if object_type in ("line", "text"):
            retval = ["fill"]
        elif object_type == "image":
            retval = []
        elif object_type == "oval" and key == "lm_display":
            retval = ["fill", "outline"]
        else:
            retval = ["outline"]
        logger.trace("returning %s for key: %s, object_type: %s", retval, key, object_type)
        return retval

    def _add_new_object(self, key, object_type, face_index, coordinates, object_kwargs):
        """ Add a new object to :attr:'_objects' for tracking.

        Parameters
        ----------
        key: str
            The key for this annotation in :attr:`_objects`
        object_type: str
            This can be any string that is a natural extension to :class:`tkinter.Canvas.create_`
        coordinates: tuple or list
            The bounding box coordinates for this object
        object_kwargs: dict
            The keyword arguments for this object
        """
        logger.trace("Adding object: (key: '%s', object_type: '%s', face_index: %s, "
                     "coordinates: %s, object_kwargs: %s)", key, object_type, face_index,
                     coordinates, object_kwargs)
        obj = getattr(self._canvas, "create_{}".format(object_type))(*coordinates, **object_kwargs)
        if key not in self._objects:
            self._objects[key] = []
        if len(self._objects[key]) - 1 < face_index:
            self._objects[key].append([obj])
        else:
            self._objects[key][face_index].append(obj)

    def _update_existing_object(self, tk_object, coordinates, object_kwargs,
                                tracking_id, object_color_keys):
        """ Update an existing tracked object.

        Parameters
        ----------
        tk_object: tkinter canvas object
            The canvas object to be updated
        coordinates: tuple or list
            The bounding box coordinates for this object
        object_kwargs: dict
            The keyword arguments for this object
        tracking_id: str
            The tracking identifier for this object
        object_color_keys: list
            The list of keyword arguments for this object to update for color

        Returns
        -------
        bool
            ``True`` if :att:`_current_color` should be updated otherwise ``False``
        """
        update_color = (object_color_keys and
                        object_kwargs[object_color_keys[0]] != self._current_color[tracking_id])
        update_kwargs = dict(state=object_kwargs.get("state", "normal"))
        if update_color:
            for key in object_color_keys:
                update_kwargs[key] = object_kwargs[object_color_keys[0]]
        if self._canvas.type(tk_object) == "image" and "image" in object_kwargs:
            update_kwargs["image"] = object_kwargs["image"]
        logger.trace("Updating coordinates: (tracking_id: %s, tk_object: '%s', object_kwargs: %s, "
                     "coordinates: %s, update_kwargs: %s", tracking_id, tk_object, object_kwargs,
                     coordinates, update_kwargs)
        self._canvas.itemconfig(tk_object, **update_kwargs)
        self._canvas.coords(tk_object, *coordinates)
        return update_color

    def _flatten_list(self, input_list):
        """ Recursively Flatten a list of lists to a single list """
        if input_list == []:
            return input_list
        if isinstance(input_list[0], list):
            return self._flatten_list(input_list[0]) + self._flatten_list(input_list[1:])
        return input_list[:1] + self._flatten_list(input_list[1:])

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

    # << ACTION CONTROL PANEL OPTIONS >>
    def _add_actions(self):
        """ Add the Action buttons for this editor's optional left hand side action sections.

        The default does nothing. Override for editor specific actions.
        """
        self._actions = self._actions

    def _add_action(self, title, icon, helptext, hotkey=None):
        """ Add an action dictionary to :attr:`_actions`. This will create a button in the optional
        actions frame to the left hand side of the frames viewer.
        """
        var = tk.BooleanVar()
        self._actions[title] = dict(icon=icon, helptext=helptext, tk_var=var, hotkey=hotkey)

    def _add_controls(self):
        """ Add the controls for this editor's control panel.

        The default does nothing. Override for editor specific controls.
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

        editor_key = self.__class__.__name__
        group_key = option.group.replace(" ", "").lower()
        group_key = "none" if group_key == "_master" else group_key
        annotation_key = option.title.replace(" ", "")
        self._canvas.control_tk_vars.setdefault(
            editor_key, dict()).setdefault(group_key, dict())[annotation_key] = option.tk_var

    def _add_annotation_format_controls(self):
        """ Add the annotation display (color/size) controls.

        These should be universal and available for all editors.
        """
        editors = ("Bounding Box", "Extract Box", "Landmarks", "Mask", "Mesh")
        if not self._annotation_formats:
            opacity = ControlPanelOption("Mask Opacity",
                                         int,
                                         group="Color",
                                         min_max=(0, 100),
                                         default=20,
                                         rounding=1,
                                         helptext="Set the mask opacity")
            for editor in editors:
                annotation_key = editor.replace(" ", "")
                logger.debug("Adding to global format controls: '%s'", editor)
                colors = ControlPanelOption(editor,
                                            str,
                                            group="Color",
                                            choices=sorted(self._canvas.colors),
                                            default=self._default_colors[annotation_key],
                                            is_radio=False,
                                            helptext="Set the annotation color")
                colors.set(self._default_colors[annotation_key])
                self._annotation_formats.setdefault(annotation_key, dict())["color"] = colors
                self._annotation_formats[annotation_key]["mask_opacity"] = opacity

        for editor in editors:
            annotation_key = editor.replace(" ", "")
            for group, ctl in self._annotation_formats[annotation_key].items():
                logger.debug("Adding global format control to editor: (editor:'%s', group: '%s')",
                             editor, group)
                self._add_control(ctl, global_control=True)


class View(Editor):
    """ The view Editor """
    def __init__(self, canvas, alignments, frames):
        control_text = "Viewer\nPreview the frame's annotations."
        super().__init__(canvas, alignments, frames, control_text)
