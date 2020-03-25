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

    Editors allow the user to use a variety of tools to manipulate alignments from the main
    display frame.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The canvas that holds the image and annotations
    detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
        The _detected_faces data for this manual session
    frames: :class:`FrameNavigation`
        The frames navigator for this manual session
    control_text: str
        The text that is to be displayed at the top of the Editor's control panel.
    """
    def __init__(self, canvas, detected_faces, frames, control_text="", key_bindings=None):
        logger.debug("Initializing %s: (canvas: '%s', detected_faces: %s, frames: %s, "
                     "control_text: %s)", self.__class__.__name__, canvas, detected_faces, frames,
                     control_text)
        self._canvas = canvas
        self._det_faces = detected_faces
        self._frames = frames

        self._current_color = dict()
        self._actions = OrderedDict()
        self._controls = dict(header=control_text, controls=[])
        self._add_key_bindings(key_bindings)

        self._add_actions()
        self._add_controls()
        self._add_annotation_format_controls()

        self._zoomed_roi = self._get_zoomed_roi()
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
    def _zoomed_dims(self):
        """ tuple: The (`width`, `height`) of the zoomed ROI """
        return (self._zoomed_roi[2] - self._zoomed_roi[0],
                self._zoomed_roi[3] - self._zoomed_roi[1])

    @property
    def _control_vars(self):
        """ dict: The tk control panel variables for the currently selected editor. """
        return self._canvas.control_tk_vars.get(self.__class__.__name__, dict())

    @property
    def controls(self):
        """ dict: The control panel options and header text for the current editor """
        return self._controls

    @property
    def _control_color(self):
        """ str: The hex color code set in the control panel for the current editor. """
        annotation = self.__class__.__name__
        return self._canvas.colors[self._annotation_formats[annotation]["color"].get()]

    @property
    def _annotation_formats(self):
        """ dict: The format (color, opacity etc.) of each editor's annotation display. """
        return self._canvas.annotation_formats

    @property
    def actions(self):
        """ list: The optional action buttons for the actions frame in the GUI for the
        current editor """
        return self._actions

    @property
    def _frame_index(self):
        """ int: The currently displayed frame index. """
        return self._frames.tk_position.get()

    def _add_key_bindings(self, key_bindings):
        """ Add the editor specific key bindings for the currently viewed editor.

        Parameters
        ----------
        key_bindings: dict
            The key binding to method dictionary for this editor.
        """
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
        self._canvas.itemconfig(self.__class__.__name__, state="hidden")

    def _object_tracker(self, key, object_type, face_index,
                        coordinates, object_kwargs):
        """ Create an annotation object and add it to :attr:`_objects` or update an existing
        annotation if it has already been created.

        Parameters
        ----------
        key: str
            The key for this annotation in :attr:`_objects`
        object_type: str
            This can be any string that is a natural extension to :class:`tkinter.Canvas.create_`
        face_index: int
            The index of the face within the current frame
        coordinates: tuple or list
            The bounding box coordinates for this object
        object_kwargs: dict
            The keyword arguments for this object

        Returns
        -------
        int:
            The tkinter canvas item identifier for the created object
        """
        object_color_keys = self._get_object_color_keys(key, object_type)
        tracking_id = "_".join((key, str(face_index)))
        face_tag = "face_{}".format(face_index)
        face_objects = set(self._canvas.find_withtag(face_tag))
        annotation_objects = set(self._canvas.find_withtag(key))
        existing_object = tuple(face_objects.intersection(annotation_objects))
        if not existing_object:
            item_id = self._add_new_object(key,
                                           object_type,
                                           face_index,
                                           coordinates,
                                           object_kwargs)
            update_color = bool(object_color_keys)
        else:
            item_id = existing_object[0]
            update_color = self._update_existing_object(
                existing_object[0],
                coordinates,
                object_kwargs,
                tracking_id,
                object_color_keys)
        if update_color:
            self._current_color[tracking_id] = object_kwargs[object_color_keys[0]]
        return item_id

    @staticmethod
    def _get_object_color_keys(key, object_type):
        """ The canvas object's parameter that needs to be adjusted for color varies based on
        the type of object that is being used. Returns the correct parameter based on object.

        Parameters
        ----------
        key: str
            The key for this annotation's tag creation
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
        elif object_type == "oval" and key.startswith("lm_dsp_"):
            retval = ["fill", "outline"]
        else:
            retval = ["outline"]
        logger.trace("returning %s for key: %s, object_type: %s", retval, key, object_type)
        return retval

    def _add_new_object(self, key, object_type, face_index, coordinates, object_kwargs):
        """ Add a new object to the canvas.

        Parameters
        ----------
        key: str
            The key for this annotation's tag creation
        object_type: str
            This can be any string that is a natural extension to :class:`tkinter.Canvas.create_`
        face_index: int
            The index of the face within the current frame
        coordinates: tuple or list
            The bounding box coordinates for this object
        object_kwargs: dict
            The keyword arguments for this object

        Returns
        -------
        int:
            The tkinter canvas item identifier for the created object
        """
        logger.debug("Adding object: (key: '%s', object_type: '%s', face_index: %s, "
                     "coordinates: %s, object_kwargs: %s)", key, object_type, face_index,
                     coordinates, object_kwargs)
        object_kwargs["tags"] = self._set_object_tags(face_index, key)
        item_id = getattr(self._canvas,
                          "create_{}".format(object_type))(*coordinates, **object_kwargs)
        return item_id

    def _set_object_tags(self, face_index, key):
        """ Create the tkinter object tags for the incoming object.

        Parameters
        ----------
        face_index: int
            The face index within the current frame for the face that tags are being created for
        key: str
            The base tag for this object, for which additional tags will be generated

        Returns
        -------
        list
            The generated tags for the current object
        """
        tags = ["face_{}".format(face_index),
                self.__class__.__name__,
                "{}_face_{}".format(self.__class__.__name__, face_index),
                key,
                "{}_face_{}".format(key, face_index)]
        if "_" in key:
            split_key = key.split("_")
            if split_key[-1].isdigit():
                base_tag = "_".join(split_key[:-1])
                tags.append(base_tag)
                tags.append("{}_face_{}".format(base_tag, face_index))
        return tags

    def _update_existing_object(self, item_id, coordinates, object_kwargs,
                                tracking_id, object_color_keys):
        """ Update an existing tracked object.

        Parameters
        ----------
        item_id: int
            The canvas object item_id to be updated
        coordinates: tuple or list
            The bounding box coordinates for this object
        object_kwargs: dict
            The keyword arguments for this object
        tracking_id: str
            The tracking identifier for this object's color
        object_color_keys: list
            The list of keyword arguments for this object to update for color

        Returns
        -------
        bool
            ``True`` if :attr:`_current_color` should be updated otherwise ``False``
        """
        update_color = (object_color_keys and
                        object_kwargs[object_color_keys[0]] != self._current_color[tracking_id])
        update_kwargs = dict(state=object_kwargs.get("state", "normal"))
        if update_color:
            for key in object_color_keys:
                update_kwargs[key] = object_kwargs[object_color_keys[0]]
        if self._canvas.type(item_id) == "image" and "image" in object_kwargs:
            update_kwargs["image"] = object_kwargs["image"]
        logger.trace("Updating coordinates: (item_id: '%s', object_kwargs: %s, "
                     "coordinates: %s, update_kwargs: %s", item_id, object_kwargs,
                     coordinates, update_kwargs)
        self._canvas.itemconfig(item_id, **update_kwargs)
        self._canvas.coords(item_id, *coordinates)
        return update_color

    # << MOUSE CALLBACKS >>
    # Mouse cursor display
    def bind_mouse_motion(self):
        """ Binds the mouse motion for the current editor's mouse <Motion> event to the editor's
        :func:`_update_cursor` function.

        Called on initialization and active editor update.
        """
        self._canvas.bind("<Motion>", self._update_cursor)

    def _update_cursor(self, event):  # pylint: disable=unused-argument
        """ The mouse cursor display as bound to the mouse's <Motion> event..

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

        Parameters
        ----------
        title: str
            The title of the action to be generated
        icon: str
            The name of the icon that is used to display this action's button
        helptext: str
            The tooltip text to display for this action
        hotkey: str, optional
            The hotkey binding for this action. Set to ``None`` if there is no hotkey binding.
            Default: ``None``
        """
        var = tk.BooleanVar()
        action = dict(icon=icon, helptext=helptext, tk_var=var, hotkey=hotkey)
        logger.debug("Adding action: %s", action)
        self._actions[title] = action

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
            logger.debug("Added global control: '%s' for editor: '%s'",
                         option.title, self.__class__.__name__)
            return
        logger.debug("Added local control: '%s' for editor: '%s'",
                     option.title, self.__class__.__name__)
        editor_key = self.__class__.__name__
        group_key = option.group.replace(" ", "").lower()
        group_key = "none" if group_key == "_master" else group_key
        annotation_key = option.title.replace(" ", "")
        self._canvas.control_tk_vars.setdefault(
            editor_key, dict()).setdefault(group_key, dict())[annotation_key] = option.tk_var

    def _add_annotation_format_controls(self):
        """ Add the annotation display (color/size) controls to :attr:`_annotation_formats`.

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
    """ The view Editor.

    Does not allow any editing, just used for previewing annotations.

    This is the default start-up editor.
    """
    def __init__(self, canvas, detected_faces, frames):
        control_text = "Viewer\nPreview the frame's annotations."
        super().__init__(canvas, detected_faces, frames, control_text)
