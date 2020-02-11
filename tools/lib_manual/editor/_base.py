#!/usr/bin/env python3
""" Editor objects for the manual adjustments tool """

import logging
import platform
import tkinter as tk

from collections import OrderedDict

import numpy as np

from lib.gui.control_helper import ControlPanelOption

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Tracking for all control panel variables, for access from all editors
_CONTROL_VARS = dict()
_ANNOTATION_FORMAT = dict()

# TODO Hide annotations for additional faces
# TODO Landmarks, Color outline and fill
# TODO dynamically bind and unbind keybindings
# TODO Global variables to canvas


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
        self._actions = OrderedDict()
        self._controls = dict(header=control_text, controls=[])

        self._add_actions()
        self._add_controls()
        self._add_annotation_format_controls()

        self._zoomed_roi = self._get_zoomed_roi()
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
    def _is_zoomed(self):
        """ bool: ``True`` if a face is currently zoomed in, ``False`` if the full frame is
        displayed """
        return self._canvas.image_is_hidden

    @property
    def _active_editor(self):
        """ str: The name of the currently active editor """
        return self._canvas.selected_action

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

    @property
    def _zoomed_dims(self):
        """ tuple: The (`width`, `height`) of the zoomed ROI """
        return (self._zoomed_roi[2] - self._zoomed_roi[0],
                self._zoomed_roi[3] - self._zoomed_roi[1])

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

    def _object_tracker(self, key, object_type, object_index, coordinates, object_kwargs):
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
        tracking_id = "_".join((key, str(object_index)))

        if key not in self._objects or len(self._objects[key]) - 1 < object_index:
            self._add_new_object(key, object_type, coordinates, object_kwargs)
            update_color = bool(object_color_keys)
        else:
            update_color = self._update_existing_object(self._objects[key][object_index],
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

    def _add_new_object(self, key, object_type, coordinates, object_kwargs):
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
        logger.trace("Adding object: (key: '%s', object_type: '%s', coordinates: %s, "
                     "object_kwargs: %s)", key, object_type, coordinates, object_kwargs)
        obj = getattr(self._canvas, "create_{}".format(object_type))
        self._objects.setdefault(key, []).append(obj(*coordinates, **object_kwargs))

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
        pos_x = event.x_root + (self.winfo_reqwidth() // 2) + 2
        pos_y = event.y_root + (self.winfo_reqheight() // 2) + 2
        self.tk_popup(pos_x, pos_y, 0)
