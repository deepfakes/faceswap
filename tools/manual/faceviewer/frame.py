#!/usr/bin/env python3
""" The Faces Viewer Frame and Canvas for Faceswap's Manual Tool. """
import logging
import platform
import tkinter as tk
from tkinter import ttk
from math import floor, ceil

import numpy as np

from lib.gui.custom_widgets import RightClickMenu, Tooltip
from lib.gui.utils import get_config, get_images

from .viewport import Viewport

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FacesFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The faces display frame (bottom section of GUI).

    Parameters
    ----------
    parent: :class:`tkinter.PanedWindow`
        The paned window that the faces frame resides in
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    detected_faces: :class:`~tool.manual.faces.DetectedFaces`
        The :class:`~lib.faces_detect.DetectedFace` objects for this video
    display_frame: :class:`DisplayFrame`
        The section of the Manual Tool that holds the frames viewer
    """
    def __init__(self, parent, tk_globals, detected_faces, display_frame):
        logger.debug("Initializing %s: (parent: %s, tk_globals: %s, detected_faces: %s, "
                     "display_frame: %s)", self.__class__.__name__, parent, tk_globals,
                     detected_faces, display_frame)
        super().__init__(parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._actions_frame = FacesActionsFrame(self)

        self._faces_frame = ttk.Frame(self)
        self._faces_frame.pack_propagate(0)
        self._faces_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._canvas = FacesViewer(self._faces_frame,
                                   tk_globals,
                                   self._actions_frame._tk_vars,
                                   detected_faces,
                                   display_frame)
        self._add_scrollbar()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _add_scrollbar(self):
        """ Add a scrollbar to the faces frame """
        logger.debug("Add Faces Viewer Scrollbar")
        scrollbar = ttk.Scrollbar(self._faces_frame, command=self._on_scroll)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.config(yscrollcommand=scrollbar.set)
        self.bind("<Configure>", self._update_viewport)
        logger.debug("Added Faces Viewer Scrollbar")
        self.update_idletasks()  # Update so scrollbar width is correct
        return scrollbar.winfo_width()

    def _on_scroll(self, *event):
        """ Callback on scrollbar scroll.
        Updates the canvas location and displays/hides thumbnail images

        Parameters
        ----------
        event :class:`tkinter.Event`
            The scrollbar callback event
        """
        self._canvas.yview(*event)
        self._canvas.viewport.update()

    def _update_viewport(self, event):  # pylint: disable=unused-argument
        """ Update the faces viewport and scrollbar.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            Unused but required
        """
        self._canvas.viewport.update()
        self._canvas.configure(scrollregion=self._canvas.bbox("backdrop"))

    def canvas_scroll(self, direction):
        """ Scroll the canvas on an up/down or page-up/page-down key press.

        Parameters
        ----------
        direction: ["up", "down", "page-up", "page-down"]
            The request page scroll direction and amount.
        """
        amount = 1 if direction.endswith("down") else -1
        units = "pages" if direction.startswith("page") else "units"
        self._canvas.canvas_scroll(amount, units)
        self._canvas.viewport.update()

    def set_annotation_display(self, key):
        """ Set the optional annotation overlay based on keyboard shortcut.

        Parameters
        ----------
        key: str
            The pressed key
        """
        self._actions_frame.on_click(self._actions_frame.key_bindings[key])


class FacesActionsFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The left hand action frame holding the action buttons.

    Parameters
    ----------
    parent: :class:`FacesFrame`
        The Faces frame that the Actions reside in
    """
    def __init__(self, parent):
        logger.debug("Initializing %s: (parent: %s)",
                     self.__class__.__name__, parent)
        super().__init__(parent)
        self.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 4), pady=2)
        self._tk_vars = dict()
        self._configure_styles()
        self._buttons = self._add_buttons()
        lockout = tk.BooleanVar()
        lockout.set(False)
        lockout.trace("w", lambda *e: self._enable_disable_buttons())
        self._tk_vars["lockout"] = lockout
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def key_bindings(self):
        """ dict: {`key`: `display`}. The mapping of key presses to optional annotations to display.
        Keyboard shortcuts utilize the function keys. """
        return {"F{}".format(idx + 9): display for idx, display in enumerate(("mesh", "mask"))}

    @property
    def _helptext(self):
        """ dict: `button key`: `button helptext`. The help text to display for each button. """
        inverse_keybindings = {val: key for key, val in self.key_bindings.items()}
        retval = dict(mesh="Display the landmarks mesh",
                      mask="Display the mask")
        for item in retval:
            retval[item] += " ({})".format(inverse_keybindings[item])
        return retval

    def _configure_styles(self):
        """ Configure the background color for button frame and the button styles. """
        style = ttk.Style()
        style.configure("display.TFrame", background='#d3d3d3')
        style.configure("display_selected.TButton", relief="flat", background="#bedaf1")
        style.configure("display_deselected.TButton", relief="flat")
        self.config(style="display.TFrame")

    def _add_buttons(self):
        """ Add the display buttons to the Faces window.
            The buttons are not activated until the faces have completed loading

        Returns
        -------
        dict:
            The display name and its associated button.
        """
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.Y)
        buttons = dict()
        for display in self.key_bindings.values():
            var = tk.BooleanVar()
            var.set(False)
            self._tk_vars[display] = var

            lookup = "landmarks" if display == "mesh" else display
            button = ttk.Button(frame,
                                image=get_images().icons[lookup],
                                command=lambda t=display: self.on_click(t),
                                style="display_deselected.TButton")
            button.state(["!pressed", "!focus"])
            button.pack()
            Tooltip(button, text=self._helptext[display])
            buttons[display] = button
        return buttons

    def on_click(self, display):
        """ Click event for the optional annotation buttons.

        Parameters
        ----------
        display: str
            The display name for the button that has called this event as exists in
            attr:`_buttons`
        """
        is_pressed = not self._tk_vars[display].get()
        style = "display_selected.TButton" if is_pressed else "display_deselected.TButton"
        state = ["pressed", "focus"] if is_pressed else ["!pressed", "!focus"]
        btn = self._buttons[display]
        btn.configure(style=style)
        btn.state(state)
        self._tk_vars[display].set(is_pressed)

    def _enable_disable_buttons(self):
        """ Enable or disable the optional annotation buttons when the face cache is idle or
        loading.
        """
        lockout_state = self._tk_vars["lockout"].get()
        state = "disabled" if lockout_state else "!disabled"
        logger.debug("lockout_state: %s, button state: %s", lockout_state, state)
        for button in self._buttons.values():
            button.state([state])


class FacesViewer(tk.Canvas):   # pylint:disable=too-many-ancestors
    """ The :class:`tkinter.Canvas` that holds the faces viewer part of the Manual Tool.

    Parameters
    ----------
    parent: :class:`tkinter.ttk.Frame`
        The parent frame for the canvas
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    tk_action_vars: dict
        The :class:`tkinter.BooleanVar` objects for selectable optional annotations
        as set by the buttons in the :class:`FacesActionsFrame`
    detected_faces: :class:`~tool.manual.faces.DetectedFaces`
        The :class:`~lib.faces_detect.DetectedFace` objects for this video
    display_frame: :class:`DisplayFrame`
        The section of the Manual Tool that holds the frames viewer
    """
    def __init__(self, parent, tk_globals, tk_action_vars, detected_faces, display_frame):
        logger.debug("Initializing %s: (parent: %s, tk_globals: %s, tk_action_vars: %s, "
                     "detected_faces: %s, display_frame: %s)", self.__class__.__name__, parent,
                     tk_globals, tk_action_vars, detected_faces, display_frame)
        super().__init__(parent, bd=0, highlightthickness=0, bg="#bcbcbc")
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor=tk.E)
        self._sizes = dict(tiny=32, small=64, medium=96, large=128, extralarge=192)

        self._globals = tk_globals
        self._tk_optional_annotations = {key: val for key, val in tk_action_vars.items()
                                         if key != "lockout"}

        self._display_frame = display_frame
        self._grid = Grid(self, detected_faces)
        self._view = Viewport(self, detected_faces.tk_edited)

        ContextMenu(self, detected_faces)
        self._bind_mouse_wheel_scrolling()
        # Set in load_frames
        self._annotation_colors = dict(mesh=self.get_muted_color("Mesh"))
        self._set_tk_callbacks(detected_faces)

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def face_size(self):
        """ int: The currently selected thumbnail size in pixels """
        scaling = get_config().scaling_factor
        size = self._sizes[self._globals.tk_faces_size.get().lower().replace(" ", "")]
        return int(round(size * scaling))

    @property
    def viewport(self):
        """ :class:`~lib.tools.manual.faceviewer.viewport.Viewport`: The viewport area of the
        faces viewer. """
        return self._view

    @property
    def grid(self):
        """ :class:`Grid`: The grid for the current :class:`FacesViewer`. """
        return self._grid

    @property
    def optional_annotations(self):
        """ dict: The values currently set for the selectable optional annotations. """
        return {opt: val.get() for opt, val in self._tk_optional_annotations.items()}

    @property
    def selected_mask(self):
        """ str: The currently selected mask from the display frame control panel. """
        return self._display_frame.tk_selected_mask.get().lower()

    @property
    def control_colors(self):
        """ :dict: Editor key with the current user selected hex code as value. """
        return {key: self._display_frame.colors[val.get()]
                for key, val in self._display_frame.tk_control_colors.items()}

    # << CALLBACK FUNCTIONS >> #
    def _set_tk_callbacks(self, detected_faces):
        """ Set the tkinter variable call backs.

        Switches the Filter view when the filter drop down is updated.
        Updates the Mesh annotation color when user amends the color drop down.
        Updates the mask type when the user changes the selected mask types
        Toggles the face viewer annotations on an optional annotation button press.
        """
        for var in (self._globals.tk_faces_size, self._globals.tk_filter_mode):
            var.trace("w", lambda *e, v=var: self.refresh_grid(v))
        var = detected_faces.tk_face_count_changed
        var.trace("w", lambda *e, v=var: self.refresh_grid(v, retain_position=True))

        self._display_frame.tk_control_colors["Mesh"].trace("w",
                                                            lambda *e: self.update_mesh_color())
        self._display_frame.tk_selected_mask.trace("w", lambda *e: self._update_mask_type())
        for opt, var in self._tk_optional_annotations.items():
            var.trace("w", lambda *e, o=opt: self._toggle_annotations(o))
        self.bind("<Configure>", lambda *e: self._view.update())

    def refresh_grid(self, trigger_var, retain_position=False):
        """ Recalculate the full grid and redraw. Used when the active filter pull down is used or
        a face has been added or removed. """
        if not trigger_var.get():
            return
        size_change = isinstance(trigger_var, tk.StringVar)
        move_to = self.yview()[0] if retain_position else 0.0
        self._grid.update()
        if move_to != 0.0:
            self.yview_moveto(move_to)
        if size_change:
            self._view.reset()
        self._view.update()
        if not size_change:
            trigger_var.set(False)

    def _update_mask_type(self):
        """ Update the displayed mask in the :class:`FacesViewer` canvas when the user changes
        the mask type. """
        logger.warning("UPDATE MASK CODE. self.selected_mask: %s, "
                       "self.optional_annotations['mask']: %s)",
                       self.selected_mask, self.optional_annotations["mask"])

    # << MOUSE HANDLING >>
    def _bind_mouse_wheel_scrolling(self):
        """ Bind mouse wheel to scroll the :class:`FacesViewer` canvas. """
        if platform.system() == "Linux":
            self.bind("<Button-4>", self._scroll)
            self.bind("<Button-5>", self._scroll)
        else:
            self.bind("<MouseWheel>", self._scroll)

    def _scroll(self, event):
        """ Handle mouse wheel scrolling over the :class:`FacesViewer` canvas.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The event fired by the mouse scrolling
        """
        if platform.system() == "Darwin":
            adjust = event.delta
        elif platform.system() == "Windows":
            adjust = event.delta / 120
        elif event.num == 5:
            adjust = -1
        else:
            adjust = 1
        self.canvas_scroll(-1 * adjust, "units", event)

    def canvas_scroll(self, amount, units, event=None):
        """ Scroll the canvas on an up/down or page-up/page-down key press.

        Parameters
        ----------
        amount: int
            The number of units to scroll the canvas
        units: ["page", "units"]
            The unit type to scroll by
        event: :class:`tkinter.Event` or ``None``, optional
            The tkinter event (if scrolling by mouse wheel) or ``None`` if the scroll action
            has been triggered by a keyboard shortcut
        """
        self.yview_scroll(int(amount), units)
        self._view.update()
        self._view.hover_box.on_hover(event)

    # << OPTIONAL ANNOTATION METHODS >> #
    def update_mesh_color(self):
        """ Update the mesh color when user updates the control panel. """
        color = self.get_muted_color("Mesh")
        if self._annotation_colors["mesh"] == color:
            return
        highlight_color = self.control_colors["Mesh"]
        self.itemconfig("mesh_polygon", outline=color)
        self.itemconfig("mesh_line", fill=color)
        self.itemconfig("highlight_mesh_polygon", outline=highlight_color)
        self.itemconfig("highlight_mesh_line", fill=highlight_color)
        self._annotation_colors["mesh"] = color

    def get_muted_color(self, color_key):
        """ Creates a muted version of the given annotation color for non-active faces.

        It is assumed that hex codes for annotations will always contain "ff" in one of the
        R, G or B channels, so the given hex code has all of the F values updated to A.

        Parameters
        ----------
        color_key: str
            The annotation key to obtain the color for from :attr:`control_colors`
        """
        return self.control_colors[color_key].replace("f", "a")

    def _toggle_annotations(self, annotation):
        """ Toggle optional annotations on or off after the user depresses an optional button.

        Parameters
        ----------
        annotation: ["mesh", "mask"]
            The optional annotation to toggle on or off
        """
        state = "normal" if self.optional_annotations[annotation] else "hidden"
        logger.debug("Toggle annotation: (annotation: %s, state: %s)", annotation, state)
        if annotation == "mesh":
            self._view.toggle_mesh(state)
        if annotation == "mask":
            self._view.toggle_mask(state, self.selected_mask)


class Grid():
    """ Holds information on current filter grid layout """
    def __init__(self, canvas, detected_faces):
        logger.debug("Initializing %s: (detected_faces: %s)",
                     self.__class__.__name__, detected_faces)
        self._canvas = canvas
        self._detected_faces = detected_faces
        self._raw_indices = detected_faces.filter.raw_indices
        self._frames_list = detected_faces.filter.frames_list

        self._landmark_mapping = dict(mouth=(48, 68),
                                      right_eyebrow=(17, 22),
                                      left_eyebrow=(22, 27),
                                      right_eye=(36, 42),
                                      left_eye=(42, 48),
                                      nose=(27, 36),
                                      jaw=(0, 17),
                                      chin=(8, 11))
        self._is_valid = False
        self._face_size = None
        self._grid = None
        self._display_faces = None

        self._canvas.update_idletasks()
        self._canvas.create_rectangle(0, 0, 0, 0, tags=["backdrop"])
        self.update()

    @property
    def is_valid(self):
        """ bool: ``True`` if the current filter means that the grid holds faces. ``False`` if
        there are no faces displayed in the grid. """
        return self._is_valid

    @property
    def columns_rows(self):
        """ tuple: the (`columns`, `rows`) required to hold all display images. """
        return tuple(reversed(self._grid.shape[1:]))

    @property
    def dimensions(self):
        """ tuple: The (`width`, `height`) required to hold all display images. """
        if self._is_valid:
            retval = tuple(dim * self._face_size for dim in reversed(self._grid.shape[1:]))
        else:
            retval = (0, 0)
        return retval

    @property
    def _visible_row_indices(self):
        """tuple: A 1 dimensional array of the (`top_row_index`, `bottom_row_index`) of the grid
        currently in the viewable area.
        """
        height = self.dimensions[1]
        visible = (max(0, floor(height * self._canvas.yview()[0]) - self._face_size),
                   ceil(height * self._canvas.yview()[1]))
        logger.trace("visible: %s", visible)
        y_points = self._grid[3, :, 1]
        top = np.searchsorted(y_points, visible[0], side="left")
        bottom = np.searchsorted(y_points, visible[1], side="right")
        return top, bottom

    @property
    def visible_area(self):
        """:class:`numpy.ndarray`:  A numpy array of shape (`4`, `rows`, `columns`) corresponding
        to the viewable area of the display grid. 1st dimension contains frame indices, 2nd
        dimension face indices. The 3rd and 4th dimension contain the x and y position of the top
        left corner of the face respectively.

        Any locations that are not populated by a face will have a frame and face index of -1
        """
        if not self._is_valid:
            retval = None, None
        else:
            top, bottom = self._visible_row_indices
            retval = self._grid[:, top:bottom, :], self._display_faces[:, top:bottom, :]
        logger.trace([r if r is None else r.shape for r in retval])
        return retval

    def update(self):
        """ Update the underlying grid.

        Called on initialization, on a filter change or on add/remove faces. Recalculates the
        underlying grid for the current filter view and updates the attributes :attr:`_grid`,
        :attr:`_display_faces`, :attr:`_raw_indices`, :attr:`_frames_list` and :attr:`is_valid`
        """
        self._face_size = self._canvas.face_size
        self._raw_indices = self._detected_faces.filter.raw_indices
        self._frames_list = self._detected_faces.filter.frames_list
        self._get_grid()
        self._get_display_faces()
        self._canvas.coords("backdrop", 0, 0, *self.dimensions)
        self._canvas.configure(scrollregion=(self._canvas.bbox("backdrop")))
        self._canvas.yview_moveto(0.0)

    def _get_grid(self):
        """ Get the grid information for faces currently displayed in the :class:`FacesViewer`.

        Returns
        :class:`numpy.ndarray`
            A numpy array of shape (`4`, `rows`, `columns`) corresponding to the display grid.
            1st dimension contains frame indices, 2nd dimension face indices. The 3rd and 4th
            dimension contain the x and y position of the top left corner of the face respectively.

            Any locations that are not populated by a face will have a frame and face index of -1
        """
        labels = self._get_labels()
        if not self._is_valid:
            logger.debug("Setting grid to None for no faces.")
            self._grid = None
            return
        x_coords = np.linspace(0,
                               labels.shape[2] * self._face_size,
                               num=labels.shape[2],
                               endpoint=False,
                               dtype="int")
        y_coords = np.linspace(0,
                               labels.shape[1] * self._face_size,
                               num=labels.shape[1],
                               endpoint=False,
                               dtype="int")
        self._grid = np.array((*labels, *np.meshgrid(x_coords, y_coords)), dtype="int")
        logger.debug(self._grid.shape)

    def _get_labels(self):
        """ Get the frame and face index for each grid position for the current filter.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of dimensions (2, rows, columns) corresponding to the display grid, with frame
            index as the first dimension and face index within the frame as the 2nd dimension.

            Any remaining placeholders at the end of the grid which are not populated with a face
            are given the index -1
        """
        face_count = len(self._raw_indices["frame"])
        self._is_valid = face_count != 0
        if not self._is_valid:
            return None
        columns = self._canvas.winfo_width() // self._face_size
        rows = ceil(face_count / columns)
        padding = [-1 for _ in range(columns - (face_count % columns))]

        labels = np.array((self._raw_indices["frame"] + padding,
                           self._raw_indices["face"] + padding),
                          dtype="int").reshape((2, rows, columns))
        logger.debug(labels.shape)
        return labels

    def _get_display_faces(self):
        """ Get the detected faces for the current filter and arrange to grid.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of dimensions (rows, columns) corresponding to the display grid, containing the
            corresponding :class:`lib.faces_detect.DetectFace` object

            Any remaining placeholders at the end of the grid which are not populated with a face
            are replaced with ``None``
        """
        if not self._is_valid:
            logger.debug("Setting display_faces to None for no faces.")
            self._display_faces = None
            return
        current_faces = self._detected_faces.current_faces
        columns, rows = self.columns_rows
        face_count = len(self._raw_indices["frame"])
        padding = [None for _ in range(face_count, columns * rows)]
        self._display_faces = np.array(
            [[None if idx is None else current_faces[idx][face_idx],
              None if idx is None else self._detected_faces.get_thumbnail(idx, face_idx)]
             for idx, face_idx
             in zip(self._raw_indices["frame"] + padding, self._raw_indices["face"] + padding)],
            dtype="object").reshape(rows, columns, 2).transpose(2, 0, 1)
        logger.debug("faces: (shape: %s, dtype: %s)",
                     self._display_faces.shape, self._display_faces.dtype)

    def transport_index_from_frame(self, frame_index):
        """ Return the main frame's transport index for given frame index based on the current
        filter criteria.

        Parameters
        ----------
        frame_index: int
            The absolute index for the frame within the full frames list

        Returns
        int
            The index of the requested frame within the filtered frames view.
        """
        retval = self._frames_list.index(frame_index) if frame_index in self._frames_list else None
        logger.trace("frame_index: %s, transport_index: %s", frame_index, retval)
        return retval


class ContextMenu():  # pylint:disable=too-few-public-methods
    """  Enables a right click context menu for the :class:`~tool.manual.FacesViewer`.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    """
    def __init__(self, canvas, detected_faces):
        logger.debug("Initializing: %s (canvas: %s, detected_faces: %s)",
                     self.__class__.__name__, canvas, detected_faces)
        self._canvas = canvas
        self._detected_faces = detected_faces
        self._menu = RightClickMenu(["Delete Face"], [self._delete_face])
        self._frame_index = None
        self._face_index = None
        self._canvas.bind("<Button-2>" if platform.system() == "Darwin" else "<Button-3>",
                          self._pop_menu)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _pop_menu(self, event):
        """ Pop up the context menu on a right click mouse event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The mouse event that has triggered the pop up menu
        """
        frame_idx, face_idx = self._canvas.viewport.face_from_point(
            self._canvas.canvasx(event.x), self._canvas.canvasy(event.y))[:2]
        if frame_idx == -1:
            logger.trace("No valid item under mouse")
            self._frame_index = self._face_index = None
            return
        self._frame_index = frame_idx
        self._face_index = face_idx
        logger.trace("Popping right click menu")
        self._menu.popup(event)

    def _delete_face(self):
        """ Delete the selected face on a right click mouse delete action. """
        logger.trace("Right click delete received. frame_id: %s, face_id: %s",
                     self._frame_index, self._face_index)
        self._detected_faces.update.delete(self._frame_index, self._face_index)
        self._frame_index = self._face_index = None
