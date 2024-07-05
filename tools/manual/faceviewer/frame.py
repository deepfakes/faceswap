#!/usr/bin/env python3
""" The Faces Viewer Frame and Canvas for Faceswap's Manual Tool. """
from __future__ import annotations
import colorsys
import gettext
import logging
import platform
import tkinter as tk
from tkinter import ttk
import typing as T
from math import floor, ceil
from threading import Thread, Event

import numpy as np

from lib.gui.custom_widgets import RightClickMenu, Tooltip
from lib.gui.utils import get_config, get_images
from lib.image import hex_to_rgb, rgb_to_hex
from lib.logger import parse_class_init

from .viewport import Viewport

if T.TYPE_CHECKING:
    from tools.manual.detected_faces import DetectedFaces
    from tools.manual.frameviewer.frame import DisplayFrame
    from tools.manual.manual import TkGlobals

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("tools.manual", localedir="locales", fallback=True)
_ = _LANG.gettext


class FacesFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The faces display frame (bottom section of GUI). This frame holds the faces viewport and
    the tkinter objects.

    Parameters
    ----------
    parent: :class:`ttk.Frame`
        The paned window that the faces frame resides in
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
        The :class:`~lib.align.DetectedFace` objects for this video
    display_frame: :class:`~tools.manual.frameviewer.frame.DisplayFrame`
        The section of the Manual Tool that holds the frames viewer
    """
    def __init__(self,
                 parent: ttk.Frame,
                 tk_globals: TkGlobals,
                 detected_faces: DetectedFaces,
                 display_frame: DisplayFrame) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._actions_frame = FacesActionsFrame(self)

        self._faces_frame = ttk.Frame(self)
        self._faces_frame.pack_propagate(False)
        self._faces_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._event = Event()
        self._canvas = FacesViewer(self._faces_frame,
                                   tk_globals,
                                   self._actions_frame._tk_vars,
                                   detected_faces,
                                   display_frame,
                                   self._event)
        self._add_scrollbar()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _add_scrollbar(self) -> None:
        """ Add a scrollbar to the faces frame """
        logger.debug("Add Faces Viewer Scrollbar")
        scrollbar = ttk.Scrollbar(self._faces_frame, command=self._on_scroll)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.config(yscrollcommand=scrollbar.set)
        self.bind("<Configure>", self._update_viewport)
        logger.debug("Added Faces Viewer Scrollbar")
        self.update_idletasks()  # Update so scrollbar width is correct

    def _on_scroll(self, *event: tk.Event) -> None:
        """ Callback on scrollbar scroll. Updates the canvas location and displays/hides
        thumbnail images.

        Parameters
        ----------
        event :class:`tkinter.Event`
            The scrollbar callback event
        """
        self._canvas.yview(*event)
        self._canvas.viewport.update()

    def _update_viewport(self, event: tk.Event) -> None:  # pylint:disable=unused-argument
        """ Update the faces viewport and scrollbar.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            Unused but required
        """
        self._canvas.viewport.update()
        self._canvas.configure(scrollregion=self._canvas.bbox("backdrop"))

    def canvas_scroll(self, direction: T.Literal["up", "down", "page-up", "page-down"]) -> None:
        """ Scroll the canvas on an up/down or page-up/page-down key press.

        Notes
        -----
        To protect against a held down key press stacking tasks and locking up the GUI
        a background thread is launched and discards subsequent key presses whilst the
        previous update occurs.

        Parameters
        ----------
        direction: ["up", "down", "page-up", "page-down"]
            The request page scroll direction and amount.
        """

        if self._event.is_set():
            logger.trace("Update already running. "  # type:ignore[attr-defined]
                         "Aborting repeated keypress")
            return
        logger.trace("Running update on received key press: %s",  # type:ignore[attr-defined]
                     direction)

        amount = 1 if direction.endswith("down") else -1
        units = "pages" if direction.startswith("page") else "units"
        self._event.set()
        thread = Thread(target=self._canvas.canvas_scroll,
                        args=(amount, units, self._event))
        thread.start()

    def set_annotation_display(self, key: str) -> None:
        """ Set the optional annotation overlay based on keyboard shortcut.

        Parameters
        ----------
        key: str
            The pressed key
        """
        self._actions_frame.on_click(self._actions_frame.key_bindings[key])


class FacesActionsFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The left hand action frame holding the optional annotation buttons.

    Parameters
    ----------
    parent: :class:`FacesFrame`
        The Faces frame that this actions frame reside in
    """
    def __init__(self, parent: FacesFrame) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(parent)
        self.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 4), pady=2)
        self._tk_vars: dict[T.Literal["mesh", "mask"], tk.BooleanVar] = {}
        self._configure_styles()
        self._buttons = self._add_buttons()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def key_bindings(self) -> dict[str, T.Literal["mask", "mesh"]]:
        """ dict: The mapping of key presses to optional annotations to display. Keyboard shortcuts
        utilize the function keys. """
        return {f"F{idx + 9}": display
                for idx, display in enumerate(T.get_args(T.Literal["mesh", "mask"]))}

    @property
    def _helptext(self) -> dict[T.Literal["mask", "mesh"], str]:
        """ dict: `button key`: `button helptext`. The help text to display for each button. """
        inverse_keybindings = {val: key for key, val in self.key_bindings.items()}
        retval: dict[T.Literal["mask", "mesh"], str] = {"mesh": _('Display the landmarks mesh'),
                                                        "mask": _('Display the mask')}
        for item in retval:
            retval[item] += f" ({inverse_keybindings[item]})"
        return retval

    def _configure_styles(self) -> None:
        """ Configure the background color for button frame and the button styles. """
        style = ttk.Style()
        style.configure("display.TFrame", background='#d3d3d3')
        style.configure("display_selected.TButton", relief="flat", background="#bedaf1")
        style.configure("display_deselected.TButton", relief="flat")
        self.config(style="display.TFrame")

    def _add_buttons(self) -> dict[T.Literal["mesh", "mask"], ttk.Button]:
        """ Add the display buttons to the Faces window.

        Returns
        -------
        dict[Literal["mesh", "mask"], tk.Button]]
            The display name and its associated button.
        """
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.Y)
        buttons = {}
        for display in self.key_bindings.values():
            var = tk.BooleanVar()
            var.set(False)
            self._tk_vars[display] = var

            lookup = "landmarks" if display == "mesh" else display
            button = ttk.Button(frame,
                                image=get_images().icons[lookup],
                                command=T.cast(T.Callable, lambda t=display: self.on_click(t)),
                                style="display_deselected.TButton")
            button.state(["!pressed", "!focus"])
            button.pack()
            Tooltip(button, text=self._helptext[display])
            buttons[display] = button
        return buttons

    def on_click(self, display: T.Literal["mesh", "mask"]) -> None:
        """ Click event for the optional annotation buttons. Loads and unloads the annotations from
        the faces viewer.

        Parameters
        ----------
        display: Literal["mesh", "mask"]
            The display name for the button that has called this event as exists in
            :attr:`_buttons`
        """
        is_pressed = not self._tk_vars[display].get()
        style = "display_selected.TButton" if is_pressed else "display_deselected.TButton"
        state = ["pressed", "focus"] if is_pressed else ["!pressed", "!focus"]
        btn = self._buttons[display]
        btn.configure(style=style)
        btn.state(state)
        self._tk_vars[display].set(is_pressed)


class FacesViewer(tk.Canvas):   # pylint:disable=too-many-ancestors
    """ The :class:`tkinter.Canvas` that holds the faces viewer section of the Manual Tool.

    Parameters
    ----------
    parent: :class:`tkinter.ttk.Frame`
        The parent frame for the canvas
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    tk_action_vars: dict
        The :class:`tkinter.BooleanVar` objects for selectable optional annotations
        as set by the buttons in the :class:`FacesActionsFrame`
    detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
        The :class:`~lib.align.DetectedFace` objects for this video
    display_frame: :class:`~tools.manual.frameviewer.frame.DisplayFrame`
        The section of the Manual Tool that holds the frames viewer
    event: :class:`threading.Event`
        The threading event object for repeated key press protection
    """
    def __init__(self, parent: ttk.Frame,
                 tk_globals: TkGlobals,
                 tk_action_vars: dict[T.Literal["mesh", "mask"], tk.BooleanVar],
                 detected_faces: DetectedFaces,
                 display_frame: DisplayFrame,
                 event: Event) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(parent,
                         bd=0,
                         highlightthickness=0,
                         bg=get_config().user_theme["group_panel"]["panel_background"])
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor=tk.E)
        self._sizes = {"tiny": 32, "small": 64, "medium": 96, "large": 128, "extralarge": 192}

        self._globals = tk_globals
        self._tk_optional_annotations = tk_action_vars
        self._event = event
        self._display_frame = display_frame
        self._grid = Grid(self, detected_faces)
        self._view = Viewport(self, detected_faces.tk_edited)
        self._annotation_colors = {"mesh": self.get_muted_color("Mesh"),
                                   "box": self.control_colors["ExtractBox"]}

        ContextMenu(self, detected_faces)
        self._bind_mouse_wheel_scrolling()
        self._set_tk_callbacks(detected_faces)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def face_size(self) -> int:
        """ int: The currently selected thumbnail size in pixels """
        scaling = get_config().scaling_factor
        size = self._sizes[self._globals.var_faces_size.get().lower().replace(" ", "")]
        scaled = size * scaling
        return int(round(scaled / 2) * 2)

    @property
    def viewport(self) -> Viewport:
        """ :class:`~tools.manual.faceviewer.viewport.Viewport`: The viewport area of the
        faces viewer. """
        return self._view

    @property
    def layout(self) -> Grid:
        """ :class:`Grid`: The grid for the current :class:`FacesViewer`. """
        return self._grid

    @property
    def optional_annotations(self) -> dict[T.Literal["mesh", "mask"], bool]:
        """ dict[Literal["mesh", "mask"], bool]: The values currently set for the
        selectable optional annotations. """
        return {opt: val.get() for opt, val in self._tk_optional_annotations.items()}

    @property
    def selected_mask(self) -> str:
        """ str: The currently selected mask from the display frame control panel. """
        return self._display_frame.tk_selected_mask.get().lower()

    @property
    def control_colors(self) -> dict[str, str]:
        """dict[str, str]: The frame Editor name as key with the current user selected hex code as
        value. """
        return ({key: val.get() for key, val in self._display_frame.tk_control_colors.items()})

    # << CALLBACK FUNCTIONS >> #
    def _set_tk_callbacks(self, detected_faces: DetectedFaces):
        """ Set the tkinter variable call backs.

        Parameters
        ----------
        detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
            The Manual Tool's Detected Faces object

        Redraw the grid on a face size change, a filter change or on add/remove faces.
        Updates the annotation colors when user amends a color drop down.
        Updates the mask type when the user changes the selected mask types
        Toggles the face viewer annotations on an optional annotation button press.
        """
        for strvar in (self._globals.var_faces_size, self._globals.var_filter_mode):
            strvar.trace_add("write", lambda *e, v=strvar: self.refresh_grid(v))
        boolvar = detected_faces.tk_face_count_changed
        boolvar.trace_add("write",
                          lambda *e, v=boolvar: self.refresh_grid(v, retain_position=True))

        self._display_frame.tk_control_colors["Mesh"].trace_add(
            "write", lambda *e: self._update_mesh_color())
        self._display_frame.tk_control_colors["ExtractBox"].trace_add(
            "write", lambda *e: self._update_box_color())
        self._display_frame.tk_selected_mask.trace_add(
            "write", lambda *e: self._update_mask_type())

        for opt, var in self._tk_optional_annotations.items():
            var.trace_add("write", lambda *e, o=opt: self._toggle_annotations(o))

        self.bind("<Configure>", lambda *e: self._view.update())

    def refresh_grid(self, trigger_var: tk.BooleanVar, retain_position: bool = False) -> None:
        """ Recalculate the full grid and redraw. Used when the active filter pull down is used, a
        face has been added or removed, or the face thumbnail size has changed.

        Parameters
        ----------
        trigger_var: :class:`tkinter.BooleanVar`
            The tkinter variable that has triggered the grid update. Will either be the variable
            indicating that the face size have been changed, or the variable indicating that the
            selected filter mode has been changed.
        retain_position: bool, optional
            ``True`` if the grid should be set back to the position it was at after the update has
            been processed, otherwise ``False``. Default: ``False``.
        """
        if not trigger_var.get():
            return
        size_change = isinstance(trigger_var, tk.StringVar)
        move_to = self.yview()[0] if retain_position else 0.0
        self._grid.update()
        if move_to != 0.0:
            self.yview_moveto(move_to)
        if size_change:
            self._view.reset()
        self._view.update(refresh_annotations=retain_position)
        if not size_change:
            trigger_var.set(False)

    def _update_mask_type(self) -> None:
        """ Update the displayed mask in the :class:`FacesViewer` canvas when the user changes
        the mask type. """
        state: T.Literal["normal", "hidden"]
        state = "normal" if self.optional_annotations["mask"] else "hidden"
        logger.debug("Updating mask type: (mask_type: %s. state: %s)", self.selected_mask, state)
        self._view.toggle_mask(state, self.selected_mask)

    # << MOUSE HANDLING >>
    def _bind_mouse_wheel_scrolling(self) -> None:
        """ Bind mouse wheel to scroll the :class:`FacesViewer` canvas. """
        if platform.system() == "Linux":
            self.bind("<Button-4>", self._scroll)
            self.bind("<Button-5>", self._scroll)
        else:
            self.bind("<MouseWheel>", self._scroll)

    def _scroll(self, event: tk.Event) -> None:
        """ Handle mouse wheel scrolling over the :class:`FacesViewer` canvas.

        Update is run in a thread to avoid repeated scroll actions stacking and locking up the GUI.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The event fired by the mouse scrolling
        """
        if self._event.is_set():
            logger.trace("Update already running. "  # type:ignore[attr-defined]
                         "Aborting repeated mousewheel")
            return
        if platform.system() == "Darwin":
            adjust = event.delta
        elif platform.system() == "Windows":
            adjust = int(event.delta / 120)
        elif event.num == 5:
            adjust = -1
        else:
            adjust = 1
        self._event.set()
        thread = Thread(target=self.canvas_scroll, args=(-1 * adjust, "units", self._event))
        thread.start()

    def canvas_scroll(self, amount: int, units: T.Literal["pages", "units"], event: Event) -> None:
        """ Scroll the canvas on an up/down or page-up/page-down key press.

        Parameters
        ----------
        amount: int
            The number of units to scroll the canvas
        units: Literal["pages", "units"]
            The unit type to scroll by
        event: :class:`threading.Event`
            event to indicate to the calling process whether the scroll is still updating
        """
        self.yview_scroll(int(amount), units)
        self._view.update()
        self._view.hover_box.on_hover(None)
        event.clear()

    # << OPTIONAL ANNOTATION METHODS >> #
    def _update_mesh_color(self) -> None:
        """ Update the mesh color when user updates the control panel. """
        color = self.get_muted_color("Mesh")
        if self._annotation_colors["mesh"] == color:
            return
        highlight_color = self.control_colors["Mesh"]

        self.itemconfig("viewport_polygon", outline=color)
        self.itemconfig("viewport_line", fill=color)
        self.itemconfig("active_mesh_polygon", outline=highlight_color)
        self.itemconfig("active_mesh_line", fill=highlight_color)
        self._annotation_colors["mesh"] = color

    def _update_box_color(self) -> None:
        """ Update the active box color when user updates the control panel. """
        color = self.control_colors["ExtractBox"]

        if self._annotation_colors["box"] == color:
            return
        self.itemconfig("active_highlighter", outline=color)
        self._annotation_colors["box"] = color

    def get_muted_color(self, color_key: str) -> str:
        """ Creates a muted version of the given annotation color for non-active faces.

        Parameters
        ----------
        color_key: str
            The annotation key to obtain the color for from :attr:`control_colors`

        Returns
        -------
        str
            The hex color code of the muted color
        """
        scale = 0.65
        hls = np.array(colorsys.rgb_to_hls(*hex_to_rgb(self.control_colors[color_key])))
        scale = (1 - scale) + 1 if hls[1] < 120 else scale
        hls[1] = max(0., min(256., scale * hls[1]))
        rgb = np.clip(np.rint(colorsys.hls_to_rgb(*hls)).astype("uint8"), 0, 255)
        retval = rgb_to_hex(rgb)
        return retval

    def _toggle_annotations(self, annotation: T.Literal["mesh", "mask"]) -> None:
        """ Toggle optional annotations on or off after the user depresses an optional button.

        Parameters
        ----------
        annotation: ["mesh", "mask"]
            The optional annotation to toggle on or off
        """
        state: T.Literal["hidden", "normal"]
        state = "normal" if self.optional_annotations[annotation] else "hidden"
        logger.debug("Toggle annotation: (annotation: %s, state: %s)", annotation, state)
        if annotation == "mesh":
            self._view.toggle_mesh(state)
        if annotation == "mask":
            self._view.toggle_mask(state, self.selected_mask)


class Grid():
    """ Holds information on the current filtered grid layout.

    The grid keeps information on frame indices, face indices, x and y positions and detected face
    objects laid out in a numpy array to reflect the current full layout of faces within the face
    viewer based on the currently selected filter and face thumbnail size.

    Parameters
    ----------
    canvas: :class:`~FacesViewer`
        The :class:`~tools.manual.faceviewer.frame.FacesViewer` canvas
    detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
        The :class:`~lib.align.DetectedFace` objects for this video
    """
    def __init__(self, canvas: FacesViewer, detected_faces: DetectedFaces):
        logger.debug(parse_class_init(locals()))
        self._canvas = canvas
        self._detected_faces = detected_faces
        self._raw_indices = detected_faces.filter.raw_indices
        self._frames_list = detected_faces.filter.frames_list

        self._is_valid: bool = False
        self._face_size: int = 0
        self._grid: np.ndarray | None = None
        self._display_faces: np.ndarray | None = None

        self._canvas.update_idletasks()
        self._canvas.create_rectangle(0, 0, 0, 0, tags=["backdrop"])
        self.update()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def face_size(self) -> int:
        """ int: The pixel size of each thumbnail within the face viewer. """
        return self._face_size

    @property
    def is_valid(self) -> bool:
        """ bool: ``True`` if the current filter means that the grid holds faces. ``False`` if
        there are no faces displayed in the grid. """
        return self._is_valid

    @property
    def columns_rows(self) -> tuple[int, int]:
        """ tuple: the (`columns`, `rows`) required to hold all display images. """
        if not self._is_valid:
            return (0, 0)
        assert self._grid is not None
        retval = tuple(reversed(self._grid.shape[1:]))
        return T.cast(tuple[int, int], retval)

    @property
    def dimensions(self) -> tuple[int, int]:
        """ tuple: The (`width`, `height`) required to hold all display images. """
        if self._is_valid:
            assert self._grid is not None
            retval = tuple(dim * self._face_size for dim in reversed(self._grid.shape[1:]))
            assert len(retval) == 2
        else:
            retval = (0, 0)
        return T.cast(tuple[int, int], retval)

    @property
    def _visible_row_indices(self) -> tuple[int, int]:
        """tuple: A 1 dimensional array of the (`top_row_index`, `bottom_row_index`) of the grid
        currently in the viewable area.
        """
        height = self.dimensions[1]
        visible = (max(0, floor(height * self._canvas.yview()[0]) - self._face_size),
                   ceil(height * self._canvas.yview()[1]))
        logger.trace("height: %s, yview: %s, face_size: %s, "  # type:ignore[attr-defined]
                     "visible: %s", height, self._canvas.yview(), self._face_size, visible)
        assert self._grid is not None
        y_points = self._grid[3, :, 1]
        top = np.searchsorted(y_points, visible[0], side="left")
        bottom = np.searchsorted(y_points, visible[1], side="right")
        return int(top), int(bottom)

    @property
    def visible_area(self) -> tuple[np.ndarray, np.ndarray]:
        """tuple[:class:`numpy.ndarray`, :class:`numpy.ndarray`]: Tuple containing 2 arrays.

        1st array contains an array of shape (`4`, `rows`, `columns`) corresponding
        to the viewable area of the display grid. 1st dimension contains frame indices, 2nd
        dimension face indices. The 3rd and 4th dimension contain the x and y position of the top
        left corner of the face respectively.

        2nd array contains :class:`~lib.align.DetectedFace` objects laid out in (rows, columns)

        Any locations that are not populated by a face will have a frame and face index of -1
        """
        if not self._is_valid:
            retval = np.zeros((4, 0, 0)), np.zeros((0, 0))
        else:
            assert self._grid is not None
            assert self._display_faces is not None
            top, bottom = self._visible_row_indices
            retval = self._grid[:, top:bottom, :], self._display_faces[top:bottom, :]
        logger.trace([r if r is None else r.shape for r in retval])  # type:ignore[attr-defined]
        return retval

    def y_coord_from_frame(self, frame_index: int) -> int:
        """ Return the y coordinate for the first face that appears in the given frame.

        Parameters
        ----------
        frame_index: int
            The frame index to locate in the grid

        Returns
        -------
        int
            The y coordinate of the first face for the given frame
        """
        assert self._grid is not None
        return min(self._grid[3][np.where(self._grid[0] == frame_index)])

    def frame_has_faces(self, frame_index: int) -> bool | np.bool_:
        """ Check whether the given frame index contains any faces.

        Parameters
        ----------
        frame_index: int
            The frame index to locate in the grid

        Returns
        -------
        bool
            ``True`` if there are faces in the given frame otherwise ``False``
        """
        if not self._is_valid:
            return False
        assert self._grid is not None
        return np.any(self._grid[0] == frame_index)

    def update(self) -> None:
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
        self._canvas.configure(scrollregion=self._canvas.bbox("backdrop"))
        self._canvas.yview_moveto(0.0)

    def _get_grid(self) -> None:
        """ Get the grid information for faces currently displayed in the :class:`FacesViewer`.
        and set to :attr:`_grid`. Creates a numpy array of shape (`4`, `rows`, `columns`)
        corresponding to the display grid. 1st dimension contains frame indices, 2nd dimension face
        indices. The 3rd and 4th dimension contain the x and y position of the top left corner of
        the face respectively.

        Any locations that are not populated by a face will have a frame and face index of -1"""
        labels = self._get_labels()
        if not self._is_valid:
            logger.debug("Setting grid to None for no faces.")
            self._grid = None
            return
        assert labels is not None
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

    def _get_labels(self) -> np.ndarray | None:
        """ Get the frame and face index for each grid position for the current filter.

        Returns
        -------
        :class:`numpy.ndarray` | None
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
        remainder = face_count % columns
        padding = [] if remainder == 0 else [-1 for _ in range(columns - remainder)]
        labels = np.array((self._raw_indices["frame"] + padding,
                           self._raw_indices["face"] + padding),
                          dtype="int").reshape((2, rows, columns))
        logger.debug("face-count: %s, columns: %s, rows: %s, remainder: %s, padding: %s, labels "
                     "shape: %s", face_count, columns, rows, remainder, padding, labels.shape)
        return labels

    def _get_display_faces(self):
        """ Get the detected faces for the current filter, arrange to grid and set to
        :attr:`_display_faces`. This is an array of dimensions (rows, columns) corresponding to the
        display grid, containing the corresponding :class:`lib.align.DetectFace` object

        Any remaining placeholders at the end of the grid which are not populated with a face are
        replaced with ``None``"""
        if not self._is_valid:
            logger.debug("Setting display_faces to None for no faces.")
            self._display_faces = None
            return
        current_faces = self._detected_faces.current_faces
        columns, rows = self.columns_rows
        face_count = len(self._raw_indices["frame"])
        padding = [None for _ in range(face_count, columns * rows)]
        self._display_faces = np.array([None if idx is None else current_faces[idx][face_idx]
                                        for idx, face_idx
                                        in zip(self._raw_indices["frame"] + padding,
                                               self._raw_indices["face"] + padding)],
                                       dtype="object").reshape(rows, columns)
        logger.debug("faces: (shape: %s, dtype: %s)",
                     self._display_faces.shape, self._display_faces.dtype)

    def transport_index_from_frame(self, frame_index: int) -> int | None:
        """ Return the main frame's transport index for the given frame index based on the current
        filter criteria.

        Parameters
        ----------
        frame_index: int
            The absolute index for the frame within the full frames list

        Returns
        -------
        int | None
            The index of the requested frame within the filtered frames view. None if no valid
            frames
        """
        retval = self._frames_list.index(frame_index) if frame_index in self._frames_list else None
        logger.trace("frame_index: %s, transport_index: %s",  # type:ignore[attr-defined]
                     frame_index, retval)
        return retval


class ContextMenu():  # pylint:disable=too-few-public-methods
    """  Enables a right click context menu for the
    :class:`~tools.manual.faceviewer.frame.FacesViewer`.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`FacesViewer` canvas
    detected_faces: :class:`~tools.manual.detected_faces`
        The manual tool's detected faces class
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
            logger.trace("No valid item under mouse")  # type:ignore[attr-defined]
            self._frame_index = self._face_index = None
            return
        self._frame_index = frame_idx
        self._face_index = face_idx
        logger.trace("Popping right click menu")  # type:ignore[attr-defined]
        self._menu.popup(event)

    def _delete_face(self):
        """ Delete the selected face on a right click mouse delete action. """
        logger.trace("Right click delete received. frame_id: %s, "  # type:ignore[attr-defined]
                     "face_id: %s", self._frame_index, self._face_index)
        self._detected_faces.update.delete(self._frame_index, self._face_index)
        self._frame_index = self._face_index = None
