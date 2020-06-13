#!/usr/bin/env python3
""" The Faces Viewer Frame and Canvas for Faceswap's Manual Tool. """
import logging
import platform
import tkinter as tk
from tkinter import ttk
from math import floor, ceil

import numpy as np

from lib.gui.custom_widgets import StatusBar, Tooltip
from lib.gui.utils import get_images, get_config

from .assets import FacesViewerLoader, ObjectCreator, UpdateFace
from .cache import FaceCache
from .display import ActiveFrame, ContextMenu, FaceFilter, HoverBox

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
    size: int
        The size, in pixels, to display the thumbnail images at
    """
    def __init__(self, parent, tk_globals, detected_faces, display_frame, size):
        logger.debug("Initializing %s: (parent: %s, tk_globals: %s, detected_faces: %s, "
                     "display_frame: %s, size: %s)", self.__class__.__name__, parent, tk_globals,
                     detected_faces, display_frame, size)
        super().__init__(parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._actions_frame = FacesActionsFrame(self)

        progress_bar = StatusBar(parent, hide_status=True)
        self._faces_frame = ttk.Frame(self)
        self._faces_frame.pack_propagate(0)
        self._faces_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._canvas = FacesViewer(self._faces_frame,
                                   tk_globals,
                                   self._actions_frame._tk_vars,
                                   detected_faces,
                                   display_frame,
                                   progress_bar,
                                   size)
        self._add_scrollbar()
        self._canvas.set_column_count()

        FacesViewerLoader(self._canvas, detected_faces)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _add_scrollbar(self):
        """ Add a scrollbar to the faces frame """
        logger.debug("Add Faces Viewer Scrollbar")
        scrollbar = ttk.Scrollbar(self._faces_frame, command=self._on_scroll)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.config(yscrollcommand=scrollbar.set)
        self.bind("<Configure>", self._update_scrollbar)
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
        self._canvas.set_visible_images(event)

    def _update_scrollbar(self, event):  # pylint: disable=unused-argument
        """ Update the faces frame scrollbar.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            Unused but required
        """
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

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
        self._canvas.set_visible_images(-1)

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
            button.state(["!pressed", "!focus", "disabled"])
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
    progress_bar: :class:`~lib.gui.custom_widgets.StatusBar`
        The progress bar object that displays in the bottom right of the GUI
    size: int
        The size, in pixels, to display the thumbnail images at
    """
    def __init__(self, parent, tk_globals, tk_action_vars, detected_faces, display_frame,
                 progress_bar, size):
        logger.debug("Initializing %s: (parent: %s, tk_globals: %s, tk_action_vars: %s, "
                     "detected_faces: %s, display_frame: %s, progress_bar: %s, size: %s)",
                     self.__class__.__name__, parent, tk_globals, tk_action_vars, detected_faces,
                     display_frame, progress_bar, size)
        super().__init__(parent, bd=0, highlightthickness=0, bg="#bcbcbc")
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor=tk.E)
        self._progress_bar = progress_bar
        self._globals = tk_globals
        self._tk_optional_annotations = {key: val for key, val in tk_action_vars.items()
                                         if key != "lockout"}
        self._faces_cache = FaceCache(self,
                                      detected_faces,
                                      tk_action_vars["lockout"],
                                      int(round(size * get_config().scaling_factor)))
        self._display_frame = display_frame
        self._utilities = dict(object_creator=ObjectCreator(self, detected_faces),
                               hover_box=HoverBox(self, detected_faces),
                               active_filter=FaceFilter(self, "all_frames"),
                               active_frame=ActiveFrame(self, detected_faces),
                               update_face=UpdateFace(self, detected_faces))
        ContextMenu(self)
        self._bind_mouse_wheel_scrolling()
        # Set in load_frames
        self._column_count = None
        self._annotation_colors = dict(mesh=self.get_muted_color("Mesh"))
        self._set_tk_callbacks()
        logger.debug("Initialized %s", self.__class__.__name__)

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

    @property
    def column_count(self):
        """ int: The number of columns in use in the :class:`FacesViewer` canvas. """
        return self._column_count

    @property
    def new_objects(self):
        """ :class:`ObjectCreator`: Class to add new objects to the :class:`FacesViewer`
        canvas. Call the :func:`create` method to add new annotations. """
        return self._utilities["object_creator"]

    @property
    def active_filter(self):
        """:class:`FaceFilter`: The currently selected filtered faces display. """
        return self._utilities["active_filter"]

    @property
    def active_frame(self):
        """:class:`ActiveFrame`: The currently active frame. """
        return self._utilities["active_frame"]

    @property
    def update_face(self):
        """:class:`UpdateFace`: Actions to add, remove or update a face in the viewer. """
        return self._utilities["update_face"]

    # << CALLBACK FUNCTIONS >> #
    def _set_tk_callbacks(self):
        """ Set the tkinter variable call backs.

        Switches the Filter view when the filter drop down is updated.
        Updates the Mesh annotation color when user amends the color drop down.
        Updates the mask type when the user changes the selected mask types
        Toggles the face viewer annotations on an optional annotation button press.
        """
        self._globals.tk_filter_mode.trace("w", lambda *e: self.switch_filter())
        self._display_frame.tk_control_colors["Mesh"].trace("w",
                                                            lambda *e: self.update_mesh_color())
        self._display_frame.tk_selected_mask.trace("w", lambda *e: self._update_mask_type())
        for opt, var in self._tk_optional_annotations.items():
            var.trace("w", lambda *e, o=opt: self._toggle_annotations(o))
        self.bind("<Configure>", self.set_visible_images)

    def set_visible_images(self, event):  # pylint:disable=unused-argument
        """ Load and unload thumbnails on a canvas resize or scroll event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter Configure event. Required but unused
        """
        if self.bbox("all") is None:
            return
        visible = (0,
                   floor(self.bbox("all")[3] * self.yview()[0]),
                   self.winfo_width(),
                   ceil(self.bbox("all")[3] * self.yview()[1]))
        displayed = set(self.find_overlapping(*visible))
        to_process = dict(show=displayed.intersection(set(self.find_withtag("not_visible"))),
                          hide=set(self.find_withtag("visible")).difference(displayed))
        funcs = dict(show=self._faces_cache.display_thumbnail,
                     hide=self._faces_cache.hide_thumbnail)
        tags = dict(show=("visible", "not_visible"), hide=("not_visible", "visible"))
        for name, process in to_process.items():
            logger.debug("processing: %s, item_ids: %s", name, process)
            for item_id in process:
                frame_idx = self.frame_index_from_object(item_id)
                face_idx = self.find_withtag("image_{}".format(frame_idx)).index(item_id)
                funcs[name](frame_idx, face_idx)
                self.dtag(item_id, tags[name][1])
                self.addtag_withtag(tags[name][0], item_id)

    def switch_filter(self):
        """ Update the :class:`FacesViewer` canvas for the active filter.
            Executed when the user changes the selected filter pull down.
         """
        filter_mode = self._globals.filter_mode.replace(" ", "_").lower()
        filter_mode = "all_frames" if filter_mode == "has_face(s)" else filter_mode
        current_dsp = self.active_filter.filter_type
        logger.debug("Current Display: '%s', Requested Display: '%s'", current_dsp, filter_mode)
        if filter_mode == current_dsp:
            return
        self.active_filter.de_initialize()
        self._utilities["active_filter"] = FaceFilter(self, filter_mode)

    def _update_mask_type(self):
        """ Update the displayed mask in the :class:`FacesViewer` canvas when the user changes
        the mask type. """
        self._faces_cache.mask_loader.update_all(self.selected_mask,
                                                 self.optional_annotations["mask"])
        self.active_frame.reload_annotations()

    # << POST INIT FUNCTIONS >> #
    def set_column_count(self):
        """ Set the column count for the displayed canvas. Must be done after
        the canvas has been packed and the scrollbar added.
        """
        self._column_count = self.winfo_width() // self._faces_cache.size
        logger.debug("Canvas width: %s, face thumbnail size: %s, column count: %s",
                     self.winfo_width(), self._faces_cache.size, self._column_count)

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
        self.set_visible_images(-1)

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
        self._utilities["hover_box"].on_hover(event)

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
        if self._faces_cache.is_loading:
            return
        if annotation == "mask":
            self._faces_cache.mask_loader.update_all(self.selected_mask,
                                                     self.optional_annotations[annotation])
        else:
            self.active_filter.toggle_annotation(self.optional_annotations[annotation])

    def frame_index_from_object(self, item_id):
        """ Retrieve the frame index that an object belongs to from it's tag.

        Parameters
        ----------
        item_id: int
            The tkinter canvas object id

        Returns
        -------
        int
            The frame index that the object belongs to or ``None`` if the tag cannot be found
        """
        retval = next((int(tag.replace("frame_id_", ""))
                       for tag in self.gettags(item_id) if tag.startswith("frame_id_")), None)
        logger.trace("item_id: %s, frame_id: %s", item_id, retval)
        return retval

    def face_index_from_object(self, item_id):
        """ Retrieve the index of the face within the current frame for the given canvas item id.

        Parameters
        ----------
        item_id: int
            The tkinter canvas object id

        Returns
        -------
        int
            The index of the face within the current frame
        """
        frame_id = self.frame_index_from_object(item_id)
        faces = self.find_withtag("image_{}".format(frame_id))
        logger.trace("frame_id: %s, face count: %s", frame_id, len(faces))
        if len(faces) == 1:  # Only 1 face, so face index is 0
            retval = 0
        else:
            face_group_id = self.face_id_from_object(item_id)
            face_groups = [next(tag
                                for tag in self.gettags(obj_id)
                                if tag.startswith("face_id"))
                           for obj_id in faces]
            logger.trace("face_groups: %s, face_group_id: %s", face_groups, face_group_id)
            retval = face_groups.index(face_group_id)
        logger.trace("item_id: %s, face_index: %s", item_id, retval)
        return retval

    def face_id_from_object(self, item_id):
        """ Retrieve the face group id tag for the given canvas item id.

        Parameters
        ----------
        item_id: int
            The tkinter canvas object id

        Returns
        -------
        str
            The face group tag for the given object, or ``None`` if a tag can't be found
        """
        retval = next((tag for tag in self.gettags(item_id)
                       if tag.startswith("face_id_")), None)
        logger.trace("item_id: %s, tag: %s", item_id, retval)
        return retval

    def coords_from_index(self, index):
        """ Returns the top left coordinates location for the canvas object based on an object's
        absolute index.

        Parameters
        ----------
        index: int
            The absolute display index of the face that the coordinates should be calculated from

        Returns
        -------
        :class:`numpy.ndarray`
            The top left (x, y) co-ordinates that an object should be placed on the canvas
            calculated from the given index.
        """
        return np.array(((index % self._column_count) * self._faces_cache.size,
                         (index // self._column_count) * self._faces_cache.size), dtype="int")
