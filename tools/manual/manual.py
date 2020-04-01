#!/usr/bin/env python3
""" Tool to manually interact with the alignments file using visual tools """
import logging
import os
import platform
import sys
import tkinter as tk
from tkinter import ttk
from time import sleep

import numpy as np

from lib.gui.control_helper import ControlPanel
from lib.gui.custom_widgets import StatusBar, Tooltip
from lib.gui.utils import get_images, get_config, initialize_config, initialize_images
from lib.multithreading import MultiThread
from lib.utils import _video_extensions
from plugins.extract.pipeline import Extractor, ExtractMedia

from .detected_faces import DetectedFaces
from .faceviewer.assets import FacesViewerLoader, ObjectCreator, UpdateFace
from .faceviewer.cache import FaceCache
from .faceviewer.display import ActiveFrame, ContextMenu, FaceFilter, HoverBox
from .frameviewer.display_frame import DisplayFrame
from .frameviewer.media import FrameNavigation

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Manual(tk.Tk):
    """ This tool is part of the Faceswap Tools suite and should be called from
    ``python tools.py manual`` command.

    Allows for visual interaction with frames, faces and alignments file to perform various
    adjustments to the alignments file.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """

    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: '%s')", self.__class__.__name__, arguments)
        super().__init__()
        self._globals = TkGlobals()
        is_video = self._check_input(arguments.frames)
        self._initialize_tkinter()

        extractor = Aligner()
        self._det_faces = DetectedFaces(self._globals,
                                        arguments.alignments_path,
                                        arguments.frames,
                                        extractor,
                                        is_video)

        video_meta_data = self._det_faces.video_meta_data
        self._frames = FrameNavigation(self._globals,
                                       arguments.frames,
                                       get_config().scaling_factor,
                                       video_meta_data)
        self._det_faces.load_faces(self._frames)

        self._containers = self._create_containers()

        self._wait_for_threads(extractor, video_meta_data)

        self._display = DisplayFrame(self._containers["top"],
                                     self._globals,
                                     self._frames,
                                     self._det_faces)
        self._faces_frame = FacesFrame(self._containers["bottom"],
                                       self._globals,
                                       self._frames,
                                       self._det_faces,
                                       self._display)

        self._options = Options(self._containers["top"], self._globals, self._display)
        self._display.tk_selected_action.set("View")

        self.bind("<Key>", self._handle_key_press)
        self._set_initial_layout()
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def _check_input(frames_location):
        """ Check whether the input is a video

        Parameters
        ----------
        frames_location: str
            The input location for video or images

        Returns
        -------
        bool: 'True' if input is a video 'False' if it is a folder.

        Raises
        ------
        FaceswapError
            If the given location is a file and does not have a valid video extension.

        """
        if os.path.isdir(frames_location):
            retval = False
        elif os.path.splitext(frames_location)[1].lower() in _video_extensions:
            retval = True
        else:
            logger.error("The input location '%s' is not valid", frames_location)
            sys.exit(1)
        logger.debug("Input '%s' is_video: %s", frames_location, retval)
        return retval

    def _wait_for_threads(self, extractor, video_meta_data):
        """ The :class:`Aligner` is launched in a background thread. Wait for it to be initialized
        prior to proceeding

        Notes
        -----
        Because some of the initialize checks perform extra work once their threads are complete,
        they should only return ``True`` once, and should not be queried again.

        """
        extractor_init = False
        frames_init = False
        while True:
            extractor_init = extractor_init if extractor_init else extractor.is_initialized
            frames_init = frames_init if frames_init else self._frames.is_initialized
            if extractor_init and frames_init:
                logger.debug("Threads inialized")
                break
            logger.debug("Threads not initialized. Waiting...")
            sleep(1)

        extractor.link_faces_and_frames(self._det_faces, self._frames)
        if any(val is None for val in video_meta_data.values()):
            self._det_faces.save_video_meta_data(**self._frames.video_meta_data)

    def _initialize_tkinter(self):
        """ Initialize a standalone tkinter instance. """
        logger.debug("Initializing tkinter")
        for widget in ("TButton", "TCheckbutton", "TRadiobutton"):
            self.unbind_class(widget, "<Key-space>")
        initialize_config(self, None, None, None)
        initialize_images()
        get_config().set_geometry(940, 600, fullscreen=True)
        self.title("Faceswap.py - Visual Alignments")
        self.tk.call(
            "wm",
            "iconphoto",
            self._w, get_images().icons["favicon"])  # pylint:disable=protected-access
        logger.debug("Initialized tkinter")

    def _create_containers(self):
        """ Create the paned window containers for various GUI elements

        Returns
        -------
        dict:
            The main containers of the manual tool.
        """
        logger.debug("Creating containers")
        main = tk.PanedWindow(self,
                              sashrelief=tk.RIDGE,
                              sashwidth=2,
                              sashpad=4,
                              orient=tk.VERTICAL,
                              name="pw_main")
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main, name="frame_top")
        main.add(top)

        bottom = ttk.Frame(main, name="frame_bottom")
        main.add(bottom)
        logger.debug("Created containers")
        return dict(main=main, top=top, bottom=bottom)

    def _handle_key_press(self, event):
        """ Keyboard shortcuts

        Parameters
        ----------
        event: :class:`tkinter.Event()`
            The tkinter key press event

        Notes
        -----
        The following keys are reserved for the :mod:`tools.lib_manual.editor` classes
            * Delete - Used for deleting faces
            * [] - decrease / increase brush size
            * B, D, E, M - Optional Actions (Brush, Drag, Erase, Zoom)
        """
        # Alt modifier appears to be broken in Windows so don't use it.
        modifiers = {0x0001: 'shift',
                     0x0004: 'ctrl'}

        tk_pos = self._globals.tk_frame_index
        bindings = {
            "z": self._display.decrement_frame,
            "x": self._display.increment_frame,
            "space": self._display.handle_play_button,
            "home": self._display.goto_first_frame,
            "end": self._display.goto_last_frame,
            "down": lambda d="down": self._faces_frame.canvas_scroll(d),
            "up": lambda d="up": self._faces_frame.canvas_scroll(d),
            "next": lambda d="page-down": self._faces_frame.canvas_scroll(d),
            "prior": lambda d="page-up": self._faces_frame.canvas_scroll(d),
            "f": self._display.cycle_filter_mode,
            "f1": lambda k=event.keysym: self._display.set_action(k),
            "f2": lambda k=event.keysym: self._display.set_action(k),
            "f3": lambda k=event.keysym: self._display.set_action(k),
            "f4": lambda k=event.keysym: self._display.set_action(k),
            "f5": lambda k=event.keysym: self._display.set_action(k),
            "f9": lambda k=event.keysym: self._faces_frame.set_annotation_display(k),
            "f10": lambda k=event.keysym: self._faces_frame.set_annotation_display(k),
            "c": lambda f=tk_pos.get(), d="previous": self._det_faces.update.copy(f, d),
            "v": lambda f=tk_pos.get(), d="next": self._det_faces.update.copy(f, d),
            "ctrl_s": self._det_faces.save,
            "r": lambda f=tk_pos.get(): self._det_faces.update.revert_to_saved(f)}

        # Allow keypad keys to be used for numbers
        press = event.keysym.replace("KP_", "") if event.keysym.startswith("KP_") else event.keysym
        modifier = "_".join(val for key, val in modifiers.items() if event.state & key != 0)
        key_press = "_".join([modifier, press]) if modifier else press
        if key_press.lower() in bindings:
            self.focus_set()
            bindings[key_press.lower()]()

    def _set_initial_layout(self):
        """ Set the bottom frame position to correct location to display full frame window. """
        self.update_idletasks()
        location = self._display.winfo_reqheight() + 5
        self._containers["main"].sash_place(0, 1, location)

    def process(self):
        """ The entry point for the Visual Alignments tool from :file:`lib.tools.cli`.

        Launch the tkinter Visual Alignments Window and run main loop.
        """
        self.mainloop()


class Options(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Control panel options for currently displayed Editor.

    parent: :class:`tkinter.ttk.Frame`
        The parent frame for the control panel options
    tk_globals: :class:`TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    display_frame: :class:`DisplayFrame`
        The frame that holds the editors
    """
    def __init__(self, parent, tk_globals, display_frame):
        super().__init__(parent)
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._globals = tk_globals
        self._display_frame = display_frame
        self._control_panels = self._initialize()
        self._set_tk_callbacks()
        self._update_options()

    def _initialize(self):
        """ Initialize all of the control panels, then display the default panel.

        Adds the control panel to :attr:`_control_panels` and sets the traceback to update
        display when a panel option has been changed.

        Notes
        -----
        All panels must be initialized at the beginning so that the global format options are not
        reset to default when the editor is first selected.

        The Traceback must be set after the panel has first been packed as otherwise it interferes
        with the loading of the faces pane.
        """
        panels = dict()
        for name, editor in self._display_frame.editors.items():
            logger.debug("Initializing control panel for '%s' editor", name)
            controls = editor.controls
            panel = ControlPanel(self, controls["controls"],
                                 option_columns=3,
                                 columns=1,
                                 max_columns=1,
                                 header_text=controls["header"],
                                 blank_nones=False,
                                 label_width=18,
                                 scrollbar=False)
            panel.pack_forget()
            panels[name] = panel
        return panels

    def _set_tk_callbacks(self):
        """ Sets the callback to change to the relevant control panel options when the selected
        editor is changed, and the display update on panel option change."""
        self._display_frame.tk_selected_action.trace("w", self._update_options)
        seen_controls = set()
        for name, editor in self._display_frame.editors.items():
            for ctl in editor.controls["controls"]:
                if ctl in seen_controls:
                    # Some controls are re-used (annotation format), so skip if trace has already
                    # been set
                    continue
                logger.debug("Adding control update callback: (editor: %s, control: %s)",
                             name, ctl.title)
                seen_controls.add(ctl)
                ctl.tk_var.trace("w", lambda *e: self._globals.tk_update.set(True))

    def _update_options(self, *args):  # pylint:disable=unused-argument
        """ Update the control panel display for the current editor.

        If the options have not already been set, then adds the control panel to
        :attr:`_control_panels`. Displays the current editor's control panel

        Parameters
        ----------
        args: tuple
            Unused but required for tkinter variable callback
        """
        self._clear_options_frame()
        editor = self._display_frame.tk_selected_action.get()
        logger.debug("Displaying control panel for editor: '%s'", editor)
        self._control_panels[editor].pack(expand=True, fill=tk.BOTH)

    def _clear_options_frame(self):
        """ Hides the currently displayed control panel """
        for editor, panel in self._control_panels.items():
            if panel.winfo_ismapped():
                logger.debug("Hiding control panel for: %s", editor)
                panel.pack_forget()


class FacesFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The faces display frame (bottom section of GUI).

    Parameters
    ----------
    parent: :class:`tkinter.PanedWindow`
        The paned window that the faces frame resides in
    tk_globals: :class:`TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    frames: :class:`FrameNavigation`
        The object that holds the cache of frames.
    detected_faces: :class:`~tool.manual.faces.DetectedFaces`
        The :class:`~lib.faces_detect.DetectedFace` objects for this video
    display_frame: :class:`DisplayFrame`
        The section of the Manual Tool that holds the frames viewer
    """
    def __init__(self, parent, tk_globals, frames, detected_faces, display_frame):
        logger.debug("Initializing %s: (parent: %s, tk_globals: %s, frames: %s, "
                     "detected_faces: %s, display_frame: %s)", self.__class__.__name__, parent,
                     tk_globals, frames, detected_faces, display_frame)
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
                                   frames,
                                   detected_faces,
                                   display_frame,
                                   progress_bar)
        scrollbar_width = self._add_scrollbar()
        self._canvas.set_column_count(self.winfo_width() - scrollbar_width)

        FacesViewerLoader(self._canvas, detected_faces)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _add_scrollbar(self):
        """ Add a scrollbar to the faces frame """
        logger.debug("Add Faces Viewer Scrollbar")
        scrollbar = ttk.Scrollbar(self._faces_frame, command=self._canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.config(yscrollcommand=scrollbar.set)
        self.bind("<Configure>", self._update_scrollbar)
        logger.debug("Added Faces Viewer Scrollbar")
        self.update_idletasks()  # Update so scrollbar width is correct
        return scrollbar.winfo_width()

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
        lockout.set(True)
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
    tk_globals: :class:`TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    tk_action_vars: dict
        The :class:`tkinter.BooleanVar` objects for selectable optional annotations
        as set by the buttons in the :class:`FacesActionsFrame`
    frames: :class:`~tools.manual.media.FrameNavigation`
        The object that holds the cache of frames and handles frame navigation.
    detected_faces: :class:`~tool.manual.faces.DetectedFaces`
        The :class:`~lib.faces_detect.DetectedFace` objects for this video
    display_frame: :class:`DisplayFrame`
        The section of the Manual Tool that holds the frames viewer
    progress_bar: :class:`~lib.gui.custom_widgets.StatusBar`
        The progress bar object that displays in the bottom right of the GUI
    """
    def __init__(self, parent, tk_globals, tk_action_vars, frames,
                 detected_faces, display_frame, progress_bar):
        logger.debug("Initializing %s: (parent: %s, tk_globals: %s, tk_action_vars: %s, "
                     "frames: %s, detected_faces: %s, display_frame: %s, progress_bar: %s)",
                     self.__class__.__name__, parent, tk_globals, tk_action_vars, frames,
                     detected_faces, display_frame, progress_bar)
        super().__init__(parent, bd=0, highlightthickness=0, bg="#bcbcbc")
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor=tk.E)
        self._progress_bar = progress_bar
        self._globals = tk_globals
        self._tk_optional_annotations = {key: val for key, val in tk_action_vars.items()
                                         if key != "lockout"}
        self._frames = frames
        self._faces_cache = FaceCache(self,
                                      get_config().scaling_factor,
                                      detected_faces,
                                      tk_action_vars["lockout"])
        self._display_frame = display_frame
        self._utilities = dict(object_creator=ObjectCreator(self),
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

    def switch_filter(self):
        """ Update the :class:`FacesViewer` canvas for the active filter.
            Executed when the user changes the selected filter pull down.
         """
        if not self._faces_cache.is_initialized:
            return
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
    def set_column_count(self, frame_width):
        """ Set the column count for the displayed canvas. Must be done after
        the canvas has been packed and the scrollbar added.

        Parameters
        ----------
        frame_width: int
            The amount of space that the canvas has available for placing faces
        """
        self._column_count = frame_width // self._faces_cache.size

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
        self._utilities["hover_box"].on_hover(event)

    # << OPTIONAL ANNOTATION METHODS >> #
    def update_mesh_color(self):
        """ Update the mesh color when user updates the control panel. """
        if not self._faces_cache.is_initialized:
            return
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


class Aligner():
    """ Handles the extraction pipeline for retrieving the alignment landmarks. """
    def __init__(self):
        self._aligner = None
        self._det_faces = None
        self._frames = None
        self._frame_index = None
        self._face_index = None
        self._init_thread = self._background_init_aligner()

    @property
    def _in_queue(self):
        """ :class:`queue.Queue` - The input queue to the aligner. """
        return self._aligner.input_queue

    @property
    def _feed_face(self):
        """ :class:`plugins.extract.pipeline.ExtractMedia`: The current face for feeding into the
        aligner, formatted for the pipeline """
        # TODO Try to remove requirement for frames here. Ultimately need a better way to access
        # a frame's image
        return ExtractMedia(
            self._frames.current_meta_data["filename"],
            self._frames.current_frame,
            detected_faces=[self._det_faces.current_faces[self._frame_index][self._face_index]])

    @property
    def is_initialized(self):
        """ bool: ``True`` if the aligner has completed initialization otherwise ``False``. """
        thread_is_alive = self._init_thread.is_alive()
        if thread_is_alive:
            self._init_thread.check_and_raise_error()
        else:
            self._init_thread.join()
        return not thread_is_alive

    def _background_init_aligner(self):
        """ Launch the aligner in a background thread so we can run other tasks whilst
        waiting for initialization """
        thread = MultiThread(self._init_aligner,
                             thread_count=1,
                             name="{}.init_aligner".format(self.__class__.__name__))
        thread.start()
        return thread

    def _init_aligner(self):
        """ Initialize Aligner in a background thread, and set it to :attr:`_aligner`. """
        logger.debug("Initialize Aligner")
        # TODO FAN
        aligner = Extractor(None, "FAN", ["components", "extended"],
                            multiprocess=True, normalize_method="hist")
        aligner.set_batchsize("align", 1)  # Set the batchsize to 1
        aligner.launch()
        logger.debug("Initialized Extractor")
        self._aligner = aligner

    def link_faces_and_frames(self, detected_faces, frames):
        """ Add the :class:`AlignmentsData` object as a property of the aligner.

        Parameters
        ----------
        detected_faces: :class:`~tool.manual.faces.DetectedFaces`
            The :class:`~lib.faces_detect.DetectedFace` objects for this video
        frames: :class:`~tools.lib_manual.media.FrameNavigation`
            The Frame Navigation object for the manual tool
        """
        self._det_faces = detected_faces
        self._frames = frames

    def get_landmarks(self, frame_index, face_index):
        """ Feed the detected face into the alignment pipeline and retrieve the landmarks

        Returns
        -------
        :class:`numpy.ndarray`
            The 68 point landmark alignments
        """
        self._frame_index = frame_index
        self._face_index = face_index
        self._in_queue.put(self._feed_face)
        detected_face = next(self._aligner.detected_faces()).detected_faces[0]
        return detected_face.landmarks_xy

    def set_normalization_method(self, method_var):
        """ Change the normalization method for faces fed into the aligner """
        method = method_var.get()
        self._aligner.set_aligner_normalization_method(method)


class TkGlobals():
    """ Tkinter Variables that need to be accessible from all areas of the GUI """
    def __init__(self):
        self._tk_frame_index = tk.IntVar()
        self._tk_frame_index.set(0)

        self._tk_transport_index = tk.IntVar()
        self._tk_transport_index.set(0)

        self._tk_update = tk.BooleanVar()
        self._tk_update.set(False)

        self._tk_filter_mode = tk.StringVar()

    @property
    def frame_index(self):
        "int: The currently displayed frame index"
        return self._tk_frame_index.get()

    @property
    def tk_frame_index(self):
        """ :class:`tkinter.IntVar`: The variable holding current frame index. """
        return self._tk_frame_index

    @property
    def filter_mode(self):
        """ str: The currently selected navigation mode. """
        return self._tk_filter_mode.get()

    @property
    def tk_filter_mode(self):
        """ :class:`tkinter.StringVar`: The variable holding the currently selected navigation
        filter mode. """
        return self._tk_filter_mode

    @property
    def tk_transport_index(self):
        """ :class:`tkinter.IntVar`: The current index of the display frame's transport slider. """
        return self._tk_transport_index

    @property
    def tk_update(self):
        """ :class:`tkinter.BooleanVar`: The variable holding the trigger that indicates that an
        update needs to occur. """
        return self._tk_update
