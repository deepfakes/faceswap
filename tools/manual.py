#!/usr/bin/env python3
""" Tool to manually interact with the alignments file using visual tools """
import logging
import os
import platform
import sys
import tkinter as tk
from tkinter import ttk
from time import sleep

from lib.gui.control_helper import ControlPanel
from lib.gui.custom_widgets import Tooltip, StatusBar
from lib.gui.utils import get_images, get_config, initialize_config, initialize_images
from lib.multithreading import MultiThread
from lib.utils import _video_extensions
from plugins.extract.pipeline import Extractor, ExtractMedia

from .lib_manual.display_frame import DisplayFrame
from .lib_manual.media import AlignmentsData, FrameNavigation, FaceCache

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
        is_video = self._check_input(arguments.frames)
        self._initialize_tkinter()

        extractor = Aligner()
        self._frames = FrameNavigation(arguments.frames, get_config().scaling_factor)
        self._alignments = AlignmentsData(arguments.alignments_path,
                                          self._frames,
                                          extractor,
                                          arguments.frames,
                                          is_video)

        self._wait_for_threads(extractor)

        self._containers = self._create_containers()

        pbar = StatusBar(self._containers["bottom"], hide_status=True)
        self._faces = FaceCache(self._alignments, pbar)

        self._display = DisplayFrame(self._containers["top"], self._frames, self._alignments)
        self._faces_frame = FacesFrame(self._containers["bottom"], self._faces, self._frames)

        self._options = Options(self._containers["top"], self._display)
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

    def _wait_for_threads(self, extractor):
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
                return
            logger.debug("Threads not initialized. Waiting...")
            sleep(1)

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
            * B, D, E, Z - Optional Actions (Brush, Drag, Erase, Zoom)
        """
        modifiers = {0x0001: 'shift',
                     0x0004: 'ctrl',
                     0x0008: 'alt',
                     0x0080: 'alt'}

        bindings = {
            "left": self._display.decrement_frame,
            "right": self._display.increment_frame,
            "space": self._display.handle_play_button,
            "home": self._display.goto_first_frame,
            "end": self._display.goto_last_frame,
            "f": self._display.cycle_navigation_mode,
            "1": lambda k=event.keysym: self._display.set_action(k),
            "2": lambda k=event.keysym: self._display.set_action(k),
            "3": lambda k=event.keysym: self._display.set_action(k),
            "4": lambda k=event.keysym: self._display.set_action(k),
            "5": lambda k=event.keysym: self._display.set_action(k),
            "c": lambda d="previous": self._alignments.copy_alignments(d),
            "v": lambda d="next": self._alignments.copy_alignments(d),
            "ctrl_s": self._alignments.save,
            "f5": self._alignments.revert_to_saved}

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
    display_frame: :class:`DisplayFrame`
        The frame that holds the editors
    """
    def __init__(self, parent, display_frame):
        super().__init__(parent)
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
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
                ctl.tk_var.trace("w", lambda *e: self._display_frame.tk_update.set(True))

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
    faces: :class:`tools.manual.lib_manual.FaceCache`
        The faces cache that holds the aligned faces
    frames: :class:`FrameNavigation`
        The object that holds the cache of frames.
    """
    def __init__(self, parent, faces, frames):
        logger.debug("Initializing %s: (parent: %s, faces: %s)",
                     self.__class__.__name__, parent, faces)
        super().__init__(parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._faces = faces

        self._actions_frame = FacesActionsFrame(self, self._faces)

        self._faces_frame = ttk.Frame(self)
        self._faces_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas = FacesViewer(self._faces_frame, self._faces, frames)
        scrollbar_width = self._add_scrollbar()
        self._canvas.load_faces(self.winfo_width() - scrollbar_width)

        logger.debug("Initialized %s", self.__class__.__name__)

    def _add_scrollbar(self):
        """ Add a scrollbar to the faces frame """
        logger.debug("Add Config Scrollbar")
        scrollbar = ttk.Scrollbar(self._faces_frame, command=self._canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.config(yscrollcommand=scrollbar.set)
        self.bind("<Configure>", self._update_scrollbar)
        logger.debug("Added Config Scrollbar")
        self.update_idletasks()  # Update so scrollbar width is correct
        return scrollbar.winfo_width()

    def _update_scrollbar(self, event):  # pylint: disable=unused-argument
        """ Update the faces frame scrollbar """
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))


class FacesActionsFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The left hand action frame holding the action buttons.

    Parameters
    ----------
    parent: :class:`FacesFrame`
        The Faces frame that the Actions reside in
    """
    def __init__(self, parent, faces):
        super().__init__(parent)
        self.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 4), pady=2)
        self._faces = faces
        self._selected_display = None
        self._configure_styles()
        self._displays = ("landmarks", "mask")
        self._initial_display = "none"
        self._buttons = self._add_buttons()
        self._optional_buttons = dict()  # Has to be set from parent after canvas is initialized

    @property
    def key_bindings(self):
        """ dict: {`key`: `display`}. The mapping of key presses to displays. Keyboard shortcut is
        the first letter of each display. """
        # TODO Key bindings
        return {"#TODO {}".format(display): display for display in self._displays}

    @property
    def _helptext(self):
        """ dict: `button key`: `button helptext`. The help text to display for each button. """
        inverse_keybindings = {val: key for key, val in self.key_bindings.items()}
        retval = dict(landmarks="Display the landmarks mesh",
                      mask="Display the mask")
        for item in retval:
            retval[item] += " ({})".format(inverse_keybindings[item])
        return retval

    def _configure_styles(self):
        """ Configure background color for Displays widget """
        style = ttk.Style()
        style.configure("display.TFrame", background='#d3d3d3')
        style.configure("display_selected.TButton", relief="flat", background="#bedaf1")
        style.configure("display_deselected.TButton", relief="flat")
        self.config(style="display.TFrame")

    def _add_buttons(self):
        """ Add the display buttons to the Faces window.

        Returns
        -------
        dict:
            The display name and its associated button.
        """
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.Y)
        buttons = dict()
        for display in self.key_bindings.values():
            if display == self._initial_display:
                btn_style = "display_selected.TButton"
                state = (["pressed", "focus"])
            else:
                btn_style = "display_deselected.TButton"
                state = (["!pressed", "!focus"])

            button = ttk.Button(frame,
                                image=get_images().icons[display],
                                command=lambda t=display: self.on_click(t),
                                style=btn_style)
            button.state(state)
            button.pack()
            Tooltip(button, text=self._helptext[display])
            buttons[display] = button
        return buttons

    def on_click(self, display):
        """ Click event for all of the main buttons.

        Parameters
        ----------
        display: str
            The display name for the button that has called this event as exists in
            attr:`_buttons`
        """
        display = None if display == self._selected_display else display
        for title, button in self._buttons.items():
            if display == title:
                button.configure(style="display_selected.TButton")
                button.state(["pressed", "focus"])
            else:
                button.configure(style="display_deselected.TButton")
                button.state(["!pressed", "!focus"])
        self._selected_display = display
        self._faces.update_all(display)


class FacesViewer(tk.Canvas):   # pylint:disable=too-many-ancestors
    """ Annotation onto tkInter Canvas.

    Parameters
    ----------
    parent: :class:`tkinter.ttk.Frame`
        The parent frame for the canvas
    faces: :class:`tools.manual.lib_manual.FaceCache`
        The faces cache that holds the aligned faces
    frames: :class:`FrameNavigation`
        The object that holds the cache of frames.
    """
    def __init__(self, parent, faces, frames):
        logger.debug("Initializing %s: (parent: %s, faces: %s)", self.__class__.__name__, parent,
                     faces)
        super().__init__(parent, bd=0, highlightthickness=0)
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor=tk.E)
        self.parent = parent
        self._faces = faces
        self._frames = frames
        self._bind_mouse()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _bind_mouse(self):
        """ Bind the mouse actions. """
        self.bind("<Motion>", self._update_cursor)
        self.bind("<ButtonPress-1>", self._select_frame)
        self.bind("<Leave>", lambda e: self._faces.clear_hovered())
        if platform.system() == "Linux":
            self.bind("<Button-4>", self._scroll)
            self.bind("<Button-5>", self._scroll)
        else:
            self.bind("<MouseWheel>", self._scroll)

    def load_faces(self, frame_width):
        """ Load the faces into the Faces Canvas in a background thread.

        Parameters
        ----------
        frame_width: int
            The width of the :class:`tkinter.ttk.Frame` that holds this canvas """
        self._faces.load_faces(self, frame_width)

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):  # pylint: disable=unused-argument
        """ The mouse cursor display as bound to the mouses <Motion> event.
        The canvas only displays faces, so if the mouse is over an object change the cursor
        otherwise use default.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Unused for default tracking, but available for specific editor
            tracking.
        """
        item_ids = self.find_withtag("current")
        if not item_ids:
            self._faces.clear_hovered()
            self.config(cursor="")
            return
        object_id = item_ids[0]
        frame_id = self._faces.frame_index_from_object(object_id)
        if frame_id is None or frame_id == self._frames.tk_position.get():
            self.config(cursor="")
            self._faces.clear_hovered()
            return
        self.config(cursor="hand1")
        self._faces.highlight_hovered(frame_id, object_id)

    def _select_frame(self, event):  # pylint: disable=unused-argument
        """ Go to the frame corresponding to the mouse click location in the faces window.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Unused but required.
        """
        item_ids = self.find_withtag("current")
        if not item_ids:
            return
        frame_id = self._faces.frame_index_from_object(item_ids[0])
        if frame_id is None or frame_id == self._frames.tk_position.get():
            return
        transport_id = self._faces.transport_index_from_frame_index(frame_id)
        if transport_id is None:
            return
        self._frames.stop_playback()
        self._frames.tk_transport_position.set(transport_id)

    def _scroll(self, event):
        """ Handle mouse wheel scrolling over the faces canvas """
        # TODO Test Windows + macOS
        if platform.system() == "Darwin":
            adjust = event.delta
        elif platform.system() == "Windows":
            adjust = event.delta / 120
        elif event.num == 5:
            adjust = -1
        else:
            adjust = 1
        self.yview_scroll(int(-1 * adjust), "units")
        self._update_cursor(event)


class Aligner():
    """ Handles the extraction pipeline for retrieving the alignment landmarks

    Parameters
    ----------
    alignments: :class:`Aligner`
        The alignments cache object for the manual tool
    """
    def __init__(self):
        self._alignments = None
        self._aligner = None
        self._init_thread = self._background_init_aligner()

    @property
    def _in_queue(self):
        """ :class:`queue.Queue` - The input queue to the aligner. """
        return self._aligner.input_queue

    @property
    def _feed_face(self):
        """ :class:`plugins.extract.pipeline.ExtractMedia`: The current face for feeding into the
        aligner, formatted for the pipeline """
        return ExtractMedia(self._alignments.frames.current_meta_data["filename"],
                            self._alignments.frames.current_frame,
                            detected_faces=[self._alignments.current_face])

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
        aligner = Extractor(None, "cv2-dnn", None, multiprocess=True, normalize_method="hist")
        aligner.set_batchsize("align", 1)  # Set the batchsize to 1
        aligner.launch()
        logger.debug("Initialized Extractor")
        self._aligner = aligner

    def link_alignments(self, alignments):
        """ Add the :class:`AlignmentsData` object as a property of the aligner.

        Parameters
        ----------
        alignments: :class:`AlignmentsData`
            The alignments cache object for the manual tool
        """
        self._alignments = alignments

    def get_landmarks(self):
        """ Feed the detected face into the alignment pipeline and retrieve the landmarks

        Returns
        -------
        :class:`numpy.ndarray`
            The 68 point landmark alignments
        """
        self._in_queue.put(self._feed_face)
        detected_face = next(self._aligner.detected_faces()).detected_faces[0]
        return detected_face.landmarks_xy

    def set_normalization_method(self, method_var):
        """ Change the normalization method for faces fed into the aligner """
        method = method_var.get()
        self._aligner.set_aligner_normalization_method(method)
