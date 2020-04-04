#!/usr/bin/env python3
""" Tool to manually interact with the alignments file using visual tools """
import logging
import os
import sys
import tkinter as tk
from tkinter import ttk
from time import sleep

from lib.gui.control_helper import ControlPanel
from lib.gui.utils import get_images, get_config, initialize_config, initialize_images
from lib.multithreading import MultiThread
from lib.utils import _video_extensions
from plugins.extract.pipeline import Extractor, ExtractMedia

from .detected_faces import DetectedFaces
from .faceviewer.frame import FacesFrame
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
                                       self._display,
                                       arguments.face_size)

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


class Aligner():
    """ Handles the extraction pipeline for retrieving the alignment landmarks. """
    def __init__(self):
        # TODO
        # self._aligners = {"cv2-dnn": None}
        # self._aligner = "cv2-dnn"
        self._aligners = {"cv2-dnn": None, "FAN": None}
        self._aligner = "FAN"
        self._det_faces = None
        self._frames = None
        self._frame_index = None
        self._face_index = None
        self._init_thread = self._background_init_aligner()

    @property
    def _in_queue(self):
        """ :class:`queue.Queue` - The input queue to the aligner. """
        return self._aligners[self._aligner].input_queue

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
        # Make sure non-GPU aligner is allocated first
        for model in sorted(self._aligners, key=str.casefold):
            aligner = Extractor(None, model, ["components", "extended"],
                                multiprocess=True, normalize_method="hist")
            aligner.set_batchsize("align", 1)  # Set the batchsize to 1
            aligner.launch()
            logger.debug("Initialized %s Extractor", model)
            self._aligners[model] = aligner

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

    def get_landmarks(self, frame_index, face_index, aligner):
        """ Feed the detected face into the alignment pipeline and retrieve the landmarks

        Parameters
        ----------
        frame_index: int
            The frame index to extract the aligned face for
        face_index: int
            The face index within the current frame to extract the face for
        aligner: ["FAN", "cv2-dnn"]
            The aligner to use to extract the face

        Returns
        -------
        :class:`numpy.ndarray`
            The 68 point landmark alignments
        """
        self._frame_index = frame_index
        self._face_index = face_index
        self._aligner = aligner
        self._in_queue.put(self._feed_face)
        detected_face = next(self._aligners[aligner].detected_faces()).detected_faces[0]
        return detected_face.landmarks_xy

    def set_normalization_method(self, method_var):
        """ Change the normalization method for faces fed into the aligner """
        method = method_var.get()
        for aligner in self._aligners.values():
            aligner.set_aligner_normalization_method(method)


class TkGlobals():
    """ Tkinter Variables that need to be accessible from all areas of the GUI """
    def __init__(self):
        self._tk_frame_index = tk.IntVar()
        self._tk_frame_index.set(0)

        self._tk_transport_index = tk.IntVar()
        self._tk_transport_index.set(0)

        self._tk_face_index = tk.IntVar()
        self._tk_face_index.set(0)

        self._tk_update = tk.BooleanVar()
        self._tk_update.set(False)

        self._tk_filter_mode = tk.StringVar()
        self._tk_is_zoomed = tk.BooleanVar()
        self._tk_is_zoomed.set(False)

    @property
    def frame_index(self):
        """ int: The currently displayed frame index """
        return self._tk_frame_index.get()

    @property
    def tk_frame_index(self):
        """ :class:`tkinter.IntVar`: The variable holding current frame index. """
        return self._tk_frame_index

    @property
    def tk_face_index(self):
        """ :class:`tkinter.IntVar`: The variable face index if the selected face when in zoomed
        mode. """
        return self._tk_face_index

    @property
    def face_index(self):
        """ int: The currently displayed face index """
        return self._tk_face_index.get()

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
    def tk_is_zoomed(self):
        """ :class:`tkinter.BooleanVar`: The variable holding the value indicating whether the
        frame viewer is zoomed into a face or zoomed out to the full frame. """
        return self._tk_is_zoomed

    @property
    def is_zoomed(self):
        """ bool: ``True`` if the frame viewer is zoomed into a face, ``False`` if the frame viewer
        is displaying a full frame. """
        return self._tk_is_zoomed.get()

    @property
    def tk_transport_index(self):
        """ :class:`tkinter.IntVar`: The current index of the display frame's transport slider. """
        return self._tk_transport_index

    @property
    def tk_update(self):
        """ :class:`tkinter.BooleanVar`: The variable holding the trigger that indicates that an
        update needs to occur. """
        return self._tk_update
