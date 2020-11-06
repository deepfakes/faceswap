#!/usr/bin/env python3
""" The Manual Tool is a tkinter driven GUI app for editing alignments files with visual tools.
This module is the main entry point into the Manual Tool. """
import logging
import os
import sys
import tkinter as tk
from tkinter import ttk
from time import sleep

import cv2
import numpy as np

from lib.gui.control_helper import ControlPanel
from lib.gui.utils import get_images, get_config, initialize_config, initialize_images
from lib.image import SingleFrameLoader
from lib.multithreading import MultiThread
from lib.utils import _video_extensions
from plugins.extract.pipeline import Extractor, ExtractMedia

from .detected_faces import DetectedFaces, ThumbsCreator
from .faceviewer.frame import FacesFrame
from .frameviewer.frame import DisplayFrame

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Manual(tk.Tk):
    """ The main entry point for Faceswap's Manual Editor Tool. This tool is part of the Faceswap
    Tools suite and should be called from ``python tools.py manual`` command.

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
        self._initialize_tkinter()
        self._globals = TkGlobals(arguments.frames)

        extractor = Aligner(self._globals, arguments.exclude_gpus)
        self._detected_faces = DetectedFaces(self._globals,
                                             arguments.alignments_path,
                                             arguments.frames,
                                             extractor)

        video_meta_data = self._detected_faces.video_meta_data
        loader = FrameLoader(self._globals, arguments.frames, video_meta_data)

        self._detected_faces.load_faces()
        self._containers = self._create_containers()
        self._wait_for_threads(extractor, loader, video_meta_data)
        self._generate_thumbs(arguments.frames, arguments.thumb_regen, arguments.single_process)

        self._display = DisplayFrame(self._containers["top"],
                                     self._globals,
                                     self._detected_faces)
        _Options(self._containers["top"], self._globals, self._display)

        self._faces_frame = FacesFrame(self._containers["bottom"],
                                       self._globals,
                                       self._detected_faces,
                                       self._display)
        self._display.tk_selected_action.set("View")

        self.bind("<Key>", self._handle_key_press)
        self._set_initial_layout()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _wait_for_threads(self, extractor, loader, video_meta_data):
        """ The :class:`Aligner` and :class:`FramesLoader` are launched in background threads.
        Wait for them to be initialized prior to proceeding.

        Parameters
        ----------
        extractor: :class:`Aligner`
            The extraction pipeline for the Manual Tool
        loader: :class:`FramesLoader`
            The frames loader for the Manual Tool
        video_meta_data: dict
            The video meta data that exists within the alignments file

        Notes
        -----
        Because some of the initialize checks perform extra work once their threads are complete,
        they should only return ``True`` once, and should not be queried again.
        """
        extractor_init = False
        frames_init = False
        while True:
            extractor_init = extractor_init if extractor_init else extractor.is_initialized
            frames_init = frames_init if frames_init else loader.is_initialized
            if extractor_init and frames_init:
                logger.debug("Threads inialized")
                break
            logger.debug("Threads not initialized. Waiting...")
            sleep(1)

        extractor.link_faces(self._detected_faces)
        if any(val is None for val in video_meta_data.values()):
            logger.debug("Saving video meta data to alignments file")
            self._detected_faces.save_video_meta_data(**loader.video_meta_data)

    def _generate_thumbs(self, input_location, force, single_process):
        """ Check whether thumbnails are stored in the alignments file and if not generate them.

        Parameters
        ----------
        input_location: str
            The input video or folder of images
        force: bool
            ``True`` if the thumbnails should be regenerated even if they exist, otherwise
            ``False``
        single_process: bool
            ``True`` will extract thumbs from a video in a single process, ``False`` will run
            parallel threads
        """
        thumbs = ThumbsCreator(self._detected_faces, input_location, single_process)
        if thumbs.has_thumbs and not force:
            return
        logger.debug("Generating thumbnails cache")
        thumbs.generate_cache()
        logger.debug("Generated thumbnails cache")

    def _initialize_tkinter(self):
        """ Initialize a standalone tkinter instance. """
        logger.debug("Initializing tkinter")
        for widget in ("TButton", "TCheckbutton", "TRadiobutton"):
            self.unbind_class(widget, "<Key-space>")
        initialize_config(self, None, None)
        initialize_images()
        get_config().set_geometry(940, 600, fullscreen=True)
        self.title("Faceswap.py - Visual Alignments")
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
        retval = dict(main=main, top=top, bottom=bottom)
        logger.debug("Created containers: %s", retval)
        return retval

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
            "z": self._display.navigation.decrement_frame,
            "x": self._display.navigation.increment_frame,
            "space": self._display.navigation.handle_play_button,
            "home": self._display.navigation.goto_first_frame,
            "end": self._display.navigation.goto_last_frame,
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
            "c": lambda f=tk_pos.get(), d="prev": self._detected_faces.update.copy(f, d),
            "v": lambda f=tk_pos.get(), d="next": self._detected_faces.update.copy(f, d),
            "ctrl_s": self._detected_faces.save,
            "r": lambda f=tk_pos.get(): self._detected_faces.revert_to_saved(f)}

        # Allow keypad keys to be used for numbers
        press = event.keysym.replace("KP_", "") if event.keysym.startswith("KP_") else event.keysym
        modifier = "_".join(val for key, val in modifiers.items() if event.state & key != 0)
        key_press = "_".join([modifier, press]) if modifier else press
        if key_press.lower() in bindings:
            logger.trace("key press: %s, action: %s", key_press, bindings[key_press.lower()])
            self.focus_set()
            bindings[key_press.lower()]()

    def _set_initial_layout(self):
        """ Set the favicon and the bottom frame position to correct location to display full
        frame window.

        Notes
        -----
        The favicon pops the tkinter GUI (without loaded elements) as soon as it is called, so
        this is set last.
        """
        logger.debug("Setting initial layout")
        self.tk.call("wm",
                     "iconphoto",
                     self._w, get_images().icons["favicon"])  # pylint:disable=protected-access
        location = int(self.winfo_screenheight() // 1.5)
        self._containers["main"].sash_place(0, 1, location)
        self.update_idletasks()

    def process(self):
        """ The entry point for the Visual Alignments tool from :mod:`lib.tools.manual.cli`.

        Launch the tkinter Visual Alignments Window and run main loop.
        """
        logger.debug("Launching mainloop")
        self.mainloop()


class _Options(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Control panel options for currently displayed Editor. This is the right hand panel of the
    GUI that holds editor specific settings and annotation display settings.

    parent: :class:`tkinter.ttk.Frame`
        The parent frame for the control panel options
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    display_frame: :class:`DisplayFrame`
        The frame that holds the editors
    """
    def __init__(self, parent, tk_globals, display_frame):
        logger.debug("Initializing %s: (parent: %s, tk_globals: %s, display_frame: %s)",
                     self.__class__.__name__, parent, tk_globals, display_frame)
        super().__init__(parent)

        self._globals = tk_globals
        self._display_frame = display_frame
        self._control_panels = self._initialize()
        self._set_tk_callbacks()
        self._update_options()
        self.pack(side=tk.RIGHT, fill=tk.Y)
        logger.debug("Initialized %s", self.__class__.__name__)

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
        self._initialize_face_options()
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        panels = dict()
        for name, editor in self._display_frame.editors.items():
            logger.debug("Initializing control panel for '%s' editor", name)
            controls = editor.controls
            panel = ControlPanel(frame, controls["controls"],
                                 option_columns=2,
                                 columns=1,
                                 max_columns=1,
                                 header_text=controls["header"],
                                 blank_nones=False,
                                 label_width=18,
                                 scrollbar=False)
            panel.pack_forget()
            panels[name] = panel
        return panels

    def _initialize_face_options(self):
        """ Set the Face Viewer options panel, beneath the standard control options. """
        frame = ttk.Frame(self)
        frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        size_frame = ttk.Frame(frame)
        size_frame.pack(side=tk.RIGHT)
        lbl = ttk.Label(size_frame, text="Face Size:")
        lbl.pack(side=tk.LEFT)
        cmb = ttk.Combobox(size_frame,
                           value=["Tiny", "Small", "Medium", "Large", "Extra Large"],
                           state="readonly",
                           textvariable=self._globals.tk_faces_size)
        self._globals.tk_faces_size.set("Medium")
        cmb.pack(side=tk.RIGHT, padx=5)

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


class TkGlobals():
    """ Holds Tkinter Variables and other frame information that need to be accessible from all
    areas of the GUI.

    Parameters
    ----------
    input_location: str
        The location of the input folder of frames or video file
    """
    def __init__(self, input_location):
        logger.debug("Initializing %s: (input_location: %s)",
                     self.__class__.__name__, input_location)
        self._tk_vars = self._get_tk_vars()

        self._is_video = self._check_input(input_location)
        self._frame_count = 0  # set by FrameLoader
        self._frame_display_dims = (int(round(896 * get_config().scaling_factor)),
                                    int(round(504 * get_config().scaling_factor)))
        self._current_frame = dict(image=None,
                                   scale=None,
                                   interpolation=None,
                                   display_dims=None,
                                   filename=None)
        logger.debug("Initialized %s", self.__class__.__name__)

    @classmethod
    def _get_tk_vars(cls):
        """ Create and initialize the tkinter variables.

        Returns
        -------
        dict
            The variable name as key, the variable as value
        """
        retval = dict()
        for name in ("frame_index", "transport_index", "face_index"):
            var = tk.IntVar()
            var.set(0)
            retval[name] = var
        for name in ("update", "update_active_viewport", "is_zoomed"):
            var = tk.BooleanVar()
            var.set(False)
            retval[name] = var
        for name in ("filter_mode", "faces_size"):
            retval[name] = tk.StringVar()
        return retval

    @property
    def current_frame(self):
        """ dict: The currently displayed frame in the frame viewer with it's meta information. Key
        and Values are as follows:

            **image** (:class:`numpy.ndarry`): The currently displayed frame in original dimensions

            **scale** (`float`): The scaling factor to use to resize the image to the display
            window

            **interpolation** (`int`): The opencv interpolator ID to use for resizing the image to
            the display window

            **display_dims** (`tuple`): The size of the currently displayed frame, sized for the
            display window

            **filename** (`str`): The filename of the currently displayed frame
        """
        return self._current_frame

    @property
    def frame_count(self):
        """ int: The total number of frames for the input location """
        return self._frame_count

    @property
    def tk_face_index(self):
        """ :class:`tkinter.IntVar`: The variable that holds the face index of the selected face
        within the current frame when in zoomed mode. """
        return self._tk_vars["face_index"]

    @property
    def tk_update_active_viewport(self):
        """ :class:`tkinter.BooleanVar`: Boolean Variable that is traced by the viewport's active
        frame to update.. """
        return self._tk_vars["update_active_viewport"]

    @property
    def face_index(self):
        """ int: The currently displayed face index when in zoomed mode. """
        return self._tk_vars["face_index"].get()

    @property
    def frame_display_dims(self):
        """ tuple: The (`width`, `height`) of the video display frame in pixels. """
        return self._frame_display_dims

    @property
    def frame_index(self):
        """ int: The currently displayed frame index. NB This returns -1 if there are no frames
        that meet the currently selected filter criteria. """
        return self._tk_vars["frame_index"].get()

    @property
    def tk_frame_index(self):
        """ :class:`tkinter.IntVar`: The variable holding the current frame index. """
        return self._tk_vars["frame_index"]

    @property
    def filter_mode(self):
        """ str: The currently selected navigation mode. """
        return self._tk_vars["filter_mode"].get()

    @property
    def tk_filter_mode(self):
        """ :class:`tkinter.StringVar`: The variable holding the currently selected navigation
        filter mode. """
        return self._tk_vars["filter_mode"]

    @property
    def tk_faces_size(self):
        """ :class:`tkinter.StringVar`: The variable holding the currently selected Faces Viewer
        thumbnail size. """
        return self._tk_vars["faces_size"]

    @property
    def is_video(self):
        """ bool: ``True`` if the input is a video file, ``False`` if it is a folder of images. """
        return self._is_video

    @property
    def tk_is_zoomed(self):
        """ :class:`tkinter.BooleanVar`: The variable holding the value indicating whether the
        frame viewer is zoomed into a face or zoomed out to the full frame. """
        return self._tk_vars["is_zoomed"]

    @property
    def is_zoomed(self):
        """ bool: ``True`` if the frame viewer is zoomed into a face, ``False`` if the frame viewer
        is displaying a full frame. """
        return self._tk_vars["is_zoomed"].get()

    @property
    def tk_transport_index(self):
        """ :class:`tkinter.IntVar`: The current index of the display frame's transport slider. """
        return self._tk_vars["transport_index"]

    @property
    def tk_update(self):
        """ :class:`tkinter.BooleanVar`: The variable holding the trigger that indicates that a
        full update needs to occur. """
        return self._tk_vars["update"]

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

    def set_frame_count(self, count):
        """ Set the count of total number of frames to :attr:`frame_count` when the
        :class:`FramesLoader` has completed loading.

        Parameters
        ----------
        count: int
            The number of frames that exist for this session
        """
        logger.debug("Setting frame_count to : %s", count)
        self._frame_count = count

    def set_current_frame(self, image, filename):
        """ Set the frame and meta information for the currently displayed frame. Populates the
        attribute :attr:`current_frame`

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            The image used to display in the Frame Viewer
        filename: str
            The filename of the current frame
        """
        scale = min(self.frame_display_dims[0] / image.shape[1],
                    self.frame_display_dims[1] / image.shape[0])
        self._current_frame["image"] = image
        self._current_frame["filename"] = filename
        self._current_frame["scale"] = scale
        self._current_frame["interpolation"] = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        self._current_frame["display_dims"] = (int(round(image.shape[1] * scale)),
                                               int(round(image.shape[0] * scale)))
        logger.trace({k: v.shape if isinstance(v, np.ndarray) else v
                      for k, v in self._current_frame.items()})

    def set_frame_display_dims(self, width, height):
        """ Set the size, in pixels, of the video frame display window and resize the displayed
        frame.

        Used on a frame resize callback, sets the :attr:frame_display_dims`.

        Parameters
        ----------
        width: int
            The width of the frame holding the video canvas in pixels
        height: int
            The height of the frame holding the video canvas in pixels
        """
        self._frame_display_dims = (int(width), int(height))
        image = self._current_frame["image"]
        scale = min(self.frame_display_dims[0] / image.shape[1],
                    self.frame_display_dims[1] / image.shape[0])
        self._current_frame["scale"] = scale
        self._current_frame["interpolation"] = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        self._current_frame["display_dims"] = (int(round(image.shape[1] * scale)),
                                               int(round(image.shape[0] * scale)))
        logger.trace({k: v.shape if isinstance(v, np.ndarray) else v
                      for k, v in self._current_frame.items()})


class Aligner():
    """ The :class:`Aligner` class sets up an extraction pipeline for each of the current Faceswap
    Aligners, along with the Landmarks based Maskers. When new landmarks are required, the bounding
    boxes from the GUI are passed to this class for pushing through the pipeline. The resulting
    Landmarks and Masks are then returned.

    Parameters
    ----------
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    exclude_gpus: list or ``None``
        A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
        ``None`` to not exclude any GPUs.
    """
    def __init__(self, tk_globals, exclude_gpus):
        logger.debug("Initializing: %s (tk_globals: %s, exclude_gpus: %s)",
                     self.__class__.__name__, tk_globals, exclude_gpus)
        self._globals = tk_globals
        self._aligners = {"cv2-dnn": None, "FAN": None, "mask": None}
        self._aligner = "FAN"
        self._exclude_gpus = exclude_gpus
        self._detected_faces = None
        self._frame_index = None
        self._face_index = None
        self._init_thread = self._background_init_aligner()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _in_queue(self):
        """ :class:`queue.Queue` - The input queue to the extraction pipeline. """
        return self._aligners[self._aligner].input_queue

    @property
    def _feed_face(self):
        """ :class:`plugins.extract.pipeline.ExtractMedia`: The current face for feeding into the
        aligner, formatted for the pipeline """
        face = self._detected_faces.current_faces[self._frame_index][self._face_index]
        return ExtractMedia(
            self._globals.current_frame["filename"],
            self._globals.current_frame["image"],
            detected_faces=[face])

    @property
    def is_initialized(self):
        """ bool: The Aligners are initialized in a background thread so that other tasks can be
        performed whilst we wait for initialization. ``True`` is returned if the aligner has
        completed initialization otherwise ``False``."""
        thread_is_alive = self._init_thread.is_alive()
        if thread_is_alive:
            logger.trace("Aligner not yet initialized")
            self._init_thread.check_and_raise_error()
        else:
            logger.trace("Aligner initialized")
            self._init_thread.join()
        return not thread_is_alive

    def _background_init_aligner(self):
        """ Launch the aligner in a background thread so we can run other tasks whilst
        waiting for initialization """
        logger.debug("Launching aligner initialization thread")
        thread = MultiThread(self._init_aligner,
                             thread_count=1,
                             name="{}.init_aligner".format(self.__class__.__name__))
        thread.start()
        logger.debug("Launched aligner initialization thread")
        return thread

    def _init_aligner(self):
        """ Initialize Aligner in a background thread, and set it to :attr:`_aligner`. """
        logger.debug("Initialize Aligner")
        # Make sure non-GPU aligner is allocated first
        for model in ("mask", "cv2-dnn", "FAN"):
            logger.debug("Initializing aligner: %s", model)
            plugin = None if model == "mask" else model
            exclude_gpus = self._exclude_gpus if model == "FAN" else None
            aligner = Extractor(None,
                                plugin,
                                ["components", "extended"],
                                exclude_gpus=exclude_gpus,
                                multiprocess=True,
                                normalize_method="hist")
            if plugin:
                aligner.set_batchsize("align", 1)  # Set the batchsize to 1
            aligner.launch()
            logger.debug("Initialized %s Extractor", model)
            self._aligners[model] = aligner

    def link_faces(self, detected_faces):
        """ As the Aligner has the potential to take the longest to initialize, it is kicked off
        as early as possible. At this time :class:`~tools.manual.detected_faces.DetectedFaces` is
        not yet available.

        Once the Aligner has initialized, this function is called to add the
        :class:`~tools.manual.detected_faces.DetectedFaces` class as a property of the Aligner.

        Parameters
        ----------
        detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
            The class that holds the :class:`~lib.faces_detect.DetectedFace` objects for the
            current Manual session
        """
        logger.debug("Linking detected_faces: %s", detected_faces)
        self._detected_faces = detected_faces

    def get_landmarks(self, frame_index, face_index, aligner):
        """ Feed the detected face into the alignment pipeline and retrieve the landmarks.

        The face to feed into the aligner is generated from the given frame and face indices.

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
        logger.trace("frame_index: %s, face_index: %s, aligner: %s",
                     frame_index, face_index, aligner)
        self._frame_index = frame_index
        self._face_index = face_index
        self._aligner = aligner
        self._in_queue.put(self._feed_face)
        detected_face = next(self._aligners[aligner].detected_faces()).detected_faces[0]
        logger.trace("landmarks: %s", detected_face.landmarks_xy)
        return detected_face.landmarks_xy

    def get_masks(self, frame_index, face_index):
        """ Feed the aligned face into the mask pipeline and retrieve the updated masks.

        The face to feed into the aligner is generated from the given frame and face indices.
        This is to be called when a manual update is done on the landmarks, and new masks need
        generating

        Parameters
        ----------
        frame_index: int
            The frame index to extract the aligned face for
        face_index: int
            The face index within the current frame to extract the face for

        Returns
        -------
        dict
            The updated masks
        """
        logger.trace("frame_index: %s, face_index: %s", frame_index, face_index)
        self._frame_index = frame_index
        self._face_index = face_index
        self._aligner = "mask"
        self._in_queue.put(self._feed_face)
        detected_face = next(self._aligners["mask"].detected_faces()).detected_faces[0]
        logger.debug("mask: %s", detected_face.mask)
        return detected_face.mask

    def set_normalization_method(self, method):
        """ Change the normalization method for faces fed into the aligner.
        The normalization method is user adjustable from the GUI. When this method is triggered
        the method is updated for all aligner pipelines.

        Parameters
        ----------
        method: str
            The normalization method to use
        """
        logger.debug("Setting normalization method to: '%s'", method)
        for plugin, aligner in self._aligners.items():
            if plugin == "mask":
                continue
            aligner.set_aligner_normalization_method(method)


class FrameLoader():
    """ Loads the frames, sets the frame count to :attr:`TkGlobals.frame_count` and handles the
    return of the correct frame for the GUI.

    Parameters
    ----------
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    frames_location: str
        The path to the input frames
    video_meta_data: dict
        The meta data held within the alignments file, if it exists and the input is a video
    """
    def __init__(self, tk_globals, frames_location, video_meta_data):
        logger.debug("Initializing %s: (tk_globals: %s, frames_location: '%s', "
                     "video_meta_data: %s)", self.__class__.__name__, tk_globals, frames_location,
                     video_meta_data)
        self._globals = tk_globals
        self._loader = None
        self._current_idx = 0
        self._init_thread = self._background_init_frames(frames_location, video_meta_data)
        self._globals.tk_frame_index.trace("w", self._set_frame)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_initialized(self):
        """ bool: ``True`` if the Frame Loader has completed initialization otherwise
        ``False``. """
        thread_is_alive = self._init_thread.is_alive()
        if thread_is_alive:
            self._init_thread.check_and_raise_error()
        else:
            self._init_thread.join()
            # Setting the initial frame cannot be done in the thread, so set when queried from main
            self._set_frame(initialize=True)
        return not thread_is_alive

    @property
    def video_meta_data(self):
        """ dict: The pts_time and key frames for the loader. """
        return self._loader.video_meta_data

    def _background_init_frames(self, frames_location, video_meta_data):
        """ Launch the images loader in a background thread so we can run other tasks whilst
        waiting for initialization. """
        thread = MultiThread(self._load_images,
                             frames_location,
                             video_meta_data,
                             thread_count=1,
                             name="{}.init_frames".format(self.__class__.__name__))
        thread.start()
        return thread

    def _load_images(self, frames_location, video_meta_data):
        """ Load the images in a background thread. """
        self._loader = SingleFrameLoader(frames_location, video_meta_data=video_meta_data)
        self._globals.set_frame_count(self._loader.count)

    def _set_frame(self, *args, initialize=False):  # pylint:disable=unused-argument
        """ Set the currently loaded frame to :attr:`_current_frame` and trigger a full GUI update.

        If the loader has not been initialized, or the navigation position is the same as the
        current position and the face is not zoomed in, then this returns having done nothing.

        Parameters
        ----------
        args: tuple
            :class:`tkinter.Event` arguments. Required but not used.
        initialize: bool, optional
            ``True`` if initializing for the first frame to be displayed otherwise ``False``.
            Default: ``False``
        """
        position = self._globals.frame_index
        if not initialize and (position == self._current_idx and not self._globals.is_zoomed):
            logger.trace("Update criteria not met. Not updating: (initialize: %s, position: %s, "
                         "current_idx: %s, is_zoomed: %s)", initialize, position,
                         self._current_idx, self._globals.is_zoomed)
            return
        if position == -1:
            filename = "No Frame"
            frame = np.ones(self._globals.frame_display_dims + (3, ), dtype="uint8")
        else:
            filename, frame = self._loader.image_from_index(position)
        logger.trace("filename: %s, frame: %s, position: %s", filename, frame.shape, position)
        self._globals.set_current_frame(frame, filename)
        self._current_idx = position
        self._globals.tk_update.set(True)
        self._globals.tk_update_active_viewport.set(True)
