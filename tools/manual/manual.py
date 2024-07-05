#!/usr/bin/env python3
""" Main entry point for the Manual Tool. A GUI app for editing alignments files """
from __future__ import annotations

import logging
import os
import sys
import typing as T
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from time import sleep

import numpy as np

from lib.gui.control_helper import ControlPanel
from lib.gui.utils import get_images, get_config, initialize_config, initialize_images
from lib.image import SingleFrameLoader, read_image_meta
from lib.logger import parse_class_init
from lib.multithreading import MultiThread
from lib.utils import handle_deprecated_cliopts
from plugins.extract import ExtractMedia, Extractor

from .detected_faces import DetectedFaces
from .faceviewer.frame import FacesFrame
from .frameviewer.frame import DisplayFrame
from .globals import TkGlobals
from .thumbnails import ThumbsCreator

if T.TYPE_CHECKING:
    from argparse import Namespace
    from lib.align import DetectedFace, Mask
    from lib.queue_manager import EventQueue

logger = logging.getLogger(__name__)

TypeManualExtractor = T.Literal["FAN", "cv2-dnn", "mask"]


@dataclass
class _Containers:
    """ Dataclass for holding the main area containers in the GUI """
    main: ttk.PanedWindow
    """:class:`tkinter.ttk.PanedWindow`: The main window holding the full GUI """
    top: ttk.Frame
    """:class:`tkinter.ttk.Frame: The top part (frame viewer) of the GUI"""
    bottom: ttk.Frame
    """:class:`tkinter.ttk.Frame: The bottom part (face viewer) of the GUI"""


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

    def __init__(self, arguments: Namespace) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__()
        arguments = handle_deprecated_cliopts(arguments)
        self._validate_non_faces(arguments.frames)

        self._initialize_tkinter()
        self._globals = TkGlobals(arguments.frames)

        extractor = Aligner(self._globals, arguments.exclude_gpus)
        self._detected_faces = DetectedFaces(self._globals,
                                             arguments.alignments_path,
                                             arguments.frames,
                                             extractor)

        video_meta_data = self._detected_faces.video_meta_data
        valid_meta = all(val is not None for val in video_meta_data.values())

        loader = FrameLoader(self._globals,
                             arguments.frames,
                             video_meta_data,
                             self._detected_faces.frame_list)

        if valid_meta:  # Load the faces whilst other threads complete if we have valid meta data
            self._detected_faces.load_faces()

        self._containers = self._create_containers()
        self._wait_for_threads(extractor, loader, valid_meta)
        if not valid_meta:  # If meta data needs updating, load faces after other threads
            self._detected_faces.load_faces()

        self._generate_thumbs(arguments.frames, arguments.thumb_regen, arguments.single_process)

        self._display = DisplayFrame(self._containers.top,
                                     self._globals,
                                     self._detected_faces)
        _Options(self._containers.top, self._globals, self._display)

        self._faces_frame = FacesFrame(self._containers.bottom,
                                       self._globals,
                                       self._detected_faces,
                                       self._display)
        self._display.tk_selected_action.set("View")

        self.bind("<Key>", self._handle_key_press)
        self._set_initial_layout()
        logger.debug("Initialized %s", self.__class__.__name__)

    @classmethod
    def _validate_non_faces(cls, frames_folder: str) -> None:
        """ Quick check on the input to make sure that a folder of extracted faces is not being
        passed in. """
        if not os.path.isdir(frames_folder):
            logger.debug("Input '%s' is not a folder", frames_folder)
            return
        test_file = next((fname
                          for fname in os.listdir(frames_folder)
                          if os.path.splitext(fname)[-1].lower() == ".png"),
                         None)
        if not test_file:
            logger.debug("Input '%s' does not contain any .pngs", frames_folder)
            return
        test_file = os.path.join(frames_folder, test_file)
        meta = read_image_meta(test_file)
        logger.debug("Test file: (filename: %s, metadata: %s)", test_file, meta)
        if "itxt" in meta and "alignments" in meta["itxt"]:
            logger.error("The input folder '%s' contains extracted faces.", frames_folder)
            logger.error("The Manual Tool works with source frames or a video file, not extracted "
                         "faces. Please update your input.")
            sys.exit(1)
        logger.debug("Test input file '%s' does not contain Faceswap header data", test_file)

    def _wait_for_threads(self, extractor: Aligner, loader: FrameLoader, valid_meta: bool) -> None:
        """ The :class:`Aligner` and :class:`FramesLoader` are launched in background threads.
        Wait for them to be initialized prior to proceeding.

        Parameters
        ----------
        extractor: :class:`Aligner`
            The extraction pipeline for the Manual Tool
        loader: :class:`FramesLoader`
            The frames loader for the Manual Tool
        valid_meta: bool
            Whether the input video had valid meta-data on import, or if it had to be created.
            ``True`` if valid meta data existed previously, ``False`` if it needed to be created

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
        if not valid_meta:
            logger.debug("Saving video meta data to alignments file")
            self._detected_faces.save_video_meta_data(
                **loader.video_meta_data)  # type:ignore[arg-type]

    def _generate_thumbs(self, input_location: str, force: bool, single_process: bool) -> None:
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

    def _initialize_tkinter(self) -> None:
        """ Initialize a standalone tkinter instance. """
        logger.debug("Initializing tkinter")
        for widget in ("TButton", "TCheckbutton", "TRadiobutton"):
            self.unbind_class(widget, "<Key-space>")
        initialize_config(self, None, None)
        initialize_images()
        get_config().set_geometry(940, 600, fullscreen=True)
        self.title("Faceswap.py - Visual Alignments")
        logger.debug("Initialized tkinter")

    def _create_containers(self) -> _Containers:
        """ Create the paned window containers for various GUI elements

        Returns
        -------
        :class:`_Containers`:
            The main containers of the manual tool.
        """
        logger.debug("Creating containers")

        main = ttk.PanedWindow(self,
                               orient=tk.VERTICAL,
                               name="pw_main")
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main, name="frame_top")
        main.add(top)

        bottom = ttk.Frame(main, name="frame_bottom")
        main.add(bottom)

        retval = _Containers(main=main, top=top, bottom=bottom)

        logger.debug("Created containers: %s", retval)
        return retval

    def _handle_key_press(self, event: tk.Event) -> None:
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

        globs = self._globals
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
            "c": lambda f=globs.frame_index, d="prev": self._detected_faces.update.copy(f, d),
            "v": lambda f=globs.frame_index, d="next": self._detected_faces.update.copy(f, d),
            "ctrl_s": self._detected_faces.save,
            "r": lambda f=globs.frame_index: self._detected_faces.revert_to_saved(f)}

        # Allow keypad keys to be used for numbers
        press = event.keysym.replace("KP_", "") if event.keysym.startswith("KP_") else event.keysym
        assert isinstance(event.state, int)
        modifier = "_".join(val for key, val in modifiers.items() if event.state & key != 0)
        key_press = "_".join([modifier, press]) if modifier else press
        if key_press.lower() in bindings:
            logger.trace("key press: %s, action: %s",  # type:ignore[attr-defined]
                         key_press, bindings[key_press.lower()])
            self.focus_set()
            bindings[key_press.lower()]()

    def _set_initial_layout(self) -> None:
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
                     self._w,  # type:ignore[attr-defined] # pylint:disable=protected-access
                     get_images().icons["favicon"])
        location = int(self.winfo_screenheight() // 1.5)
        self._containers.main.sashpos(0, location)
        self.update_idletasks()

    def process(self) -> None:
        """ The entry point for the Visual Alignments tool from :mod:`lib.tools.manual.cli`.

        Launch the tkinter Visual Alignments Window and run main loop.
        """
        logger.debug("Launching mainloop")
        self.mainloop()


class _Options(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Control panel options for currently displayed Editor. This is the right hand panel of the
    GUI that holds editor specific settings and annotation display settings.

    Parameters
    ----------
    parent: :class:`tkinter.ttk.Frame`
        The parent frame for the control panel options
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    display_frame: :class:`DisplayFrame`
        The frame that holds the editors
    """
    def __init__(self,
                 parent: ttk.Frame,
                 tk_globals: TkGlobals,
                 display_frame: DisplayFrame) -> None:
        logger.debug(parse_class_init(locals()))
        super().__init__(parent)

        self._globals = tk_globals
        self._display_frame = display_frame
        self._control_panels = self._initialize()
        self._set_tk_callbacks()
        self._update_options()
        self.pack(side=tk.RIGHT, fill=tk.Y)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _initialize(self) -> dict[str, ControlPanel]:
        """ Initialize all of the control panels, then display the default panel.

        Adds the control panel to :attr:`_control_panels` and sets the traceback to update
        display when a panel option has been changed.

        Notes
        -----
        All panels must be initialized at the beginning so that the global format options are not
        reset to default when the editor is first selected.

        The Traceback must be set after the panel has first been packed as otherwise it interferes
        with the loading of the faces pane.

        Returns
        -------
        dict[str, :class:`~lib.gui.control_helper.ControlPanel`]
            The configured control panels
        """
        self._initialize_face_options()
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        panels = {}
        for name, editor in self._display_frame.editors.items():
            logger.debug("Initializing control panel for '%s' editor", name)
            controls = editor.controls
            panel = ControlPanel(frame, controls["controls"],
                                 option_columns=2,
                                 columns=1,
                                 max_columns=1,
                                 header_text=controls["header"],
                                 blank_nones=False,
                                 label_width=12,
                                 style="CPanel",
                                 scrollbar=False)
            panel.pack_forget()
            panels[name] = panel
        return panels

    def _initialize_face_options(self) -> None:
        """ Set the Face Viewer options panel, beneath the standard control options. """
        frame = ttk.Frame(self)
        frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        size_frame = ttk.Frame(frame)
        size_frame.pack(side=tk.RIGHT)
        lbl = ttk.Label(size_frame, text="Face Size:")
        lbl.pack(side=tk.LEFT)
        cmb = ttk.Combobox(size_frame,
                           values=["Tiny", "Small", "Medium", "Large", "Extra Large"],
                           state="readonly",
                           textvariable=self._globals.var_faces_size)
        self._globals.var_faces_size.set("Medium")
        cmb.pack(side=tk.RIGHT, padx=5)

    def _set_tk_callbacks(self) -> None:
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
                ctl.tk_var.trace("w", lambda *e: self._globals.var_full_update.set(True))

    def _update_options(self, *args) -> None:  # pylint:disable=unused-argument
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

    def _clear_options_frame(self) -> None:
        """ Hides the currently displayed control panel """
        for editor, panel in self._control_panels.items():
            if panel.winfo_ismapped():
                logger.debug("Hiding control panel for: %s", editor)
                panel.pack_forget()


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
    def __init__(self, tk_globals: TkGlobals, exclude_gpus: list[int] | None) -> None:
        logger.debug("Initializing: %s (tk_globals: %s, exclude_gpus: %s)",
                     self.__class__.__name__, tk_globals, exclude_gpus)
        self._globals = tk_globals
        self._exclude_gpus = exclude_gpus

        self._detected_faces: DetectedFaces | None = None
        self._frame_index: int | None = None
        self._face_index: int | None = None

        self._aligners: dict[TypeManualExtractor, Extractor | None] = {"cv2-dnn": None,
                                                                       "FAN": None,
                                                                       "mask": None}
        self._aligner: TypeManualExtractor = "FAN"

        self._init_thread = self._background_init_aligner()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _in_queue(self) -> EventQueue:
        """ :class:`queue.Queue` - The input queue to the extraction pipeline. """
        aligner = self._aligners[self._aligner]
        assert aligner is not None
        return aligner.input_queue

    @property
    def _feed_face(self) -> ExtractMedia:
        """ :class:`~plugins.extract.extract_media.ExtractMedia`: The current face for feeding into
        the aligner, formatted for the pipeline """
        assert self._frame_index is not None
        assert self._face_index is not None
        assert self._detected_faces is not None
        face = self._detected_faces.current_faces[self._frame_index][self._face_index]
        return ExtractMedia(
            self._globals.current_frame.filename,
            self._globals.current_frame.image,
            detected_faces=[face])

    @property
    def is_initialized(self) -> bool:
        """ bool: The Aligners are initialized in a background thread so that other tasks can be
        performed whilst we wait for initialization. ``True`` is returned if the aligner has
        completed initialization otherwise ``False``."""
        thread_is_alive = self._init_thread.is_alive()
        if thread_is_alive:
            logger.trace("Aligner not yet initialized")  # type:ignore[attr-defined]
            self._init_thread.check_and_raise_error()
        else:
            logger.trace("Aligner initialized")  # type:ignore[attr-defined]
            self._init_thread.join()
        return not thread_is_alive

    def _background_init_aligner(self) -> MultiThread:
        """ Launch the aligner in a background thread so we can run other tasks whilst
        waiting for initialization

        Returns
        -------
        :class:`lib.multithreading.MultiThread
            The background aligner loader thread
        """
        logger.debug("Launching aligner initialization thread")
        thread = MultiThread(self._init_aligner,
                             thread_count=1,
                             name=f"{self.__class__.__name__}.init_aligner")
        thread.start()
        logger.debug("Launched aligner initialization thread")
        return thread

    def _init_aligner(self) -> None:
        """ Initialize Aligner in a background thread, and set it to :attr:`_aligner`. """
        logger.debug("Initialize Aligner")
        # Make sure non-GPU aligner is allocated first
        for model in T.get_args(TypeManualExtractor):
            logger.debug("Initializing aligner: %s", model)
            plugin = None if model == "mask" else model
            exclude_gpus = self._exclude_gpus if model == "FAN" else None
            aligner = Extractor(None,
                                plugin,
                                ["components", "extended"],
                                exclude_gpus=exclude_gpus,
                                multiprocess=True,
                                normalize_method="hist",
                                disable_filter=True)
            if plugin:
                aligner.set_batchsize("align", 1)  # Set the batchsize to 1
            aligner.launch()
            logger.debug("Initialized %s Extractor", model)
            self._aligners[model] = aligner

    def link_faces(self, detected_faces: DetectedFaces) -> None:
        """ As the Aligner has the potential to take the longest to initialize, it is kicked off
        as early as possible. At this time :class:`~tools.manual.detected_faces.DetectedFaces` is
        not yet available.

        Once the Aligner has initialized, this function is called to add the
        :class:`~tools.manual.detected_faces.DetectedFaces` class as a property of the Aligner.

        Parameters
        ----------
        detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
            The class that holds the :class:`~lib.align.DetectedFace` objects for the
            current Manual session
        """
        logger.debug("Linking detected_faces: %s", detected_faces)
        self._detected_faces = detected_faces

    def get_landmarks(self, frame_index: int, face_index: int, aligner: TypeManualExtractor
                      ) -> np.ndarray:
        """ Feed the detected face into the alignment pipeline and retrieve the landmarks.

        The face to feed into the aligner is generated from the given frame and face indices.

        Parameters
        ----------
        frame_index: int
            The frame index to extract the aligned face for
        face_index: int
            The face index within the current frame to extract the face for
        aligner: Literal["FAN", "cv2-dnn"]
            The aligner to use to extract the face

        Returns
        -------
        :class:`numpy.ndarray`
            The 68 point landmark alignments
        """
        logger.trace("frame_index: %s, face_index: %s, aligner: %s",  # type:ignore[attr-defined]
                     frame_index, face_index, aligner)
        self._frame_index = frame_index
        self._face_index = face_index
        self._aligner = aligner
        self._in_queue.put(self._feed_face)
        extractor = self._aligners[aligner]
        assert extractor is not None
        detected_face = next(extractor.detected_faces()).detected_faces[0]
        logger.trace("landmarks: %s", detected_face.landmarks_xy)  # type:ignore[attr-defined]
        return detected_face.landmarks_xy

    def _remove_nn_masks(self, detected_face: DetectedFace) -> None:
        """ Remove any non-landmarks based masks on a landmark edit

        Parameters
        ----------
        detected_face:
            The detected face object to remove masks from
        """
        del_masks = {m for m in detected_face.mask if m not in ("components", "extended")}
        logger.debug("Removing masks after landmark update: %s", del_masks)
        for mask in del_masks:
            del detected_face.mask[mask]

    def get_masks(self, frame_index: int, face_index: int) -> dict[str, Mask]:
        """ Feed the aligned face into the mask pipeline and retrieve the updated masks.

        The face to feed into the aligner is generated from the given frame and face indices.
        This is to be called when a manual update is done on the landmarks, and new masks need
        generating.

        Parameters
        ----------
        frame_index: int
            The frame index to extract the aligned face for
        face_index: int
            The face index within the current frame to extract the face for

        Returns
        -------
        dict[str, :class:`~lib.align.aligned_mask.Mask`]
            The updated masks
        """
        logger.trace("frame_index: %s, face_index: %s",  # type:ignore[attr-defined]
                     frame_index, face_index)
        self._frame_index = frame_index
        self._face_index = face_index
        self._aligner = "mask"
        self._in_queue.put(self._feed_face)
        assert self._aligners["mask"] is not None
        detected_face = next(self._aligners["mask"].detected_faces()).detected_faces[0]
        self._remove_nn_masks(detected_face)
        logger.debug("mask: %s", detected_face.mask)
        return detected_face.mask

    def set_normalization_method(self, method: T.Literal["none", "clahe", "hist", "mean"]) -> None:
        """ Change the normalization method for faces fed into the aligner.
        The normalization method is user adjustable from the GUI. When this method is triggered
        the method is updated for all aligner pipelines.

        Parameters
        ----------
        method: Literal["none", "clahe", "hist", "mean"]
            The normalization method to use
        """
        logger.debug("Setting normalization method to: '%s'", method)
        for plugin, aligner in self._aligners.items():
            assert aligner is not None
            if plugin == "mask":
                continue
            logger.debug("Setting to: '%s'", method)
            aligner.aligner.set_normalize_method(method)


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
    file_list: list[str]
        The list of filenames that exist within the alignments file
    """
    def __init__(self,
                 tk_globals: TkGlobals,
                 frames_location: str,
                 video_meta_data: dict[str, list[int] | list[float] | None],
                 file_list: list[str]) -> None:
        logger.debug(parse_class_init(locals()))
        self._globals = tk_globals
        self._loader: SingleFrameLoader | None = None
        self._current_idx = 0
        self._init_thread = self._background_init_frames(frames_location,
                                                         video_meta_data,
                                                         file_list)
        self._globals.var_frame_index.trace_add("write", self._set_frame)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_initialized(self) -> bool:
        """ bool: ``True`` if the Frame Loader has completed initialization. """
        thread_is_alive = self._init_thread.is_alive()
        if thread_is_alive:
            self._init_thread.check_and_raise_error()
        else:
            self._init_thread.join()
            self._set_frame(initialize=True)  # Setting initial frame must be done from main thread
        return not thread_is_alive

    @property
    def video_meta_data(self) -> dict[str, list[int] | list[float] | None]:
        """ dict: The pts_time and key frames for the loader. """
        assert self._loader is not None
        return self._loader.video_meta_data

    def _background_init_frames(self,
                                frames_location: str,
                                video_meta_data: dict[str, list[int] | list[float] | None],
                                frame_list: list[str]) -> MultiThread:
        """ Launch the images loader in a background thread so we can run other tasks whilst
        waiting for initialization.

        Parameters
        ----------
        frame_location: str
            The location of the source video file/frames folder
        video_meta_data: dict
            The meta data for video file sources
        frame_list: list[str]
            The list of frames that exist in the alignments file
        """
        thread = MultiThread(self._load_images,
                             frames_location,
                             video_meta_data,
                             frame_list,
                             thread_count=1,
                             name=f"{self.__class__.__name__}.init_frames")
        thread.start()
        return thread

    def _load_images(self,
                     frames_location: str,
                     video_meta_data: dict[str, list[int] | list[float] | None],
                     frame_list: list[str]) -> None:
        """ Load the images in a background thread.

        Parameters
        ----------
        frame_location: str
            The location of the source video file/frames folder
        video_meta_data: dict
            The meta data for video file sources
        frame_list: list[str]
            The list of frames that exist in the alignments file
        """
        self._loader = SingleFrameLoader(frames_location, video_meta_data=video_meta_data)
        if not self._loader.is_video and len(frame_list) < self._loader.count:
            files = [os.path.basename(f) for f in self._loader.file_list]
            skip_list = [idx for idx, fname in enumerate(files) if fname not in frame_list]
            logger.debug("Adding %s entries to skip list for images not in alignments file",
                         len(skip_list))
            self._loader.add_skip_list(skip_list)
        self._globals.set_frame_count(self._loader.process_count)

    def _set_frame(self,  # pylint:disable=unused-argument
                   *args,
                   initialize: bool = False) -> None:
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
            logger.trace("Update criteria not met. Not updating: "  # type:ignore[attr-defined]
                         "(initialize: %s, position: %s, current_idx: %s, is_zoomed: %s)",
                         initialize, position, self._current_idx, self._globals.is_zoomed)
            return
        if position == -1:
            filename = "No Frame"
            frame = np.ones(self._globals.frame_display_dims + (3, ), dtype="uint8")
        else:
            assert self._loader is not None
            filename, frame = self._loader.image_from_index(position)
        logger.trace("filename: %s, frame: %s, position: %s",  # type:ignore[attr-defined]
                     filename, frame.shape, position)
        self._globals.set_current_frame(frame, filename)
        self._current_idx = position
        self._globals.var_full_update.set(True)
        self._globals.var_update_active_viewport.set(True)
