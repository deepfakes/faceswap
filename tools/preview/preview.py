#!/usr/bin/env python3
""" Tool to preview swaps and tweak configuration prior to running a convert """
from __future__ import annotations
import gettext
import logging
import random
import tkinter as tk
import typing as T

from tkinter import ttk
import os
import sys

from threading import Event, Lock, Thread

import numpy as np

from lib.align import DetectedFace
from lib.cli.args_extract_convert import ConvertArgs
from lib.gui.utils import get_images, get_config, initialize_config, initialize_images
from lib.convert import Converter
from lib.utils import FaceswapError, handle_deprecated_cliopts
from lib.queue_manager import queue_manager
from scripts.fsmedia import Alignments, Images
from scripts.convert import Predict, ConvertItem

from plugins.extract import ExtractMedia

from .control_panels import ActionFrame, ConfigTools, OptionsBook
from .viewer import FacesDisplay, ImagesCanvas

if T.TYPE_CHECKING:
    from argparse import Namespace
    from lib.queue_manager import EventQueue
    from .control_panels import BusyProgressBar

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("tools.preview", localedir="locales", fallback=True)
_ = _LANG.gettext


class Preview(tk.Tk):
    """ This tool is part of the Faceswap Tools suite and should be called from
    ``python tools.py preview`` command.

    Loads up 5 semi-random face swaps and displays them, cropped, in place in the final frame.
    Allows user to live tweak settings, before saving the final config to
    :file:`./config/convert.ini`

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """
    _w: str

    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        super().__init__()
        arguments = handle_deprecated_cliopts(arguments)
        self._config_tools = ConfigTools()
        self._lock = Lock()
        self._dispatcher = Dispatcher(self)
        self._display = FacesDisplay(self, 256, 64)
        self._samples = Samples(self, arguments, 5)
        self._patch = Patch(self, arguments)

        self._initialize_tkinter()
        self._image_canvas: ImagesCanvas | None = None
        self._opts_book: OptionsBook | None = None
        self._cli_frame: ActionFrame | None = None  # cli frame holds cli options
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def config_tools(self) -> "ConfigTools":
        """ :class:`ConfigTools`: The object responsible for parsing configuration options and
        updating to/from the GUI """
        return self._config_tools

    @property
    def dispatcher(self) -> "Dispatcher":
        """ :class:`Dispatcher`: The object responsible for triggering events and variables and
        handling global GUI state """
        return self._dispatcher

    @property
    def display(self) -> FacesDisplay:
        """ :class:`~tools.preview.viewer.FacesDisplay`: The object that holds the sample,
        converted and patched faces """
        return self._display

    @property
    def lock(self) -> Lock:
        """ :class:`threading.Lock`: The threading lock object for the Preview GUI """
        return self._lock

    @property
    def progress_bar(self) -> BusyProgressBar:
        """ :class:`~tools.preview.control_panels.BusyProgressBar`: The progress bar that indicates
        a swap/patch thread is running """
        assert self._cli_frame is not None
        return self._cli_frame.busy_progress_bar

    def update_display(self):
        """ Update the images in the canvas and redraw """
        if not hasattr(self, "_image_canvas"):  # On first call object not yet created
            return
        assert self._image_canvas is not None
        self._image_canvas.reload()

    def _initialize_tkinter(self) -> None:
        """ Initialize a standalone tkinter instance. """
        logger.debug("Initializing tkinter")
        initialize_config(self, None, None)
        initialize_images()
        get_config().set_geometry(940, 600, fullscreen=False)
        self.title("Faceswap.py - Convert Settings")
        self.tk.call(
            "wm",
            "iconphoto",
            self._w,
            get_images().icons["favicon"])  # pylint:disable=protected-access
        logger.debug("Initialized tkinter")

    def process(self) -> None:
        """ The entry point for the Preview tool from :file:`lib.tools.cli`.

        Launch the tkinter preview Window and run main loop.
        """
        self._build_ui()
        self.mainloop()

    def _refresh(self, *args) -> None:
        """ Patch faces with current convert settings.

        Parameters
        ----------
        *args: tuple
            Unused, but required for tkinter callback.
        """
        logger.debug("Patching swapped faces. args: %s", args)
        self._dispatcher.set_busy()
        self._config_tools.update_config()
        with self._lock:
            assert self._cli_frame is not None
            self._patch.converter_arguments = self._cli_frame.convert_args

        self._dispatcher.set_needs_patch()
        logger.debug("Patched swapped faces")

    def _build_ui(self) -> None:
        """ Build the elements for displaying preview images and options panels. """
        container = ttk.PanedWindow(self,
                                    orient=tk.VERTICAL)
        container.pack(fill=tk.BOTH, expand=True)
        setattr(container, "preview_display", self._display)  # TODO subclass not setattr
        self._image_canvas = ImagesCanvas(self, container)
        container.add(self._image_canvas, weight=3)

        options_frame = ttk.Frame(container)
        self._cli_frame = ActionFrame(self, options_frame)
        self._opts_book = OptionsBook(options_frame,
                                      self._config_tools,
                                      self._refresh)
        container.add(options_frame, weight=1)
        self.update_idletasks()
        container.sashpos(0, int(400 * get_config().scaling_factor))


class Dispatcher():
    """ Handles the app level tk.Variables and the threading events. Dispatches events to the
    correct location and handles GUI state whilst events are handled

    Parameters
    ----------
    app: :class:`Preview`
        The main tkinter Preview app
    """
    def __init__(self, app: Preview):
        logger.debug("Initializing %s: (app: %s)", self.__class__.__name__, app)
        self._app = app
        self._tk_busy = tk.BooleanVar(value=False)
        self._evnt_needs_patch = Event()
        self._is_updating = False
        self._stacked_event = False
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def needs_patch(self) -> Event:
        """:class:`threading.Event`. Set by the parent and cleared by the child. Informs the child
        patching thread that a run needs to be processed """
        return self._evnt_needs_patch

    # TKInter Variables
    def set_busy(self) -> None:
        """ Set the tkinter busy variable to ``True`` and display the busy progress bar """
        if self._tk_busy.get():
            logger.debug("Busy event is already set. Doing nothing")
            return
        if not hasattr(self._app, "progress_bar"):
            logger.debug("Not setting busy during initial startup")
            return

        logger.debug("Setting busy event to True")
        self._tk_busy.set(True)
        self._app.progress_bar.start()
        self._app.update_idletasks()

    def _unset_busy(self) -> None:
        """ Set the tkinter busy variable to ``False`` and hide the busy progress bar """
        self._is_updating = False
        if not self._tk_busy.get():
            logger.debug("busy unset when already unset. Doing nothing")
            return
        logger.debug("Setting busy event to False")
        self._tk_busy.set(False)
        self._app.progress_bar.stop()
        self._app.update_idletasks()

    # Threading Events
    def _wait_for_patch(self) -> None:
        """ Wait for a patch thread to complete before triggering a display refresh and unsetting
        the busy indicators """
        logger.debug("Checking for patch completion...")
        if self._evnt_needs_patch.is_set():
            logger.debug("Samples not patched. Waiting...")
            self._app.after(1000, self._wait_for_patch)
            return

        logger.debug("Patch completion detected")
        self._app.update_display()
        self._unset_busy()

        if self._stacked_event:
            logger.debug("Processing last stacked event")
            self.set_busy()
            self._stacked_event = False
            self.set_needs_patch()
            return

    def set_needs_patch(self) -> None:
        """ Sends a trigger to the patching thread that it needs to be run. Waits for the patching
        to complete prior to triggering a display refresh and unsetting the busy indicators """
        if self._is_updating:
            logger.debug("Request to run patch when it is already running. Adding stacked event.")
            self._stacked_event = True
            return
        self._is_updating = True
        logger.debug("Triggering patch")
        self._evnt_needs_patch.set()
        self._wait_for_patch()


class Samples():
    """ The display samples.

    Obtains and holds :attr:`sample_size` semi random test faces for displaying in the
    preview GUI.

    The file list is split into evenly sized groups of :attr:`sample_size`. When a display set is
    generated, a random image from each of the groups is selected to provide an array of images
    across the length of the video.

    Parameters
    ----------
    app: :class:`Preview`
        The main tkinter Preview app
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    sample_size: int
        The number of samples to take from the input video/images
    """

    def __init__(self, app: Preview, arguments: Namespace, sample_size: int) -> None:
        logger.debug("Initializing %s: (app: %s, arguments: '%s', sample_size: %s)",
                     self.__class__.__name__, app, arguments, sample_size)
        self._sample_size = sample_size
        self._app = app
        self._input_images: list[ConvertItem] = []
        self._predicted_images: list[tuple[ConvertItem, np.ndarray]] = []

        self._images = Images(arguments)
        self._alignments = Alignments(arguments,
                                      is_extract=False,
                                      input_is_video=self._images.is_video)
        if self._alignments.version == 1.0:
            logger.error("The alignments file format has been updated since the given alignments "
                         "file was generated. You need to update the file to proceed.")
            logger.error("To do this run the 'Alignments Tool' > 'Extract' Job.")
            sys.exit(1)

        if not self._alignments.have_alignments_file:
            logger.error("Alignments file not found at: '%s'", self._alignments.file)
            sys.exit(1)

        if self._images.is_video:
            assert isinstance(self._images.input_images, str)
            self._alignments.update_legacy_has_source(os.path.basename(self._images.input_images))

        self._filelist = self._get_filelist()
        self._indices = self._get_indices()

        self._predictor = Predict(self._sample_size, arguments)
        self._predictor.launch(queue_manager.get_queue("preview_predict_in"))
        self._app._display.set_centering(self._predictor.centering)
        self.generate()

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def available_masks(self) -> list[str]:
        """ list: The mask names that are available for every face in the alignments file """
        retval = [key
                  for key, val in self.alignments.mask_summary.items()
                  if val == self.alignments.faces_count]
        return retval

    @property
    def sample_size(self) -> int:
        """ int: The number of samples to take from the input video/images """
        return self._sample_size

    @property
    def predicted_images(self) -> list[tuple[ConvertItem, np.ndarray]]:
        """ list: The predicted faces output from the Faceswap model """
        return self._predicted_images

    @property
    def alignments(self) -> Alignments:
        """ :class:`~lib.align.Alignments`: The alignments for the preview faces """
        return self._alignments

    @property
    def predictor(self) -> Predict:
        """ :class:`~scripts.convert.Predict`: The Predictor for the Faceswap model """
        return self._predictor

    @property
    def _random_choice(self) -> list[int]:
        """ list: Random indices from the :attr:`_indices` group """
        retval = [random.choice(indices) for indices in self._indices]
        logger.debug(retval)
        return retval

    def _get_filelist(self) -> list[str]:
        """ Get a list of files for the input, filtering out those frames which do
        not contain faces.

        Returns
        -------
        list
            A list of filenames of frames that contain faces.
        """
        logger.debug("Filtering file list to frames with faces")
        if isinstance(self._images.input_images, str):
            vid_name, ext = os.path.splitext(self._images.input_images)
            filelist = [f"{vid_name}_{frame_no:06d}{ext}"
                        for frame_no in range(1, self._images.images_found + 1)]
        else:
            filelist = self._images.input_images

        retval = [filename for filename in filelist
                  if self._alignments.frame_has_faces(os.path.basename(filename))]
        logger.debug("Filtered out frames: %s", self._images.images_found - len(retval))
        try:
            assert retval
        except AssertionError as err:
            msg = ("No faces were found in any of the frames passed in. Make sure you are passing "
                   "in a frames source rather than extracted faces, and that you have provided "
                   "the correct alignments file.")
            raise FaceswapError(msg) from err
        return retval

    def _get_indices(self) -> list[list[int]]:
        """ Get indices for each sample group.

        Obtain :attr:`self.sample_size` evenly sized groups of indices
        pertaining to the filtered :attr:`self._file_list`

        Returns
        -------
        list
            list of indices relating to the filtered file list, split into groups
        """
        # Remove start and end values to get a list divisible by self.sample_size
        no_files = len(self._filelist)
        self._sample_size = min(self._sample_size, no_files)
        crop = no_files % self._sample_size
        top_tail = list(range(no_files))[
            crop // 2:no_files - (crop - (crop // 2))]
        # Partition the indices
        size = len(top_tail)
        retval = [top_tail[start:start + size // self._sample_size]
                  for start in range(0, size, size // self._sample_size)]
        logger.debug("Indices pools: %s", [f"{idx}: (start: {min(pool)}, "
                                           f"end: {max(pool)}, size: {len(pool)})"
                                           for idx, pool in enumerate(retval)])
        return retval

    def generate(self) -> None:
        """ Generate a sample set.

        Selects :attr:`sample_size` random faces. Runs them through prediction to obtain the
        swap, then trigger the patch event to run the faces through patching.
        """
        logger.debug("Generating new random samples")
        self._app.dispatcher.set_busy()
        self._load_frames()
        self._predict()
        self._app.dispatcher.set_needs_patch()
        logger.debug("Generated new random samples")

    def _load_frames(self) -> None:
        """ Load a sample of random frames.

        * Picks a random face from each indices group.

        * Takes the first face from the image (if there are multiple faces). Adds the images to \
        :attr:`self._input_images`.

        * Sets :attr:`_display.source` to the input images and flags that the display should be \
        updated
        """
        self._input_images = []
        for selection in self._random_choice:
            filename = os.path.basename(self._filelist[selection])
            image = self._images.load_one_image(self._filelist[selection])
            # Get first face only
            face = self._alignments.get_faces_in_frame(filename)[0]
            detected_face = DetectedFace()
            detected_face.from_alignment(face, image=image)
            inbound = ExtractMedia(filename=filename, image=image, detected_faces=[detected_face])
            self._input_images.append(ConvertItem(inbound=inbound))
        self._app.display.source = self._input_images
        self._app.display.update_source = True
        logger.debug("Selected frames: %s",
                     [frame.inbound.filename for frame in self._input_images])

    def _predict(self) -> None:
        """ Predict from the loaded frames.

        With a threading lock (to prevent stacking), run the selected faces through the Faceswap
        model predict function and add the output to :attr:`predicted`
        """
        with self._app.lock:
            self._predicted_images = []
            for frame in self._input_images:
                self._predictor.in_queue.put(frame)
            idx = 0
            while idx < self._sample_size:
                logger.debug("Predicting face %s of %s", idx + 1, self._sample_size)
                items: (T.Literal["EOF"] |
                        list[tuple[ConvertItem, np.ndarray]]) = self._predictor.out_queue.get()
                if items == "EOF":
                    logger.debug("Received EOF")
                    break
                for item in items:
                    self._predicted_images.append(item)
                    logger.debug("Predicted face %s of %s", idx + 1, self._sample_size)
                    idx += 1
        logger.debug("Predicted faces")


class Patch():
    """ The Patch pipeline

    Runs in it's own thread. Takes the output from the Faceswap model predictor and runs the faces
    through the convert pipeline using the currently selected options.

    Parameters
    ----------
    app: :class:`Preview`
        The main tkinter Preview app
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`

    Attributes
    ----------
    converter_arguments: dict
        The currently selected converter command line arguments for the patch queue
    """
    def __init__(self, app: Preview, arguments: Namespace) -> None:
        logger.debug("Initializing %s: (app: %s, arguments: '%s')",
                     self.__class__.__name__, app, arguments)
        self._app = app
        self._queue_patch_in = queue_manager.get_queue("preview_patch_in")
        self.converter_arguments: dict[str, T.Any] | None = None  # Updated converter args

        configfile = arguments.configfile if hasattr(arguments, "configfile") else None
        self._converter = Converter(output_size=app._samples.predictor.output_size,
                                    coverage_ratio=app._samples.predictor.coverage_ratio,
                                    centering=app._samples.predictor.centering,
                                    draw_transparent=False,
                                    pre_encode=None,
                                    arguments=self._generate_converter_arguments(
                                        arguments,
                                        app._samples.available_masks),
                                    configfile=configfile)
        self._thread = Thread(target=self._process,
                              name="patch_thread",
                              args=(self._queue_patch_in,
                                    self._app.dispatcher.needs_patch,
                                    app._samples),
                              daemon=True)
        self._thread.start()
        logger.debug("Initializing %s", self.__class__.__name__)

    @property
    def converter(self) -> Converter:
        """ :class:`lib.convert.Converter`: The converter to use for patching the images. """
        return self._converter

    @staticmethod
    def _generate_converter_arguments(arguments: Namespace,
                                      available_masks: list[str]) -> Namespace:
        """ Add the default converter arguments to the initial arguments. Ensure the mask selection
        is available.

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The :mod:`argparse` arguments as passed in from :mod:`tools.py`
        available_masks: list
            The masks that are available for convert
        Returns
        ----------
        arguments: :class:`argparse.Namespace`
            The :mod:`argparse` arguments as passed in with converter default
            arguments added
        """
        valid_masks = available_masks + ["none"]
        converter_arguments = ConvertArgs(None, "convert").get_optional_arguments()  # type: ignore
        for item in converter_arguments:
            value = item.get("default", None)
            # Skip options without a default value
            if value is None:
                continue
            option = item.get("dest", item["opts"][1].replace("--", ""))
            if option == "mask_type" and value not in valid_masks:
                logger.debug("Amending default mask from '%s' to '%s'", value, valid_masks[0])
                value = valid_masks[0]
            # Skip options already in arguments
            if hasattr(arguments, option):
                continue
            # Add option to arguments
            setattr(arguments, option, value)
        logger.debug(arguments)
        return arguments

    def _process(self,
                 patch_queue_in: EventQueue,
                 trigger_event: Event,
                 samples: Samples) -> None:
        """ The face patching process.

        Runs in a thread, and waits for an event to be set. Once triggered, runs a patching
        cycle and sets the :class:`Display` destination images.

        Parameters
        ----------
        patch_queue_in: :class:`~lib.queue_manager.EventQueue`
            The input queue for the patching process
        trigger_event: :class:`threading.Event`
            The event that indicates a patching run needs to be processed
        samples: :class:`Samples`
            The Samples for display.
        """
        logger.debug("Launching patch process thread: (patch_queue_in: %s, trigger_event: %s, "
                     "samples: %s)", patch_queue_in, trigger_event, samples)
        patch_queue_out = queue_manager.get_queue("preview_patch_out")
        while True:
            trigger = trigger_event.wait(1)
            if not trigger:
                continue
            logger.debug("Patch Triggered")
            queue_manager.flush_queue("preview_patch_in")
            self._feed_swapped_faces(patch_queue_in, samples)
            with self._app.lock:
                self._update_converter_arguments()
                self._converter.reinitialize(config=self._app.config_tools.config)
            swapped = self._patch_faces(patch_queue_in, patch_queue_out, samples.sample_size)
            with self._app.lock:
                self._app.display.destination = swapped

            logger.debug("Patch complete")
            trigger_event.clear()

        logger.debug("Closed patch process thread")

    def _update_converter_arguments(self) -> None:
        """ Update the converter arguments to the currently selected values. """
        logger.debug("Updating Converter cli arguments")
        if self.converter_arguments is None:
            logger.debug("No arguments to update")
            return
        for key, val in self.converter_arguments.items():
            logger.debug("Updating %s to %s", key, val)
            setattr(self._converter.cli_arguments, key, val)
        logger.debug("Updated Converter cli arguments")

    @staticmethod
    def _feed_swapped_faces(patch_queue_in: EventQueue, samples: Samples) -> None:
        """ Feed swapped faces to the converter's in-queue.

        Parameters
        ----------
        patch_queue_in: :class:`~lib.queue_manager.EventQueue`
            The input queue for the patching process
        samples: :class:`Samples`
            The Samples for display.
        """
        logger.debug("feeding swapped faces to converter")
        for item in samples.predicted_images:
            patch_queue_in.put(item)
        logger.debug("fed %s swapped faces to converter",
                     len(samples.predicted_images))
        logger.debug("Putting EOF to converter")
        patch_queue_in.put("EOF")

    def _patch_faces(self,
                     queue_in: EventQueue,
                     queue_out: EventQueue,
                     sample_size: int) -> list[np.ndarray]:
        """ Patch faces.

        Run the convert process on the swapped faces and return the patched faces.

        patch_queue_in: :class:`~lib.queue_manager.EventQueue`
            The input queue for the patching process
        queue_out: :class:`~lib.queue_manager.EventQueue`
            The output queue from the patching process
        sample_size: int
            The number of samples to be displayed

        Returns
        -------
        list
            The swapped faces patched with the selected convert settings
        """
        logger.debug("Patching faces")
        self._converter.process(queue_in, queue_out)
        swapped = []
        idx = 0
        while idx < sample_size:
            logger.debug("Patching image %s of %s", idx + 1, sample_size)
            item = queue_out.get()
            swapped.append(item[1])
            logger.debug("Patched image %s of %s", idx + 1, sample_size)
            idx += 1
        logger.debug("Patched faces")
        return swapped
