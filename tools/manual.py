#!/usr/bin/env python3
""" Tool to manually interact with the alignments file using visual tools """
import logging
import os

import tkinter as tk
from tkinter import ttk
from concurrent import futures
from functools import partial
from time import time

import cv2
from tqdm import tqdm
from PIL import Image, ImageTk

from lib.alignments import Alignments
from lib.gui.control_helper import set_slider_rounding
from lib.gui.custom_widgets import Tooltip
from lib.gui.utils import get_images, get_config, initialize_config, initialize_images
from lib.image import ImagesLoader

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
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        super().__init__()
        self._frame_cache = FrameCache(arguments.frames)
        self._alignments = self._get_alignments(arguments.alignments_path)
        self._frame_cache.cache_frames()

        self._initialize_tkinter()
        self._containers = self._create_containers()

        self._display = DisplayFrame(self._containers["top"], self._frame_cache)

        lbl = ttk.Label(self._containers["top"], text="Top Right")
        self._containers["top"].add(lbl)

        self._set_layout()
        self._set_keybindings()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_alignments(self, alignments_path):
        """ Get the alignments object """
        if alignments_path:
            folder, filename = os.path.split(self._arguments.alignments_path)
        else:
            filename = "alignments.fsa"
            if self._frame_cache.is_video:
                folder, vid = os.path.split(os.path.splitext(self._frame_cache.location)[0])
                filename = "{}_{}".format(vid, filename)
            else:
                folder = self._frame_cache.location
        return Alignments(folder, filename)

    def _initialize_tkinter(self):
        """ Initialize a standalone tkinter instance. """
        logger.debug("Initializing tkinter")
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

        top = tk.PanedWindow(main,
                             sashrelief=tk.RIDGE,
                             sashwidth=2,
                             sashpad=4,
                             orient=tk.HORIZONTAL,
                             name="pw_top")
        main.add(top)

        bottom = ttk.Frame(main, name="frame_bottom")
        main.add(bottom)
        logger.debug("Created containers")
        return dict(main=main, top=top, bottom=bottom)

    def _set_layout(self):
        """ Place the sashes of the paned window """
        self.update_idletasks()
        self._containers["top"].sash_place(0, (self._frame_cache.display_dims[0]) + 8, 1)
        self._containers["main"].sash_place(0, 1, self._frame_cache.display_dims[1] + 8)

    def _set_keybindings(self):
        """ Set the keybindings for keyboard shortcuts """
        self.bind("<Key>", self._handle_key_press)

    def _handle_key_press(self, event):
        key = event.keysym
        if key.lower() == "left":
            self._frame_cache.set_prev_frame()
        elif key.lower() == "right":
            self._frame_cache.set_next_frame()
        elif key.lower() == "space":
            self._display.handle_play_button()

    def process(self):
        """ The entry point for the Visual Alignments tool from :file:`lib.tools.cli`.

        Launch the tkinter Visual Alignments Window and run main loop.
        """
        lbl = ttk.Label(self._containers["bottom"], text="Bottom")
        lbl.pack()
        # self._build_ui()
        self.mainloop()

    def _build_ui(self):
        """ Build the page elements for displaying the Visual Alignments tool. """
        pass


class DisplayFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The main video display frame (top left section of GUI).

    Parameters
    ----------
    parent: :class:`tkinter.PanedWindow`
        The paned window that the display frame resides in
    frames_location: str
        The path to the input frames
    """
    def __init__(self, parent, frame_cache):
        logger.debug("Initializing %s: (parent: %s, frame_cache: %s)",
                     self.__class__.__name__, parent, frame_cache)
        super().__init__(parent)
        parent.add(self)
        self._frame_cache = frame_cache
        self._viewer = self._add_viewer()

        transport_frame = ttk.Frame(self)
        transport_frame.pack(side=tk.BOTTOM, padx=5, pady=5, fill=tk.X)

        self._add_nav(transport_frame)
        self._play_button = self._add_transport(transport_frame)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _add_viewer(self):
        """ Adds the frames viewer window

        Returns
        -------
        dict:
            The `canvas` (:class:`tk.canvas`) and `image_canvas`

        """
        canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor=tk.E)
        imgcanvas = canvas.create_image(self._frame_cache.display_dims[0] // 2,
                                        self._frame_cache.display_dims[1] // 2,
                                        image=self._frame_cache.current_frame, anchor=tk.CENTER)

        needs_update = self._frame_cache.tk_update
        needs_update.trace("w", self._update_display)
        return dict(canvas=canvas, image_canvas=imgcanvas)

    def _add_nav(self, transport_frame):
        """ Add the slider to navigate through frames """
        var = self._frame_cache.tk_position
        max_frame = self._frame_cache.frame_count - 1

        frame = ttk.Frame(transport_frame)

        frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        lbl_frame = ttk.Frame(frame)
        lbl_frame.pack(side=tk.RIGHT)
        tbox = ttk.Entry(lbl_frame,
                         width=7,
                         textvariable=var,
                         justify=tk.RIGHT)
        tbox.pack(padx=0, side=tk.LEFT)
        lbl = ttk.Label(lbl_frame, text="/{}".format(max_frame))
        lbl.pack(side=tk.RIGHT)

        cmd = partial(set_slider_rounding,
                      var=var,
                      d_type=int,
                      round_to=1,
                      min_max=(0, max_frame))

        nav = ttk.Scale(frame, variable=var, from_=0, to=max_frame, command=cmd)
        nav.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _add_transport(self, transport_frame):
        """ Add video transport controls """
        frame = ttk.Frame(transport_frame)
        frame.pack(side=tk.BOTTOM, fill=tk.X)
        icons = get_images().icons

        for action in ("play", "prev", "next", "speed"):
            if action == "play":
                play_button = ttk.Button(frame,
                                         image=icons[action],
                                         width=14,
                                         command=self.handle_play_button)
                play_button.pack(side=tk.LEFT, padx=(0, 6))
                Tooltip(play_button, text="Play/Pause (SPACE)")
                self._frame_cache.tk_is_playing.trace("w", self._play)
            if action in ("prev", "next"):
                cmd = getattr(self._frame_cache, "set_{}_frame".format(action))
                if action == "prev":
                    helptext = "Go to Previous Frame (LEFT)"
                else:
                    helptext = "Go to Next Frame (RIGHT)"
                btn = ttk.Button(frame, image=icons[action], width=14, command=cmd)
                btn.pack(side=tk.LEFT)
                Tooltip(btn, text=helptext)
            elif action == "speed":
                self._add_speed_combo(frame)
        return play_button

    def handle_play_button(self):
        """ Handle the play button.

        Switches the :attr:`_frame_cache.is_playing` variable.
        """
        is_playing = self._frame_cache.tk_is_playing.get()
        self._frame_cache.tk_is_playing.set(not is_playing)

    def _add_speed_combo(self, frame):
        """ Adds the speed control Combo box and links to
        :attr:`_frame_cache.tk_playback_speed`. """
        tk_var = self._frame_cache.tk_playback_speed
        tk_var.set("1x")
        sframe = ttk.Frame(frame)
        sframe.pack(side=tk.RIGHT)
        lbl = ttk.Label(sframe, text="Playback Speed")
        lbl.pack(side=tk.LEFT, padx=(0, 5))
        combo = ttk.Combobox(sframe, textvariable=tk_var, values=["1x", "2x"], width=3)
        combo.pack(side=tk.RIGHT)
        Tooltip(combo, text="Set Playback Speed")

    def _update_display(self, *args):  # pylint:disable=unused-argument
        """ Update the display on frame cache update """
        if not self._frame_cache.tk_update.get():
            return
        self._viewer["canvas"].itemconfig(self._viewer["image_canvas"],
                                          image=self._frame_cache.current_frame)
        self._frame_cache.tk_update.set(False)

    def _play(self, *args):  # pylint:disable=unused-argument
        """ Play the video file at the selected speed """
        start = time()
        is_playing = self._frame_cache.tk_is_playing.get()
        icon = "pause" if is_playing else "play"
        self._play_button.config(image=get_images().icons[icon])

        if not is_playing:
            logger.debug("Pause detected. Stopping.")
            return

        self._frame_cache.set_next_frame(is_playing=True)
        self._viewer["canvas"].itemconfig(self._viewer["image_canvas"],
                                          image=self._frame_cache.current_frame)
        speed = self._frame_cache.tk_playback_speed.get().replace("x", "")
        delay = self._frame_cache.delay // int(speed)
        duration = int((time() - start) * 1000)
        delay = max(1, delay - duration)
        self.after(delay, self._play)


class FrameCache():
    """ Caches all video frames to compressed JPGs for full transport control.

    Handles the return of the correct frame for the GUI.

    Parameters
    ----------
    frames_location: str
        The path to the input frames
    """
    def __init__(self, frames_location):
        logger.debug("Initializing %s: (frames_location: '%s')",
                     self.__class__.__name__, frames_location)
        self._loader = ImagesLoader(frames_location)
        self._delay = int(round(1000 / self._loader.fps))
        self._frames = list()
        self._current_idx = 0
        self._tk_vars = self._set_tk_vars()
        self._current_frame = None
        self._display_dims = (960, 540)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_video(self):
        """ bool: 'True' if input is a video 'False' if it is a folder. """
        return self._loader.is_video

    @property
    def location(self):
        """ str: The input folder or video location. """
        return self._loader.location

    @property
    def current_frame(self):
        """ :class:`ImageTk.PhotoImage`: The currently loaded, decompressed frame. """
        return self._current_frame

    @property
    def delay(self):
        """ int: The number of milliseconds between updates to playback the video at the
        correct fps """
        return self._delay

    @property
    def display_dims(self):
        """ tuple: The (`width`, `height`) of the display image. """
        return self._display_dims

    @property
    def frame_count(self):
        """ int: The total number of frames """
        return len(self._frames)

    @property
    def tk_playback_speed(self):
        """ :class:`tkinter.IntVar`: Multiplier to playback speed. """
        return self._tk_vars["speed"]

    @property
    def tk_position(self):
        """ :class:`tkinter.IntVar`: The current frame position. """
        return self._tk_vars["position"]

    @property
    def tk_is_playing(self):
        """ :class:`tkinter.BooleanVar`: Whether the stream is currently playing. """
        return self._tk_vars["is_playing"]

    @property
    def tk_update(self):
        """ :class:`tkinter.BooleanVar`: Whether the display needs to be updated. """
        return self._tk_vars["updated"]

    def _set_tk_vars(self):
        """ Set the initial tkinter variables and add traces. """
        logger.debug("Setting tkinter variables")
        position = tk.IntVar()
        position.set(self._current_idx)
        position.trace("w", self._set_current_frame)

        speed = tk.StringVar()

        is_playing = tk.BooleanVar()
        is_playing.set(False)

        updated = tk.BooleanVar()
        updated.set(False)

        retval = dict(position=position, is_playing=is_playing, speed=speed, updated=updated)
        logger.debug("Set tkinter variables: %s", retval)
        return retval

    def _set_current_frame(self, *args,  # pylint:disable=unused-argument
                           initialize=False, is_playing=False):
        """ Set the currently loaded, decompressed frame to :attr:`_current_frame`

        Parameters
        ----------
        args: tuple
            Required for event callback. Unused.
        initialize: bool, optional
            ``True`` if initializing for the first frame to be displayed otherwise ``False``.
            Default: ``False``
        is_playing: bool, optional
            ``True`` if the frame is being incremented because the Play button has been pressed.
            ``False`` if incremented for other reasons. Default: ``False``
        """
        position = self.tk_position.get()
        if not initialize and position == self._current_idx:
            return
        frame = cv2.imdecode(self._frames[position]["image"], cv2.IMREAD_UNCHANGED)
        self._current_frame = ImageTk.PhotoImage(Image.fromarray(frame))
        if not is_playing:
            self.tk_update.set(True)

    def cache_frames(self):
        """ Increment through all frames JPG compressing each and caching to a list in frame order
        and assign to :attr:`_frames`
        """
        executor = futures.ThreadPoolExecutor(max_workers=3)
        images = dict()
        with executor:
            for filename, frame in tqdm(self._loader.load(),
                                        total=self._loader.count,
                                        desc="Analyzing Video..."):
                self._frames.append(filename)
                images[executor.submit(self._encode_frame, frame)] = (filename, frame.shape[:2])
            for future in tqdm(futures.as_completed(images), total=self._loader.count):
                filename, dims = images[future]
                img = future.result()
                self._frames[self._frames.index(filename)] = dict(filename=filename,
                                                                  image=img,
                                                                  dims=dims)
        self._set_current_frame(initialize=True)

    def _encode_frame(self, frame):
        """ Resize frame and encode to jpg

        Parameters
        ----------
        frame: :class:`numpy.ndarray`
            The frame to be compressed and resized

        Returns
        -------
        bytes:
            Compressed and resized frame
        """
        img = frame[..., 2::-1]
        scale = min(self._display_dims[0] / img.shape[1], self._display_dims[1] / img.shape[0])
        dst_dims = (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale)))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, dst_dims, interpolation=interp)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 20])[1]

    def set_next_frame(self, is_playing=False):
        """ Update :attr:`self.current_frame` to the next frame

        Parameters
        ----------
        is_playing: bool
            ``True`` if the frame is being incremented because the Play button has been pressed.
            ``False`` if incremented for other reasons
        """
        position = self.tk_position.get()
        if position == self.frame_count - 1:
            logger.trace("End of stream. Not incrementing")
            if is_playing:
                self.tk_is_playing.set(False)
        else:
            if not is_playing and self.tk_is_playing.get():
                # Stop playback
                self.tk_is_playing.set(False)
            self.tk_position.set(position + 1)
        self._set_current_frame(is_playing)

    def set_prev_frame(self):
        """ Update :attr:`self.current_frame` to the previous frame """
        position = self.tk_position.get()
        if self.tk_is_playing.get():
            # Stop playback
            self.tk_is_playing.set(False)
        if position == 0:
            logger.trace("Beginning of stream. Not decrementing")
        else:
            self.tk_position.set(position - 1)
        self._set_current_frame()
