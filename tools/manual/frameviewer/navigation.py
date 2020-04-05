#!/usr/bin/env python3
""" Media objects for the manual adjustments tool """
import logging
import tkinter as tk

from lib.image import SingleFrameLoader
from lib.multithreading import MultiThread

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FrameNavigation():
    """Handles the return of the correct frame for the GUI.

    Parameters
    ----------
    tk_globals: :class:`TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    frames_location: str
        The path to the input frames
    """
    def __init__(self, tk_globals, frames_location, video_meta_data):
        logger.debug("Initializing %s: (tk_globals: %s, frames_location: '%s', "
                     "video_meta_data: %s)", self.__class__.__name__, tk_globals, frames_location,
                     video_meta_data)
        self._globals = tk_globals
        self._video_meta_data = video_meta_data
        self._loader = None
        self._current_idx = 0

        self._tk_is_playing = tk.BooleanVar()
        self._tk_is_playing.set(False)

        self._init_thread = self._background_init_frames(frames_location)
        self._globals.tk_frame_index.trace("w", self._set_current_frame)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_initialized(self):
        """ bool: ``True`` if the aligner has completed initialization otherwise ``False``. """
        thread_is_alive = self._init_thread.is_alive()
        if thread_is_alive:
            self._init_thread.check_and_raise_error()
        else:
            self._init_thread.join()
            # Setting the initial frame cannot be done in the thread, so set when queried from main
            self._set_current_frame(initialize=True)
        return not thread_is_alive

    @property
    def is_video(self):
        """ bool: 'True' if input is a video 'False' if it is a folder. """
        return self._loader.is_video

    @property
    def location(self):
        """ str: The input folder or video location. """
        return self._loader.location

    @property
    def frame_count(self):
        """ int: The total number of frames """
        return self._loader.count

    @property
    def video_meta_data(self):
        """ dict: The pts_time and key frames for the loader. """
        return self._loader.video_meta_data

    @property
    def tk_is_playing(self):
        """ :class:`tkinter.BooleanVar`: Whether the stream is currently playing. """
        return self._tk_is_playing

    def _background_init_frames(self, frames_location):
        """ Launch the images loader in a background thread so we can run other tasks whilst
        waiting for initialization. """
        thread = MultiThread(self._load_images,
                             frames_location,
                             self._video_meta_data,
                             thread_count=1,
                             name="{}.init_frames".format(self.__class__.__name__))
        thread.start()
        return thread

    def _load_images(self, frames_location, video_meta_data):
        """ Load the images in a background thread. """
        self._loader = SingleFrameLoader(frames_location, video_meta_data=video_meta_data)

    def _set_current_frame(self, *args,  # pylint:disable=unused-argument
                           initialize=False):
        """ Set the currently loaded frame to :attr:`_current_frame`

        Parameters
        ----------
        args: tuple
            Required for event callback. Unused.
        initialize: bool, optional
            ``True`` if initializing for the first frame to be displayed otherwise ``False``.
            Default: ``False``
        """
        position = self._globals.frame_index
        if not initialize and (position == self._current_idx and not self._globals.is_zoomed):
            return
        filename, frame = self._loader.image_from_index(position)
        self._globals.set_current_frame(frame, filename)
        self._current_idx = position
        self._globals.tk_update.set(True)

    def stop_playback(self):
        """ Stop play back if playing """
        if self.tk_is_playing.get():
            logger.trace("Stopping playback")
            self.tk_is_playing.set(False)
