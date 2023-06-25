#!/usr/bin/python
""" The pop up preview window for Faceswap.

If Tkinter is installed, then this will be used to manage the preview image, otherwise we
fallback to opencv's imshow
"""
from __future__ import annotations
import logging
import typing as T

from threading import Event, Lock
from time import sleep

import cv2

if T.TYPE_CHECKING:
    from collections.abc import Generator
    import numpy as np

logger = logging.getLogger(__name__)
TriggerType = dict[T.Literal["toggle_mask", "refresh", "save", "quit", "shutdown"], Event]
TriggerKeysType = T.Literal["m", "r", "s", "enter"]
TriggerNamesType = T.Literal["toggle_mask", "refresh", "save", "quit"]


class PreviewBuffer():
    """ A thread safe class for holding preview images """
    def __init__(self) -> None:
        logger.debug("Initializing: %s", self.__class__.__name__)
        self._images: dict[str, np.ndarray] = {}
        self._lock = Lock()
        self._updated = Event()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def is_updated(self) -> bool:
        """ bool: ``True`` when new images have been loaded into the  preview buffer """
        return self._updated.is_set()

    def add_image(self, name: str, image: np.ndarray) -> None:
        """ Add an image to the preview buffer in a thread safe way """
        logger.debug("Adding image: (name: '%s', shape: %s)", name, image.shape)
        with self._lock:
            self._images[name] = image
        logger.debug("Added images: %s", list(self._images))
        self._updated.set()

    def get_images(self) -> Generator[tuple[str, np.ndarray], None, None]:
        """ Get the latest images from the preview buffer. When iterator is exhausted clears the
        :attr:`updated` event.

        Yields
        ------
        name: str
            The name of the image
        :class:`numpy.ndarray`
            The image in BGR format
        """
        logger.debug("Retrieving images: %s", list(self._images))
        with self._lock:
            for name, image in self._images.items():
                logger.debug("Yielding: '%s' (%s)", name, image.shape)
                yield name, image
            if self.is_updated:
                logger.debug("Clearing updated event")
                self._updated.clear()
                logger.debug("Retrieved images")


class PreviewBase():  # pylint:disable=too-few-public-methods
    """ Parent class for OpenCV and Tkinter Preview Windows

    Parameters
    ----------
    preview_buffer: :class:`PreviewBuffer`
        The thread safe object holding the preview images
    triggers: dict, optional
        Dictionary of event triggers for pop-up preview. Not required when running inside the GUI.
        Default: `None`
     """
    def __init__(self,
                 preview_buffer: PreviewBuffer,
                 triggers: TriggerType | None = None) -> None:
        logger.debug("Initializing %s parent (triggers: %s)",
                     self.__class__.__name__, triggers)
        self._triggers = triggers
        self._buffer = preview_buffer
        self._keymaps: dict[TriggerKeysType, TriggerNamesType] = {"m": "toggle_mask",
                                                                  "r": "refresh",
                                                                  "s": "save",
                                                                  "enter": "quit"}
        self._title = ""
        logger.debug("Initialized %s parent", self.__class__.__name__)

    @property
    def _should_shutdown(self) -> bool:
        """ bool: ``True`` if the preview has received an external signal to shutdown otherwise
        ``False`` """
        if self._triggers is None or not self._triggers["shutdown"].is_set():
            return False
        logger.debug("Shutdown signal received")
        return True

    def _launch(self) -> None:
        """ Wait until an image is loaded into the preview buffer and call the child's
        :func:`_display_preview` function """
        logger.debug("Launching %s", self.__class__.__name__)
        while True:
            if not self._buffer.is_updated:
                logger.debug("Waiting for preview image")
                sleep(1)
                continue
            break
        logger.debug("Launching preview")
        self._display_preview()

    def _display_preview(self) -> None:
        """ Override for preview viewer's display loop """
        raise NotImplementedError()


class PreviewCV(PreviewBase):  # pylint:disable=too-few-public-methods
    """ Simple fall back preview viewer using OpenCV for when TKinter is not available

    Parameters
    ----------
    preview_buffer: :class:`PreviewBuffer`
        The thread safe object holding the preview images
    triggers: dict
        Dictionary of event triggers for pop-up preview.
     """
    def __init__(self,
                 preview_buffer: PreviewBuffer,
                 triggers: TriggerType) -> None:
        logger.debug("Unable to import Tkinter. Falling back to OpenCV")
        super().__init__(preview_buffer, triggers=triggers)
        self._triggers: TriggerType = self._triggers
        self._windows: list[str] = []

        self._lookup = {ord(key): val
                        for key, val in self._keymaps.items() if key != "enter"}
        self._lookup[ord("\n")] = self._keymaps["enter"]
        self._lookup[ord("\r")] = self._keymaps["enter"]

        self._launch()

    @property
    def _window_closed(self) -> bool:
        """ bool: ``True`` if any window has been closed otherwise ``False`` """
        retval = any(cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1 for win in self._windows)
        if retval:
            logger.debug("Window closed detected")
        return retval

    def _check_keypress(self, key: int):
        """ Check whether we have received a valid key press from OpenCV window and handle
        accordingly.

        Parameters
        ----------
        key_press: int
            The key press received from OpenCV
        """
        if not key or key == -1 or key not in self._lookup:
            return

        if key == ord("r"):
            print("")  # Let log print on different line from loss output
            logger.info("Refresh preview requested...")

        self._triggers[self._lookup[key]].set()
        logger.debug("Processed keypress '%s'. Set event for '%s'", key, self._lookup[key])

    def _display_preview(self):
        """ Handle the displaying of the images currently in :attr:`_preview_buffer`"""
        while True:
            if self._buffer.is_updated or self._window_closed:
                for name, image in self._buffer.get_images():
                    logger.debug("showing image: '%s' (%s)", name, image.shape)
                    cv2.imshow(name, image)
                    self._windows.append(name)

            key = cv2.waitKey(1000)
            self._check_keypress(key)

            if self._triggers["shutdown"].is_set():
                logger.debug("Shutdown received")
                break
        logger.debug("%s shutdown", self.__class__.__name__)
