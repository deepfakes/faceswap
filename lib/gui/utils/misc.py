#!/usr/bin/env python3
""" Miscellaneous Utility functions for the GUI. Includes LongRunningTask object """
from __future__ import annotations
import logging
import sys
import typing as T

from threading import Event, Thread
from queue import Queue

from .config import get_config

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType
    from lib.multithreading import _ErrorType


logger = logging.getLogger(__name__)


class LongRunningTask(Thread):
    """ Runs long running tasks in a background thread to prevent the GUI from becoming
    unresponsive.

    This is sub-classed from :class:`Threading.Thread` so check documentation there for base
    parameters. Additional parameters listed below.

    Parameters
    ----------
    widget: tkinter object, optional
        The widget that this :class:`LongRunningTask` is associated with. Used for setting the busy
        cursor in the correct location. Default: ``None``.
    """
    _target: Callable
    _args: tuple
    _kwargs: dict[str, T.Any]
    _name: str

    def __init__(self,
                 target: Callable | None = None,
                 name: str | None = None,
                 args: tuple = (),
                 kwargs: dict[str, T.Any] | None = None,
                 *,
                 daemon: bool = True,
                 widget=None):
        logger.debug("Initializing %s: (target: %s, name: %s, args: %s, kwargs: %s, "
                     "daemon: %s)", self.__class__.__name__, target, name, args, kwargs,
                     daemon)
        super().__init__(target=target, name=name, args=args, kwargs=kwargs,
                         daemon=daemon)
        self.err: _ErrorType = None
        self._widget = widget
        self._config = get_config()
        self._config.set_cursor_busy(widget=self._widget)
        self._complete = Event()
        self._queue: Queue = Queue()
        logger.debug("Initialized %s", self.__class__.__name__,)

    @property
    def complete(self) -> Event:
        """ :class:`threading.Event`:  Event is set if the thread has completed its task,
        otherwise it is unset.
        """
        return self._complete

    def run(self) -> None:
        """ Commence the given task in a background thread. """
        try:
            if self._target is not None:
                retval = self._target(*self._args, **self._kwargs)
                self._queue.put(retval)
        except Exception:  # pylint:disable=broad-except
            self.err = T.cast(tuple[type[BaseException], BaseException, "TracebackType"],
                              sys.exc_info())
            assert self.err is not None
            logger.debug("Error in thread (%s): %s", self._name,
                         self.err[1].with_traceback(self.err[2]))
        finally:
            self._complete.set()
            # Avoid a ref-cycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def get_result(self) -> T.Any:
        """ Return the result from the given task.

        Returns
        -------
        varies:
            The result of the thread will depend on the given task. If a call is made to
            :func:`get_result` prior to the thread completing its task then ``None`` will be
            returned
        """
        if not self._complete.is_set():
            logger.warning("Aborting attempt to retrieve result from a LongRunningTask that is "
                           "still running")
            return None
        if self.err:
            logger.debug("Error caught in thread")
            self._config.set_cursor_default(widget=self._widget)
            raise self.err[1].with_traceback(self.err[2])

        logger.debug("Getting result from thread")
        retval = self._queue.get()
        logger.debug("Got result from thread")
        self._config.set_cursor_default(widget=self._widget)
        return retval
