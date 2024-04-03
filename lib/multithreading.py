#!/usr/bin/env python3
""" Multithreading/processing utils for faceswap """
from __future__ import annotations
import logging
import typing as T
from multiprocessing import cpu_count

import queue as Queue
import sys
import threading
from types import TracebackType

if T.TYPE_CHECKING:
    from collections.abc import Callable, Generator

logger = logging.getLogger(__name__)
_ErrorType: T.TypeAlias = tuple[type[BaseException],
                                BaseException,
                                TracebackType] | tuple[T.Any, T.Any, T.Any] | None
_THREAD_NAMES: set[str] = set()


def total_cpus():
    """ Return total number of cpus """
    return cpu_count()


def _get_name(name: str) -> str:
    """ Obtain a unique name for a thread

    Parameters
    ----------
    name: str
        The requested name

    Returns
    -------
    str
        The request name with "_#" appended (# being an integer) making the name unique
    """
    idx = 0
    real_name = name
    while True:
        if real_name in _THREAD_NAMES:
            real_name = f"{name}_{idx}"
            idx += 1
            continue
        _THREAD_NAMES.add(real_name)
        return real_name


class FSThread(threading.Thread):
    """ Subclass of thread that passes errors back to parent

    Parameters
    ----------
    target: callable object, Optional
        The callable object to be invoked by the run() method. If ``None`` nothing is called.
        Default: ``None``
    name: str, optional
        The thread name. if ``None`` a unique name is constructed of the form "Thread-N" where N
        is a small decimal number. Default: ``None``
    args: tuple
        The argument tuple for the target invocation. Default: ().
    kwargs: dict
        keyword arguments for the target invocation. Default: {}.
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
                 daemon: bool | None = None) -> None:
        super().__init__(target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
        self.err: _ErrorType = None

    def check_and_raise_error(self) -> None:
        """ Checks for errors in thread and raises them in caller.

        Raises
        ------
        Error
            Re-raised error from within the thread
        """
        if not self.err:
            return
        logger.debug("Thread error caught: %s", self.err)
        raise self.err[1].with_traceback(self.err[2])

    def run(self) -> None:
        """ Runs the target, reraising any errors from within the thread in the caller. """
        try:
            if self._target is not None:
                self._target(*self._args, **self._kwargs)
        except Exception as err:  # pylint:disable=broad-except
            self.err = sys.exc_info()
            logger.debug("Error in thread (%s): %s", self._name, str(err))
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs


class MultiThread():
    """ Threading for IO heavy ops. Catches errors in thread and rethrows to parent.

    Parameters
    ----------
    target: callable object
        The callable object to be invoked by the run() method.
    args: tuple
        The argument tuple for the target invocation. Default: ().
    thread_count: int, optional
        The number of threads to use. Default: 1
    name: str, optional
        The thread name. if ``None`` a unique name is constructed of the form {target.__name__}_N
        where N is an incrementing integer. Default: ``None``
    kwargs: dict
        keyword arguments for the target invocation. Default: {}.
    """
    def __init__(self,
                 target: Callable,
                 *args,
                 thread_count: int = 1,
                 name: str | None = None,
                 **kwargs) -> None:
        self._name = _get_name(name if name else target.__name__)
        logger.debug("Initializing %s: (target: '%s', thread_count: %s)",
                     self.__class__.__name__, self._name, thread_count)
        logger.trace("args: %s, kwargs: %s", args, kwargs)  # type:ignore
        self.daemon = True
        self._thread_count = thread_count
        self._threads: list[FSThread] = []
        self._target = target
        self._args = args
        self._kwargs = kwargs
        logger.debug("Initialized %s: '%s'", self.__class__.__name__, self._name)

    @property
    def has_error(self) -> bool:
        """ bool: ``True`` if a thread has errored, otherwise ``False`` """
        return any(thread.err for thread in self._threads)

    @property
    def errors(self) -> list[_ErrorType]:
        """ list: List of thread error values """
        return [thread.err for thread in self._threads if thread.err]

    @property
    def name(self) -> str:
        """ :str: The name of the thread """
        return self._name

    def check_and_raise_error(self) -> None:
        """ Checks for errors in thread and raises them in caller.

        Raises
        ------
        Error
            Re-raised error from within the thread
        """
        if not self.has_error:
            return
        logger.debug("Thread error caught: %s", self.errors)
        error = self.errors[0]
        assert error is not None
        raise error[1].with_traceback(error[2])

    def is_alive(self) -> bool:
        """ Check if any threads are still alive

        Returns
        -------
        bool
            ``True`` if any threads are alive. ``False`` if no threads are alive
        """
        return any(thread.is_alive() for thread in self._threads)

    def start(self) -> None:
        """ Start all the threads for the given method, args and kwargs """
        logger.debug("Starting thread(s): '%s'", self._name)
        for idx in range(self._thread_count):
            name = self._name if self._thread_count == 1 else f"{self._name}_{idx}"
            logger.debug("Starting thread %s of %s: '%s'",
                         idx + 1, self._thread_count, name)
            thread = FSThread(name=name,
                              target=self._target,
                              args=self._args,
                              kwargs=self._kwargs)
            thread.daemon = self.daemon
            thread.start()
            self._threads.append(thread)
        logger.debug("Started all threads '%s': %s", self._name, len(self._threads))

    def completed(self) -> bool:
        """ Check if all threads have completed

        Returns
        -------
        ``True`` if all threads have completed otherwise ``False``
        """
        retval = all(not thread.is_alive() for thread in self._threads)
        logger.debug(retval)
        return retval

    def join(self) -> None:
        """ Join the running threads, catching and re-raising any errors

        Clear the list of threads for class instance re-use
        """
        logger.debug("Joining Threads: '%s'", self._name)
        for thread in self._threads:
            logger.debug("Joining Thread: '%s'", thread._name)  # pylint:disable=protected-access
            thread.join()
            if thread.err:
                logger.error("Caught exception in thread: '%s'",
                             thread._name)  # pylint:disable=protected-access
                raise thread.err[1].with_traceback(thread.err[2])
        del self._threads
        self._threads = []
        logger.debug("Joined all Threads: '%s'", self._name)


class BackgroundGenerator(MultiThread):
    """ Run a task in the background background and queue data for consumption

    Parameters
    ----------
    generator: iterable
        The generator to run in the background
    prefetch, int, optional
        The number of items to pre-fetch from the generator before blocking (see Notes). Default: 1
    name: str, optional
        The thread name. if ``None`` a unique name is constructed of the form
        {generator.__name__}_N where N is an incrementing integer. Default: ``None``
    args: tuple, Optional
        The argument tuple for generator invocation. Default: ``None``.
    kwargs: dict, Optional
        keyword arguments for the generator invocation. Default: ``None``.

    Notes
    -----
    Putting to the internal queue only blocks if put is called while queue has already
    reached max size. Therefore this means prefetch is actually 1 more than the parameter
    supplied (N in the queue, one waiting for insertion)

    References
    ----------
    https://stackoverflow.com/questions/7323664/
    """
    def __init__(self,
                 generator: Callable,
                 prefetch: int = 1,
                 name: str | None = None,
                 args: tuple | None = None,
                 kwargs: dict[str, T.Any] | None = None) -> None:
        super().__init__(name=name, target=self._run)
        self.queue: Queue.Queue = Queue.Queue(prefetch)
        self.generator = generator
        self._gen_args = args or tuple()
        self._gen_kwargs = kwargs or {}
        self.start()

    def _run(self) -> None:
        """ Run the :attr:`_generator` and put into the queue until until queue size is reached.

        Raises
        ------
        Exception
            If there is a failure to run the generator and put to the queue
        """
        try:
            for item in self.generator(*self._gen_args, **self._gen_kwargs):
                self.queue.put(item)
            self.queue.put(None)
        except Exception:
            self.queue.put(None)
            raise

    def iterator(self) -> Generator:
        """ Iterate items out of the queue

        Yields
        ------
        Any
            The items from the generator
        """
        while True:
            next_item = self.queue.get()
            self.check_and_raise_error()
            if next_item is None or next_item == "EOF":
                logger.debug("Got EOF OR NONE in BackgroundGenerator")
                break
            yield next_item
