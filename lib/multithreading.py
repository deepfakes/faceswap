#!/usr/bin/env python3
""" Multithreading/processing utils for faceswap """

import logging
from multiprocessing import cpu_count

import queue as Queue
import sys
import threading

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def total_cpus():
    """ Return total number of cpus """
    return cpu_count()


class FSThread(threading.Thread):
    """ Subclass of thread that passes errors back to parent """
    def __init__(self, group=None, target=None, name=None,  # pylint: disable=too-many-arguments
                 args=(), kwargs=None, *, daemon=None):
        super().__init__(group=group, target=target, name=name,
                         args=args, kwargs=kwargs, daemon=daemon)
        self.err = None

    def check_and_raise_error(self):
        """ Checks for errors in thread and raises them in caller """
        if not self.err:
            return
        logger.debug("Thread error caught: %s", self.err)
        raise self.err[1].with_traceback(self.err[2])

    def run(self):
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception as err:  # pylint: disable=broad-except
            self.err = sys.exc_info()
            logger.debug("Error in thread (%s): %s", self._name, str(err))
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs


class MultiThread():
    """ Threading for IO heavy ops
        Catches errors in thread and rethrows to parent """
    def __init__(self, target, *args, thread_count=1, name=None, **kwargs):
        self._name = name if name else target.__name__
        logger.debug("Initializing %s: (target: '%s', thread_count: %s)",
                     self.__class__.__name__, self._name, thread_count)
        logger.trace("args: %s, kwargs: %s", args, kwargs)
        self.daemon = True
        self._thread_count = thread_count
        self._threads = list()
        self._target = target
        self._args = args
        self._kwargs = kwargs
        logger.debug("Initialized %s: '%s'", self.__class__.__name__, self._name)

    @property
    def has_error(self):
        """ Return true if a thread has errored, otherwise false """
        return any(thread.err for thread in self._threads)

    @property
    def errors(self):
        """ Return a list of thread errors """
        return [thread.err for thread in self._threads if thread.err]

    @property
    def name(self):
        """ Return thread name """
        return self._name

    def check_and_raise_error(self):
        """ Checks for errors in thread and raises them in caller """
        if not self.has_error:
            return
        logger.debug("Thread error caught: %s", self.errors)
        error = self.errors[0]
        raise error[1].with_traceback(error[2])

    def is_alive(self):
        """ Return true if any thread is alive else false """
        return any(thread.is_alive() for thread in self._threads)

    def start(self):
        """ Start a thread with the given method and args """
        logger.debug("Starting thread(s): '%s'", self._name)
        for idx in range(self._thread_count):
            name = "{}_{}".format(self._name, idx)
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

    def completed(self):
        """ Return False if there are any alive threads else True """
        retval = all(not thread.is_alive() for thread in self._threads)
        logger.debug(retval)
        return retval

    def join(self):
        """ Join the running threads, catching and re-raising any errors """
        logger.debug("Joining Threads: '%s'", self._name)
        for thread in self._threads:
            logger.debug("Joining Thread: '%s'", thread._name)  # pylint: disable=protected-access
            thread.join()
            if thread.err:
                logger.error("Caught exception in thread: '%s'",
                             thread._name)  # pylint: disable=protected-access
                raise thread.err[1].with_traceback(thread.err[2])
        logger.debug("Joined all Threads: '%s'", self._name)


class BackgroundGenerator(MultiThread):
    """ Run a queue in the background. From:
        https://stackoverflow.com/questions/7323664/ """
    # See below why prefetch count is flawed
    def __init__(self, generator, prefetch=1, thread_count=2,
                 queue=None, args=None, kwargs=None):
        # pylint:disable=too-many-arguments
        super().__init__(target=self._run, thread_count=thread_count)
        self.queue = queue or Queue.Queue(prefetch)
        self.generator = generator
        self._gen_args = args or tuple()
        self._gen_kwargs = kwargs or dict()
        self.start()

    def _run(self):
        """ Put until queue size is reached.
            Note: put blocks only if put is called while queue has already
            reached max size => this makes prefetch + thread_count prefetched items!
            N in the the queue, one waiting for insertion per thread! """
        try:
            for item in self.generator(*self._gen_args, **self._gen_kwargs):
                self.queue.put(item)
            self.queue.put(None)
        except Exception:
            self.queue.put(None)
            raise

    def iterator(self):
        """ Iterate items out of the queue """
        while True:
            next_item = self.queue.get()
            self.check_and_raise_error()
            if next_item is None or next_item == "EOF":
                logger.debug("Got EOF OR NONE in BackgroundGenerator")
                break
            yield next_item
