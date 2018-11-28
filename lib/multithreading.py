#!/usr/bin/env python3
""" Multithreading/processing utils for faceswap """

import logging
import multiprocessing as mp
import queue as Queue
import threading
from lib.logger import LOG_QUEUE, set_root_logger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_launched_threads = set()  # pylint: disable=invalid-name


class PoolProcess():
    """ Pool multiple processes """
    def __init__(self, method, in_queue, out_queue, *args, processes=None, **kwargs):
        self._name = method.__name__
        logger.debug("Initializing '%s': (target: '%s', processes: %s)",
                     self.__class__.__name__, self._name, processes)

        self.procs = self.set_procs(processes)
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(processes=self.procs)

        self._method = method
        self._kwargs = self.build_target_kwargs(in_queue, out_queue, kwargs)
        self._args = args

        logger.debug("Initialized '%s': '%s'", self.__class__.__name__, self._name)

    @staticmethod
    def build_target_kwargs(in_queue, out_queue, kwargs):
        """ Add standard kwargs to passed in kwargs list """
        kwargs["log_init"] = set_root_logger
        kwargs["log_queue"] = LOG_QUEUE
        kwargs["in_queue"] = in_queue
        kwargs["out_queue"] = out_queue
        return kwargs

    def set_procs(self, processes):
        """ Set the number of processes to use """
        if processes is None:
            running_processes = len(mp.active_children())
            processes = max(mp.cpu_count() - running_processes, 1)
        logger.verbose("Processing '%s' in %s processes", self._name, processes)
        return processes

    def start(self):
        """ Run the processing pool """
        logging.debug("Pooling Processes: (target: '%s', args: %s, kwargs: %s)",
                      self._name, self._args, self._kwargs)
        for idx in range(self.procs):
            logger.debug("Adding process %s of %s to mp.Pool '%s'",
                         idx + 1, self.procs, self._name)
            self.pool.apply_async(self._method, args=self._args, kwds=self._kwargs)
        logging.debug("Pooled Processes: '%s'", self._name)

    def join(self):
        """ Join the process """
        logger.debug("Joining Pooled Process: '%s'", self._name)
        self.pool.close()
        self.pool.join()
        logger.debug("Joined Pooled Process: '%s'", self._name)


class SpawnProcess(mp.context.SpawnProcess):
    """ Process in spawnable context
        Must be spawnable to share CUDA across processes """
    def __init__(self, target, in_queue, out_queue, *args, **kwargs):
        name = target.__name__
        logger.debug("Initializing '%s': (target: '%s', args: %s, kwargs: %s)",
                     self.__class__.__name__, name, args, kwargs)
        ctx = mp.get_context("spawn")
        self.event = ctx.Event()
        kwargs = self.build_target_kwargs(in_queue, out_queue, kwargs)
        super().__init__(target=target, name=name, args=args, kwargs=kwargs)
        self.daemon = True
        logger.debug("Initialized '%s': '%s'", self.__class__.__name__, name)

    def build_target_kwargs(self, in_queue, out_queue, kwargs):
        """ Add standard kwargs to passed in kwargs list """
        kwargs["event"] = self.event
        kwargs["log_init"] = set_root_logger
        kwargs["log_queue"] = LOG_QUEUE
        kwargs["in_queue"] = in_queue
        kwargs["out_queue"] = out_queue
        return kwargs

    def start(self):
        """ Add logging to start function """
        logger.debug("Spawning Process: (name: '%s', args: %s, kwargs: %s, daemon: %s)",
                     self._name, self._args, self._kwargs, self.daemon)
        super().start()
        logger.debug("Spawned Process: (name: '%s', PID: %s)", self._name, self.pid)

    def join(self, timeout=None):
        """ Add logging to join function """
        logger.debug("Joining Process: (name: '%s', PID: %s)", self._name, self.pid)
        super().join(timeout=timeout)
        logger.debug("Joined Process: (name: '%s', PID: %s)", self._name, self.pid)


class MultiThread():
    """ Threading for IO heavy ops """
    def __init__(self, target, *args, thread_count=1, **kwargs):
        self._name = target.__name__
        logger.debug("Initializing '%s': (target: '%s', thread_count: %s, args: %s, kwargs: %s)",
                     self.__class__.__name__, self._name, thread_count, args, kwargs)
        self.daemon = True
        self._thread_count = thread_count
        self._threads = list()
        self._target = target
        self._args = args
        self._kwargs = kwargs
        logger.debug("Initialized '%s': '%s'", self.__class__.__name__, self._name)

    def start(self):
        """ Start a thread with the given method and args """
        logger.debug("Starting thread(s): '%s'", self._name)
        for idx in range(self._thread_count):
            name = "{}_{}".format(self._name, idx)
            logger.debug("Starting thread %s of %s: '%s'",
                         idx + 1, self._thread_count, name)
            thread = threading.Thread(name=name,
                                      target=self._target,
                                      args=self._args,
                                      kwargs=self._kwargs)
            thread.daemon = self.daemon
            thread.start()
            _launched_threads.add(thread)
            self._threads.append(thread)
        logger.debug("Started all threads '%s': %s", self._name, len(self._threads))

    def join(self):
        """ Join the running threads """
        logger.debug("Joining Threads: '%s'", self._name)
        for thread in self._threads:
            logger.debug("Joining Thread: '%s'", thread._name)  # pylint: disable=protected-access
            thread.join()
            _launched_threads.remove(thread)
        logger.debug("Joined all Threads: '%s'", self._name)


class BackgroundGenerator(threading.Thread):
    """ Run a queue in the background. From:
        https://stackoverflow.com/questions/7323664/ """
    # See below why prefetch count is flawed
    def __init__(self, generator, prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(maxsize=prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        """ Put until queue size is reached.
            Note: put blocks only if put is called while queue has already
            reached max size => this makes 2 prefetched items! One in the
            queue, one waiting for insertion! """
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def iterator(self):
        """ Iterate items out of the queue """
        while True:
            next_item = self.queue.get()
            if next_item is None:
                break
            yield next_item


def terminate_threads():
    """ Join all active threads on unexpected shutdown

        If the thread is doing long running work, make sure you
        have a mechanism in place to terminate this work to avoid
        long blocks
    """
    for thread in _launched_threads:
        if thread.is_alive():
            thread.join()
