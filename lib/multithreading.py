#!/usr/bin/env python3
""" Multithreading/processing utils for faceswap """

import logging
import multiprocessing as mp
import queue as Queue
import threading

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PoolProcess():
    """ Pool multiple processes """
    def __init__(self, method, processes=None, verbose=False):
        logger.debug("Initializing %s: %s", self.__class__.__name__, locals())
        self.verbose = verbose
        self.method = method
        self.procs = self.set_procs(processes)
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def set_procs(processes):
        """ Set the number of processes to use """
        if processes is None:
            running_processes = len(mp.active_children())
            processes = max(mp.cpu_count() - running_processes, 1)
        logger.verbose("Processing in %s processes", processes)
        return processes

    def in_process(self, *args, **kwargs):
        """ Run the processing pool """
        pool = mp.Pool(processes=self.procs)
        for idx in range(self.procs):
            logger.debug("Adding process %s of %s to mp.Pool: %s",
                         idx + 1, self.procs, locals())
            pool.apply_async(self.method, args=args, kwds=kwargs)


class SpawnProcess():
    """ Process in spawnable context
        Must be spawnable to share CUDA across processes """
    def __init__(self):
        logger.debug("Initializing %s: %s", self.__class__.__name__, locals())
        self.context = mp.get_context("spawn")
        self.daemonize = True
        self.process = None
        self.event = self.context.Event()
        logger.debug("Initialized %s", self.__class__.__name__)

    def in_process(self, target, *args, **kwargs):
        """ Start a process in the spawn context """
        kwargs["event"] = self.event
        logger.debug("Spawning Process: %s", locals())
        self.process = self.context.Process(target=target,
                                            args=args,
                                            kwargs=kwargs)
        self.process.daemon = self.daemonize
        self.process.start()

    def join(self):
        """ Join the process """
        logger.debug("Joining Process: %s", locals())
        self.process.join()


class MultiThread():
    """ Threading for IO heavy ops """
    def __init__(self, thread_count=1):
        logger.debug("Initializing %s: %s", self.__class__.__name__, locals())
        self.thread_count = thread_count
        self.threads = list()
        logger.debug("Initialized %s", self.__class__.__name__)

    def in_thread(self, target, *args, **kwargs):
        """ Start a thread with the given method and args """
        for idx in range(self.thread_count):
            name = "{}_{}".format(target.__name__, idx)
            logger.debug("Adding thread %s of %s to Threads: %s",
                         idx + 1, self.thread_count, locals())
            thread = threading.Thread(name=name, target=target, args=args, kwargs=kwargs)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def join_threads(self):
        """ Join the running threads """
        for idx, thread in enumerate(self.threads):
            logger.debug("Joining Thread: %s: %s", idx + 1, locals())
            thread.join()


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
