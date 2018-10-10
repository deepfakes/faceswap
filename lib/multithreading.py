#!/usr/bin/env python3
""" Multithreading/processing utils for faceswap """

import multiprocessing as mp
import queue as Queue
import threading


class PoolProcess():
    """ Pool multiple processes """
    def __init__(self, method,
                 initializer=None, processes=None, verbose=False):
        self.verbose = verbose
        self.initializer = initializer
        self.method = method
        self.procs = self.set_procs(processes)

    def set_procs(self, processes):
        """ Set the number of processes to use """
        if processes is None:
            processes = mp.cpu_count()
        if self.verbose:
            print("Processing in {} processes".format(processes))
        return processes

    def process(self, data):
        """ Run the processing pool """
        pool = mp.Pool(processes=self.procs, initializer=self.initializer)
        for item in pool.imap_unordered(self.method, data):
            yield item if item is not None else 0


class MultiThread():
    """ Threading for IO heavy ops """
    def __init__(self, thread_count=1, queue_size=100):
        self.thread_count = thread_count
        self.queue = mp.Queue(maxsize=queue_size)
        self.threads = list()

    def in_thread(self, target=None, args=(), kwargs=None):
        """ Start a thread with the given method and args """
        for _ in range(self.thread_count):
            thread = threading.Thread(target=target, args=args, kwargs=kwargs)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def join_threads(self):
        """ Join the running threads """
        for thread in self.threads:
            thread.join()


class BackgroundGenerator(threading.Thread):
    """ Run a queue in the background. From:
        https://stackoverflow.com/questions/7323664/ """
    # See below why prefetch count is flawed
    def __init__(self, generator, prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(prefetch)
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
