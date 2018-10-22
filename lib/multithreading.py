#!/usr/bin/env python3
""" Multithreading/processing utils for faceswap """

import multiprocessing as mp
import queue as Queue
from queue import Empty as QueueEmpty  # Used for imports
import threading
from time import sleep


class QueueManager():
    """ Manage queues for availabilty across processes
        Don't import this class directly, instead
        import the variable: queue_manager """
    def __init__(self):
        self.manager = mp.Manager()
        self.queues = dict()

    def add_queue(self, name, maxsize=0):
        """ Add a queue to the manager """
        if name in self.queues.keys():
            raise ValueError("Queue '{}' already exists.".format(name))
        queue = self.manager.Queue(maxsize=maxsize)
        self.queues[name] = queue

    def del_queue(self, name):
        """ remove a queue from the manager """
        del self.queues[name]

    def get_queue(self, name, maxsize=0):
        """ Return a queue from the manager
            If it doesn't exist, create it """
        queue = self.queues.get(name, None)
        if queue:
            return queue
        self.add_queue(name, maxsize)
        return self.queues[name]

    def terminate_queues(self):
        """ Clear all queues and send EOF
            To be called if there is an error """
        for queue in self.queues.values():
            while not queue.empty():
                queue.get()
            queue.put("EOF")

    def debug_monitor(self, update_secs=2):
        """ Debug tool for monitoring queues """
        thread = MultiThread(thread_count=update_secs)
        thread.in_thread(self.debug_queue_sizes)

    def debug_queue_sizes(self):
        """ Output the queue sizes """
        while True:
            print("=== QUEUE SIZES ===")
            for name in sorted(self.queues.keys()):
                print(name, self.queues[name].qsize())
            print("====================\n")
            sleep(2)


queue_manager = QueueManager()


class PoolProcess():
    """ Pool multiple processes """
    def __init__(self, method, processes=None, verbose=False):
        self.verbose = verbose
        self.method = method
        self.procs = self.set_procs(processes)

    def set_procs(self, processes):
        """ Set the number of processes to use """
        if processes is None:
            running_processes = len(mp.active_children())
            processes = max(mp.cpu_count() - running_processes, 1)
        if self.verbose:
            print("Processing in {} processes".format(processes))
        return processes

    def in_process(self, *args, **kwargs):
        """ Run the processing pool """
        pool = mp.Pool(processes=self.procs)
        for _ in range(self.procs):
            pool.apply_async(self.method, args=args, kwds=kwargs)


class SpawnProcess():
    """ Process in spawnable context
        Must be spawnable to share CUDA across processes """
    def __init__(self):
        self.context = mp.get_context("spawn")
        self.daemonize = True
        self.process = None
        self.event = self.context.Event()

    def in_process(self, target, *args, **kwargs):
        """ Start a process in the spawn context """
        kwargs["event"] = self.event
        self.process = self.context.Process(target=target,
                                            args=args,
                                            kwargs=kwargs)
        self.process.daemon = self.daemonize
        self.process.start()

    def join(self):
        """ Join the process """
        self.process.join()


class MultiThread():
    """ Threading for IO heavy ops """
    def __init__(self, thread_count=1):
        self.thread_count = thread_count
        self.threads = list()

    def in_thread(self, target, *args, **kwargs):
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
