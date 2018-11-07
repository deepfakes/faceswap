#!/usr/bin/env python3
""" Queue Manager for faceswap

    NB: Keep this in it's own module! If it gets loaded from
    a multiprocess on a Windows System it will break Faceswap"""

import multiprocessing as mp
import threading

from queue import Empty as QueueEmpty  # Used for imports
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
        thread = threading.Thread(target=self.debug_queue_sizes,
                                  args=(update_secs, ))
        thread.daemon = True
        thread.start()

    def debug_queue_sizes(self, update_secs):
        """ Output the queue sizes """
        while True:
            print("=== QUEUE SIZES ===")
            for name in sorted(self.queues.keys()):
                print(name, self.queues[name].qsize())
            print("====================\n")
            sleep(update_secs)


queue_manager = QueueManager()
