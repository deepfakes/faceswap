#!/usr/bin/env python3
""" Queue Manager for faceswap

    NB: Keep this in it's own module! If it gets loaded from
    a multiprocess on a Windows System it will break Faceswap"""

import logging
import multiprocessing as mp
import threading

from queue import Empty as QueueEmpty  # pylint: disable=unused-import; # noqa
from time import sleep

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class QueueManager():
    """ Manage queues for availabilty across processes
        Don't import this class directly, instead
        import the variable: queue_manager """
    def __init__(self):
        logger.debug("Initializing '%s'", self.__class__.__name__)
        self.manager = mp.Manager()
        self.queues = dict()
        logger.debug("Initialized '%s'", self.__class__.__name__)

    def add_queue(self, name, maxsize=0):
        """ Add a queue to the manager """
        logger.debug("QueueManager adding: (name: '%s', maxsize: %s)", name, maxsize)
        if name in self.queues.keys():
            raise ValueError("Queue '{}' already exists.".format(name))
        queue = self.manager.Queue(maxsize=maxsize)
        self.queues[name] = queue
        logger.debug("QueueManager added: (name: '%s')", name)

    def del_queue(self, name):
        """ remove a queue from the manager """
        logger.debug("QueueManager deleting: '%s'", name)
        del self.queues[name]
        logger.debug("QueueManager deleted: '%s'", name)

    def get_queue(self, name, maxsize=0):
        """ Return a queue from the manager
            If it doesn't exist, create it """
        logger.debug("QueueManager getting: '%s'", name)
        queue = self.queues.get(name, None)
        if not queue:
            self.add_queue(name, maxsize)
            queue = self.queues[name]
        logger.debug("QueueManager got: '%s'", name)
        return queue

    def terminate_queues(self):
        """ Clear all queues and send EOF
            To be called if there is an error """
        logger.debug("QueueManager terminating all queues")
        for q_name, queue in self.queues.items():
            if q_name == "logger":
                continue
            logger.debug("QueueManager terminating: '%s'", q_name)
            while not queue.empty():
                queue.get()
            queue.put("EOF")
        logger.debug("QueueManager terminated all queues")

    def del_queues(self):
        """ remove all queue from the manager """
        for q_name in list(self.queues.keys()):
            self.del_queue(q_name)

    def debug_monitor(self, update_secs=2):
        """ Debug tool for monitoring queues """
        thread = threading.Thread(target=self.debug_queue_sizes,
                                  args=(update_secs, ))
        thread.daemon = True
        thread.start()

    def debug_queue_sizes(self, update_secs):
        """ Output the queue sizes """
        while True:
            for name in sorted(self.queues.keys()):
                logger.debug("%s: %s", name, self.queues[name].qsize())
            sleep(update_secs)


queue_manager = QueueManager()  # pylint: disable=invalid-name
