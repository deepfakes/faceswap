#!/usr/bin/env python3
""" Queue Manager for faceswap

    NB: Keep this in it's own module! If it gets loaded from
    a multiprocess on a Windows System it will break Faceswap"""

import logging
import threading

from queue import Queue, Empty as QueueEmpty  # pylint: disable=unused-import; # noqa
from time import sleep

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class QueueManager():
    """ Manage queues for availabilty across processes
        Don't import this class directly, instead
        import the variable: queue_manager """
    def __init__(self):
        logger.debug("Initializing %s", self.__class__.__name__)

        self.shutdown = threading.Event()
        self.queues = dict()
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_queue(self, name, maxsize=0):
        """ Add a queue to the manager

            Adds an event "shutdown" to the queue that can be used to indicate
            to a process that any activity on the queue should cease """

        logger.debug("QueueManager adding: (name: '%s', maxsize: %s)", name, maxsize)
        if name in self.queues.keys():
            raise ValueError("Queue '{}' already exists.".format(name))

        queue = Queue(maxsize=maxsize)

        setattr(queue, "shutdown", self.shutdown)
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
        """ Set shutdown event, clear and send EOF to all queues
            To be called if there is an error """
        logger.debug("QueueManager terminating all queues")
        self.shutdown.set()
        self.flush_queues()
        for q_name, queue in self.queues.items():
            logger.debug("QueueManager terminating: '%s'", q_name)
            queue.put("EOF")
        logger.debug("QueueManager terminated all queues")

    def flush_queues(self):
        """ Empty out all queues """
        for q_name in self.queues.keys():
            self.flush_queue(q_name)
        logger.debug("QueueManager flushed all queues")

    def flush_queue(self, q_name):
        """ Empty out a specific queue """
        logger.debug("QueueManager flushing: '%s'", q_name)
        queue = self.queues[q_name]
        while not queue.empty():
            queue.get(True, 1)

    def debug_monitor(self, update_secs=2):
        """ Debug tool for monitoring queues """
        thread = threading.Thread(target=self.debug_queue_sizes,
                                  args=(update_secs, ))
        thread.daemon = True
        thread.start()

    def debug_queue_sizes(self, update_secs):
        """ Output the queue sizes
            logged to INFO so it also displays in console
        """
        while True:
            logger.info("====================================================")
            for name in sorted(self.queues.keys()):
                logger.info("%s: %s", name, self.queues[name].qsize())
            sleep(update_secs)


queue_manager = QueueManager()  # pylint: disable=invalid-name
