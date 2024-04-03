#!/usr/bin/env python3
""" Queue Manager for faceswap

    NB: Keep this in it's own module! If it gets loaded from
    a multiprocess on a Windows System it will break Faceswap"""

import logging
import threading

from queue import Queue, Empty as QueueEmpty  # pylint:disable=unused-import; # noqa
from time import sleep

logger = logging.getLogger(__name__)


class EventQueue(Queue):
    """ Standard Queue object with a separate global shutdown parameter indicating that the main
    process, and by extension this queue, should be shut down.

    Parameters
    ----------
    shutdown_event: :class:`threading.Event`
        The global shutdown event common to all managed queues
    maxsize: int, Optional
        Upperbound limit on the number of items that can be placed in the queue. Default: `0`
    """
    def __init__(self, shutdown_event: threading.Event, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._shutdown = shutdown_event

    @property
    def shutdown(self) -> threading.Event:
        """ :class:`threading.Event`: The global shutdown event """
        return self._shutdown


class _QueueManager():
    """ Manage :class:`EventQueue` objects for availabilty across processes.

        Notes
        -----
        Don't import this class directly, instead import via :func:`queue_manager` """
    def __init__(self) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)

        self.shutdown = threading.Event()
        self.queues: dict[str, EventQueue] = {}
        logger.debug("Initialized %s", self.__class__.__name__)

    def add_queue(self, name: str, maxsize: int = 0, create_new: bool = False) -> str:
        """ Add a :class:`EventQueue` to the manager.

        Parameters
        ----------
        name: str
            The name of the queue to create
        maxsize: int, optional
            The maximum queue size. Set to `0` for unlimited. Default: `0`
        create_new: bool, optional
            If a queue of the given name exists, and this value is ``False``, then an error is
            raised preventing the creation of duplicate queues. If this value is ``True`` and
            the given name exists then an integer is appended to the end of the queue name and
            incremented until the given name is unique. Default: ``False``

        Returns
        -------
        str
            The final generated name for the queue
        """
        logger.debug("QueueManager adding: (name: '%s', maxsize: %s, create_new: %s)",
                     name, maxsize, create_new)
        if not create_new and name in self.queues:
            raise ValueError(f"Queue '{name}' already exists.")
        if create_new and name in self.queues:
            i = 0
            while name in self.queues:
                name = f"{name}{i}"
            logger.debug("Duplicate queue name. Updated to: '%s'", name)

        self.queues[name] = EventQueue(self.shutdown, maxsize=maxsize)
        logger.debug("QueueManager added: (name: '%s')", name)
        return name

    def del_queue(self, name: str) -> None:
        """ Remove a queue from the manager

        Parameters
        ----------
        name: str
            The name of the queue to be deleted. Must exist within the queue manager.
        """
        logger.debug("QueueManager deleting: '%s'", name)
        del self.queues[name]
        logger.debug("QueueManager deleted: '%s'", name)

    def get_queue(self, name: str, maxsize: int = 0) -> EventQueue:
        """ Return a :class:`EventQueue` from the manager. If it doesn't exist, create it.

        Parameters
        ----------
        name: str
            The name of the queue to obtain
        maxsize: int, Optional
            The maximum queue size. Set to `0` for unlimited. Only used if the requested queue
            does not already exist. Default: `0`
         """
        logger.debug("QueueManager getting: '%s'", name)
        queue = self.queues.get(name)
        if not queue:
            self.add_queue(name, maxsize)
            queue = self.queues[name]
        logger.debug("QueueManager got: '%s'", name)
        return queue

    def terminate_queues(self) -> None:
        """ Terminates all managed queues.

        Sets the global shutdown event, clears and send EOF to all queues.  To be called if there
        is an error """
        logger.debug("QueueManager terminating all queues")
        self.shutdown.set()
        self._flush_queues()
        for q_name, queue in self.queues.items():
            logger.debug("QueueManager terminating: '%s'", q_name)
            queue.put("EOF")
        logger.debug("QueueManager terminated all queues")

    def _flush_queues(self):
        """ Empty out the contents of every managed queue. """
        for q_name in self.queues:
            self.flush_queue(q_name)
        logger.debug("QueueManager flushed all queues")

    def flush_queue(self, name: str) -> None:
        """ Flush the contents from a managed queue.

        Parameters
        ----------
        name: str
            The name of the managed :class:`EventQueue` to flush
        """
        logger.debug("QueueManager flushing: '%s'", name)
        queue = self.queues[name]
        while not queue.empty():
            queue.get(True, 1)

    def debug_monitor(self, update_interval: int = 2) -> None:
        """ A debug tool for monitoring managed :class:`EventQueues`.

        Prints queue sizes to the console for all managed queues.

        Parameters
        ----------
        update_interval: int, Optional
            The number of seconds between printing information to the console. Default: 2
        """
        thread = threading.Thread(target=self._debug_queue_sizes,
                                  args=(update_interval, ))
        thread.daemon = True
        thread.start()

    def _debug_queue_sizes(self, update_interval) -> None:
        """ Print the queue size for each managed queue to console.

        Parameters
        ----------
        update_interval: int
            The number of seconds between printing information to the console
        """
        while True:
            logger.info("====================================================")
            for name in sorted(self.queues.keys()):
                logger.info("%s: %s", name, self.queues[name].qsize())
            sleep(update_interval)


queue_manager = _QueueManager()  # pylint:disable=invalid-name
