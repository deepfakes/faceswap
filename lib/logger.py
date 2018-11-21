#!/usr/bin/python
""" Logging Setup """
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import os
import sys

from lib.queue_manager import queue_manager


class MultiProcessingLogger(logging.Logger):
    """ Create custom logger  with custom levels """
    def __init__(self, name):
        for new_level in (('VERBOSE', 15), ('TRACE', 5)):
            level_name, level_num = new_level
            if hasattr(logging, level_name):
                continue
            logging.addLevelName(level_num, level_name)
            setattr(logging, level_name, level_num)
        super().__init__(name)

    def verbose(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'VERBOSE'.
        """
        if self.isEnabledFor(15):
            self._log(15, msg, args, **kwargs)

    def trace(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'VERBOSE'.
        """
        if self.isEnabledFor(5):
            self._log(5, msg, args, **kwargs)


logging.setLoggerClass(MultiProcessingLogger)


def log_setup():
    """ initial log set up. """
    log_queue = queue_manager.get_queue("logger")
    q_handler = QueueHandler(log_queue)
    f_handler = file_handler()
    s_handler = stream_handler()
    q_listener = QueueListener(log_queue, f_handler, s_handler)
#    q_listener._sentinel = "EOF"

    rootlogger = logging.getLogger()
    rootlogger.setLevel(logging.INFO)

    rootlogger.addHandler(q_handler)
    q_listener.start()


def file_handler():
    """ Add a logging rotating file handler """
    log_format = logging.Formatter("%(asctime)s %(threadName)-15s %(module)-15s %(funcName)-25s "
                                   "%(levelname)-8s %(message)s", datefmt="%m/%d/%Y %H:%M:%S")

    filename = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap.log")
    should_rotate = os.path.isfile(filename)
    log_file = RotatingFileHandler(filename, backupCount=1)
    if should_rotate:
        log_file.doRollover()
    log_file.setFormatter(log_format)
    return log_file


def stream_handler():
    """ Add a logging cli handler """
    log_format = logging.Formatter("%(asctime)s %(module)-20s %(levelname)-8s %(message)s",
                                   datefmt="%m/%d/%Y %H:%M:%S")

    log_console = logging.StreamHandler()
    log_console.setFormatter(log_format)
    return log_console


def set_loglevel(loglevel):
    ''' Check valid log level supplied and set log level '''
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    logging.info('Log level set to: %s', loglevel.upper())
