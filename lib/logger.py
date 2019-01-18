#!/usr/bin/python
""" Logging Setup """
import collections
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import os
import re
import sys
import traceback

from datetime import datetime
from time import sleep

from lib.queue_manager import queue_manager
from lib.sysinfo import sysinfo

LOG_QUEUE = queue_manager._log_queue  # pylint: disable=protected-access


class MultiProcessingLogger(logging.Logger):
    """ Create custom logger  with custom levels """
    def __init__(self, name):
        for new_level in (("VERBOSE", 15), ("TRACE", 5)):
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


class FaceswapFormatter(logging.Formatter):
    """ Override formatter to strip newlines and multiple spaces from logger
        Messages that begin with "R|" should be handled as is
    """
    def format(self, record):
        if record.msg.startswith("R|"):
            record.msg = record.msg[2:]
            record.strip_spaces = False
        elif record.strip_spaces:
            record.msg = re.sub(" +", " ", record.msg.replace("\n", "\\n").replace("\r", "\\r"))
        return super().format(record)


class RollingBuffer(collections.deque):
    """File-like that keeps a certain number of lines of text in memory."""
    def write(self, buffer):
        """ Write line to buffer """
        for line in buffer.rstrip().splitlines():
            self.append(line + "\n")


def set_root_logger(loglevel=logging.INFO, queue=LOG_QUEUE):
    """ Setup the root logger.
        Loaded in main process and into any spawned processes
        Automatically added in multithreading.py"""
    rootlogger = logging.getLogger()
    q_handler = QueueHandler(queue)
    rootlogger.addHandler(q_handler)
    rootlogger.setLevel(loglevel)


def log_setup(loglevel, logfile, command):
    """ initial log set up. """
    numeric_loglevel = get_loglevel(loglevel)
    root_loglevel = min(logging.DEBUG, numeric_loglevel)
    set_root_logger(loglevel=root_loglevel)
    log_format = FaceswapFormatter("%(asctime)s %(processName)-15s %(threadName)-15s "
                                   "%(module)-15s %(funcName)-25s %(levelname)-8s %(message)s",
                                   datefmt="%m/%d/%Y %H:%M:%S")
    f_handler = file_handler(numeric_loglevel, logfile, log_format, command)
    s_handler = stream_handler(numeric_loglevel)
    c_handler = crash_handler(log_format)

    q_listener = QueueListener(LOG_QUEUE, f_handler, s_handler, c_handler,
                               respect_handler_level=True)
    q_listener.start()
    logging.info("Log level set to: %s", loglevel.upper())


def file_handler(loglevel, logfile, log_format, command):
    """ Add a logging rotating file handler """
    if logfile is not None:
        filename = logfile
    else:
        filename = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap")
        # Windows has issues sharing the log file with subprocesses, so log GUI separately
        filename += "_gui.log" if command == "gui" else ".log"

    should_rotate = os.path.isfile(filename)
    log_file = RotatingFileHandler(filename, backupCount=1)
    if should_rotate:
        log_file.doRollover()
    log_file.setFormatter(log_format)
    log_file.setLevel(loglevel)
    return log_file


def stream_handler(loglevel):
    """ Add a logging cli handler """
    # Don't set stdout to lower than verbose
    loglevel = max(loglevel, 15)
    log_format = FaceswapFormatter("%(asctime)s %(levelname)-8s %(message)s",
                                   datefmt="%m/%d/%Y %H:%M:%S")

    log_console = logging.StreamHandler(sys.stdout)
    log_console.setFormatter(log_format)
    log_console.setLevel(loglevel)
    return log_console


def crash_handler(log_format):
    """ Add a handler that sores the last 50 debug lines to `debug_buffer`
        for use in crash reports """
    log_crash = logging.StreamHandler(debug_buffer)
    log_crash.setFormatter(log_format)
    log_crash.setLevel(logging.DEBUG)
    return log_crash


def get_loglevel(loglevel):
    """ Check valid log level supplied and return numeric log level """
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)

    return numeric_level


def crash_log():
    """ Write debug_buffer to a crash log on crash """
    path = os.getcwd()
    filename = os.path.join(path, datetime.now().strftime("crash_report.%Y.%m.%d.%H%M%S%f.log"))

    # Wait until all log items have been processed
    while not LOG_QUEUE.empty():
        sleep(1)

    freeze_log = list(debug_buffer)
    with open(filename, "w") as outfile:
        outfile.writelines(freeze_log)
        traceback.print_exc(file=outfile)
        outfile.write(sysinfo.full_info())
    return filename


# Add a flag to logging.LogRecord to not strip formatting from particular records
old_factory = logging.getLogRecordFactory()


def faceswap_logrecord(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.strip_spaces = True
    return record


logging.setLogRecordFactory(faceswap_logrecord)

# Set logger class to custom logger
logging.setLoggerClass(MultiProcessingLogger)

# Stores the last 50 debug messages
debug_buffer = RollingBuffer(maxlen=50)  # pylint: disable=invalid-name
