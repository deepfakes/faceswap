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

# ### << START: ROTATING FILE HANDLER - WINDOWS PERMISSION FIX >> ###
# This works around file locking issue on Windows specifically in the case of
# long lived child processes.
#
# Python opens files with inheritable handle and without file sharing by
# default. This causes the RotatingFileHandler file handle to be duplicated in
# the subprocesses even if the log file is not used in it. Because of this
# handle in the child process, when the RotatingFileHandler tries to os.rename()
# the file in the parent process, it fails with:
#     WindowsError: [Error 32] The process cannot access the file because
#     it is being used by another process
# Taken from: https://github.com/luci/client-py/blob/master/utils/logging_utils.py
# # Copyright 2015 The LUCI Authors. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0


if sys.platform == "win32":
    import codecs
    import ctypes
    import msvcrt  # pylint: disable=F0401
    import _subprocess  # noqa pylint: disable=F0401,W0611

    FILE_ATTRIBUTE_NORMAL = 0x00000080
    FILE_SHARE_READ = 1
    FILE_SHARE_WRITE = 2
    FILE_SHARE_DELETE = 4
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    OPEN_ALWAYS = 4

    def shared_open(path):
        """Opens a file with full sharing mode and without inheritance.

        The file is open for both read and write.

        See https://bugs.python.org/issue15244 for inspiration.
        """
        path = str(path)
        handle = ctypes.windll.kernel32.CreateFileW(
            path,
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            None,
            OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            None)
        ctr_handle = msvcrt.open_osfhandle(
            handle,
            os.O_BINARY | os.O_NOINHERIT)  # pylint: disable=no-member
        return os.fdopen(ctr_handle, "r+b")

    class NoInheritRotatingFileHandler(RotatingFileHandler):
        """ Overide Rotating FileHandler for Windows """
        def _open(self):
            """Opens the log file without handle inheritance but with file sharing.

            Ignores self.mode.
            """
            winf = shared_open(self.baseFilename)
            if self.encoding:
                # Do the equivalent of
                # codecs.open(self.baseFilename, self.mode, self.encoding)
                info = codecs.lookup(self.encoding)
                winf = codecs.StreamReaderWriter(
                    winf, info.streamreader, info.streamwriter, "replace")
                winf.encoding = self.encoding
            return winf
else:  # Not Windows.
    NoInheritRotatingFileHandler = RotatingFileHandler
# ### << END: ROTATING FILE HANDLER - WINDOWS PERMISSION FIX >> ###


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
    """ Override formatter to strip newlines and multiple spaces from logger """
    def format(self, record):
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


def log_setup(loglevel):
    """ initial log set up. """
    numeric_loglevel = get_loglevel(loglevel)
    root_loglevel = min(logging.DEBUG, numeric_loglevel)
    set_root_logger(loglevel=root_loglevel)
    log_format = FaceswapFormatter("%(asctime)s %(processName)-15s %(threadName)-15s "
                                   "%(module)-15s %(funcName)-25s %(levelname)-8s %(message)s",
                                   datefmt="%m/%d/%Y %H:%M:%S")
    f_handler = file_handler(numeric_loglevel, log_format)
    s_handler = stream_handler(numeric_loglevel)
    c_handler = crash_handler(log_format)

    q_listener = QueueListener(LOG_QUEUE, f_handler, s_handler, c_handler,
                               respect_handler_level=True)
    q_listener.start()
    logging.info("Log level set to: %s", loglevel.upper())


def file_handler(loglevel, log_format):
    """ Add a logging rotating file handler """
    filename = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap.log")
    should_rotate = os.path.isfile(filename)
    log_file = NoInheritRotatingFileHandler(filename, backupCount=1)
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


# Set logger class to custom logger
logging.setLoggerClass(MultiProcessingLogger)

# Stores the last 50 debug messages
debug_buffer = RollingBuffer(maxlen=50)  # pylint: disable=invalid-name
