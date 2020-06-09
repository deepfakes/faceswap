#!/usr/bin/python
""" Logging Setup """
import collections
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import traceback

from datetime import datetime
from tqdm import tqdm


class FaceswapLogger(logging.Logger):
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
    """ Override formatter to strip newlines the final message """

    def format(self, record):
        record.message = record.getMessage()
        record = self.rewrite_tf_deprecation(record)
        # strip newlines
        if "\n" in record.message or "\r" in record.message:
            record.message = record.message.replace("\n", "\\n").replace("\r", "\\r")

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        msg = self.formatMessage(record)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if msg[-1:] != "\n":
                msg = msg + "\n"
            msg = msg + record.exc_text
        if record.stack_info:
            if msg[-1:] != "\n":
                msg = msg + "\n"
            msg = msg + self.formatStack(record.stack_info)
        return msg

    @staticmethod
    def rewrite_tf_deprecation(record):
        """ Change TF deprecation messages from WARNING to DEBUG """
        if record.levelno == 30 and (record.funcName == "_tfmw_add_deprecation_warning" or
                                     record.module in("deprecation", "deprecation_wrapper")):
            record.levelno = 10
            record.levelname = "DEBUG"
        return record


class RollingBuffer(collections.deque):
    """File-like that keeps a certain number of lines of text in memory."""
    def write(self, buffer):
        """ Write line to buffer """
        for line in buffer.rstrip().splitlines():
            self.append(line + "\n")


class TqdmHandler(logging.StreamHandler):
    """ Use TQDM Write for outputting to console """
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def set_root_logger(loglevel=logging.INFO):
    """ Setup the root logger. """
    rootlogger = logging.getLogger()
    rootlogger.setLevel(loglevel)
    return rootlogger


def log_setup(loglevel, logfile, command, is_gui=False):
    """ initial log set up. """
    numeric_loglevel = get_loglevel(loglevel)
    root_loglevel = min(logging.DEBUG, numeric_loglevel)
    rootlogger = set_root_logger(loglevel=root_loglevel)
    log_format = FaceswapFormatter("%(asctime)s %(processName)-15s %(threadName)-15s "
                                   "%(module)-15s %(funcName)-25s %(levelname)-8s %(message)s",
                                   datefmt="%m/%d/%Y %H:%M:%S")
    f_handler = file_handler(numeric_loglevel, logfile, log_format, command)
    s_handler = stream_handler(numeric_loglevel, is_gui)
    c_handler = crash_handler(log_format)
    rootlogger.addHandler(f_handler)
    rootlogger.addHandler(s_handler)
    rootlogger.addHandler(c_handler)
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


def stream_handler(loglevel, is_gui):
    """ Add a logging cli handler """
    # Don't set stdout to lower than verbose
    loglevel = max(loglevel, 15)
    log_format = FaceswapFormatter("%(asctime)s %(levelname)-8s %(message)s",
                                   datefmt="%m/%d/%Y %H:%M:%S")

    if is_gui:
        # tqdm.write inserts extra lines in the GUI, so use standard output as
        # it is not needed there.
        log_console = logging.StreamHandler(sys.stdout)
    else:
        log_console = TqdmHandler(sys.stdout)
    log_console.setFormatter(log_format)
    log_console.setLevel(loglevel)
    return log_console


def crash_handler(log_format):
    """ Add a handler that sores the last 100 debug lines to 'debug_buffer'
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
    original_traceback = traceback.format_exc()
    path = os.path.dirname(os.path.realpath(sys.argv[0]))
    filename = os.path.join(path, datetime.now().strftime("crash_report.%Y.%m.%d.%H%M%S%f.log"))
    freeze_log = list(debug_buffer)
    try:
        from lib.sysinfo import sysinfo  # pylint:disable=import-outside-toplevel
    except Exception:  # pylint:disable=broad-except
        sysinfo = ("\n\nThere was an error importing System Information from lib.sysinfo. This is "
                   "probably a bug which should be fixed:\n{}".format(traceback.format_exc()))
    with open(filename, "w") as outfile:
        outfile.writelines(freeze_log)
        outfile.write(original_traceback)
        outfile.write(sysinfo)
    return filename


old_factory = logging.getLogRecordFactory()  # pylint: disable=invalid-name


def faceswap_logrecord(*args, **kwargs):
    """ Add a flag to logging.LogRecord to not strip formatting from particular records """
    record = old_factory(*args, **kwargs)
    record.strip_spaces = True
    return record


logging.setLogRecordFactory(faceswap_logrecord)

# Set logger class to custom logger
logging.setLoggerClass(FaceswapLogger)

# Stores the last 100 debug messages
debug_buffer = RollingBuffer(maxlen=100)  # pylint: disable=invalid-name
