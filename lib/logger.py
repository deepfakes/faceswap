#!/usr/bin/python
""" Logging Functions for Faceswap. """
import collections
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import traceback

from datetime import datetime
from tqdm import tqdm


class FaceswapLogger(logging.Logger):
    """ A standard :class:`logging.logger` with additional "verbose" and "trace" levels added. """
    def __init__(self, name):
        for new_level in (("VERBOSE", 15), ("TRACE", 5)):
            level_name, level_num = new_level
            if hasattr(logging, level_name):
                continue
            logging.addLevelName(level_num, level_name)
            setattr(logging, level_name, level_num)
        super().__init__(name)

    def verbose(self, msg, *args, **kwargs):
        # pylint:disable=wrong-spelling-in-docstring
        """ Create a log message at severity level 15.

        Parameters
        ----------
        msg: str
            The log message to be recorded at Verbose level
        args: tuple
            Standard logging arguments
        kwargs: dict
            Standard logging key word arguments
        """
        if self.isEnabledFor(15):
            self._log(15, msg, args, **kwargs)

    def trace(self, msg, *args, **kwargs):
        # pylint:disable=wrong-spelling-in-docstring
        """ Create a log message at severity level 5.

        Parameters
        ----------
        msg: str
            The log message to be recorded at Trace level
        args: tuple
            Standard logging arguments
        kwargs: dict
            Standard logging key word arguments
        """
        if self.isEnabledFor(5):
            self._log(5, msg, args, **kwargs)


class FaceswapFormatter(logging.Formatter):
    """ Overrides the standard :class:`logging.Formatter`.

    Strip newlines from incoming log messages.

    Rewrites some upstream warning messages to debug level to avoid spamming the console.
    """

    def format(self, record):
        """ Strip new lines from log records and rewrite certain warning messages to debug level.

        Parameters
        ----------
        record : :class:`logging.LogRecord`
            The incoming log record to be formatted for entry into the logger.

        Returns
        -------
        str
            The formatted log message
        """
        record.message = record.getMessage()
        record = self._rewrite_warnings(record)
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

    @classmethod
    def _rewrite_warnings(cls, record):
        """ Change certain warning messages from WARNING to DEBUG to avoid passing non-important
        information to output.

        Parameters
        ----------
        record: :class:`logging.LogRecord`
            The log record to check for rewriting
        """
        if record.levelno == 30 and (record.funcName == "_tfmw_add_deprecation_warning" or
                                     record.module in ("deprecation", "deprecation_wrapper")):
            record.levelno = 10
            record.levelname = "DEBUG"
        return record


class RollingBuffer(collections.deque):
    """File-like that keeps a certain number of lines of text in memory for writing out to the
    crash log. """

    def write(self, buffer):
        """ Splits lines from the incoming buffer and writes them out to the rolling buffer.

        Parameters
        ----------
        buffer: str
            The log messages to write to the rolling buffer
        """
        for line in buffer.rstrip().splitlines():
            self.append(line + "\n")


class TqdmHandler(logging.StreamHandler):
    """ Overrides :class:`logging.StreamHandler` to use :func:`tqdm.tqdm.write` rather than writing
    to :func:`sys.stderr` so that log messages do not mess up tqdm progress bars. """

    def emit(self, record):
        """ Format the incoming message and pass to :func:`tqdm.tqdm.write`.

        Parameters
        ----------
        record : :class:`logging.LogRecord`
            The incoming log record to be formatted for entry into the logger.
        """
        msg = self.format(record)
        tqdm.write(msg)


def _set_root_logger(loglevel=logging.INFO):
    """ Setup the root logger.

    Parameters
    ----------
    loglevel: int, optional
        The log level to set the root logger to. Default :attr:`logging.INFO`

    Returns
    -------
    :class:`logging.Logger`
        The root logger for Faceswap
    """
    rootlogger = logging.getLogger()
    rootlogger.setLevel(loglevel)
    return rootlogger


def log_setup(loglevel, log_file, command, is_gui=False):
    """ Set up logging for Faceswap.

    Sets up the root logger, the formatting for the crash logger and the file logger, and sets up
    the crash, file and stream log handlers.

    Parameters
    ----------
    loglevel: str
        The requested log level that Faceswap should be run at.
    log_file: str
        The location of the log file to write Faceswap's log to
    command: str
        The Faceswap command that is being run. Used to dictate whether the log file should
        have "_gui" appended to the filename or not.
    is_gui: bool, optional
        Whether Faceswap is running in the GUI or not. Dictates where the stream handler should
        output messages to. Default: ``False``
     """
    numeric_loglevel = get_loglevel(loglevel)
    root_loglevel = min(logging.DEBUG, numeric_loglevel)
    rootlogger = _set_root_logger(loglevel=root_loglevel)
    log_format = FaceswapFormatter("%(asctime)s %(processName)-15s %(threadName)-15s "
                                   "%(module)-15s %(funcName)-25s %(levelname)-8s %(message)s",
                                   datefmt="%m/%d/%Y %H:%M:%S")
    f_handler = _file_handler(numeric_loglevel, log_file, log_format, command)
    s_handler = _stream_handler(numeric_loglevel, is_gui)
    c_handler = _crash_handler(log_format)
    rootlogger.addHandler(f_handler)
    rootlogger.addHandler(s_handler)
    rootlogger.addHandler(c_handler)
    logging.info("Log level set to: %s", loglevel.upper())


def _file_handler(loglevel, log_file, log_format, command):
    """ Add a rotating file handler for the current Faceswap session. 1 backup is always kept.

    Parameters
    ----------
    loglevel: str
        The requested log level that messages should be logged at.
    log_file: str
        The location of the log file to write Faceswap's log to
    log_format: :class:`FaceswapFormatter:
        The formatting to store log messages as
    command: str
        The Faceswap command that is being run. Used to dictate whether the log file should
        have "_gui" appended to the filename or not.

    Returns
    -------
    :class:`logging.RotatingFileHandler`
        The logging file handler
    """
    if log_file is not None:
        filename = log_file
    else:
        filename = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap")
        # Windows has issues sharing the log file with sub-processes, so log GUI separately
        filename += "_gui.log" if command == "gui" else ".log"

    should_rotate = os.path.isfile(filename)
    log_file = RotatingFileHandler(filename, backupCount=1)
    if should_rotate:
        log_file.doRollover()
    log_file.setFormatter(log_format)
    log_file.setLevel(loglevel)
    return log_file


def _stream_handler(loglevel, is_gui):
    """ Add a stream handler for the current Faceswap session. The stream handler will only ever
    output at a maximum of VERBOSE level to avoid spamming the console.

    Parameters
    ----------
    loglevel: str
        The requested log level that messages should be logged at.
    is_gui: bool, optional
        Whether Faceswap is running in the GUI or not. Dictates where the stream handler should
        output messages to.

    Returns
    -------
    :class:`TqdmHandler` or :class:`logging.StreamHandler`
        The stream handler to use
    """
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


def _crash_handler(log_format):
    """ Add a handler that stores the last 100 debug lines to :attr:'_debug_buffer' for use in
    crash reports.

    Parameters
    ----------
    log_format: :class:`FaceswapFormatter:
        The formatting to store log messages as

    Returns
    -------
    :class:`logging.StreamHandler`
        The crash log handler
    """
    log_crash = logging.StreamHandler(_debug_buffer)
    log_crash.setFormatter(log_format)
    log_crash.setLevel(logging.DEBUG)
    return log_crash


def get_loglevel(loglevel):
    """ Check whether a valid log level has been supplied, and return the numeric log level that
    corresponds to the given string level.

    Parameters
    ----------
    loglevel: str
        The loglevel that has been requested

    Returns
    -------
    int
        The numeric representation of the given loglevel
    """
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    return numeric_level


def crash_log():
    """ On a crash, write out the contents of :func:`_debug_buffer` containing the last 100 lines
    of debug messages to a crash report in the root Faceswap folder.

    Returns
    -------
    str
        The filename of the file that contains the crash report
    """
    original_traceback = traceback.format_exc()
    path = os.path.dirname(os.path.realpath(sys.argv[0]))
    filename = os.path.join(path, datetime.now().strftime("crash_report.%Y.%m.%d.%H%M%S%f.log"))
    freeze_log = list(_debug_buffer)
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


_old_factory = logging.getLogRecordFactory()


def _faceswap_logrecord(*args, **kwargs):
    """ Add a flag to :class:`logging.LogRecord` to not strip formatting from particular
    records. """
    record = _old_factory(*args, **kwargs)
    record.strip_spaces = True
    return record


logging.setLogRecordFactory(_faceswap_logrecord)

# Set logger class to custom logger
logging.setLoggerClass(FaceswapLogger)

# Stores the last 100 debug messages
_debug_buffer = RollingBuffer(maxlen=100)
