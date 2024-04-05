#!/usr/bin/python
""" Logging Functions for Faceswap. """
# NOTE: Don't import non stdlib packages. This module is accessed by setup.py
import collections
import logging
from logging.handlers import RotatingFileHandler
import os
import platform
import re
import sys
import typing as T
import time
import traceback

from datetime import datetime


# TODO - Remove this monkey patch when TF autograph fixed to handle newer logging lib
def _patched_format(self, record):
    """ Autograph tf-2.10 has a bug with the 3.10 version of logging.PercentStyle._format(). It is
    non-critical but spits out warnings. This is the Python 3.9 version of the function and should
    be removed once fixed """
    return self._fmt % record.__dict__  # pylint:disable=protected-access


setattr(logging.PercentStyle, "_format", _patched_format)


class FaceswapLogger(logging.Logger):
    """ A standard :class:`logging.logger` with additional "verbose" and "trace" levels added. """
    def __init__(self, name: str) -> None:
        for new_level in (("VERBOSE", 15), ("TRACE", 5)):
            level_name, level_num = new_level
            if hasattr(logging, level_name):
                continue
            logging.addLevelName(level_num, level_name)
            setattr(logging, level_name, level_num)
        super().__init__(name)

    def verbose(self, msg: str, *args, **kwargs) -> None:
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

    def trace(self, msg: str, *args, **kwargs) -> None:
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


class ColoredFormatter(logging.Formatter):
    """ Overrides the stand :class:`logging.Formatter` to enable colored labels for message level
    labels on supported platforms

    Parameters
    ----------
    fmt: str
        The format string for the message as a whole
    pad_newlines: bool, Optional
        If ``True`` new lines will be padded to appear in line with the log message, if ``False``
        they will be left aligned

    kwargs: dict
        Standard :class:`logging.Formatter` keyword arguments
    """
    def __init__(self, fmt: str, pad_newlines: bool = False, **kwargs) -> None:
        super().__init__(fmt, **kwargs)
        self._use_color = self._get_color_compatibility()
        self._level_colors = {"CRITICAL": "\033[31m",  # red
                              "ERROR": "\033[31m",  # red
                              "WARNING": "\033[33m",  # yellow
                              "INFO": "\033[32m",  # green
                              "VERBOSE": "\033[34m"}  # blue
        self._default_color = "\033[0m"
        self._newline_padding = self._get_newline_padding(pad_newlines, fmt)

    @classmethod
    def _get_color_compatibility(cls) -> bool:
        """ Return whether the system supports color ansi codes. Most OSes do other than Windows
        below Windows 10 version 1511.

        Returns
        -------
        bool
            ``True`` if the system supports color ansi codes otherwise ``False``
        """
        if platform.system().lower() != "windows":
            return True
        try:
            win = sys.getwindowsversion()  # type:ignore # pylint:disable=no-member
            if win.major >= 10 and win.build >= 10586:
                return True
        except Exception:  # pylint:disable=broad-except
            return False
        return False

    def _get_newline_padding(self, pad_newlines: bool, fmt: str) -> int:
        """ Parses the format string to obtain padding for newlines if requested

        Parameters
        ----------
        fmt: str
            The format string for the message as a whole
        pad_newlines: bool, Optional
            If ``True`` new lines will be padded to appear in line with the log message, if
            ``False`` they will be left aligned

        Returns
        -------
        int
            The amount of padding to apply to the front of newlines
        """
        if not pad_newlines:
            return 0
        msg_idx = fmt.find("%(message)") + 1
        filtered = fmt[:msg_idx - 1]
        spaces = filtered.count(" ")
        pads = [int(pad.replace("s", "")) for pad in re.findall(r"\ds", filtered)]
        if "asctime" in filtered:
            pads.append(self._get_sample_time_string())
        return sum(pads) + spaces

    def _get_sample_time_string(self) -> int:
        """ Obtain a sample time string and calculate correct padding.

        This may be inaccurate when ticking over an integer from single to double digits, but that
        shouldn't be a huge issue.

        Returns
        -------
        int
            The length of the formatted date-time string
        """
        sample_time = time.time()
        date_format = self.datefmt if self.datefmt else self.default_time_format
        datestring = time.strftime(date_format, logging.Formatter.converter(sample_time))
        if not self.datefmt and self.default_msec_format:
            msecs = (sample_time - int(sample_time)) * 1000
            datestring = self.default_msec_format % (datestring, msecs)
        return len(datestring)

    def format(self, record: logging.LogRecord) -> str:
        """ Color the log message level if supported otherwise return the standard log message.

        Parameters
        ----------
        record: :class:`logging.LogRecord`
            The incoming log record to be formatted for entry into the logger.

        Returns
        -------
        str
            The formatted log message
        """
        formatted = super().format(record)
        levelname = record.levelname
        if self._use_color and levelname in self._level_colors:
            formatted = re.sub(levelname,
                               f"{self._level_colors[levelname]}{levelname}{self._default_color}",
                               formatted,
                               1)
        if self._newline_padding:
            formatted = formatted.replace("\n", f"\n{' ' * self._newline_padding}")
        return formatted


class FaceswapFormatter(logging.Formatter):
    """ Overrides the standard :class:`logging.Formatter`.

    Strip newlines from incoming log messages.

    Rewrites some upstream warning messages to debug level to avoid spamming the console.
    """

    def format(self, record: logging.LogRecord) -> str:
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
        record = self._lower_external(record)
        # strip newlines
        if record.levelno < 30 and ("\n" in record.message or "\r" in record.message):
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
    def _rewrite_warnings(cls, record: logging.LogRecord) -> logging.LogRecord:
        """ Change certain warning messages from WARNING to DEBUG to avoid passing non-important
        information to output.

        Parameters
        ----------
        record: :class:`logging.LogRecord`
            The log record to check for rewriting

        Returns
        -------
        :class:`logging.LogRecord`
            The log rewritten or untouched record

        """
        if record.levelno == 30 and record.funcName == "warn" and record.module == "ag_logging":
            # TF 2.3 in Conda is imported with the wrong gast(0.4 when 0.3.3 should be used). This
            # causes warnings in autograph. They don't appear to impact performance so de-elevate
            # warning to debug
            record.levelno = 10
            record.levelname = "DEBUG"

        if record.levelno == 30 and (record.funcName == "_tfmw_add_deprecation_warning" or
                                     record.module in ("deprecation", "deprecation_wrapper")):
            # Keras Deprecations.
            record.levelno = 10
            record.levelname = "DEBUG"

        return record

    @classmethod
    def _lower_external(cls, record: logging.LogRecord) -> logging.LogRecord:
        """ Some external libs log at a higher level than we would really like, so lower their
        log level.

        Specifically: Matplotlib font properties

        Parameters
        ----------
        record: :class:`logging.LogRecord`
            The log record to check for rewriting

        Returns
        ----------
        :class:`logging.LogRecord`
            The log rewritten or untouched record
        """
        if (record.levelno == 20 and record.funcName == "__init__"
                and record.module == "font_manager"):
            # Matplotlib font manager
            record.levelno = 10
            record.levelname = "DEBUG"

        return record


class RollingBuffer(collections.deque):
    """File-like that keeps a certain number of lines of text in memory for writing out to the
    crash log. """

    def write(self, buffer: str) -> None:
        """ Splits lines from the incoming buffer and writes them out to the rolling buffer.

        Parameters
        ----------
        buffer: str
            The log messages to write to the rolling buffer
        """
        for line in buffer.rstrip().splitlines():
            self.append(f"{line}\n")


class TqdmHandler(logging.StreamHandler):
    """ Overrides :class:`logging.StreamHandler` to use :func:`tqdm.tqdm.write` rather than writing
    to :func:`sys.stderr` so that log messages do not mess up tqdm progress bars. """

    def emit(self, record: logging.LogRecord) -> None:
        """ Format the incoming message and pass to :func:`tqdm.tqdm.write`.

        Parameters
        ----------
        record : :class:`logging.LogRecord`
            The incoming log record to be formatted for entry into the logger.
        """
        # tqdm is imported here as it won't be installed when setup.py is running
        from tqdm import tqdm  # pylint:disable=import-outside-toplevel
        msg = self.format(record)
        tqdm.write(msg)


def _set_root_logger(loglevel: int = logging.INFO) -> logging.Logger:
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


def log_setup(loglevel, log_file: str, command: str, is_gui: bool = False) -> None:
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

    if command == "setup":
        log_format = FaceswapFormatter("%(asctime)s %(module)-16s %(funcName)-30s %(levelname)-8s "
                                       "%(message)s", datefmt="%m/%d/%Y %H:%M:%S")
        s_handler = _stream_setup_handler(numeric_loglevel)
        f_handler = _file_handler(root_loglevel, log_file, log_format, command)
    else:
        log_format = FaceswapFormatter("%(asctime)s %(processName)-15s %(threadName)-30s "
                                       "%(module)-15s %(funcName)-30s %(levelname)-8s %(message)s",
                                       datefmt="%m/%d/%Y %H:%M:%S")
        s_handler = _stream_handler(numeric_loglevel, is_gui)
        f_handler = _file_handler(numeric_loglevel, log_file, log_format, command)

    rootlogger.addHandler(f_handler)
    rootlogger.addHandler(s_handler)

    if command != "setup":
        c_handler = _crash_handler(log_format)
        rootlogger.addHandler(c_handler)
        logging.info("Log level set to: %s", loglevel.upper())


def _file_handler(loglevel,
                  log_file: str,
                  log_format: FaceswapFormatter,
                  command: str) -> RotatingFileHandler:
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
    if log_file:
        filename = log_file
    else:
        filename = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap")
        # Windows has issues sharing the log file with sub-processes, so log GUI separately
        filename += "_gui.log" if command == "gui" else ".log"

    should_rotate = os.path.isfile(filename)
    handler = RotatingFileHandler(filename, backupCount=1, encoding="utf-8")
    if should_rotate:
        handler.doRollover()
    handler.setFormatter(log_format)
    handler.setLevel(loglevel)
    return handler


def _stream_handler(loglevel: int, is_gui: bool) -> logging.StreamHandler | TqdmHandler:
    """ Add a stream handler for the current Faceswap session. The stream handler will only ever
    output at a maximum of VERBOSE level to avoid spamming the console.

    Parameters
    ----------
    loglevel: int
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


def _stream_setup_handler(loglevel: int) -> logging.StreamHandler:
    """ Add a stream handler for faceswap's setup.py script
    This stream handler outputs a limited set of easy to use information using colored labels
    if available. It will only ever output at a minimum of INFO level

    Parameters
    ----------
    loglevel: int
        The requested log level that messages should be logged at.

    Returns
    -------
    :class:`logging.StreamHandler`
        The stream handler to use
    """
    loglevel = max(loglevel, 15)
    log_format = ColoredFormatter("%(levelname)-8s %(message)s", pad_newlines=True)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_format)
    handler.setLevel(loglevel)
    return handler


def _crash_handler(log_format: FaceswapFormatter) -> logging.StreamHandler:
    """ Add a handler that stores the last 100 debug lines to :attr:'_DEBUG_BUFFER' for use in
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
    log_crash = logging.StreamHandler(_DEBUG_BUFFER)
    log_crash.setFormatter(log_format)
    log_crash.setLevel(logging.DEBUG)
    return log_crash


def get_loglevel(loglevel: str) -> int:
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
        raise ValueError(f"Invalid log level: {loglevel}")
    return numeric_level


def crash_log() -> str:
    """ On a crash, write out the contents of :func:`_DEBUG_BUFFER` containing the last 100 lines
    of debug messages to a crash report in the root Faceswap folder.

    Returns
    -------
    str
        The filename of the file that contains the crash report
    """
    original_traceback = traceback.format_exc().encode("utf-8")
    path = os.path.dirname(os.path.realpath(sys.argv[0]))
    filename = os.path.join(path, datetime.now().strftime("crash_report.%Y.%m.%d.%H%M%S%f.log"))
    freeze_log = [line.encode("utf-8") for line in _DEBUG_BUFFER]
    try:
        from lib.sysinfo import sysinfo  # pylint:disable=import-outside-toplevel
    except Exception:  # pylint:disable=broad-except
        sysinfo = ("\n\nThere was an error importing System Information from lib.sysinfo. This is "
                   f"probably a bug which should be fixed:\n{traceback.format_exc()}")
    with open(filename, "wb") as outfile:
        outfile.writelines(freeze_log)
        outfile.write(original_traceback)
        outfile.write(sysinfo.encode("utf-8"))
    return filename


def _process_value(value: T.Any) -> T.Any:
    """ Process the values from a local dict and return in a loggable format

    Parameters
    ----------
    value: Any
        The dictionary value

    Returns
    -------
    Any
        The original or ammended value
    """
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple, set)) and len(value) > 10:
        return f'[type: "{type(value).__name__}" len: {len(value)}'

    try:
        import numpy as np  # pylint:disable=import-outside-toplevel
    except ImportError:
        return value

    if isinstance(value, np.ndarray) and np.prod(value.shape) > 10:
        return f'[type: "{type(value).__name__}" shape: {value.shape}, dtype: "{value.dtype}"]'
    return value


def parse_class_init(locals_dict: dict[str, T.Any]) -> str:
    """ Parse a locals dict from a class and return in a format suitable for logging
    Parameters
    ----------
    locals_dict: dict[str, T.Any]
        A locals() dictionary from a newly initialized class
    Returns
    -------
    str
        The locals information suitable for logging
    """
    delimit = {k: _process_value(v)
               for k, v in locals_dict.items() if k != "self"}
    dsp = ", ".join(f"{k}: {v}" for k, v in delimit.items())
    dsp = f" ({dsp})" if dsp else ""
    return f"Initializing {locals_dict['self'].__class__.__name__}{dsp}"


_OLD_FACTORY = logging.getLogRecordFactory()


def _faceswap_logrecord(*args, **kwargs) -> logging.LogRecord:
    """ Add a flag to :class:`logging.LogRecord` to not strip formatting from particular
    records. """
    record = _OLD_FACTORY(*args, **kwargs)
    record.strip_spaces = True  # type:ignore
    return record


logging.setLogRecordFactory(_faceswap_logrecord)

# Set logger class to custom logger
logging.setLoggerClass(FaceswapLogger)

# Stores the last 100 debug messages
_DEBUG_BUFFER = RollingBuffer(maxlen=100)
