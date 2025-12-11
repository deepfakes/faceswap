#!/usr/bin/env python3
""" Launches the correct script with the given Command Line Arguments """
from __future__ import annotations
import logging
import os
import platform
import sys
import typing as T

from importlib import import_module

from lib.gpu_stats import GPUStats
from lib.logger import crash_log, log_setup
from lib.utils import (FaceswapError, get_backend, get_torch_version,
                       get_module_objects, safe_shutdown, set_backend)

if T.TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ScriptExecutor():
    """ Loads the relevant script modules and executes the script.

        This class is initialized in each of the argparsers for the relevant
        command, then execute script is called within their set_default
        function.

        Parameters
        ----------
        command: str
            The faceswap command that is being executed
        """
    def __init__(self, command: str) -> None:
        self._command = command.lower()

    def _set_environment_variables(self) -> None:
        """ Set the number of threads that numexpr can use. """
        # Allocate a decent number of threads to numexpr to suppress warnings
        cpu_count = os.cpu_count()
        allocate = max(1, cpu_count - cpu_count // 3 if cpu_count is not None else 1)
        if "OMP_NUM_THREADS" in os.environ:
            # If this is set above NUMEXPR_MAX_THREADS, numexpr will error.
            # ref: https://github.com/pydata/numexpr/issues/322
            os.environ.pop("OMP_NUM_THREADS")
        logger.debug("Setting NUMEXPR_MAX_THREADS to %s", allocate)
        os.environ["NUMEXPR_MAX_THREADS"] = str(allocate)

        if get_backend() == "apple_silicon":  # Let apple put unsupported ops on the CPU
            logger.debug("Enabling unsupported Ops on CPU for Apple Silicon")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    def _import_script(self) -> Callable:
        """ Imports the relevant script as indicated by :attr:`_command` from the scripts folder.

        Returns
        -------
        class: Faceswap Script
            The uninitialized script from the faceswap scripts folder.
        """
        self._set_environment_variables()
        self._test_for_torch_version()
        self._test_for_gui()
        cmd = os.path.basename(sys.argv[0])
        src = f"tools.{self._command.lower()}" if cmd == "tools.py" else "scripts"
        mod = ".".join((src, self._command.lower()))
        module = import_module(mod)
        script = getattr(module, self._command.title())
        return script

    def _test_for_torch_version(self) -> None:
        """ Check that the required PyTorch version is installed.

        Raises
        ------
        FaceswapError
            If PyTorch is not found, or is not between versions 2.3 and 2.9
        """
        min_ver = (2, 3)
        max_ver = (2, 9)
        try:
            import torch  # noqa:F401 pylint:disable=unused-import,import-outside-toplevel
        except ImportError as err:
            msg = (
                f"There was an error importing PyTorch. This is most likely because you do "
                f"not have PyTorch installed. Original import error: {str(err)}")
            self._handle_import_error(msg)

        torch_ver = get_torch_version()
        if torch_ver < min_ver:
            msg = (f"The minimum supported PyTorch is version {min_ver} but you have version "
                   f"{torch_ver} installed. Please upgrade PyTorch.")
            self._handle_import_error(msg)
        if torch_ver > max_ver:
            msg = (f"The maximum supported PyTorch is version {max_ver} but you have version "
                   f"{torch_ver} installed. Please downgrade PyTorch.")
            self._handle_import_error(msg)
        logger.debug("Installed PyTorch Version: %s", torch_ver)

    @classmethod
    def _handle_import_error(cls, message: str) -> None:
        """ Display the error message to the console and wait for user input to dismiss it, if
        running GUI under Windows, otherwise use standard error handling.

        Parameters
        ----------
        message: str
            The error message to display
        """
        if "gui" in sys.argv and platform.system() == "Windows":
            logger.error(message)
            logger.info("Press \"ENTER\" to dismiss the message and close FaceSwap")
            input()
            sys.exit(1)
        else:
            raise FaceswapError(message)

    def _test_for_gui(self) -> None:
        """ If running the gui, performs check to ensure necessary prerequisites are present. """
        if self._command != "gui":
            return
        self._test_tkinter()
        self._check_display()

    @classmethod
    def _test_tkinter(cls) -> None:
        """ If the user is running the GUI, test whether the tkinter app is available on their
        machine. If not exit gracefully.

        This avoids having to import every tkinter function within the GUI in a wrapper and
        potentially spamming traceback errors to console.

        Raises
        ------
        FaceswapError
            If tkinter cannot be imported
        """
        try:
            import tkinter  # noqa pylint:disable=unused-import,import-outside-toplevel
        except ImportError as err:
            logger.error("It looks like TkInter isn't installed for your OS, so the GUI has been "
                         "disabled. To enable the GUI please install the TkInter application. You "
                         "can try:")
            logger.info("Anaconda: conda install tk")
            logger.info("Windows/macOS: Install ActiveTcl Community Edition from "
                        "http://www.activestate.com")
            logger.info("Ubuntu/Mint/Debian: sudo apt install python3-tk")
            logger.info("Arch: sudo pacman -S tk")
            logger.info("CentOS/Redhat: sudo yum install tkinter")
            logger.info("Fedora: sudo dnf install python3-tkinter")
            raise FaceswapError("TkInter not found") from err

    @classmethod
    def _check_display(cls) -> None:
        """ Check whether there is a display to output the GUI to.

        If running on Windows then it is assumed that we are not running in headless mode

        Raises
        ------
        FaceswapError
            If a DISPLAY environmental cannot be found
        """
        if not os.environ.get("DISPLAY", None) and os.name != "nt":
            if platform.system() == "Darwin":
                logger.info("macOS users need to install XQuartz. "
                            "See https://support.apple.com/en-gb/HT201341")
            raise FaceswapError("No display detected. GUI mode has been disabled.")

    def execute_script(self, arguments: argparse.Namespace) -> None:
        """ Performs final set up and launches the requested :attr:`_command` with the given
        command line arguments.

        Monitors for errors and attempts to shut down the process cleanly on exit.

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The command line arguments to be passed to the executing script.
        """
        is_gui = hasattr(arguments, "redirect_gui") and arguments.redirect_gui
        log_setup(arguments.loglevel, arguments.logfile, self._command, is_gui)
        success = False

        if self._command != "gui":
            self._configure_backend(arguments)
        try:
            script = self._import_script()
            process = script(arguments)
            process.process()
            success = True
        except FaceswapError as err:
            for line in str(err).splitlines():
                logger.error(line)
        except KeyboardInterrupt:  # pylint:disable=try-except-raise
            raise
        except SystemExit:
            pass
        except Exception:  # pylint:disable=broad-except
            crash_file = crash_log()
            logger.exception("Got Exception on main handler:")
            logger.critical("An unexpected crash has occurred. Crash report written to '%s'. "
                            "You MUST provide this file if seeking assistance. Please verify you "
                            "are running the latest version of faceswap before reporting",
                            crash_file)

        finally:
            safe_shutdown(got_error=not success)

    def _configure_backend(self, arguments: argparse.Namespace) -> None:
        """ Configure the backend.

        Exclude any GPUs for use by Faceswap when requested.

        Set Faceswap backend to CPU if all GPUs have been deselected.

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The command line arguments passed to Faceswap.
        """
        if not hasattr(arguments, "exclude_gpus"):
            # CPU backends and systems where no GPU was detected will not have this attribute
            logger.debug("Adding missing exclude gpus argument to namespace")
            setattr(arguments, "exclude_gpus", None)
            return

        assert GPUStats is not None
        if arguments.exclude_gpus:
            if not all(idx.isdigit() for idx in arguments.exclude_gpus):
                logger.error("GPUs passed to the ['-X', '--exclude-gpus'] argument must all be "
                             "integers.")
                sys.exit(1)
            arguments.exclude_gpus = [int(idx) for idx in arguments.exclude_gpus]
            GPUStats().exclude_devices(arguments.exclude_gpus)

        if GPUStats().exclude_all_devices:
            msg = "Switching backend to CPU"
            set_backend("cpu")
            logger.info(msg)

        logger.debug("Executing: %s. PID: %s", self._command, os.getpid())


__all__ = get_module_objects(__name__)
