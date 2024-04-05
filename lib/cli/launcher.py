#!/usr/bin/env python3
""" Launches the correct script with the given Command Line Arguments """
from __future__ import annotations
import logging
import os
import platform
import sys
import typing as T

from importlib import import_module

from lib.gpu_stats import set_exclude_devices, GPUStats
from lib.logger import crash_log, log_setup
from lib.utils import (FaceswapError, get_backend, get_tf_version,
                       safe_shutdown, set_backend, set_system_verbosity)

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

    def _import_script(self) -> Callable:
        """ Imports the relevant script as indicated by :attr:`_command` from the scripts folder.

        Returns
        -------
        class: Faceswap Script
            The uninitialized script from the faceswap scripts folder.
        """
        self._set_environment_variables()
        self._test_for_tf_version()
        self._test_for_gui()
        cmd = os.path.basename(sys.argv[0])
        src = f"tools.{self._command.lower()}" if cmd == "tools.py" else "scripts"
        mod = ".".join((src, self._command.lower()))
        module = import_module(mod)
        script = getattr(module, self._command.title())
        return script

    def _set_environment_variables(self) -> None:
        """ Set the number of threads that numexpr can use and TF environment variables. """
        # Allocate a decent number of threads to numexpr to suppress warnings
        cpu_count = os.cpu_count()
        allocate = cpu_count - cpu_count // 3 if cpu_count is not None else 1
        if "OMP_NUM_THREADS" in os.environ:
            # If this is set above NUMEXPR_MAX_THREADS, numexpr will error.
            # ref: https://github.com/pydata/numexpr/issues/322
            os.environ.pop("OMP_NUM_THREADS")
        os.environ["NUMEXPR_MAX_THREADS"] = str(max(1, allocate))

        # Ensure tensorflow doesn't pin all threads to one core when using Math Kernel Library
        os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
        os.environ["KMP_AFFINITY"] = "disabled"

        # If running under CPU on Windows, the following error can be encountered:
        # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5 already initialized.
        # OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into
        # the program. That is dangerous, since it can degrade performance or cause incorrect
        # results. The best thing to do is to ensure that only a single OpenMP runtime is linked
        # into the process, e.g. by avoiding static linking of the OpenMP runtime in any library.
        # As an unsafe, unsupported, undocumented workaround you can set the environment variable
        # KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause
        # crashes or silently produce incorrect results. For more information,
        # please see http://www.intel.com/software/products/support/.
        #
        # TODO find a better way than just allowing multiple libs
        if get_backend() == "cpu" and platform.system() == "Windows":
            logger.debug("Setting `KMP_DUPLICATE_LIB_OK` environment variable to `TRUE`")
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # There is a memory leak in TF2.10+ predict function. This fix will work for tf2.10 but not
        # for later versions. This issue has been patched recently, but we'll probably need to
        # skip some TF versions
        # ref: https://github.com/tensorflow/tensorflow/issues/58676
        # TODO remove this fix post TF2.10 and check memleak is fixed
        logger.debug("Setting TF_RUN_EAGER_OP_AS_FUNCTION env var to False")
        os.environ["TF_RUN_EAGER_OP_AS_FUNCTION"] = "false"

    def _test_for_tf_version(self) -> None:
        """ Check that the required Tensorflow version is installed.

        Raises
        ------
        FaceswapError
            If Tensorflow is not found, or is not between versions 2.4 and 2.9
        """
        min_ver = (2, 10)
        max_ver = (2, 10)
        try:
            import tensorflow as tf  # noqa pylint:disable=import-outside-toplevel,unused-import
        except ImportError as err:
            if "DLL load failed while importing" in str(err):
                msg = (
                    f"A DLL library file failed to load. Make sure that you have Microsoft Visual "
                    "C++ Redistributable (2015, 2017, 2019) installed for your machine from: "
                    "https://support.microsoft.com/en-gb/help/2977003. Original error: "
                    f"{str(err)}")
            else:
                msg = (
                    f"There was an error importing Tensorflow. This is most likely because you do "
                    "not have TensorFlow installed, or you are trying to run tensorflow-gpu on a "
                    "system without an Nvidia graphics card. Original import "
                    f"error: {str(err)}")
            self._handle_import_error(msg)

        tf_ver = get_tf_version()
        if tf_ver < min_ver:
            msg = (f"The minimum supported Tensorflow is version {min_ver} but you have version "
                   f"{tf_ver} installed. Please upgrade Tensorflow.")
            self._handle_import_error(msg)
        if tf_ver > max_ver:
            msg = (f"The maximum supported Tensorflow is version {max_ver} but you have version "
                   f"{tf_ver} installed. Please downgrade Tensorflow.")
            self._handle_import_error(msg)
        logger.debug("Installed Tensorflow Version: %s", tf_ver)

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
        set_system_verbosity(arguments.loglevel)
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

        if arguments.exclude_gpus:
            if not all(idx.isdigit() for idx in arguments.exclude_gpus):
                logger.error("GPUs passed to the ['-X', '--exclude-gpus'] argument must all be "
                             "integers.")
                sys.exit(1)
            arguments.exclude_gpus = [int(idx) for idx in arguments.exclude_gpus]
            set_exclude_devices(arguments.exclude_gpus)

        if GPUStats().exclude_all_devices:
            msg = "Switching backend to CPU"
            set_backend("cpu")
            logger.info(msg)

        logger.debug("Executing: %s. PID: %s", self._command, os.getpid())
