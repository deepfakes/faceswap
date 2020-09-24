#!/usr/bin/env python3
""" Launches the correct script with the given Command Line Arguments """
import logging
import os
import platform
import sys

from importlib import import_module

from lib.gpu_stats import set_exclude_devices, GPUStats
from lib.logger import crash_log, log_setup
from lib.utils import (FaceswapError, get_backend, KerasFinder, safe_shutdown, set_backend,
                       set_system_verbosity)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ScriptExecutor():  # pylint:disable=too-few-public-methods
    """ Loads the relevant script modules and executes the script.

        This class is initialized in each of the argparsers for the relevant
        command, then execute script is called within their set_default
        function.

        Parameters
        ----------
        command: str
            The faceswap command that is being executed
        """
    def __init__(self, command):
        self._command = command.lower()

    def _import_script(self):
        """ Imports the relevant script as indicated by :attr:`_command` from the scripts folder.

        Returns
        -------
        class: Faceswap Script
            The uninitialized script from the faceswap scripts folder.
        """
        self._test_for_tf_version()
        self._test_for_gui()
        cmd = os.path.basename(sys.argv[0])
        src = "tools.{}".format(self._command.lower()) if cmd == "tools.py" else "scripts"
        mod = ".".join((src, self._command.lower()))
        module = import_module(mod)
        script = getattr(module, self._command.title())
        return script

    def _test_for_tf_version(self):
        """ Check that the required Tensorflow version is installed.

        Raises
        ------
        FaceswapError
            If Tensorflow is not found, or is not between versions 2.2 and 2.2
        """
        min_ver = 2.2
        max_ver = 2.2
        try:
            # Ensure tensorflow doesn't pin all threads to one core when using Math Kernel Library
            os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
            os.environ["KMP_AFFINITY"] = "disabled"
            import tensorflow as tf  # pylint:disable=import-outside-toplevel
        except ImportError as err:
            if "DLL load failed while importing" in str(err):
                msg = (
                    "A DLL library file failed to load. Make sure that you have Microsoft Visual "
                    "C++ Redistributable (2015, 2017, 2019) installed for your machine from: "
                    "https://support.microsoft.com/en-gb/help/2977003")
            else:
                msg = (
                    "There was an error importing Tensorflow. This is most likely because you do "
                    "not have TensorFlow installed, or you are trying to run tensorflow-gpu on a "
                    "system without an Nvidia graphics card. Original import "
                    "error: {}".format(str(err)))
            self._handle_import_error(msg)

        tf_ver = float(".".join(tf.__version__.split(".")[:2]))  # pylint:disable=no-member
        if tf_ver < min_ver:
            msg = ("The minimum supported Tensorflow is version {} but you have version {} "
                   "installed. Please upgrade Tensorflow.".format(min_ver, tf_ver))
            self._handle_import_error(msg)
        if tf_ver > max_ver:
            msg = ("The maximumum supported Tensorflow is version {} but you have version {} "
                   "installed. Please downgrade Tensorflow.".format(max_ver, tf_ver))
            self._handle_import_error(msg)
        logger.debug("Installed Tensorflow Version: %s", tf_ver)

    @classmethod
    def _handle_import_error(cls, message):
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

    def _test_for_gui(self):
        """ If running the gui, performs check to ensure necessary prerequisites are present. """
        if self._command != "gui":
            return
        self._test_tkinter()
        self._check_display()

    @staticmethod
    def _test_tkinter():
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
            # pylint: disable=unused-variable
            import tkinter  # noqa pylint: disable=unused-import,import-outside-toplevel
        except ImportError:
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
            raise FaceswapError("TkInter not found")

    @staticmethod
    def _check_display():
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

    def execute_script(self, arguments):
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
        except KeyboardInterrupt:  # pylint: disable=try-except-raise
            raise
        except SystemExit:
            pass
        except Exception:  # pylint: disable=broad-except
            crash_file = crash_log()
            logger.exception("Got Exception on main handler:")
            logger.critical("An unexpected crash has occurred. Crash report written to '%s'. "
                            "You MUST provide this file if seeking assistance. Please verify you "
                            "are running the latest version of faceswap before reporting",
                            crash_file)

        finally:
            safe_shutdown(got_error=not success)

    def _configure_backend(self, arguments):
        """ Configure the backend.

        Exclude any GPUs for use by Faceswap when requested.

        Set Faceswap backend to CPU if all GPUs have been deselected.

        Add the Keras import interception code.

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The command line arguments passed to Faceswap.
        """
        if not hasattr(arguments, "exclude_gpus"):
            # Cpu backends will not have this attribute
            logger.debug("Adding missing exclude gpus argument to namespace")
            setattr(arguments, "exclude_gpus", None)

        if arguments.exclude_gpus:
            if not all(idx.isdigit() for idx in arguments.exclude_gpus):
                logger.error("GPUs passed to the ['-X', '--exclude-gpus'] argument must all be "
                             "integers.")
                sys.exit(1)
            arguments.exclude_gpus = [int(idx) for idx in arguments.exclude_gpus]
            set_exclude_devices(arguments.exclude_gpus)

        if ((get_backend() == "cpu" or GPUStats().exclude_all_devices) and
                (self._command == "extract" and arguments.detector == "s3fd")):
            logger.error("Extracting on CPU is not currently for detector: '%s'",
                         arguments.detector.upper())
            sys.exit(0)

        if GPUStats().exclude_all_devices and get_backend() != "cpu":
            msg = "Switching backend to CPU"
            if get_backend() == "amd":
                msg += (". Using Tensorflow for CPU operations.")
                os.environ["KERAS_BACKEND"] = "tensorflow"
            set_backend("cpu")
            logger.info(msg)

        # Add Keras finder to the meta_path list as the first item
        sys.meta_path.insert(0, KerasFinder())

        logger.debug("Executing: %s. PID: %s", self._command, os.getpid())

        if get_backend() == "amd":
            plaidml_found = self._setup_amd(arguments)
            if not plaidml_found:
                safe_shutdown(got_error=True)
                sys.exit(1)

    @classmethod
    def _setup_amd(cls, arguments):
        """ Test for plaidml and perform setup for AMD.

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The command line arguments passed to Faceswap.
        """
        logger.debug("Setting up for AMD")
        try:
            import plaidml  # noqa pylint:disable=unused-import,import-outside-toplevel
        except ImportError:
            logger.error("PlaidML not found. Run `pip install plaidml-keras` for AMD support")
            return False
        from lib.plaidml_tools import setup_plaidml  # pylint:disable=import-outside-toplevel
        setup_plaidml(arguments.loglevel, arguments.exclude_gpus)
        logger.debug("setup up for PlaidML")
        return True
