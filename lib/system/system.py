#! /usr/env/bin/python3
"""
Holds information about the running system. Used in setup.py and lib.sysinfo
NOTE: Only packages from Python's Standard Library should be imported in this module
"""
from __future__ import annotations

import ctypes
import locale
import logging
import os
import platform
import re
import sys
import typing as T

from shutil import which
from subprocess import CalledProcessError, run

from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


VALID_PYTHON = ((3, 11), (3, 13))
""" tuple[tuple[int, int], tuple[int, int]] : The minimum and maximum versions of Python that can
run Faceswap """
VALID_TORCH = ((2, 3), (2, 9))
""" tuple[tuple[int, int], tuple[int, int]] : The minimum and maximum versions of Torch that can
run Faceswap """
VALID_KERAS = ((3, 12), (3, 12))
""" tuple[tuple[int, int], tuple[int, int]] : The minimum and maximum versions of Keras that can
run Faceswap """


def _lines_from_command(command: list[str]) -> list[str]:
    """ Output stdout lines from an executed command.

    Parameters
    ----------
    command : list[str]
        The command to run

    Returns
    -------
    list[str]
        The output lines from the given command
    """
    logger.debug("Running command %s", command)
    try:
        proc = run(command,
                   capture_output=True,
                   check=True,
                   encoding=locale.getpreferredencoding(),
                   errors="replace")
    except (FileNotFoundError, CalledProcessError) as err:
        logger.debug("Error from command: %s", str(err))
        return []
    return proc.stdout.splitlines()


class System:  # pylint:disable=too-many-instance-attributes
    """ Holds information about the currently running system and environment """
    def __init__(self) -> None:
        self.platform = platform.platform()
        """ str : Human readable platform identifier """
        self.system: T.Literal["darwin", "linux", "windows"] = T.cast(
            T.Literal["darwin", "linux", "windows"], platform.system().lower())
        """ str : The system (OS type) that this code is running on. Always lowercase """
        self.machine = platform.machine()
        """ str : The machine type (eg: "x86_64") """
        self.release = platform.release()
        """ str : The OS Release that this code is running on """
        self.processor = platform.processor()
        """ str : The processor in use, if detected """
        self.cpu_count = os.cpu_count()
        """ int : The number of CPU cores on the system """
        self.python_implementation = platform.python_implementation()
        """ str : The python implementation in use"""
        self.python_version = platform.python_version()
        """ str : The <major>.<minor>.<release> version of Python that is running """
        self.python_architecture = platform.architecture()[0]
        """ str : The Python architecture that is running (eg: 64bit/32bit)"""
        self.encoding = locale.getpreferredencoding()
        """ str : The system encoding """
        self.is_conda = ("conda" in sys.version.lower() or
                         os.path.exists(os.path.join(sys.prefix, 'conda-meta')))
        """ bool : ``True`` if running under Conda otherwise ``False`` """
        self.is_admin = self._get_permissions()
        """ bool : ``True`` if we are running with Admin privileges """
        self.is_virtual_env = self._check_virtual_env()
        """ bool : ``True`` if Python is being run inside a virtual environment """

    @property
    def is_linux(self) -> bool:
        """ bool : `True` if running on a Linux system otherwise ``False``. """
        return self.system == "linux"

    @property
    def is_macos(self) -> bool:
        """ bool : `True` if running on a macOS system otherwise ``False``. """
        return self.system == "darwin"

    @property
    def is_windows(self) -> bool:
        """ bool : `True` if running on a Windows system otherwise ``False``. """
        return self.system == "windows"

    def __repr__(self) -> str:
        """ Pretty print the system information for logging """
        attrs = ", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items()
                          if not k.startswith("_"))
        return f"{self.__class__.__name__}({attrs})"

    def _get_permissions(self) -> bool:
        """ Check whether user is admin

        Returns
        -------
        bool
            ``True`` if we are running with Admin privileges
        """
        if self.is_windows:
            retval = ctypes.windll.shell32.IsUserAnAdmin() != 0  # type:ignore[attr-defined]
        else:
            retval = os.getuid() == 0  # type:ignore[attr-defined]  # pylint:disable=no-member
        return retval

    def _check_virtual_env(self) -> bool:
        """ Check whether we are in a virtual environment

        Returns
        -------
        bool
             ``True`` if Python is being run inside a virtual environment
        """
        if not self.is_conda:
            retval = (hasattr(sys, "real_prefix") or
                      (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
        else:
            prefix = os.path.dirname(sys.prefix)
            retval = os.path.basename(prefix) == "envs"
        return retval

    def validate_python(self, max_version: tuple[int, int] | None = None) -> bool:
        """ Check that the running Python version is valid

        Parameters
        ----------
        max_version: tuple[int, int] | None, Optional
            The max version to validate Python against. ``None`` for the project Maximum.
            Default: ``None`` (project maximum)

        Returns
        -------
        bool
            ``True`` if the running Python version is valid, otherwise logs an error and exits
        """
        max_python = VALID_PYTHON[1] if max_version is None else max_version
        retval = (VALID_PYTHON[0] <= sys.version_info[:2] <= max_python
                  and self.python_architecture == "64bit")
        logger.debug("Python version %s(%s) within %s - %s(64bit): %s",
                     self.python_version,
                     self.python_architecture,
                     VALID_PYTHON[0],
                     max_python,
                     retval)
        if not retval:
            print()
            logger.error("Your Python version %s(%s) is unsupported. Please run with Python "
                         "version %s to %s 64bit.",
                         self.python_version,
                         self.python_architecture,
                         ".".join(str(x) for x in VALID_PYTHON[0]),
                         ".".join(str(x) for x in max_python))
            print()
            logger.error("If you have recently upgraded faceswap, then you will need to create a "
                         "new virtual environment.")
            logger.error("The easiest way to do this is to run the latest version of the Faceswap "
                         "installer from:")
            logger.error("https://github.com/deepfakes/faceswap/releases")
            print()
            input("Press <Enter> to close")
            sys.exit(1)

        return retval

    def validate(self) -> None:
        """ Perform validation that the running system can be used for faceswap. Log an error and
        exit if it cannot """
        if not any((self.is_linux, self.is_macos, self.is_windows)):
            logger.error("Your system %s is not supported!", self.system.title())
            sys.exit(1)
        if self.is_macos and self.machine == "arm64" and not self.is_conda:
            logger.error("Setting up Faceswap for Apple Silicon outside of a Conda "
                         "environment is unsupported")
            sys.exit(1)
        self.validate_python()


class Packages():
    """ Holds information about installed python and conda packages.

    Note: Packaging library is lazy loaded as it may not be available during setup.py
    """
    def __init__(self) -> None:
        self._conda_exe = which("conda")
        self._installed_python = self._get_installed_python()
        self._installed_conda: list[str] | None = None
        self._get_installed_conda()

    @property
    def installed_python(self) -> dict[str, str]:
        """ dict[str, str] : Installed Python package names to Python package versions """
        return self._installed_python

    @property
    def installed_python_pretty(self) -> str:
        """ str: A pretty printed representation of installed Python packages """
        pkgs = self._installed_python
        align = max(len(x) for x in pkgs) + 1
        return "\n".join(f"{k.ljust(align)} {v}" for k, v in pkgs.items())

    @property
    def installed_conda(self) -> dict[str, tuple[str, str, str]]:
        """ dict[str, tuple[str, str]] : Installed Conda package names to the version and
        channel """
        if not self._installed_conda:
            return {}

        installed = [re.sub(" +", " ", line.strip())
                     for line in self._installed_conda if not line.startswith("#")]
        retval = {}
        for pkg in installed:
            item = pkg.split(" ")
            assert len(item) == 4
            retval[item[0]] = T.cast(tuple[str, str, str], tuple(item[1:]))
        return retval

    @property
    def installed_conda_pretty(self) -> str:
        """ str: A pretty printed representation of installed conda packages """
        if not self._installed_conda:
            return "Could not get Conda package list"
        return "\n".join(self._installed_conda)

    def __repr__(self) -> str:
        """ Pretty print the installed packages for logging """
        props = ", ".join(
            f"{k}={repr(getattr(self, k))}"
            for k, v in self.__class__.__dict__.items()
            if isinstance(v, property) and not k.startswith("_") and "pretty" not in k)
        return f"{self.__class__.__name__}({props})"

    def _get_installed_python(self) -> dict[str, str]:
        """ Parse the installed python modules

        Returns
        -------
        dict[str, str]
            Installed Python package names to Python package versions
        """
        installed = _lines_from_command([sys.executable, "-m", "pip", "freeze", "--local"])
        retval = {}
        for pkg in installed:
            if "==" not in pkg:
                continue
            item = pkg.split("==")
            retval[item[0].lower()] = item[1]
        logger.debug("Installed Python packages: %s", retval)
        return retval

    def _get_installed_conda(self) -> None:
        """ Collect the output from 'conda list' for the installed Conda packages and
        populate :attr:`_installed_conda`

        Returns
        -------
        list[str]
            Each line of output from the 'conda list' command
        """
        if not self._conda_exe:
            logger.debug("Conda not found. Not collecting packages")
            return

        lines = _lines_from_command([self._conda_exe, "list", "--show-channel-urls"])
        if not lines:
            self._installed_conda = ["Could not get Conda package list"]
            return
        self._installed_conda = lines
        logger.debug("Installed Conda packages: %s", self.installed_conda)


__all__ = get_module_objects(__name__)


if __name__ == "__main__":
    print(System())
    print(Packages())
