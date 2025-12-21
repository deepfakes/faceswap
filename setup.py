#!/usr/bin/env python3
""" Install packages for faceswap.py """
# pylint:disable=too-many-lines
from __future__ import annotations

import logging
import json
import os
import re
import sys
import typing as T
from importlib import import_module
from shutil import which
from string import printable
from subprocess import PIPE, Popen

from lib.logger import log_setup
from lib.system import Cuda, Packages, ROCm, System
from lib.utils import get_module_objects, PROJECT_ROOT
from requirements.requirements import Requirements, PYTHON_VERSIONS

if T.TYPE_CHECKING:
    from packaging.requirements import Requirement
    import pip
    import lib.utils as lib_utils

logger = logging.getLogger(__name__)
BackendType: T.TypeAlias = T.Literal['nvidia', 'apple_silicon', 'cpu', 'rocm', "all"]

# Conda packages that are required for a specific backend
_CONDA_BACKEND_REQUIRED: dict[BackendType, list[str]] = {
    "all": ["tk", "git"]}

# Conda packages that are required for a specific OS
_CONDA_OS_REQUIRED: dict[T.Literal["darwin", "linux", "windows"], list[str]] = {
    "linux": ["xorg-libxft"]}  # required to fix TK fonts on Linux

# Mapping of Conda packages to channel if in not conda-forge
_CONDA_MAPPING: dict[str, str] = {}

# Force output to utf-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type:ignore[union-attr]


class _InstallState:  # pylint:disable=too-few-public-methods
    """ Marker to track if a step has failed installing """
    failed = False
    messages: list[str] = []


class Environment():
    """ The current install environment

    Parameters
    ----------
    updater : bool, Optional
        ``True`` if the script is being called by Faceswap's internal updater. ``False`` if full
        setup is running. Default: ``False``
    """
    _backends = (("nvidia", "apple_silicon", "rocm", "cpu"))

    def __init__(self, updater: bool = False) -> None:
        self.updater = updater
        self.system = System()
        logger.debug("Running on: %s", self.system)
        if not updater:
            self.system.validate()
        self.is_installer: bool = False  # Flag setup is being run by installer to skip steps
        self.include_dev_tools: bool = False
        self.backend: T.Literal["nvidia", "apple_silicon", "cpu", "rocm"] | None = None
        self.enable_docker: bool = False
        self.cuda_cudnn = ["", ""]
        self.requirement_version = ""
        self.rocm_version: tuple[int, ...] = (0, 0, 0)
        self._process_arguments()
        self._output_runtime_info()
        self._check_pip()

    @property
    def cuda_version(self) -> str:
        """ str : The detected globally installed Cuda Version """
        return self.cuda_cudnn[0]

    @property
    def cudnn_version(self) -> str:
        """ str : The detected globally installed cuDNN Version """
        return self.cuda_cudnn[1]

    def set_backend(self, backend: T.Literal["nvidia", "apple_silicon", "cpu", "rocm"]) -> None:
        """ Set the backend to install for

        Parameters
        ----------
        backend : Literal["nvidia", "apple_silicon", "cpu", "rocm"]
            The backend to setup faceswap for
        """
        logger.debug("Setting backend to '%s'", backend)
        self.backend = backend

    def set_requirements(self, requirements: str) -> None:
        """ Validate that the requirements are compatible with the running Python version and
        set the requirements file version to install use

        Parameters
        ----------
        backend : str
            The requirements file version to use for install
        """
        if requirements in PYTHON_VERSIONS:
            self.system.validate_python(max_version=PYTHON_VERSIONS[requirements])
        logger.debug("Setting requirements to '%s'", requirements)
        self.requirement_version = requirements

    def _parse_backend_from_cli(self, arg: str) -> None:
        """ Parse a command line argument and populate :attr:`backend` if valid

        Parameters
        ----------
        arg : str
            The command line argument to parse
        """
        arg = arg.lower()
        if not any(arg.startswith(b) for b in self._backends):
            return
        self.set_backend(next(b for b in self._backends if arg.startswith(b)))  # type:ignore[misc]
        if arg == "cpu":
            self.set_requirements("cpu")
            return
        # Get Cuda/ROCm requirements file
        assert self.backend is not None
        req_files = sorted([os.path.splitext(f)[0].replace("requirements_", "")
                            for f in os.listdir(os.path.join(PROJECT_ROOT, "requirements"))
                            if os.path.splitext(f)[-1] == ".txt"
                            and f.startswith("requirements_")
                            and self.backend in f])
        if arg == self.backend:  # Default to latest
            logger.debug("No version specified. Defaulting to latest requirements")
            self.set_requirements(req_files[-1])
            return
        lookup = [r.replace("_", "") for r in req_files]
        if arg not in lookup:
            logger.debug("Defaulting to latest requirements for unknown lookup '%s'", arg)
            self.set_requirements(req_files[-1])
            return
        self.set_requirements(req_files[lookup.index(arg)])

    def _process_arguments(self) -> None:
        """ Process any cli arguments and dummy in cli arguments if calling from updater. """
        args = sys.argv[:]
        if self.updater:
            get_backend = T.cast("lib_utils",  # type:ignore[attr-defined,valid-type]
                                 import_module("lib.utils")).get_backend
            args.append(f"--{get_backend()}")
        logger.debug(args)
        if self.system.is_macos and self.system.machine == "arm64":
            self.set_backend("apple_silicon")
            self.set_requirements("apple-silicon")
        for arg in args:
            if arg == "--installer":
                self.is_installer = True
                continue
            if arg == "--dev":
                self.include_dev_tools = True
                continue
            if not self.backend and arg.startswith("--"):
                self._parse_backend_from_cli(arg[2:])

    def _output_runtime_info(self) -> None:
        """ Output run time info """
        logger.info("Setup in %s %s", self.system.system.title(), self.system.release)
        logger.info("Running as %s", "Root/Admin" if self.system.is_admin else "User")
        if self.system.is_conda:
            logger.info("Running in Conda")
        if self.system.is_virtual_env:
            logger.info("Running in a Virtual Environment")
        logger.info("Encoding: %s", self.system.encoding)

    def _check_pip(self) -> None:
        """ Check installed pip version """
        try:
            _pip = T.cast("pip", import_module("pip"))  # type:ignore[valid-type]
        except ModuleNotFoundError:
            logger.error("Import pip failed. Please Install python3-pip and try again")
            sys.exit(1)
        logger.info("Pip version: %s", _pip.__version__)  # type:ignore[attr-defined]

    def _configure_keras(self) -> None:
        """ Set up the keras.json file to use Torch as the backend """
        if "KERAS_HOME" in os.environ:
            keras_dir = os.environ["KERAS_HOME"]
        else:
            keras_base_dir = os.path.expanduser("~")
            if not os.access(keras_base_dir, os.W_OK):
                keras_base_dir = "/tmp"
            keras_dir = os.path.join(keras_base_dir, ".keras")
        keras_dir = os.path.expanduser(keras_dir)
        os.makedirs(keras_dir, exist_ok=True)
        conf_file = os.path.join(keras_dir, "keras.json")
        config = {}
        if os.path.exists(conf_file):
            try:
                with open(conf_file, "r", encoding="utf-8") as c_file:
                    config = json.load(c_file)
            except ValueError:
                pass
        config["backend"] = "torch"
        with open(conf_file, "w", encoding="utf-8") as c_file:
            c_file.write(json.dumps(config, indent=4))
        logger.info("Keras config written to: %s", conf_file)

    def set_config(self) -> None:
        """ Set the backend in the faceswap config file """
        config = {"backend": self.backend}
        pypath = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(pypath, "config", ".faceswap")
        with open(config_file, "w", encoding="utf8") as cnf:
            json.dump(config, cnf)
        logger.info("Faceswap config written to: %s", config_file)
        self._configure_keras()


class RequiredPackages():
    """ Holds information about installed and required packages.
    Handles updating dependencies based on running platform/backend

    Parameters
    ----------
    environment : :class:`Environment`
        Environment class holding information about the running system
    """
    def __init__(self, environment: Environment) -> None:
        self._env = environment
        self._packages = Packages()
        self._requirements = Requirements(include_dev=self._env.include_dev_tools)
        self._check_packaging()
        self.conda = self._get_missing_conda()
        self.python = self._get_missing_python(
            self._requirements.requirements[self._env.requirement_version])
        self.pip_arguments = [
            x.strip()
            for p in self._requirements.global_options[self._env.requirement_version]
            for x in p.split()]
        """ list[str] : Any additional pip arguments that are required for installing from pip for
        the given backend """

    @property
    def packages_need_install(self) -> bool:
        """bool : ``True`` if there are packages available that need to be installed """
        return bool(self.conda or self.python)

    def _check_packaging(self) -> None:
        """ Install packaging if it is not available  """
        if self._requirements.packaging_available:
            return
        cmd = [sys.executable, "-u", "-m", "pip", "install", "--no-cache-dir"]
        if self._env.system.is_admin and not self._env.system.is_virtual_env:
            cmd.append("--user")
        cmd.append("packaging")
        logger.info("Installing required package...")
        installer = Installer(self._env, ["Packaging"], cmd, False, False)
        if installer() != 0:
            logger.error("Unable to install package: %s. Process aborted", "packaging")
            sys.exit(1)

    def _get_missing_python(self, requirements: list[Requirement]
                            ) -> list[dict[T.Literal["name", "package"], str]]:
        """ Check for missing Python dependencies

        Parameters
        ----------
        requirements : list[:class:`packaging.requirements.Requirement]`
            The packages that are required to be installed

        Returns
        -------
        list[dict[Literal["name", "package"], str]]
            List of missing Python packages to install
        """
        retval: list[dict[T.Literal["name", "package"], str]] = []
        for req in requirements:
            package: dict[T.Literal["name", "package"], str] = {
                "name": req.name.title(),
                "package": f"{req.name}{req.specifier}"}
            installed_version = self._packages.installed_python.get(req.name, "")
            if not installed_version:
                logger.debug("Adding new Python package '%s'", package["package"])
                retval.append(package)
                continue
            if not req.specifier.contains(installed_version):
                logger.debug("Adding Python package '%s' for specifier change from '%s' to '%s'",
                             package["package"], installed_version, str(req.specifier))
                retval.append(package)
                continue
            logger.debug("Skipping installed Python package '%s'", package["package"])
        logger.debug("Selected missing Python packages: %s", retval)
        return retval

    def _get_required_conda(self) -> list[dict[T.Literal["package", "channel"], str]]:
        """ Add backend specific packages to Conda required packages

        Returns
        -------
        list[tuple[Literal["package", "channel"], str]]
            List of required Conda package names and the channel to install from
        """
        retval: list[dict[T.Literal["package", "channel"], str]] = []
        assert self._env.backend is not None
        to_add = (_CONDA_BACKEND_REQUIRED.get(self._env.backend, []) +
                  _CONDA_BACKEND_REQUIRED.get("all", []) +
                  _CONDA_OS_REQUIRED.get(self._env.system.system, []))
        if not to_add:
            logger.debug("No packages to add for '%s'('%s'). All backend packages: %s. All OS "
                         "packages: %s",
                         self._env.backend, self._env.system,
                         _CONDA_BACKEND_REQUIRED, _CONDA_OS_REQUIRED)
            return retval
        for pkg in to_add:
            channel = _CONDA_MAPPING.get(pkg, "conda-forge")
            retval.append({"package": pkg, "channel": channel})
            logger.debug("Adding conda required package '%s' for system '%s'('%s'))",
                         pkg, self._env.backend, self._env.system.system)
        return retval

    def _get_missing_conda(self) -> dict[str, list[dict[T.Literal["name", "package"], str]]]:
        """ Check for conda missing dependencies

        Returns
        -------
        dict[str, list[dict[Literal["name", "package"], str]]]
            The Conda packages to install grouped by channel
        """
        retval: dict[str, list[dict[T.Literal["name", "package"], str]]] = {}
        if not self._env.system.is_conda:
            return retval
        required = self._get_required_conda()
        requirements = self._requirements.parse_requirements(
            [p["package"] for p in required])
        channels = [p["channel"] for p in required]
        installed = {k: v for k, v in self._packages.installed_conda.items() if v[1] != "pypi"}
        for req, channel in zip(requirements, channels):
            spec_str = str(req.specifier).replace("==", "=") if req.specifier else ""
            package: dict[T.Literal["name", "package"], str] = {"name": req.name.title(),
                                                                "package": f"{req.name}{spec_str}"}
            exists = installed.get(req.name)
            if req.name == "tk" and self._env.system.is_linux:
                # Default TK has bad fonts under Linux.
                # Ref: https://github.com/ContinuumIO/anaconda-issues/issues/6833
                # This versioning will fail in parse_requirements, so we need to do it here
                package["package"] = f"{req.name}=*=xft_*"  # Swap out for explicit XFT version
                if exists is not None and not exists[1].startswith("xft"):  # Replace noxft version
                    exists = None
            if not exists:
                logger.debug("Adding new Conda package '%s'", package["package"])
                retval.setdefault(channel, []).append(package)
                continue
            if exists[-1] != channel:
                logger.debug("Adding Conda package '%s' for channel change from '%s' to '%s'",
                             package["package"], exists[-1], channel)
                retval.setdefault(channel, []).append(package)
                continue
            if not req.specifier.contains(exists[0]):
                logger.debug("Adding Conda package '%s' for specifier change from '%s' to '%s'",
                             package["package"], exists[0], spec_str)
                retval.setdefault(channel, []).append(package)
                continue
            logger.debug("Skipping installed Conda package '%s'", package["package"])
        logger.debug("Selected missing Conda packages: %s", retval)
        return retval


class Checks():  # pylint:disable=too-few-public-methods
    """ Pre-installation checks

    Parameters
    ----------
    environment : :class:`Environment`
        Environment class holding information about the running system
    """
    def __init__(self, environment: Environment) -> None:
        self._env: Environment = environment
        self._tips: Tips = Tips()
    # Checks not required for installer
        if self._env.is_installer:
            return
    # Checks not required for Apple Silicon
        if self._env.backend == "apple_silicon":
            return
        self._user_input()
        self._check_cuda()
        self._check_rocm()
        if self._env.system.is_windows:
            self._tips.pip()

    def _rocm_ask_enable(self) -> None:
        """ Set backend to 'rocm' if OS is Linux and ROCm support required """
        if not self._env.system.is_linux:
            return
        logger.info("ROCm support:\r\nIf you are using an AMD GPU, then select 'yes'."
                    "\r\nCPU/non-AMD GPU users should answer 'no'.\r\n")
        i = input("Enable ROCm Support? [y/N] ").strip()
        if i not in ("", "Y", "y", "n", "N"):
            logger.warning("Invalid selection '%s'", i)
            self._rocm_ask_enable()
            return
        if i not in ("Y", "y"):
            return
        logger.info("ROCm Support Enabled")
        self._env.set_backend("rocm")
        versions = ["6.0", "6.1", "6.2", "6.3", "6.4"]
        i = input(f"Which ROCm version? [{', '.join(versions)}] ").strip()
        i = versions[-1] if not i else i
        print(i, i in versions, versions)
        if i not in versions:
            logger.warning("Invalid selection '%s'", i)
            self._rocm_ask_enable()
            return
        logger.info("ROCm Version %s Selected", i)
        self._env.set_requirements(f"rocm_{i.replace('.', '')}")

    def _docker_ask_enable(self) -> None:
        """ Enable or disable Docker """
        i = input("Enable  Docker? [y/N] ").strip()
        if i not in ("", "Y", "y", "n", "N"):
            logger.warning("Invalid selection '%s'", i)
            self._docker_ask_enable()
            return
        if i in ("Y", "y"):
            logger.info("Docker Enabled")
            self._env.enable_docker = True
        else:
            logger.info("Docker Disabled")
            self._env.enable_docker = False

    def _cuda_ask_enable(self) -> None:
        """ Enable or disable CUDA """
        i = input("Enable  CUDA? [Y/n] ").strip()
        if i not in ("", "Y", "y", "n", "N"):
            logger.warning("Invalid selection '%s'", i)
            self._cuda_ask_enable()
            return
        if i not in ("", "Y", "y"):
            return
        logger.info("CUDA Enabled")
        self._env.set_backend("nvidia")
        versions = ["11", "12", "13"]
        i = input("Which Cuda version: 11 (GTX7xx-8xx), 12 (GTX9xx-10xx) or 13 (RTX20xx-)? "
                  f"[{', '.join(versions)}] ").strip()
        i = "13" if not i else i
        if i not in versions:
            logger.warning("Invalid selection '%s'", i)
            self._cuda_ask_enable()
            return
        logger.info("CUDA Version %s Selected", i)
        self._env.set_requirements(f"nvidia_{i}")

    def _docker_confirm(self) -> None:
        """ Warn if nvidia-docker on non-Linux system """
        logger.warning("Nvidia-Docker is only supported on Linux.\r\n"
                       "Only CPU is supported in Docker for your system")
        self._docker_ask_enable()
        if self._env.enable_docker:
            logger.warning("CUDA Disabled")
            self._env.set_backend("cpu")

    def _docker_tips(self) -> None:
        """ Provide tips for Docker use """
        if self._env.backend != "nvidia":
            self._tips.docker_no_cuda()
        else:
            self._tips.docker_cuda()

    def _user_input(self) -> None:
        """ Get user input for AMD/ROCm/Cuda/Docker """
        if self._env.backend is None:
            self._rocm_ask_enable()
        if self._env.backend is None:
            self._docker_ask_enable()
            self._cuda_ask_enable()
        if not self._env.system.is_linux and (self._env.enable_docker
                                              and self._env.backend == "nvidia"):
            self._docker_confirm()
        if self._env.enable_docker:
            self._docker_tips()
            self._env.set_config()
            sys.exit(0)

    def _check_cuda(self) -> None:
        """ Check for Cuda and cuDNN Locations. """
        if self._env.backend != "nvidia":
            logger.debug("Skipping Cuda checks as not enabled")
            return
        if not any((self._env.system.is_linux, self._env.system.is_windows)):
            return
        cuda = Cuda()
        if cuda.versions:
            str_vers = ", ".join(".".join(str(x) for x in v) for v in cuda.versions)
            msg = (f"Globally installed Cuda version{'s' if len(cuda.versions) > 1 else ''} "
                   f"{str_vers} found. PyTorch uses it's own version of Cuda, so if you have "
                   "GPU issues, you should remove these global installs")
            _InstallState.messages.append(msg)
            self._env.cuda_cudnn[0] = str_vers
            logger.debug("CUDA version: %s", self._env.cuda_version)
        if cuda.cudnn_versions:
            str_vers = ", ".join(".".join(str(x) for x in v)
                                 for v in cuda.cudnn_versions.values())
            msg = ("Globally installed CuDNN version"
                   f"{'s' if len(cuda.cudnn_versions) > 1 else ''} {str_vers} found. PyTorch uses "
                   "its own version of Cuda, so if you have GPU issues, you should remove these "
                   "global installs")
            _InstallState.messages.append(msg)
            self._env.cuda_cudnn[1] = str_vers
            logger.debug("cuDNN version: %s", self._env.cudnn_version)

    def _check_rocm(self) -> None:
        """ Check for ROCm version """
        if self._env.backend != "rocm" or not self._env.system.is_linux:
            logger.debug("Skipping ROCm checks as not enabled")
            return
        rocm = ROCm()

        if rocm.is_valid or rocm.valid_installed:
            self._env.rocm_version = max(rocm.valid_versions)
            logger.info("ROCm version: %s", ".".join(str(v) for v in self._env.rocm_version))
        if rocm.is_valid:
            return
        if rocm.valid_installed:
            str_vers = ".".join(str(v) for v in self._env.rocm_version)
            _InstallState.messages.append(
                f"Valid ROCm version {str_vers} is installed, but is not your default version.\n"
                "You may need to change this to enable GPU acceleration")
            return

        if rocm.versions:
            str_vers = ", ".join(".".join(str(x) for x in v) for v in rocm.versions)
            msg = f"Incompatible ROCm version{'s' if len(rocm.versions) > 1 else ''}: {str_vers}\n"
        else:
            msg = "ROCm not found\n"
            _InstallState.messages.append(f"{msg}\n")
        str_min = ".".join(str(v) for v in rocm.version_min)
        str_max = ".".join(str(v) for v in rocm.version_max)
        valid = f"{str_min} to {str_max}" if str_min != str_max else str_min
        msg += ("The installation can proceed, but you will need to install ROCm version "
                f"{valid} to enable GPU acceleration")
        _InstallState.messages.append(msg)


class Status():
    """ Simple Status output for intercepting Conda/Pip installs and keeping the terminal clean

    Parameters
    ----------
    is_conda : bool
        ``True`` if installing packages from Conda. ``False`` if installing from pip
    """
    def __init__(self, is_conda: bool):
        self._is_conda = is_conda
        self._last_line = ""
        self._max_width = 79  # Keep short because of NSIS Details window size
        self._prefix = "> "
        self._conda_tracked: dict[str, dict[T.Literal["size", "done"], float]] = {}
        self._re_pip_pkg = re.compile(r"^Downloading\s(?P<lib>\w+)\b.*?\s\((?P<size>.+)\)")
        self._re_pip_http = re.compile(r"https?://[^\s]*/([^/\s]+)")
        self._re_pip_progress = re.compile(r"^Progress\s+(?P<done>\d+).+?(?P<total>\d+)")
        self._re_conda = re.compile(
            r"(?P<lib>^\S+)\s+\|\s+(?P<tot>\d+\.?\d*\s\w+).*\|\s+(?P<prg>\d+)%")

    def _clear_line(self) -> None:
        """ Clear the last printed line from the console """
        print(" " * self._max_width, end="\r")

    def _print(self, line: str) -> None:
        """ Clear the last line and print the new line to the console

        Parameters
        ----------
        line : str
            The line to print
        """
        full_line = f"{self._prefix}{line}"
        output = full_line
        if len(output) > self._max_width:
            output = f"{output[:self._max_width - 3]}..."
        if len(output) < len(self._last_line):
            self._clear_line()
        self._last_line = full_line
        print(output, end="\r")

    def _parse_size(self, size: str) -> float:
        """ Parse the string representation of a package size and return as megabytes

        Parameters
        ----------
        size : str
            The string representation of a package size

        Returns
        -------
        float
            The size in megabytes
        """
        size, unit = size.strip().split(" ", maxsplit=1)
        if unit.lower() == "b":
            return float(size) / 1024 / 1024
        if unit.lower() == "kb":
            return float(size) / 1024
        if unit.lower() == "mb":
            return float(size)
        if unit.lower() == "gb":
            return float(size) * 1024
        return float(size)  # Should never happen, but to prevent error

    def _print_conda(self, line: str) -> None:
        """ Output progress for Conda installs

        Parameters
        ----------
        line : str
            The conda install line to parse
        """
        progress = self._re_conda.match(line)
        if progress is None:
            self._print(line)
            return
        info = progress.groupdict()
        if info["lib"] not in self._conda_tracked:
            self._conda_tracked[info["lib"]] = {"size": self._parse_size(info["tot"]),
                                                "done": float(info["prg"])}
        else:
            self._conda_tracked[info["lib"]]["done"] = float(info["prg"])
        count = len(self._conda_tracked)
        total_size = sum(v["size"] for v in self._conda_tracked.values())
        prog = min(sum(v["done"] for v in self._conda_tracked.values()) / count, 100.)
        self._print(f"Downloading {count} packages ({total_size:.1f} MB) {prog:.1f}%")

    def _print_pip(self, line: str) -> None:
        """ Output progress for Pip installs

        Parameters
        ----------
        line : str
            The pip install line to parse
        """
        if (line.lower().startswith("installing collected packages:") and
                len(line) > self._max_width):
            count = len(line.split(":", maxsplit=1)[-1].split(","))
            line = f"Installing {count} collected packages..."
        progress = self._re_pip_progress.match(line)
        if progress is None:
            self._print(line)
            return
        info = progress.groupdict()
        done = (int(info["done"]) / int(info["total"])) * 100.0
        last_line = self._last_line.strip()[len(self._prefix):]
        pkg = self._re_pip_pkg.match(self._re_pip_http.sub(r"\1", last_line))
        if pkg is not None:
            info = pkg.groupdict()
            last_line = f"Downloading {info['lib']} ({info['size']})"
        self._print(f"{last_line} {done:.1f}%")

    def __call__(self, line: str) -> None:
        """ Update the output status with the given line

        Parameters
        ----------
        line : str
            A cleansed line from either Conda or Pip installers
        """
        if self._is_conda:
            self._print_conda(line.strip())
        else:
            self._print_pip(line.strip())

    def close(self) -> None:
        """ Reset all progress bars and re-enable the cursor """
        self._clear_line()


class Installer():
    """ Uses the python Subprocess module to install packages.

    Parameters
    ----------
    environment : :class:`Environment`
        Environment class holding information about the running system
    packages : list[str]
        The list of package names that are to be installed
    command : list
        The command to run
    is_conda : bool
        ``True`` if conda install command is running. ``False`` if pip install command is running
    is_gui : bool
        ``True`` if the process is being called from the Faceswap GUI
    """
    def __init__(self,  # pylint:disable=too-many-positional-arguments
                 environment: Environment,
                 packages: list[str],
                 command: list[str],
                 is_conda: bool,
                 is_gui: bool) -> None:
        self._output_information(packages)
        logger.debug("argv: %s", command)
        self._env = environment
        self._packages = packages
        self._command = command
        self._is_conda = is_conda
        self._is_gui = is_gui
        self._status = Status(is_conda)
        self._re_ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self._seen_lines: set[str] = set()
        self.error_lines: list[str] = []

    @classmethod
    def _output_information(cls, packages: list[str]):
        """ INFO log the packages to be installed, splitting along multiple lines for long package
        lists (68 chars = 79 chars - (log-level spacing + indent))

        Parameters
        ----------
        packages : list[str]
            The list of package names that are to be installed
        """
        output = ""
        sep = ", "
        for pkg in packages:
            current = pkg + sep
            if len(output) + len(current) > 68:
                logger.info("  %s", output)
                output = current
            else:
                output += current
        if output:
            logger.info("  %s", output[:-len(sep)])

    def _clean_line(self, text: str) -> str:
        """Remove ANSI escape sequences and special characters from text.

        Parameters
        ----------
        text : str
            The text to clean

        Returns
        -------
        str
            The cleansed text
        """
        clean = self._re_ansi_escape.sub("", text.rstrip())
        return ''.join(c for c in clean if c in set(printable))

    def _seen_line_log(self, text: str, is_error: bool = False) -> str:
        """ Output gets spammed to the log file when conda is waiting/processing. Only log each
        unique line once.

        Parameters
        ----------
        text : str
            The text to log
        is_error : bool, optional
            ``True`` if the line comes from an error. Default: ``False``

        Returns
        -------
        str
            The cleansed log line

        """
        clean = self._clean_line(text)
        if clean in self._seen_lines:
            return ""
        clean = f"ERROR: {clean}" if is_error else clean
        logger.debug(clean)
        self._seen_lines.add(clean)
        return clean

    def __call__(self) -> int:
        """ Install a package using the Subprocess module

        Returns
        -------
        int
            The return code of the package install process
        """
        with Popen(self._command,
                   bufsize=0, stdout=PIPE, stderr=PIPE) as proc:
            lines = b""
            while True:
                if proc.stdout is not None:
                    lines = proc.stdout.readline()
                returncode = proc.poll()
                if lines == b"" and returncode is not None:
                    break
                for line in lines.split(b"\r"):
                    clean = self._seen_line_log(line.decode("utf-8", errors="replace"))
                    if not self._is_gui and clean:
                        self._status(clean)
            if returncode and proc.stderr is not None:
                for line in proc.stderr.readlines():
                    clean = self._seen_line_log(line.decode("utf-8", errors="replace"),
                                                is_error=True)
                    if clean:
                        self.error_lines.append(clean.replace("ERROR:", "").strip())

        logger.debug("Packages: %s, returncode: %s", self._packages, returncode)
        if not self._is_gui:
            self._status.close()
        return returncode


class Install():  # pylint:disable=too-few-public-methods
    """ Handles installation of Faceswap requirements

    Parameters
    ----------
    environment : :class:`Environment`
        Environment class holding information about the running system
    is_gui : bool, Optional
        ``True`` if the caller is the Faceswap GUI. Used to prevent output of progress bars
        which get scrambled in the GUI
     """
    def __init__(self, environment: Environment, is_gui: bool = False) -> None:
        self._env = environment
        self._is_gui = is_gui
        if not self._env.is_installer and not self._env.updater:
            self._ask_continue()
        self._packages = RequiredPackages(environment)
        if self._env.updater and not self._packages.packages_need_install:
            logger.info("All Dependencies are up to date")
            return
        self._install_packages()
        self._finalize()

    def _ask_continue(self) -> None:
        """ Ask Continue with Install """
        if _InstallState.messages:
            for msg in _InstallState.messages:
                logger.warning(msg)
        text = "Please ensure your System Dependencies are met."
        if self._env.backend == "rocm":
            text += ("\r\nPlease ensure that your AMD GPU is supported by the "
                     "installed ROCm version before proceeding.")
        text += "\r\nContinue? [y/N] "
        inp = input(text)
        if inp in ("", "N", "n"):
            logger.info("Installation cancelled")
            sys.exit(0)

    def _from_pip(self,
                  packages: list[dict[T.Literal["name", "package"], str]],
                  extra_args: list[str] | None = None) -> None:
        """ Install packages from pip

        Parameters
        ----------
        packages : list[dict[T.Literal["name", "package"], str]
            The formatted list of packages to be installed
        extra_args : list[str] | None, optional
            Any extra arguments to provide to pip. Default: ``None`` (no extra arguments)
        """
        pipexe = [sys.executable,
                  "-u", "-m", "pip", "install", "--no-cache-dir", "--progress-bar=raw"]

        if not self._env.system.is_admin and not self._env.system.is_virtual_env:
            pipexe.append("--user")  # install as user to solve perm restriction
        if extra_args is not None:
            pipexe.extend(extra_args)
        pipexe.extend([p["package"] for p in packages])
        names = [p["name"] for p in packages]
        installer = Installer(self._env, names, pipexe, False, self._is_gui)
        if installer() != 0:
            msg = f"Unable to install Python packages: {', '.join(names)}"
            logger.warning("%s. Please install these packages manually", msg)
            for line in installer.error_lines:
                _InstallState.messages.append(line)
            _InstallState.failed = True

    def _from_conda(self,
                    packages: list[dict[T.Literal["name", "package"], str]],
                    channel: str) -> None:
        """ Install packages from conda

        Parameters
        ----------
        packages : list[dict[T.Literal["name", "package"], str]]
            The full formatted packages to be installed
        channel : str
            The Conda channel to install from.

        Returns
        -------
        bool
            ``True`` if the package was succesfully installed otherwise ``False``
        """
        conda = which("conda")
        assert conda is not None
        condaexe = [conda, "install", "-y", "-c", channel,
                    "--override-channels", "--strict-channel-priority"]
        condaexe += [p["package"] for p in packages]
        names = [p["name"] for p in packages]
        retcode = Installer(self._env, names, condaexe, True, self._is_gui)()
        if retcode != 0:
            logger.warning("Unable to install Conda packages: %s. "
                           "Please install these packages manually", ', '.join(names))
            _InstallState.failed = True

    def _install_packages(self) -> None:
        """ Install the required packages """
        if self._packages.conda:
            logger.info("Installing Conda packages...")
            for channel, packages in self._packages.conda.items():
                self._from_conda(packages, channel)
        if self._packages.python:
            logger.info("Installing Python packages...")
            packages = [p for p in self._packages.python if p["name"] != "Packaging"]
            self._from_pip(packages, extra_args=self._packages.pip_arguments)

    def _finalize(self) -> None:
        """ Output final information on completion """
        if self._env.updater:
            return
        if not _InstallState.failed:
            if _InstallState.messages:
                for msg in _InstallState.messages:
                    logger.warning(msg)
            logger.info("All Faceswap dependencies are met. You are good to go.\r\n\r\n"
                        "Enter:  'python faceswap.py -h' to see the options\r\n"
                        "        'python faceswap.py gui' to launch the GUI")
        else:
            msg = "Some packages failed to install. "
            if not _InstallState.messages:
                msg += ("This may be temporary and might be fixed by re-running this script. "
                        "Otherwise check 'faceswap_setup.log' to see which failed and install "
                        "these packages manually.")
            else:
                msg += ("Further information can be found in 'faceswap_setup.log'. The following "
                        "output shows specific error(s) that were collected:\r\n")
                msg += "\r\n".join(_InstallState.messages)
            logger.error(msg)
            sys.exit(1)


class Tips():
    """ Display installation Tips """
    @classmethod
    def docker_no_cuda(cls) -> None:
        """ Output Tips for Docker without Cuda """
        logger.info(
            "1. Install Docker from: https://www.docker.com/get-started\n\n"
            "2. Enter the Faceswap folder and build the Docker Image For Faceswap:\n"
            "   docker build -t faceswap-cpu -f Dockerfile.cpu .\n\n"
            "3. Launch and enter the Faceswap container:\n"
            "  a. Headless:\n"
            "     docker run --rm -it -v ./:/srv faceswap-cpu\n\n"
            "  b. GUI:\n"
            "     xhost +local: && \\ \n"
            "     docker run --rm -it \\ \n"
            "     -v ./:/srv \\ \n"
            "     -v /tmp/.X11-unix:/tmp/.X11-unix \\ \n"
            "     -e DISPLAY=${DISPLAY} \\ \n"
            "     faceswap-cpu \n")
        logger.info("That's all you need to do with docker. Have fun.")

    @classmethod
    def docker_cuda(cls) -> None:
        """ Output Tips for Docker with Cuda"""
        logger.info(
            "1. Install Docker from: https://www.docker.com/get-started\n\n"
            "2. Install latest CUDA 11 and cuDNN 8 from: https://developer.nvidia.com/cuda-"
            "downloads\n\n"
            "3. Install the the Nvidia Container Toolkit from https://docs.nvidia.com/datacenter/"
            "cloud-native/container-toolkit/latest/install-guide\n\n"
            "4. Restart Docker Service\n\n"
            "5. Enter the Faceswap folder and build the Docker Image For Faceswap:\n"
            "   docker build -t faceswap-gpu -f Dockerfile.gpu .\n\n"
            "6. Launch and enter the Faceswap container:\n"
            "  a. Headless:\n"
            "     docker run --runtime=nvidia --rm -it -v ./:/srv faceswap-gpu\n\n"
            "  b. GUI:\n"
            "     xhost +local: && \\ \n"
            "     docker run --runtime=nvidia --rm -it \\ \n"
            "     -v ./:/srv \\ \n"
            "     -v /tmp/.X11-unix:/tmp/.X11-unix \\ \n"
            "     -e DISPLAY=${DISPLAY} \\ \n"
            "     faceswap-gpu \n")
        logger.info("That's all you need to do with docker. Have fun.")

    @classmethod
    def macos(cls) -> None:
        """ Output Tips for macOS"""
        logger.info(
            "setup.py does not directly support macOS. The following tips should help:\n\n"
            "1. Install system dependencies:\n"
            "XCode from the Apple Store\n"
            "XQuartz: https://www.xquartz.org/\n\n")

    @classmethod
    def pip(cls) -> None:
        """ Pip Tips """
        logger.info("1. Install PIP requirements\n"
                    "You may want to execute `chcp 65001` in cmd line\n"
                    "to fix Unicode issues on Windows when installing dependencies")


if __name__ == "__main__":
    logfile = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap_setup.log")
    log_setup("INFO", logfile, "setup")
    logger.debug("Setup called with args: %s", sys.argv)
    ENV = Environment()
    Checks(ENV)
    ENV.set_config()
    if _InstallState.failed:
        sys.exit(1)
    Install(ENV)


__all__ = get_module_objects(__name__)
