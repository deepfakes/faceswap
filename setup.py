#!/usr/bin/env python3
""" Install packages for faceswap.py """
# pylint:disable=too-many-lines

import logging
import ctypes
import json
import locale
import platform
import operator
import os
import re
import sys
import typing as T
from shutil import which
from subprocess import list2cmdline, PIPE, Popen, run, STDOUT

from pkg_resources import parse_requirements

from lib.logger import log_setup

logger = logging.getLogger(__name__)
backend_type: T.TypeAlias = T.Literal['nvidia', 'apple_silicon', 'directml', 'cpu', 'rocm', "all"]

_INSTALL_FAILED = False
# Packages that are explicitly required for setup.py
_INSTALLER_REQUIREMENTS: list[tuple[str, str]] = [("pexpect>=4.8.0", "!Windows"),
                                                  ("pywinpty==2.0.2", "Windows")]
# Conda packages that are required for a specific backend
# TODO zlib-wapi is required on some Windows installs where cuDNN complains:
# Could not locate zlibwapi.dll. Please make sure it is in your library path!
# This only seems to occur on Anaconda cuDNN not conda-forge
_BACKEND_SPECIFIC_CONDA: dict[backend_type, list[str]] = {
    "nvidia": ["cudatoolkit", "cudnn", "zlib-wapi"],
    "apple_silicon": ["libblas"]}
# Packages that should only be installed through pip
_FORCE_PIP: dict[backend_type, list[str]] = {
    "nvidia": ["tensorflow"],
    "all": [
        "tensorflow-cpu",  # conda-forge leads to flatbuffer errors because of mixed sources
        "imageio-ffmpeg"]}  # 17/11/23 Conda forge uses incorrect ffmpeg, so fallback to pip
# Revisions of tensorflow GPU and cuda/cudnn requirements. These relate specifically to the
# Tensorflow builds available from pypi
_TENSORFLOW_REQUIREMENTS = {">=2.10.0,<2.11.0": [">=11.2,<11.3", ">=8.1,<8.2"]}
# ROCm min/max version requirements for Tensorflow
_TENSORFLOW_ROCM_REQUIREMENTS = {">=2.10.0,<2.11.0": ((5, 2, 0), (5, 4, 0))}
# TODO tensorflow-metal versioning

# Mapping of Python packages to their conda names if different from pip or in non-default channel
_CONDA_MAPPING: dict[str, tuple[str, str]] = {
    "cudatoolkit": ("cudatoolkit", "conda-forge"),
    "cudnn": ("cudnn", "conda-forge"),
    "fastcluster": ("fastcluster", "conda-forge"),
    "ffmpy": ("ffmpy", "conda-forge"),
    # "imageio-ffmpeg": ("imageio-ffmpeg", "conda-forge"),
    "nvidia-ml-py": ("nvidia-ml-py", "conda-forge"),
    "tensorflow-deps": ("tensorflow-deps", "apple"),
    "libblas": ("libblas", "conda-forge"),
    "zlib-wapi": ("zlib-wapi", "conda-forge"),
    "xorg-libxft": ("xorg-libxft", "conda-forge")}

# Force output to utf-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type:ignore[attr-defined]


class Environment():
    """ The current install environment

    Parameters
    ----------
    updater: bool, Optional
        ``True`` if the script is being called by Faceswap's internal updater. ``False`` if full
        setup is running. Default: ``False``
    """

    _backends = (("nvidia", "apple_silicon", "directml", "rocm", "cpu"))

    def __init__(self, updater: bool = False) -> None:
        self.updater = updater
        # Flag that setup is being run by installer so steps can be skipped
        self.is_installer: bool = False
        self.backend: backend_type | None = None
        self.enable_docker: bool = False
        self.cuda_cudnn = ["", ""]
        self.rocm_version: tuple[int, ...] = (0, 0, 0)

        self._process_arguments()
        self._check_permission()
        self._check_system()
        self._check_python()
        self._output_runtime_info()
        self._check_pip()
        self._upgrade_pip()
        self._set_env_vars()

    @property
    def encoding(self) -> str:
        """ Get system encoding """
        return locale.getpreferredencoding()

    @property
    def os_version(self) -> tuple[str, str]:
        """ Get OS Version """
        return platform.system(), platform.release()

    @property
    def py_version(self) -> tuple[str, str]:
        """ Get Python Version """
        return platform.python_version(), platform.architecture()[0]

    @property
    def is_conda(self) -> bool:
        """ Check whether using Conda """
        return ("conda" in sys.version.lower() or
                os.path.exists(os.path.join(sys.prefix, 'conda-meta')))

    @property
    def is_admin(self) -> bool:
        """ Check whether user is admin """
        try:
            retval = os.getuid() == 0  # type: ignore
        except AttributeError:
            retval = ctypes.windll.shell32.IsUserAnAdmin() != 0  # type: ignore
        return retval

    @property
    def cuda_version(self) -> str:
        """ str: The detected globally installed Cuda Version """
        return self.cuda_cudnn[0]

    @property
    def cudnn_version(self) -> str:
        """ str: The detected globally installed cuDNN Version """
        return self.cuda_cudnn[1]

    @property
    def is_virtualenv(self) -> bool:
        """ Check whether this is a virtual environment """
        if not self.is_conda:
            retval = (hasattr(sys, "real_prefix") or
                      (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
        else:
            prefix = os.path.dirname(sys.prefix)
            retval = os.path.basename(prefix) == "envs"
        return retval

    def _process_arguments(self) -> None:
        """ Process any cli arguments and dummy in cli arguments if calling from updater. """
        args = [arg for arg in sys.argv]  # pylint:disable=unnecessary-comprehension
        if self.updater:
            from lib.utils import get_backend  # pylint:disable=import-outside-toplevel
            args.append(f"--{get_backend()}")

        logger.debug(args)
        for arg in args:
            if arg == "--installer":
                self.is_installer = True
            if not self.backend and (arg.startswith("--") and
                                     arg.replace("--", "") in self._backends):
                self.backend = arg.replace("--", "").lower()  # type:ignore

    def _check_permission(self) -> None:
        """ Check for Admin permissions """
        if self.updater:
            return
        if self.is_admin:
            logger.info("Running as Root/Admin")
        else:
            logger.info("Running without root/admin privileges")

    def _check_system(self) -> None:
        """ Check the system """
        if not self.updater:
            logger.info("The tool provides tips for installation and installs required python "
                        "packages")
        logger.info("Setup in %s %s", self.os_version[0], self.os_version[1])
        if not self.updater and not self.os_version[0] in ["Windows", "Linux", "Darwin"]:
            logger.error("Your system %s is not supported!", self.os_version[0])
            sys.exit(1)
        if self.os_version[0].lower() == "darwin" and platform.machine() == "arm64":
            self.backend = "apple_silicon"

            if not self.updater and not self.is_conda:
                logger.error("Setting up Faceswap for Apple Silicon outside of a Conda "
                             "environment is unsupported")
                sys.exit(1)

    def _check_python(self) -> None:
        """ Check python and virtual environment status """
        logger.info("Installed Python: %s %s", self.py_version[0], self.py_version[1])

        if self.updater:
            return

        if not ((3, 10) <= sys.version_info < (3, 11) and self.py_version[1] == "64bit"):
            logger.error("Please run this script with Python version 3.10 64bit and try "
                         "again.")
            sys.exit(1)

    def _output_runtime_info(self) -> None:
        """ Output run time info """
        if self.is_conda:
            logger.info("Running in Conda")
        if self.is_virtualenv:
            logger.info("Running in a Virtual Environment")
        logger.info("Encoding: %s", self.encoding)

    def _check_pip(self) -> None:
        """ Check installed pip version """
        if self.updater:
            return
        try:
            import pip  # noqa pylint:disable=unused-import,import-outside-toplevel
        except ImportError:
            logger.error("Import pip failed. Please Install python3-pip and try again")
            sys.exit(1)

    def _upgrade_pip(self) -> None:
        """ Upgrade pip to latest version """
        if not self.is_conda:
            # Don't do this with Conda, as we must use Conda version of pip
            logger.info("Upgrading pip...")
            pipexe = [sys.executable, "-m", "pip"]
            pipexe.extend(["install", "--no-cache-dir", "-qq", "--upgrade"])
            if not self.is_admin and not self.is_virtualenv:
                pipexe.append("--user")
            pipexe.append("pip")
            run(pipexe, check=True)
        import pip  # pylint:disable=import-outside-toplevel
        pip_version = pip.__version__
        logger.info("Installed pip: %s", pip_version)

    def set_config(self) -> None:
        """ Set the backend in the faceswap config file """
        config = {"backend": self.backend}
        pypath = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(pypath, "config", ".faceswap")
        with open(config_file, "w", encoding="utf8") as cnf:
            json.dump(config, cnf)
        logger.info("Faceswap config written to: %s", config_file)

    def _set_env_vars(self) -> None:
        """ There are some foibles under Conda which need to be worked around in different
        situations.

        Linux:
        Update the LD_LIBRARY_PATH environment variable when activating a conda environment
        and revert it when deactivating.

        Notes
        -----
        From Tensorflow 2.7, installing Cuda Toolkit from conda-forge and tensorflow from pip
        causes tensorflow to not be able to locate shared libs and hence not use the GPU.
        We update the environment variable for all instances using Conda as it shouldn't hurt
        anything and may help avoid conflicts with globally installed Cuda
        """
        if not self.is_conda:
            return

        linux_update = self.os_version[0].lower() == "linux" and self.backend == "nvidia"

        if not linux_update:
            return

        conda_prefix = os.environ["CONDA_PREFIX"]
        activate_folder = os.path.join(conda_prefix, "etc", "conda", "activate.d")
        deactivate_folder = os.path.join(conda_prefix, "etc", "conda", "deactivate.d")
        os.makedirs(activate_folder, exist_ok=True)
        os.makedirs(deactivate_folder, exist_ok=True)

        activate_script = os.path.join(conda_prefix, activate_folder, "env_vars.sh")
        deactivate_script = os.path.join(conda_prefix, deactivate_folder, "env_vars.sh")

        if os.path.isfile(activate_script):
            # Only create file if it does not already exist. There may be instances where people
            # have created their own scripts, but these should be few and far between and those
            # people should already know what they are doing.
            return

        conda_libs = os.path.join(conda_prefix, "lib")
        activate = ["#!/bin/sh\n\n",
                    "export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}\n",
                    f"export LD_LIBRARY_PATH='{conda_libs}':${{LD_LIBRARY_PATH}}\n"]
        deactivate = ["#!/bin/sh\n\n",
                      "export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}\n",
                      "unset OLD_LD_LIBRARY_PATH\n"]
        logger.info("Cuda search path set to '%s'", conda_libs)

        with open(activate_script, "w", encoding="utf8") as afile:
            afile.writelines(activate)
        with open(deactivate_script, "w", encoding="utf8") as afile:
            afile.writelines(deactivate)


class Packages():
    """ Holds information about installed and required packages.
    Handles updating dependencies based on running platform/backend

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    """
    def __init__(self, environment: Environment) -> None:
        self._env = environment

        # Default TK has bad fonts under Linux. There is a better build in Conda-Forge, so set
        # channel accordingly
        tk_channel = "conda-forge" if self._env.os_version[0].lower() == "linux" else "default"
        self._conda_required_packages: list[tuple[list[str] | str, str]] = [("tk", tk_channel),
                                                                            ("git", "default")]
        self._update_backend_specific_conda()
        self._installed_packages = self._get_installed_packages()
        self._conda_installed_packages = self._get_installed_conda_packages()
        self._required_packages: list[tuple[str, list[tuple[str, str]]]] = []
        self._missing_packages: list[tuple[str, list[tuple[str, str]]]] = []
        self._conda_missing_packages: list[tuple[list[str] | str, str]] = []

    @property
    def prerequisites(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """ list: Any required packages that the installer needs prior to installing the faceswap
        environment on the specific platform that are not already installed """
        all_installed = self._all_installed_packages
        candidates = self._format_requirements(
            [pkg for pkg, plat in _INSTALLER_REQUIREMENTS
             if self._env.os_version[0] == plat or (plat[0] == "!" and
                                                    self._env.os_version[0] != plat[1:])])
        retval = [(pkg, spec) for pkg, spec in candidates
                  if pkg not in all_installed or (
                    pkg in all_installed and
                    not self._validate_spec(spec, all_installed.get(pkg, ""))
                  )]
        return retval

    @property
    def packages_need_install(self) -> bool:
        """bool: ``True`` if there are packages available that need to be installed """
        return bool(self._missing_packages or self._conda_missing_packages)

    @property
    def to_install(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """ list: The required packages that need to be installed """
        return self._missing_packages

    @property
    def to_install_conda(self) -> list[tuple[list[str] | str, str]]:
        """ list: The required conda packages that need to be installed """
        return self._conda_missing_packages

    @property
    def _all_installed_packages(self) -> dict[str, str]:
        """ dict[str, str]: The package names and version string for all installed packages across
        pip and conda """
        return {**self._installed_packages, **self._conda_installed_packages}

    def _update_backend_specific_conda(self) -> None:
        """ Add backend specific packages to Conda required packages """
        assert self._env.backend is not None
        to_add = _BACKEND_SPECIFIC_CONDA.get(self._env.backend)
        if not to_add:
            logger.debug("No backend packages to add for '%s'. All optional packages: %s",
                         self._env.backend, _BACKEND_SPECIFIC_CONDA)
            return

        combined_cuda = []
        for pkg in to_add:
            pkg, channel = _CONDA_MAPPING.get(pkg, (pkg, ""))
            if pkg == "zlib-wapi" and self._env.os_version[0].lower() != "windows":
                # TODO move this front and center
                continue
            if pkg in ("cudatoolkit", "cudnn"):  # TODO Handle multiple cuda/cudnn requirements
                idx = 0 if pkg == "cudatoolkit" else 1
                pkg = f"{pkg}{list(_TENSORFLOW_REQUIREMENTS.values())[0][idx]}"

                combined_cuda.append(pkg)
                continue

            self._conda_required_packages.append((pkg, channel))
            logger.info("Adding conda required package '%s' for backend '%s')",
                        pkg, self._env.backend)

        if combined_cuda:
            self._conda_required_packages.append((combined_cuda, channel))
            logger.info("Adding conda required package '%s' for backend '%s')",
                        combined_cuda, self._env.backend)

    @classmethod
    def _format_requirements(cls, packages: list[str]
                             ) -> list[tuple[str, list[tuple[str, str]]]]:
        """ Parse a list of requirements.txt formatted package strings to a list of pkgresource
        formatted requirements """
        return [(package.unsafe_name, package.specs)
                for package in parse_requirements(packages)
                if package.marker is None or package.marker.evaluate()]

    @classmethod
    def _validate_spec(cls,
                       required: list[tuple[str, str]],
                       existing: str) -> bool:
        """ Validate whether the required specification for a package is met by the installed
        version.

        required: list[tuple[str, str]]
            The required package version spec to check
        existing: str
            The version of the installed package

        Returns
        -------
        bool
            ``True`` if the required specification is met by the existing specification
        """
        ops = {"==": operator.eq, ">=": operator.ge, "<=": operator.le,
               ">": operator.gt, "<": operator.lt}
        if not required:
            return True

        return all(ops[spec[0]]([int(s) for s in existing.split(".")],
                                [int(s) for s in spec[1].split(".")])
                   for spec in required)

    def _get_installed_packages(self) -> dict[str, str]:
        """ Get currently installed packages and add to :attr:`_installed_packages`

        Returns
        -------
        dict[str, str]
            The installed package name and version string
        """
        installed_packages = {}
        with Popen(f"\"{sys.executable}\" -m pip freeze --local", shell=True, stdout=PIPE) as chk:
            installed = chk.communicate()[0].decode(self._env.encoding,
                                                    errors="ignore").splitlines()

        for pkg in installed:
            if "==" not in pkg:
                continue
            item = pkg.split("==")
            installed_packages[item[0]] = item[1]
        logger.debug(installed_packages)
        return installed_packages

    def _get_installed_conda_packages(self) -> dict[str, str]:
        """ Get currently installed conda packages

        Returns
        -------
        dict[str, str]
            The installed package name and version string
        """
        if not self._env.is_conda:
            return {}
        chk = os.popen("conda list").read()
        installed = [re.sub(" +", " ", line.strip())
                     for line in chk.splitlines() if not line.startswith("#")]
        retval = {}
        for pkg in installed:
            item = pkg.split(" ")
            retval[item[0]] = item[1]
        logger.debug(retval)
        return retval

    def get_required_packages(self) -> None:
        """ Load the requirements from the backend specific requirements list """
        req_files = ["_requirements_base.txt", f"requirements_{self._env.backend}.txt"]
        pypath = os.path.dirname(os.path.realpath(__file__))
        requirements = []
        for req_file in req_files:
            requirements_file = os.path.join(pypath, "requirements", req_file)
            with open(requirements_file, encoding="utf8") as req:
                for package in req.readlines():
                    package = package.strip()
                    if package and (not package.startswith(("#", "-r"))):
                        requirements.append(package)

        self._required_packages = self._format_requirements(requirements)
        logger.debug(self._required_packages)

    def _update_tf_dep_nvidia(self) -> None:
        """ Update the Tensorflow dependency for global Cuda installs """
        if self._env.is_conda:  # Conda handles Cuda and cuDNN so nothing to do here
            return
        tf_ver = None
        cuda_inst = self._env.cuda_version
        cudnn_inst = self._env.cudnn_version
        if len(cudnn_inst) == 1:  # Sometimes only major version is reported
            cudnn_inst = f"{cudnn_inst}.0"
        for key, val in _TENSORFLOW_REQUIREMENTS.items():
            cuda_req = next(parse_requirements(f"cuda{val[0]}")).specs
            cudnn_req = next(parse_requirements(f"cudnn{val[1]}")).specs
            if (self._validate_spec(cuda_req, cuda_inst)
                    and self._validate_spec(cudnn_req, cudnn_inst)):
                tf_ver = key
                break

        if tf_ver:
            # Remove the version of tensorflow in requirements file and add the correct version
            # that corresponds to the installed Cuda/cuDNN versions
            self._required_packages = [pkg for pkg in self._required_packages
                                       if pkg[0] != "tensorflow"]
            tf_ver = f"tensorflow{tf_ver}"
            self._required_packages.append(("tensorflow", next(parse_requirements(tf_ver)).specs))
            return

        logger.warning(
            "The minimum Tensorflow requirement is 2.10 \n"
            "Tensorflow currently has no official prebuild for your CUDA, cuDNN combination.\n"
            "Either install a combination that Tensorflow supports or build and install your own "
            "tensorflow.\r\n"
            "CUDA Version: %s\r\n"
            "cuDNN Version: %s\r\n"
            "Help:\n"
            "Building Tensorflow: https://www.tensorflow.org/install/install_sources\r\n"
            "Tensorflow supported versions: "
            "https://www.tensorflow.org/install/source#tested_build_configurations",
            self._env.cuda_version, self._env.cudnn_version)

        custom_tf = input("Location of custom tensorflow wheel (leave blank to manually "
                          "install): ")
        if not custom_tf:
            return

        custom_tf = os.path.realpath(os.path.expanduser(custom_tf))
        global _INSTALL_FAILED  # pylint:disable=global-statement
        if not os.path.isfile(custom_tf):
            logger.error("%s not found", custom_tf)
            _INSTALL_FAILED = True
        elif os.path.splitext(custom_tf)[1] != ".whl":
            logger.error("%s is not a valid pip wheel", custom_tf)
            _INSTALL_FAILED = True
        elif custom_tf:
            self._required_packages.append((custom_tf, [(custom_tf, "")]))

    def _update_tf_dep_rocm(self) -> None:
        """ Update the Tensorflow dependency for global ROCm installs """
        if not any(self._env.rocm_version):  # ROCm was not found and the install will be aborted
            return

        global _INSTALL_FAILED  # pylint:disable=global-statement
        candidates = [key for key, val in _TENSORFLOW_ROCM_REQUIREMENTS.items()
                      if val[0] <= self._env.rocm_version <= val[1]]

        if not candidates:
            _INSTALL_FAILED = True
            logger.error("No matching Tensorflow candidates found for ROCm %s in %s",
                         ".".join(str(v) for v in self._env.rocm_version),
                         _TENSORFLOW_ROCM_REQUIREMENTS)
            return

        # set tf_ver to the minimum and maximum compatible range
        tf_ver = f"{candidates[0].split(',')[0]},{candidates[-1].split(',')[-1]}"
        # Remove the version of tensorflow-rocm in requirements file and add the correct version
        # that corresponds to the installed ROCm version
        self._required_packages = [pkg for pkg in self._required_packages
                                   if not pkg[0].startswith("tensorflow-rocm")]
        tf_ver = f"tensorflow-rocm{tf_ver}"
        self._required_packages.append(("tensorflow-rocm",
                                        next(parse_requirements(tf_ver)).specs))

    def update_tf_dep(self) -> None:
        """ Update Tensorflow Dependency.

        Selects a compatible version of Tensorflow for a globally installed GPU library
        """
        if self._env.backend == "nvidia":
            self._update_tf_dep_nvidia()
        if self._env.backend == "rocm":
            self._update_tf_dep_rocm()

    def _check_conda_missing_dependencies(self) -> None:
        """ Check for conda missing dependencies and add to :attr:`_conda_missing_packages` """
        if not self._env.is_conda:
            return
        for pkg in self._conda_required_packages:
            reqs = next(parse_requirements(pkg[0]))  # TODO Handle '=' vs '==' for conda
            key = reqs.unsafe_name
            specs = reqs.specs

            if pkg[0] == "tk" and self._env.os_version[0].lower() == "linux":
                # Default tk has bad fonts under Linux. We pull in an explicit build from
                # Conda-Forge that is compiled with better fonts.
                # Ref: https://github.com/ContinuumIO/anaconda-issues/issues/6833
                newpkg = (f"{pkg[0]}=*=xft_*", pkg[1])  # Swap out package for explicit XFT version
                self._conda_missing_packages.append(newpkg)
                # We also need to bring in xorg-libxft incase libXft does not exist on host system
                self._conda_missing_packages.append(_CONDA_MAPPING["xorg-libxft"])
                continue

            if key not in self._conda_installed_packages:
                self._conda_missing_packages.append(pkg)
                continue

            if not self._validate_spec(specs, self._conda_installed_packages[key]):
                self._conda_missing_packages.append(pkg)
        logger.debug(self._conda_missing_packages)

    def check_missing_dependencies(self) -> None:
        """ Check for missing dependencies and add to :attr:`_missing_packages` """
        for key, specs in self._required_packages:

            if self._env.is_conda:  # Get Conda alias for Key
                key = _CONDA_MAPPING.get(key, (key, None))[0]

            if key not in self._all_installed_packages:
                # Add not installed packages to missing packages list
                self._missing_packages.append((key, specs))
                continue

            if not self._validate_spec(specs, self._all_installed_packages.get(key, "")):
                self._missing_packages.append((key, specs))

        logger.debug(self._missing_packages)
        self._check_conda_missing_dependencies()


class Checks():  # pylint:disable=too-few-public-methods
    """ Pre-installation checks

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    """
    def __init__(self, environment: Environment) -> None:
        self._env:  Environment = environment
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
        if self._env.os_version[0] == "Windows":
            self._tips.pip()

    def _rocm_ask_enable(self) -> None:
        """ Set backend to 'rocm' if OS is Linux and ROCm support required """
        if self._env.os_version[0] != "Linux":
            return
        logger.info("ROCm support:\r\nIf you are using an AMD GPU, then select 'yes'."
                    "\r\nCPU/non-AMD GPU users should answer 'no'.\r\n")
        i = input("Enable ROCm Support? [y/N] ")
        if i in ("Y", "y"):
            logger.info("ROCm Support Enabled")
            self._env.backend = "rocm"

    def _directml_ask_enable(self) -> None:
        """ Set backend to 'directml' if OS is Windows and DirectML support required """
        if self._env.os_version[0] != "Windows":
            return
        logger.info("DirectML support:\r\nIf you are using an AMD or Intel GPU, then select 'yes'."
                    "\r\nNvidia users should answer 'no'.")
        i = input("Enable DirectML Support? [y/N] ")
        if i in ("Y", "y"):
            logger.info("DirectML Support Enabled")
            self._env.backend = "directml"

    def _user_input(self) -> None:
        """ Get user input for AMD/DirectML/ROCm/Cuda/Docker """
        self._directml_ask_enable()
        self._rocm_ask_enable()
        if not self._env.backend:
            self._docker_ask_enable()
            self._cuda_ask_enable()
        if self._env.os_version[0] != "Linux" and (self._env.enable_docker
                                                   and self._env.backend == "nvidia"):
            self._docker_confirm()
        if self._env.enable_docker:
            self._docker_tips()
            self._env.set_config()
            sys.exit(0)

    def _docker_ask_enable(self) -> None:
        """ Enable or disable Docker """
        i = input("Enable  Docker? [y/N] ")
        if i in ("Y", "y"):
            logger.info("Docker Enabled")
            self._env.enable_docker = True
        else:
            logger.info("Docker Disabled")
            self._env.enable_docker = False

    def _docker_confirm(self) -> None:
        """ Warn if nvidia-docker on non-Linux system """
        logger.warning("Nvidia-Docker is only supported on Linux.\r\n"
                       "Only CPU is supported in Docker for your system")
        self._docker_ask_enable()
        if self._env.enable_docker:
            logger.warning("CUDA Disabled")
            self._env.backend = "cpu"

    def _docker_tips(self) -> None:
        """ Provide tips for Docker use """
        if self._env.backend != "nvidia":
            self._tips.docker_no_cuda()
        else:
            self._tips.docker_cuda()

    def _cuda_ask_enable(self) -> None:
        """ Enable or disable CUDA """
        i = input("Enable  CUDA? [Y/n] ")
        if i in ("", "Y", "y"):
            logger.info("CUDA Enabled")
            self._env.backend = "nvidia"

    def _check_cuda(self) -> None:
        """ Check for Cuda and cuDNN Locations. """
        if self._env.backend != "nvidia":
            logger.debug("Skipping Cuda checks as not enabled")
            return

        if self._env.is_conda:
            logger.info("Skipping Cuda/cuDNN checks for Conda install")
            return

        if self._env.os_version[0] in ("Linux", "Windows"):
            global _INSTALL_FAILED  # pylint:disable=global-statement
            check = CudaCheck()
            if check.cuda_version:
                self._env.cuda_cudnn[0] = check.cuda_version
                logger.info("CUDA version: %s", self._env.cuda_version)
            else:
                logger.error("CUDA not found. Install and try again.\n"
                             "Recommended version:      CUDA 10.1     cuDNN 7.6\n"
                             "CUDA: https://developer.nvidia.com/cuda-downloads\n"
                             "cuDNN: https://developer.nvidia.com/rdp/cudnn-download")
                _INSTALL_FAILED = True
                return

            if check.cudnn_version:
                self._env.cuda_cudnn[1] = ".".join(check.cudnn_version.split(".")[:2])
                logger.info("cuDNN version: %s", self._env.cudnn_version)
            else:
                logger.error("cuDNN not found. See "
                             "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#"
                             "cudnn for instructions")
                _INSTALL_FAILED = True
            return

        # If we get here we're on MacOS
        self._tips.macos()
        logger.warning("Cannot find CUDA on macOS")
        self._env.cuda_cudnn[0] = input("Manually specify CUDA version: ")

    def _check_rocm(self) -> None:
        """ Check for ROCm version """
        if self._env.backend != "rocm" or self._env.os_version[0] != "Linux":
            logger.info("Skipping ROCm checks as not enabled")
            return

        global _INSTALL_FAILED  # pylint:disable=global-statement
        check = ROCmCheck()

        str_min = ".".join(str(v) for v in check.version_min)
        str_max = ".".join(str(v) for v in check.version_max)

        if check.is_valid:
            self._env.rocm_version = check.rocm_version
            logger.info("ROCm version: %s", ".".join(str(v) for v in self._env.rocm_version))
        else:
            if check.rocm_version:
                msg = f"Incompatible ROCm version: {'.'.join(str(v) for v in check.rocm_version)}"
            else:
                msg = "ROCm not found"
            logger.error("%s.\n"
                         "A compatible version of ROCm must be installed to proceed.\n"
                         "ROCm versions between %s and %s are supported.\n"
                         "ROCm install guide: https://docs.amd.com/bundle/ROCm_Installation_Guide"
                         "v5.0/page/Overview_of_ROCm_Installation_Methods.html",
                         msg,
                         str_min,
                         str_max)
            _INSTALL_FAILED = True


def _check_ld_config(lib: str) -> str:
    """ Locate a library in ldconfig

    Parameters
    ----------
    lib: str The library to locate

    Returns
    -------
    str
        The library from ldconfig, or empty string if not found
    """
    retval = ""
    ldconfig = which("ldconfig")
    if not ldconfig:
        return retval

    retval = next((line.decode("utf-8", errors="replace").strip()
                  for line in run([ldconfig, "-p"],
                                  capture_output=True,
                                  check=False).stdout.splitlines()
                  if lib.encode("utf-8") in line), "")

    if retval or (not retval and not os.environ.get("LD_LIBRARY_PATH")):
        return retval

    for path in os.environ["LD_LIBRARY_PATH"].split(":"):
        if not path or not os.path.exists(path):
            continue

        retval = next((fname.strip() for fname in reversed(os.listdir(path))
                       if lib in fname), "")
        if retval:
            break

    return retval


class ROCmCheck():  # pylint:disable=too-few-public-methods
    """ Find the location of system installed ROCm on Linux """
    def __init__(self) -> None:
        self.version_min = min(v[0] for v in _TENSORFLOW_ROCM_REQUIREMENTS.values())
        self.version_max = max(v[1] for v in _TENSORFLOW_ROCM_REQUIREMENTS.values())
        self.rocm_version: tuple[int, ...] = (0, 0, 0)
        if platform.system() == "Linux":
            self._rocm_check()

    @property
    def is_valid(self):
        """ bool: `True` if ROCm has been detected and is between the minimum and maximum
        compatible versions otherwise ``False`` """
        return self.version_min <= self.rocm_version <= self.version_max

    def _rocm_check(self) -> None:
        """ Attempt to locate the installed ROCm version from the dynamic link loader. If not found
        with ldconfig then attempt to find it in LD_LIBRARY_PATH. If found, set the
        :attr:`rocm_version` to the discovered version
        """
        chk = _check_ld_config("librocm-core.so.")
        if not chk:
            return

        rocm_vers = chk.strip()
        version = re.search(r"rocm\-(\d+\.\d+\.\d+)", rocm_vers)
        if version is None:
            return
        try:
            self.rocm_version = tuple(int(v) for v in version.groups()[0].split("."))
        except ValueError:
            return


class CudaCheck():  # pylint:disable=too-few-public-methods
    """ Find the location of system installed Cuda and cuDNN on Windows and Linux. """

    def __init__(self) -> None:
        self.cuda_path: str | None = None
        self.cuda_version: str | None = None
        self.cudnn_version: str | None = None

        self._os: str = platform.system().lower()
        self._cuda_keys: list[str] = [key
                                      for key in os.environ
                                      if key.lower().startswith("cuda_path_v")]
        self._cudnn_header_files: list[str] = ["cudnn_version.h", "cudnn.h"]
        logger.debug("cuda keys: %s, cudnn header files: %s",
                     self._cuda_keys, self._cudnn_header_files)
        if self._os in ("windows", "linux"):
            self._cuda_check()
            self._cudnn_check()

    def _cuda_check(self) -> None:
        """ Obtain the location and version of Cuda and populate :attr:`cuda_version` and
        :attr:`cuda_path`

        Initially just calls `nvcc -V` to get the installed version of Cuda currently in use.
        If this fails, drills down to more OS specific checking methods.
        """
        with Popen("nvcc -V", shell=True, stdout=PIPE, stderr=PIPE) as chk:
            stdout, stderr = chk.communicate()
        if not stderr:
            version = re.search(r".*release (?P<cuda>\d+\.\d+)",
                                stdout.decode(locale.getpreferredencoding(), errors="ignore"))
            if version is not None:
                self.cuda_version = version.groupdict().get("cuda", None)
            path = which("nvcc")
            if path:
                path = path.split("\n")[0]  # Split multiple entries and take first found
                while True:  # Get Cuda root folder
                    path, split = os.path.split(path)
                    if split == "bin":
                        break
                self.cuda_path = path
            return

        # Failed to load nvcc, manual check
        getattr(self, f"_cuda_check_{self._os}")()
        logger.debug("Cuda Version: %s, Cuda Path: %s", self.cuda_version, self.cuda_path)

    def _cuda_check_linux(self) -> None:
        """ For Linux check the dynamic link loader for libcudart. If not found with ldconfig then
        attempt to find it in LD_LIBRARY_PATH. """
        chk = _check_ld_config("libcudart.so.")
        if not chk:  # Cuda not found
            return

        cudavers = chk.strip().replace("libcudart.so.", "")
        self.cuda_version = cudavers[:cudavers.find(" ")] if " " in cudavers else cudavers
        cuda_path = chk[chk.find("=>") + 3:chk.find("targets") - 1]
        if os.path.exists(cuda_path):
            self.cuda_path = cuda_path

    def _cuda_check_windows(self) -> None:
        """ Check Windows CUDA Version and path from Environment Variables"""
        if not self._cuda_keys:  # Cuda environment variable not found
            return
        self.cuda_version = self._cuda_keys[0].lower().replace("cuda_path_v", "").replace("_", ".")
        self.cuda_path = os.environ[self._cuda_keys[0][0]]

    def _cudnn_check_files(self) -> bool:
        """ Check header files for cuDNN version """
        cudnn_checkfiles = getattr(self, f"_get_checkfiles_{self._os}")()
        cudnn_checkfile = next((hdr for hdr in cudnn_checkfiles if os.path.isfile(hdr)), None)
        logger.debug("cudnn checkfiles: %s", cudnn_checkfile)
        if not cudnn_checkfile:
            return False

        found = 0
        with open(cudnn_checkfile, "r", encoding="utf8") as ofile:
            for line in ofile:
                if line.lower().startswith("#define cudnn_major"):
                    major = line[line.rfind(" ") + 1:].strip()
                    found += 1
                elif line.lower().startswith("#define cudnn_minor"):
                    minor = line[line.rfind(" ") + 1:].strip()
                    found += 1
                elif line.lower().startswith("#define cudnn_patchlevel"):
                    patchlevel = line[line.rfind(" ") + 1:].strip()
                    found += 1
                if found == 3:
                    break
        if found != 3:  # Full version not determined
            return False

        self.cudnn_version = ".".join([str(major), str(minor), str(patchlevel)])
        logger.debug("cudnn version: %s", self.cudnn_version)
        return True

    def _cudnn_check(self) -> None:
        """ Check Linux or Windows cuDNN Version from cudnn.h and add to :attr:`cudnn_version`. """
        if self._cudnn_check_files():
            return
        if self._os == "windows":
            return

        chk = _check_ld_config("libcudnn.so.")
        if not chk:
            return
        cudnnvers = chk.strip().replace("libcudnn.so.", "").split()[0]
        if not cudnnvers:
            return

        self.cudnn_version = cudnnvers
        logger.debug("cudnn version: %s", self.cudnn_version)

    def _get_checkfiles_linux(self) -> list[str]:
        """ Return the the files to check for cuDNN locations for Linux by querying
        the dynamic link loader.

        Returns
        -------
        list
            List of header file locations to scan for cuDNN versions
        """
        chk = _check_ld_config("libcudnn.so.")
        chk = chk.strip().replace("libcudnn.so.", "")
        if not chk:
            return []

        cudnn_vers = chk[0]
        header_files = [f"cudnn_v{cudnn_vers}.h"] + self._cudnn_header_files

        cudnn_path = os.path.realpath(chk[chk.find("=>") + 3:chk.find("libcudnn") - 1])
        cudnn_path = cudnn_path.replace("lib", "include")
        cudnn_checkfiles = [os.path.join(cudnn_path, header) for header in header_files]
        return cudnn_checkfiles

    def _get_checkfiles_windows(self) -> list[str]:
        """ Return the check-file locations for Windows. Just looks inside the include folder of
        the discovered :attr:`cuda_path`

        Returns
        -------
        list
            List of header file locations to scan for cuDNN versions
        """
        # TODO A more reliable way of getting the windows location
        if not self.cuda_path or not os.path.exists(self.cuda_path):
            return []
        scandir = os.path.join(self.cuda_path, "include")
        cudnn_checkfiles = [os.path.join(scandir, header) for header in self._cudnn_header_files]
        return cudnn_checkfiles


class Install():  # pylint:disable=too-few-public-methods
    """ Handles installation of Faceswap requirements

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    is_gui: bool, Optional
        ``True`` if the caller is the Faceswap GUI. Used to prevent output of progress bars
        which get scrambled in the GUI
     """
    def __init__(self, environment: Environment, is_gui: bool = False) -> None:
        self._env = environment
        self._packages = Packages(environment)
        self._is_gui = is_gui

        if self._env.os_version[0] == "Windows":
            self._installer: type[Installer] = WinPTYInstaller
        else:
            self._installer = PexpectInstaller

        if not self._env.is_installer and not self._env.updater:
            self._ask_continue()

        self._packages.get_required_packages()
        self._packages.update_tf_dep()
        self._packages.check_missing_dependencies()

        if self._env.updater and not self._packages.packages_need_install:
            logger.info("All Dependencies are up to date")
            return

        logger.info("Installing Required Python Packages. This may take some time...")
        self._install_setup_packages()
        self._install_missing_dep()
        if self._env.updater:
            return
        if not _INSTALL_FAILED:
            logger.info("All python3 dependencies are met.\r\nYou are good to go.\r\n\r\n"
                        "Enter:  'python faceswap.py -h' to see the options\r\n"
                        "        'python faceswap.py gui' to launch the GUI")
        else:
            logger.error("Some packages failed to install. This may be a temporary error which "
                         "might be fixed by re-running this script. Otherwise please install "
                         "these packages manually.")
            sys.exit(1)

    def _ask_continue(self) -> None:
        """ Ask Continue with Install """
        text = "Please ensure your System Dependencies are met"
        if self._env.backend == "rocm":
            text += ("\r\nROCm users: Please ensure that your AMD GPU is supported by the "
                     "installed ROCm version before proceeding.")
        text += "\r\nContinue? [y/N] "
        inp = input(text)
        if inp in ("", "N", "n"):
            logger.error("Please install system dependencies to continue")
            sys.exit(1)

    @classmethod
    def _format_package(cls, package: str, version: list[tuple[str, str]]) -> str:
        """ Format a parsed requirement package and version string to a format that can be used by
        the installer.

        Parameters
        ----------
        package: str
            The package name
        version: list
            The parsed requirement version strings

        Returns
        -------
        str
            The formatted full package and version string
        """
        return f"{package}{','.join(''.join(spec) for spec in version)}"

    def _install_setup_packages(self) -> None:
        """ Install any packages that are required for the setup.py installer to work. This
        includes the pexpect package if it is not already installed.

        Subprocess is used as we do not currently have pexpect
        """
        for pkg in self._packages.prerequisites:
            pkg_str = self._format_package(*pkg)
            if self._env.is_conda:
                cmd = ["conda", "install", "-y"]
                if any(char in pkg_str for char in (" ", "<", ">", "*", "|")):
                    pkg_str = f"\"{pkg_str}\""
            else:
                cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
                if self._env.is_admin:
                    cmd.append("--user")
            cmd.append(pkg_str)

            clean_pkg = pkg_str.replace("\"", "")
            installer = SubProcInstaller(self._env, clean_pkg, cmd, self._is_gui)
            if installer() != 0:
                logger.error("Unable to install package: %s. Process aborted", clean_pkg)
                sys.exit(1)

    def _install_conda_packages(self) -> None:
        """ Install required conda packages """
        logger.info("Installing Required Conda Packages. This may take some time...")
        for pkg in self._packages.to_install_conda:
            channel = "" if len(pkg) != 2 else pkg[1]
            self._from_conda(pkg[0], channel=channel, conda_only=True)

    def _install_python_packages(self) -> None:
        """ Install required pip packages """
        conda_only = False
        assert self._env.backend is not None
        for pkg, version in self._packages.to_install:
            if self._env.is_conda:
                mapping = _CONDA_MAPPING.get(pkg, (pkg, ""))
                channel = "" if mapping[1] is None else mapping[1]
                pkg = mapping[0]
                pip_only = pkg in _FORCE_PIP.get(self._env.backend, []) or pkg in _FORCE_PIP["all"]
            pkg = self._format_package(pkg, version) if version else pkg
            if self._env.is_conda and not pip_only:
                if self._from_conda(pkg, channel=channel, conda_only=conda_only):
                    continue
            self._from_pip(pkg)

    def _install_missing_dep(self) -> None:
        """ Install missing dependencies """
        self._install_conda_packages()  # Install conda packages first
        self._install_python_packages()

    def _from_conda(self,
                    package: list[str] | str,
                    channel: str = "",
                    conda_only: bool = False) -> bool:
        """ Install a conda package

        Parameters
        ----------
        package: list[str] | str
            The full formatted package(s), with version(s), to be installed
        channel: str, optional
            The Conda channel to install from. Select empty string for default channel.
            Default: ``""`` (empty string)
        conda_only: bool, optional
            ``True`` if the package is only available in Conda. Default: ``False``

        Returns
        -------
        bool
            ``True`` if the package was succesfully installed otherwise ``False``
        """
        #  Packages with special characters need to be enclosed in double quotes
        success = True
        condaexe = ["conda", "install", "-y"]
        if channel:
            condaexe.extend(["-c", channel])

        pkgs = package if isinstance(package, list) else [package]

        for i, pkg in enumerate(pkgs):
            if any(char in pkg for char in (" ", "<", ">", "*", "|")):
                pkgs[i] = f"\"{pkg}\""
        condaexe.extend(pkgs)

        clean_pkg = " ".join([p.replace("\"", "") for p in pkgs])
        installer = self._installer(self._env, clean_pkg, condaexe, self._is_gui)
        retcode = installer()

        if retcode != 0 and not conda_only:
            logger.info("%s not available in Conda. Installing with pip", package)
        elif retcode != 0:
            logger.warning("Couldn't install %s with Conda. Please install this package "
                           "manually", package)
        success = retcode == 0 and success
        return success

    def _from_pip(self, package: str) -> None:
        """ Install a pip package

        Parameters
        ----------
        package: str
            The full formatted package, with version, to be installed
        """
        pipexe = [sys.executable, "-u", "-m", "pip", "install", "--no-cache-dir"]
        # install as user to solve perm restriction
        if not self._env.is_admin and not self._env.is_virtualenv:
            pipexe.append("--user")
        pipexe.append(package)

        installer = self._installer(self._env, package, pipexe, self._is_gui)
        if installer() != 0:
            logger.warning("Couldn't install %s with pip. Please install this package manually",
                           package)
            global _INSTALL_FAILED  # pylint:disable=global-statement
            _INSTALL_FAILED = True


class ProgressBar():
    """ Simple progress bar using STDLib for intercepting Conda installs and keeping the
    terminal from getting jumbled """
    def __init__(self):
        self._width_desc = 21
        self._width_size = 9
        self._width_bar = 35
        self._width_pct = 4
        self._marker = "█"

        self._cursor_visible = True
        self._current_pos = 0
        self._bars = []

    @classmethod
    def _display_cursor(cls, visible: bool) -> None:
        """ Sends ANSI code to display or hide the cursor

        Parameters
        ----------
        visible: bool
            ``True`` to display the cursor. ``False`` to hide the cursor
        """
        code = "\x1b[?25h" if visible else "\x1b[?25l"
        print(code, end="\r")

    def _format_bar(self, description: str, size: str, percent: int) -> str:
        """ Format the progress bar for display

        Parameters
        ----------
        description: str
            The description to display for the progress bar
        size: str
            The size of the download, including units
        percent: int
            The percentage progress of the bar
        """
        size = size[:self._width_size].ljust(self._width_size)
        bar_len = int(self._width_bar * (percent / 100))
        progress = f"{self._marker * bar_len}"[:self._width_bar].ljust(self._width_bar)
        pct = f"{percent}%"[:self._width_pct].rjust(self._width_pct)
        return f"  {description}| {size} | {progress} | {pct}"

    def _move_cursor(self, position: int) -> str:
        """ Generate ANSI code for moving the cursor to the given progress bar's position

        Parameters
        ----------
        position: int
            The progress bar position to move to

        Returns
        -------
        str
            The ansi code to move to the given position
        """
        move = position - self._current_pos
        retval = "\x1b[A" if move < 0 else "\x1b[B" if move > 0 else ""
        retval *= abs(move)
        return retval

    def __call__(self, description: str, size: str, percent: int) -> None:
        """ Create or update a progress bar

        Parameters
        ----------
        description: str
            The description to display for the progress bar
        size: str
            The size of the download, including units
        percent: int
            The percentage progress of the bar
        """
        if self._cursor_visible:
            self._display_cursor(visible=False)

        desc = description[:self._width_desc].ljust(self._width_desc)
        if desc not in self._bars:
            self._bars.append(desc)

        position = self._bars.index(desc)
        pbar = self._format_bar(desc, size, percent)

        output = f"{self._move_cursor(position)} {pbar}"

        print(output)
        self._current_pos = position + 1

    def close(self) -> None:
        """ Reset all progress bars and re-enable the cursor """
        print(self._move_cursor(len(self._bars)), end="\r")
        self._display_cursor(True)
        self._cursor_visible = True
        self._current_pos = 0
        self._bars = []


class Installer():
    """ Parent class for package installers.

    PyWinPty is used for Windows, Pexpect is used for Linux, as these can provide us with realtime
    output.

    Subprocess is used as a fallback if any of the above fail, but this caches output, so it can
    look like the process has hung to the end user

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True`` if the process is being called from the Faceswap GUI
    """
    def __init__(self,
                 environment: Environment,
                 package: str,
                 command: list[str],
                 is_gui: bool) -> None:
        logger.info("Installing %s", package)
        logger.debug("argv: %s", command)
        self._env = environment
        self._package = package
        self._command = command
        self._is_conda = "conda" in command
        self._is_gui = is_gui

        self._progess_bar = ProgressBar()
        self._re_conda = re.compile(
            rb"(?P<lib>^\S+)\s+\|\s+(?P<tot>\d+\.?\d*\s\w+).*\|\s+(?P<prg>\d+%)")
        self._re_pip_pkg = re.compile(rb"^\s*Downloading\s(?P<lib>\w+-.+?)-")
        self._re_pip = re.compile(rb"(?P<done>\d+\.?\d*)/(?P<tot>\d+\.?\d*\s\w+)")
        self._pip_pkg = ""
        self._seen_lines: set[str] = set()

    def __call__(self) -> int:
        """ Call the subclassed call function

        Returns
        -------
        int
            The return code of the package install process
        """
        try:
            returncode = self.call()
        except Exception as err:  # pylint:disable=broad-except
            logger.debug("Failed to install with %s. Falling back to subprocess. Error: %s",
                         self.__class__.__name__, str(err))
            self._progess_bar.close()
            returncode = SubProcInstaller(self._env, self._package, self._command, self._is_gui)()

        logger.debug("Package: %s, returncode: %s", self._package, returncode)
        self._progess_bar.close()
        return returncode

    def call(self) -> int:
        """ Override for package installer specific logic.

        Returns
        -------
        int
            The return code of the package install process
        """
        raise NotImplementedError()

    def _print_conda(self, text: bytes) -> None:
        """ Output progress for Conda installs

        Parameters
        ----------
        text: bytes
            The text to print
        """
        data = self._re_conda.match(text)
        if not data:
            return
        lib = data.groupdict()["lib"].decode("utf-8", errors="replace")
        size = data.groupdict()["tot"].decode("utf-8", errors="replace")
        progress = int(data.groupdict()["prg"].decode("utf-8", errors="replace")[:-1])
        self._progess_bar(lib, size, progress)

    def _print_pip(self, text: bytes) -> None:
        """ Output progress for Pip installs

        Parameters
        ----------
        text: bytes
            The text to print
        """
        pkg = self._re_pip_pkg.match(text)
        if pkg:
            logger.debug("Collected pip package '%s'", pkg)
            self._pip_pkg = pkg.groupdict()["lib"].decode("utf-8", errors="replace")
            return
        data = self._re_pip.search(text)
        if not data:
            return
        done = float(data.groupdict()["done"].decode("utf-8", errors="replace"))
        size = data.groupdict()["tot"].decode("utf-8", errors="replace")
        progress = int(round(done / float(size.split()[0]) * 100, 0))
        self._progess_bar(self._pip_pkg, size, progress)

    def _non_gui_print(self, text: bytes) -> None:
        """ Print output to console if not running in the GUI

        Parameters
        ----------
        text: bytes
            The text to print
        """
        if self._is_gui:
            return
        if self._is_conda:
            self._print_conda(text)
        else:
            self._print_pip(text)

    def _seen_line_log(self, text: str) -> None:
        """ Output gets spammed to the log file when conda is waiting/processing. Only log each
        unique line once.

        Parameters
        ----------
        text: str
            The text to log
        """
        if text in self._seen_lines:
            return
        logger.debug(text)
        self._seen_lines.add(text)


class PexpectInstaller(Installer):  # pylint:disable=too-few-public-methods
    """ Package installer for Linux/macOS using Pexpect

    Uses Pexpect for installing packages allowing access to realtime feedback

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True`` if the process is being called from the Faceswap GUI
    """
    def call(self) -> int:
        """ Install a package using the Pexpect module

        Returns
        -------
        int
            The return code of the package install process
        """
        import pexpect  # pylint:disable=import-outside-toplevel,import-error
        proc = pexpect.spawn(" ".join(self._command), timeout=None)
        while True:
            try:
                proc.expect([b"\r\n", b"\r"])
                line: bytes = proc.before
                self._seen_line_log(line.decode("utf-8", errors="replace").rstrip())
                self._non_gui_print(line)
            except pexpect.EOF:
                break
        proc.close()
        return proc.exitstatus


class WinPTYInstaller(Installer):  # pylint:disable=too-few-public-methods
    """ Package installer for Windows using WinPTY

    Spawns a pseudo PTY for installing packages allowing access to realtime feedback

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True`` if the process is being called from the Faceswap GUI
    """
    def __init__(self,
                 environment: Environment,
                 package: str,
                 command: list[str],
                 is_gui: bool) -> None:
        super().__init__(environment, package, command, is_gui)
        self._cmd = which(command[0], path=os.environ.get('PATH', os.defpath))
        self._cmdline = list2cmdline(command)
        logger.debug("cmd: '%s', cmdline: '%s'", self._cmd, self._cmdline)

        self._pbar = re.compile(r"(?:eta\s[\d\W]+)|(?:\s+\|\s+\d+%)\Z")
        self._eof = False
        self._read_bytes = 1024

        self._lines: list[str] = []
        self._out = ""

    def _read_from_pty(self, proc: T.Any, winpty_error: T.Any) -> None:
        """ Read :attr:`_num_bytes` from WinPTY. If there is an error reading, recursively halve
        the number of bytes read until we get a succesful read. If we get down to 1 byte without a
        succesful read, assume we are at EOF.

        Parameters
        ----------
        proc: :class:`winpty.PTY`
            The WinPTY process
        winpty_error: :class:`winpty.WinptyError`
            The winpty error exception. Passed in as WinPTY is not in global scope
        """
        try:
            from_pty = proc.read(self._read_bytes)
        except winpty_error:
            # TODO Reinsert this check
            # The error message "pipe has been ended" is language specific so this check
            # fails on non english systems. For now we just swallow all errors until no
            # bytes are left to read and then check the return code
            # if any(val in str(err) for val in ["EOF", "pipe has been ended"]):
            #    # Get remaining bytes. On a comms error, the buffer remains unread so keep
            #    # halving buffer amount until down to 1 when we know we have everything
            #     if self._read_bytes == 1:
            #         self._eof = True
            #     from_pty = ""
            #     self._read_bytes //= 2
            # else:
            #     raise

            # Get remaining bytes. On a comms error, the buffer remains unread so keep
            # halving buffer amount until down to 1 when we know we have everything
            if self._read_bytes == 1:
                self._eof = True
            from_pty = ""
            self._read_bytes //= 2

        self._out += from_pty

    def _out_to_lines(self) -> None:
        """ Process the winpty output into separate lines. Roll over any semi-consumed lines to the
        next proc call. """
        if "\n" not in self._out:
            return

        self._lines.extend(self._out.split("\n"))

        if self._out.endswith("\n") or self._eof:  # Ends on newline or is EOF
            self._out = ""
        else:  # roll over semi-consumed line to next read
            self._out = self._lines[-1]
            self._lines = self._lines[:-1]

    def call(self) -> int:
        """ Install a package using the PyWinPTY module

        Returns
        -------
        int
            The return code of the package install process
        """
        import winpty  # pylint:disable=import-outside-toplevel,import-error
        # For some reason with WinPTY we need to pass in the full command. Probably a bug
        proc = winpty.PTY(
            100,
            24,
            backend=winpty.enums.Backend.WinPTY,  # ConPTY hangs and has lots of Ansi Escapes
            agent_config=winpty.enums.AgentConfig.WINPTY_FLAG_PLAIN_OUTPUT)  # Strip all Ansi

        if not proc.spawn(self._cmd, cmdline=self._cmdline):
            del proc
            raise RuntimeError("Failed to spawn winpty")

        while True:
            self._read_from_pty(proc, winpty.WinptyError)
            self._out_to_lines()
            for line in self._lines:
                self._seen_line_log(line.rstrip())
                self._non_gui_print(line.encode("utf-8", errors="replace"))
            self._lines = []

            if self._eof:
                returncode = proc.get_exitstatus()
                break

        del proc
        return returncode


class SubProcInstaller(Installer):
    """ The fallback package installer if either of the OS specific installers fail.

    Uses the python Subprocess module to install packages. Feedback does not return in realtime
    so the process can look like it has hung to the end user

    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True`` if the process is being called from the Faceswap GUI
    """
    def __init__(self,
                 environment: Environment,
                 package: str,
                 command: list[str],
                 is_gui: bool) -> None:
        super().__init__(environment, package, command, is_gui)
        self._shell = self._env.os_version[0] == "Windows" and command[0] == "conda"

    def __call__(self) -> int:
        """ Override default call function so we don't recursively call ourselves on failure. """
        returncode = self.call()
        logger.debug("Package: %s, returncode: %s", self._package, returncode)
        return returncode

    def call(self) -> int:
        """ Install a package using the Subprocess module

        Returns
        -------
        int
            The return code of the package install process
        """
        with Popen(self._command,
                   bufsize=0, stdout=PIPE, stderr=STDOUT, shell=self._shell) as proc:
            while True:
                if proc.stdout is not None:
                    lines = proc.stdout.readline()
                returncode = proc.poll()
                if lines == b"" and returncode is not None:
                    break

                for line in lines.split(b"\r"):
                    self._seen_line_log(line.decode("utf-8", errors="replace").rstrip())
                    self._non_gui_print(line)

        return returncode


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
            "XQuartz: https://www.xquartz.org/\n\n"

            "2a. It is recommended to use Anaconda for your Python Virtual Environment as this\n"
            "will handle the installation of CUDA and cuDNN for you:\n"
            "https://www.anaconda.com/distribution/\n\n"

            "2b. If you do not want to use Anaconda you will need to manually install CUDA and "
            "cuDNN:\n"
            "CUDA: https://developer.nvidia.com/cuda-downloads"
            "cuDNN: https://developer.nvidia.com/rdp/cudnn-download\n\n")

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
    if _INSTALL_FAILED:
        sys.exit(1)
    Install(ENV)
