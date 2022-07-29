#!/usr/bin/env python3
""" Install packages for faceswap.py """

# >>> Environment
import logging
import ctypes
import json
import locale
import platform
import operator
import os
import re
import sys
from subprocess import run, PIPE, Popen, STDOUT
from typing import Dict, List, Optional, Tuple

from pkg_resources import parse_requirements, Requirement

from lib.logger import log_setup

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_INSTALL_FAILED = False
# Revisions of tensorflow GPU and cuda/cudnn requirements. These relate specifically to the
# Tensorflow builds available from pypi
_TENSORFLOW_REQUIREMENTS = {">=2.4.0,<2.5.0": ["11.0", "8.0"],
                            ">=2.5.0,<2.9.0": ["11.2", "8.1"]}
# Packages that are explicitly required for setup.py
_INSTALLER_REQUIREMENTS = ["pexpect>=4.8.0"]

# Mapping of Python packages to their conda names if different from pip or in non-default channel
_CONDA_MAPPING: Dict[str, Tuple[str, str]] = {
    # "opencv-python": ("opencv", "conda-forge"),  # Periodic issues with conda-forge opencv
    "fastcluster": ("fastcluster", "conda-forge"),
    "imageio-ffmpeg": ("imageio-ffmpeg", "conda-forge"),
    "tensorflow-deps": ("tensorflow-deps", "apple"),
    "libblas": ("libblas", "conda-forge")}


class Environment():
    """ The current install environment

    Parameters
    ----------
    updater: bool, Optional
        ``True`` of the script is being called by Faceswap's internal updater. ``False`` if full
        setup is running. Default: ``False``
    """
    def __init__(self, updater: bool = False) -> None:
        self.conda_required_packages: List[Tuple[str, ...]] = [("tk", )]
        self.updater = updater
        # Flag that setup is being run by installer so steps can be skipped
        self.is_installer: bool = False
        self.cuda_version: str = ""
        self.cudnn_version: str = ""
        self.enable_amd: bool = False
        self.enable_apple_silicon: bool = False
        self.enable_docker: bool = False
        self.enable_cuda: bool = False
        self.required_packages: List[Tuple[str, List[Tuple[str, str]]]] = []
        self.missing_packages: List[Tuple[str, List[Tuple[str, str]]]] = []
        self.conda_missing_packages: List[Tuple[str, ...]] = []

        self._process_arguments()
        self._check_permission()
        self._check_system()
        self._check_python()
        self._output_runtime_info()
        self._check_pip()
        self._upgrade_pip()
        self._set_ld_library_path()

        self.installed_packages = self.get_installed_packages()
        self.installed_packages.update(self.get_installed_conda_packages())

    @property
    def encoding(self) -> str:
        """ Get system encoding """
        return locale.getpreferredencoding()

    @property
    def os_version(self) -> Tuple[str, str]:
        """ Get OS Version """
        return platform.system(), platform.release()

    @property
    def py_version(self) -> Tuple[str, str]:
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
    def is_virtualenv(self) -> bool:
        """ Check whether this is a virtual environment """
        if not self.is_conda:
            retval = (hasattr(sys, "real_prefix") or
                      (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
        else:
            prefix = os.path.dirname(sys.prefix)
            retval = (os.path.basename(prefix) == "envs")
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
            if arg == "--nvidia":
                self.enable_cuda = True
            if arg == "--amd":
                self.enable_amd = True
            if arg == "--apple-silicon":
                self.enable_apple_silicon = True

    def get_required_packages(self) -> None:
        """ Load requirements list """
        if self.enable_amd:
            suffix = "amd.txt"
        elif self.enable_cuda:
            suffix = "nvidia.txt"
        elif self.enable_apple_silicon:
            suffix = "apple_silicon.txt"
        else:
            suffix = "cpu.txt"
        req_files = ["_requirements_base.txt", f"requirements_{suffix}"]
        pypath = os.path.dirname(os.path.realpath(__file__))
        requirements = []
        for req_file in req_files:
            requirements_file = os.path.join(pypath, "requirements", req_file)
            with open(requirements_file, encoding="utf8") as req:
                for package in req.readlines():
                    package = package.strip()
                    if package and (not package.startswith(("#", "-r"))):
                        requirements.append(package)

        # Add required installer packages
        if self.os_version[0] != "Windows":
            for inst in _INSTALLER_REQUIREMENTS:
                requirements.insert(0, inst)

        self.required_packages = [(pkg.unsafe_name, pkg.specs)
                                  for pkg in parse_requirements(requirements)
                                  if pkg.marker is None or pkg.marker.evaluate()]
        logger.debug(self.required_packages)

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
            self.enable_apple_silicon = True

            if not self.updater and not self.is_conda:
                logger.error("Setting up Faceswap for Apple Silicon outside of a Conda "
                             "environment is unsupported")
                sys.exit(1)

    def _check_python(self) -> None:
        """ Check python and virtual environment status """
        logger.info("Installed Python: %s %s", self.py_version[0], self.py_version[1])

        if self.updater:
            return

        if not ((3, 7) <= sys.version_info < (3, 10) and self.py_version[1] == "64bit"):
            logger.error("Please run this script with Python version 3.7 to 3.9 64bit and try "
                         "again.")
            sys.exit(1)
        if self.enable_amd and sys.version_info >= (3, 9):
            logger.error("The AMD version of Faceswap cannot be installed on versions of Python "
                         "higher than 3.8")
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

    def get_installed_packages(self) -> Dict[str, str]:
        """ Get currently installed packages """
        installed_packages = {}
        with Popen(f"\"{sys.executable}\" -m pip freeze --local", shell=True, stdout=PIPE) as chk:
            installed = chk.communicate()[0].decode(self.encoding).splitlines()

        for pkg in installed:
            if "==" not in pkg:
                continue
            item = pkg.split("==")
            installed_packages[item[0]] = item[1]
        logger.debug(installed_packages)
        return installed_packages

    def get_installed_conda_packages(self) -> Dict[str, str]:
        """ Get currently installed conda packages """
        if not self.is_conda:
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

    def update_tf_dep(self) -> None:
        """ Update Tensorflow Dependency """
        if self.is_conda or not self.enable_cuda:
            # CPU/AMD doesn't need Cuda and Conda handles Cuda and cuDNN so nothing to do here
            return

        tf_ver = None
        cudnn_inst = self.cudnn_version.split(".")
        for key, val in _TENSORFLOW_REQUIREMENTS.items():
            cuda_req = val[0]
            cudnn_req = val[1].split(".")
            if cuda_req == self.cuda_version and (cudnn_req[0] == cudnn_inst[0] and
                                                  cudnn_req[1] <= cudnn_inst[1]):
                tf_ver = key
                break
        if tf_ver:
            # Remove the version of tensorflow in requirements file and add the correct version
            # that corresponds to the installed Cuda/cuDNN versions
            self.required_packages = [pkg for pkg in self.required_packages
                                      if not pkg[0].startswith("tensorflow-gpu")]
            tf_ver = f"tensorflow-gpu{tf_ver}"

            tf_ver = f"tensorflow-gpu{tf_ver}"
            self.required_packages.append(("tensorflow-gpu",
                                           next(parse_requirements(tf_ver)).specs))
            return

        logger.warning(
            "The minimum Tensorflow requirement is 2.4 \n"
            "Tensorflow currently has no official prebuild for your CUDA, cuDNN combination.\n"
            "Either install a combination that Tensorflow supports or build and install your own "
            "tensorflow-gpu.\r\n"
            "CUDA Version: %s\r\n"
            "cuDNN Version: %s\r\n"
            "Help:\n"
            "Building Tensorflow: https://www.tensorflow.org/install/install_sources\r\n"
            "Tensorflow supported versions: "
            "https://www.tensorflow.org/install/source#tested_build_configurations",
            self.cuda_version, self.cudnn_version)

        custom_tf = input("Location of custom tensorflow-gpu wheel (leave "
                          "blank to manually install): ")
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
            self.required_packages.append((custom_tf, [(custom_tf, "")]))

    def set_config(self) -> None:
        """ Set the backend in the faceswap config file """
        if self.enable_amd:
            backend = "amd"
        elif self.enable_cuda:
            backend = "nvidia"
        elif self.enable_apple_silicon:
            backend = "apple_silicon"
        else:
            backend = "cpu"
        config = {"backend": backend}
        pypath = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(pypath, "config", ".faceswap")
        with open(config_file, "w", encoding="utf8") as cnf:
            json.dump(config, cnf)
        logger.info("Faceswap config written to: %s", config_file)

    def _set_ld_library_path(self) -> None:
        """ Update the LD_LIBRARY_PATH environment variable when activating a conda environment
        and revert it when deactivating. Linux/conda only

        Notes
        -----
        From Tensorflow 2.7, installing Cuda Toolkit from conda-forge and tensorflow from pip
        causes tensorflow to not be able to locate shared libs and hence not use the GPU.
        We update the environment variable for all instances using Conda as it shouldn't hurt
        anything and may help avoid conflicts with globally installed Cuda
        """
        if not self.is_conda or not self.enable_cuda or self.os_version[0].lower() != "linux":
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
        shebang = "#!/bin/sh\n\n"

        with open(activate_script, "w", encoding="utf8") as afile:
            afile.write(f"{shebang}")
            afile.write("export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}\n")
            afile.write(f"export LD_LIBRARY_PATH='{conda_libs}':${{LD_LIBRARY_PATH}}\n")

        with open(deactivate_script, "w", encoding="utf8") as afile:
            afile.write(f"{shebang}")
            afile.write("export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}\n")
            afile.write("unset OLD_LD_LIBRARY_PATH\n")

        logger.info("Cuda search path set to '%s'", conda_libs)


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
        if self._env.enable_apple_silicon:
            return
        self._user_input()
        self._check_cuda()
        self._env.update_tf_dep()
        if self._env.os_version[0] == "Windows":
            self._tips.pip()

    def _user_input(self) -> None:
        """ Get user input for AMD/Cuda/Docker """
        self._amd_ask_enable()
        if not self._env.enable_amd:
            self._docker_ask_enable()
            self._cuda_ask_enable()
        if self._env.os_version[0] != "Linux" and (self._env.enable_docker
                                                   and self._env.enable_cuda):
            self._docker_confirm()
        if self._env.enable_docker:
            self._docker_tips()
            self._env.set_config()
            sys.exit(0)

    def _amd_ask_enable(self) -> None:
        """ Enable or disable Plaidml for AMD"""
        logger.info("AMD Support: AMD GPU support is currently limited.\r\n"
                    "Nvidia Users MUST answer 'no' to this option.")
        i = input("Enable AMD Support? [y/N] ")
        if i in ("Y", "y"):
            logger.info("AMD Support Enabled")
            self._env.enable_amd = True
        else:
            logger.info("AMD Support Disabled")
            self._env.enable_amd = False

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
            self._env.enable_cuda = False

    def _docker_tips(self) -> None:
        """ Provide tips for Docker use """
        if not self._env.enable_cuda:
            self._tips.docker_no_cuda()
        else:
            self._tips.docker_cuda()

    def _cuda_ask_enable(self) -> None:
        """ Enable or disable CUDA """
        i = input("Enable  CUDA? [Y/n] ")
        if i in ("", "Y", "y"):
            logger.info("CUDA Enabled")
            self._env.enable_cuda = True
        else:
            logger.info("CUDA Disabled")
            self._env.enable_cuda = False

    def _check_cuda(self) -> None:
        """ Check for Cuda and cuDNN Locations. """
        if not self._env.enable_cuda:
            logger.debug("Skipping Cuda checks as not enabled")
            return

        if self._env.is_conda:
            logger.info("Skipping Cuda/cuDNN checks for Conda install")
            return

        if self._env.os_version[0] in ("Linux", "Windows"):
            global _INSTALL_FAILED  # pylint:disable=global-statement
            check = CudaCheck()
            if check.cuda_version:
                self._env.cuda_version = check.cuda_version
                logger.info("CUDA version: %s", self._env.cuda_version)
            else:
                logger.error("CUDA not found. Install and try again.\n"
                             "Recommended version:      CUDA 10.1     cuDNN 7.6\n"
                             "CUDA: https://developer.nvidia.com/cuda-downloads\n"
                             "cuDNN: https://developer.nvidia.com/rdp/cudnn-download")
                _INSTALL_FAILED = True
                return

            if check.cudnn_version:
                self._env.cudnn_version = ".".join(check.cudnn_version.split(".")[:2])
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
        self._env.cuda_version = input("Manually specify CUDA version: ")


class CudaCheck():  # pylint:disable=too-few-public-methods
    """ Find the location of system installed Cuda and cuDNN on Windows and Linux. """

    def __init__(self) -> None:
        self.cuda_path: Optional[str] = None
        self.cuda_version: Optional[str] = None
        self.cudnn_version: Optional[str] = None

        self._os: str = platform.system().lower()
        self._cuda_keys: List[str] = [key
                                      for key in os.environ
                                      if key.lower().startswith("cuda_path_v")]
        self._cudnn_header_files: List[str] = ["cudnn_version.h", "cudnn.h"]
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
                                stdout.decode(locale.getpreferredencoding()))
            if version is not None:
                self.cuda_version = version.groupdict().get("cuda", None)
            locate = "where" if self._os == "windows" else "which"
            path = os.popen(f"{locate} nvcc").read()
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
        chk = os.popen("ldconfig -p | grep -P \"libcudart.so.\\d+.\\d+\" | head -n 1").read()
        if not chk and os.environ.get("LD_LIBRARY_PATH"):
            for path in os.environ["LD_LIBRARY_PATH"].split(":"):
                chk = os.popen(f"ls {path} | grep -P -o \"libcudart.so.\\d+.\\d+\" | "
                               "head -n 1").read()
                if chk:
                    break
        if not chk:  # Cuda not found
            return

        cudavers = chk.strip().replace("libcudart.so.", "")
        self.cuda_version = cudavers[:cudavers.find(" ")]
        self.cuda_path = chk[chk.find("=>") + 3:chk.find("targets") - 1]

    def _cuda_check_windows(self) -> None:
        """ Check Windows CUDA Version and path from Environment Variables"""
        if not self._cuda_keys:  # Cuda environment variable not found
            return
        self.cuda_version = self._cuda_keys[0].lower().replace("cuda_path_v", "").replace("_", ".")
        self.cuda_path = os.environ[self._cuda_keys[0][0]]

    def _cudnn_check(self):
        """ Check Linux or Windows cuDNN Version from cudnn.h and add to :attr:`cudnn_version`. """
        cudnn_checkfiles = getattr(self, f"_get_checkfiles_{self._os}")()
        cudnn_checkfile = next((hdr for hdr in cudnn_checkfiles if os.path.isfile(hdr)), None)
        logger.debug("cudnn checkfiles: %s", cudnn_checkfile)
        if not cudnn_checkfile:
            return
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
        if found != 3:  # Full version could not be determined
            return
        self.cudnn_version = ".".join([str(major), str(minor), str(patchlevel)])
        logger.debug("cudnn version: %s", self.cudnn_version)

    def _get_checkfiles_linux(self) -> List[str]:
        """ Return the the files to check for cuDNN locations for Linux by querying
        the dynamic link loader.

        Returns
        -------
        list
            List of header file locations to scan for cuDNN versions
        """
        chk = os.popen("ldconfig -p | grep -P \"libcudnn.so.\\d+\" | head -n 1").read()
        chk = chk.strip().replace("libcudnn.so.", "")
        if not chk:
            return []

        cudnn_vers = chk[0]
        header_files = [f"cudnn_v{cudnn_vers}.h"] + self._cudnn_header_files

        cudnn_path = os.path.realpath(chk[chk.find("=>") + 3:chk.find("libcudnn") - 1])
        cudnn_path = cudnn_path.replace("lib", "include")
        cudnn_checkfiles = [os.path.join(cudnn_path, header) for header in header_files]
        return cudnn_checkfiles

    def _get_checkfiles_windows(self) -> List[str]:
        """ Return the check-file locations for Windows. Just looks inside the include folder of
        the discovered :attr:`cuda_path`

        Returns
        -------
        list
            List of header file locations to scan for cuDNN versions
        """
        # TODO A more reliable way of getting the windows location
        if not self.cuda_path:
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
        self._operators = {"==": operator.eq,
                           ">=": operator.ge,
                           "<=": operator.le,
                           ">": operator.gt,
                           "<": operator.lt}
        self._env = environment
        self._is_gui = is_gui
        if self._env.os_version[0] == "Windows":
            self._installer = self._subproc_installer
        else:
            self._installer = self._pexpect_installer

        if not self._env.is_installer and not self._env.updater:
            self._ask_continue()
        self._env.get_required_packages()
        self._check_missing_dep()
        self._check_conda_missing_dep()
        if (self._env.updater and
                not self._env.missing_packages and not self._env.conda_missing_packages):
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

    @classmethod
    def _ask_continue(cls) -> None:
        """ Ask Continue with Install """
        inp = input("Please ensure your System Dependencies are met. Continue? [y/N] ")
        if inp in ("", "N", "n"):
            logger.error("Please install system dependencies to continue")
            sys.exit(1)

    def _check_missing_dep(self) -> None:
        """ Check for missing dependencies """
        for key, specs in self._env.required_packages:

            if self._env.is_conda:  # Get Conda alias for Key
                key = _CONDA_MAPPING.get(key, (key, None))[0]

            if key not in self._env.installed_packages:
                # Add not installed packages to missing packages list
                self._env.missing_packages.append((key, specs))
                continue

            installed_vers = self._env.installed_packages.get(key, "")

            if specs and not all(self._operators[spec[0]](
                [int(s) for s in installed_vers.split(".")],
                [int(s) for s in spec[1].split(".")])
                                 for spec in specs):
                self._env.missing_packages.append((key, specs))
        logger.debug(self._env.missing_packages)

    def _check_conda_missing_dep(self) -> None:
        """ Check for conda missing dependencies """
        if not self._env.is_conda:
            return
        installed_conda_packages = self._env.get_installed_conda_packages()
        for pkg in self._env.conda_required_packages:
            key = pkg[0].split("==")[0]
            if key not in self._env.installed_packages:
                self._env.conda_missing_packages.append(pkg)
                continue
            if len(pkg[0].split("==")) > 1:
                if pkg[0].split("==")[1] != installed_conda_packages.get(key):
                    self._env.conda_missing_packages.append(pkg)
                    continue
        logger.debug(self._env.conda_missing_packages)

    @classmethod
    def _format_package(cls, package: str, version: List[Tuple[str, str]]) -> str:
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
        setup_packages = [(pkg.unsafe_name, pkg.specs)
                          for pkg in parse_requirements(_INSTALLER_REQUIREMENTS)]

        for pkg in setup_packages:
            if pkg not in self._env.missing_packages:
                continue
            self._env.missing_packages.pop(self._env.missing_packages.index(pkg))
            pkg_str = self._format_package(*pkg)
            if self._env.is_conda:
                cmd = ["conda", "install", "-y"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
                if self._env.is_admin:
                    cmd.append("--user")
            cmd.append(pkg_str)

            clean_pkg = pkg_str.replace("\"", "")
            if self._subproc_installer(cmd, clean_pkg) != 0:
                logger.error("Unable to install package: %s. Process aborted", clean_pkg)
                sys.exit(1)

    def _install_missing_dep(self) -> None:
        """ Install missing dependencies """
        # Install conda packages first
        if self._env.conda_missing_packages:
            self._install_conda_packages()
        if self._env.missing_packages:
            self._install_python_packages()

    def _install_python_packages(self) -> None:
        """ Install required pip packages """
        conda_only = False
        for pkg, version in self._env.missing_packages:
            if self._env.is_conda:
                mapping = _CONDA_MAPPING.get(pkg, (pkg, ""))
                channel = None if mapping[1] == "" else mapping[1]
                pkg = mapping[0]
            pkg = self._format_package(pkg, version) if version else pkg
            if self._env.is_conda:
                if pkg.startswith("tensorflow-gpu"):
                    # From TF 2.4 onwards, Anaconda Tensorflow becomes a mess. The version of 2.5
                    # installed by Anaconda is compiled against an incorrect numpy version which
                    # breaks Tensorflow. Coupled with this the versions of cudatoolkit and cudnn
                    # available in the default Anaconda channel are not compatible with the
                    # official PyPi versions of Tensorflow. With this in mind we will pull in the
                    # required Cuda/cuDNN from conda-forge, and install Tensorflow with pip
                    # TODO Revert to Conda if they get their act together

                    # Rewrite tensorflow requirement to versions from highest available cuda/cudnn
                    highest_cuda = sorted(_TENSORFLOW_REQUIREMENTS.values())[-1]
                    compat_tf = next(k for k, v in _TENSORFLOW_REQUIREMENTS.items()
                                     if v == highest_cuda)
                    pkg = f"tensorflow-gpu{compat_tf}"
                    conda_only = True

                if self._conda_installer(pkg, channel=channel, conda_only=conda_only):
                    continue
            self._pip_installer(pkg)

    def _install_conda_packages(self) -> None:
        """ Install required conda packages """
        logger.info("Installing Required Conda Packages. This may take some time...")
        for pkg in self._env.conda_missing_packages:
            channel = None if len(pkg) != 2 else pkg[1]
            self._conda_installer(pkg[0], channel=channel, conda_only=True)

    def _conda_installer(self,
                         package: str,
                         channel: Optional[str] = None,
                         conda_only: bool = False) -> bool:
        """ Install a conda package

        Parameters
        ----------
        package: str
            The full formatted package, with version, to be installed
        channel: str, optional
            The Conda channel to install from. Select ``None`` for default channel.
            Default: ``None``
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

        if package.startswith("tensorflow-gpu"):
            # Here we will install the cuda/cudnn toolkits, currently only available from
            # conda-forge, but fail tensorflow itself so that it can be handled by pip.
            specs = Requirement.parse(package).specs
            for key, val in _TENSORFLOW_REQUIREMENTS.items():
                req_specs = Requirement.parse("foobar" + key).specs
                if all(item in req_specs for item in specs):
                    cuda, cudnn = val
                    break
            condaexe.extend(["-c", "conda-forge", f"cudatoolkit={cuda}", f"cudnn={cudnn}"])
            package = "Cuda Toolkit"
            success = False

        if package != "Cuda Toolkit":
            if any(char in package for char in (" ", "<", ">", "*", "|")):
                package = f"\"{package}\""
            condaexe.append(package)

        clean_pkg = package.replace("\"", "")
        retcode = self._installer(condaexe, clean_pkg)

        if retcode != 0 and not conda_only:
            logger.info("%s not available in Conda. Installing with pip", package)
        elif retcode != 0:
            logger.warning("Couldn't install %s with Conda. Please install this package "
                           "manually", package)
        success = retcode == 0 and success
        return success

    def _pip_installer(self, package: str) -> None:
        """ Install a pip package

        Parameters
        ----------
        package: str
            The full formatted package, with version, to be installed
        """
        pipexe = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
        # install as user to solve perm restriction
        if not self._env.is_admin and not self._env.is_virtualenv:
            pipexe.append("--user")
        pipexe.append(package)

        if self._installer(pipexe, package) != 0:
            logger.warning("Couldn't install %s with pip. Please install this package manually",
                           package)
            global _INSTALL_FAILED  # pylint:disable=global-statement
            _INSTALL_FAILED = True

    def _pexpect_installer(self, command: List[str], package: str) -> int:
        """ Run an install command using pexpect and log output.

        Pexpect is used so we can get unbuffered output to display updates

        Parameters
        ----------
        command: list
            The command to run
        package: str
            The package name that is being installed

        Returns
        -------
        int
            The return code from the subprocess
        """
        import pexpect  # pylint:disable=import-outside-toplevel,import-error
        logger.info("Installing %s", package)

        proc = pexpect.spawn(" ".join(command),
                             encoding=self._env.encoding,
                             codec_errors="replace",
                             timeout=None)
        last_line_cr = False
        while True:
            try:
                idx = proc.expect(["\r\n", "\r"])
                line = proc.before.rstrip()
                if line and idx == 0:
                    if last_line_cr:
                        last_line_cr = False
                        # Output last line of progress bar and go to next line
                        if not self._is_gui:
                            print(line)
                    logger.verbose(line)  # type:ignore
                elif line and idx == 1:
                    last_line_cr = True
                    logger.debug(line)
                    if not self._is_gui:
                        print(line, end="\r")
            except pexpect.EOF:
                break
        proc.close()
        returncode = proc.exitstatus
        logger.debug("Package: %s, returncode: %s", package, returncode)
        return returncode

    def _subproc_installer(self, command: List[str], package: str) -> int:
        """ Run an install command using subprocess Popen.

        pexpect uses pty which is not useable in Windows. The pexpect popen_spawn module does not
        give easy access to the return code, and also dumps stdout to console so we use subprocess
        for Windows. The downside of this is that we cannot do unbuffered reads, so the process can
        look like it hangs.

        #TODO Implement real time read functionality for windows

        Parameters
        ----------
        command: list
            The command to run
        package: str
            The package name that is being installed

        Returns
        -------
        int
            The return code from the subprocess
        """
        logger.info("Installing %s", package)
        shell = self._env.os_version[0] == "Windows" and command[0] == "conda"
        with Popen(command, bufsize=0, stdout=PIPE, stderr=STDOUT, shell=shell) as proc:
            last_line_cr = False
            while True:
                if proc.stdout is not None:
                    line = proc.stdout.readline().decode(self._env.encoding, errors="replace")
                if line == "" and proc.poll is not None:
                    break

                is_cr = line.startswith("\r")
                line = line.rstrip()

                if line and not is_cr:
                    if last_line_cr:
                        last_line_cr = False
                        # Go to next line
                        if not self._is_gui:
                            print("")
                    logger.verbose(line)  # type:ignore
                elif line:
                    last_line_cr = True
                    logger.debug(line)
                    if not self._is_gui:
                        print(line, end="\r")
            returncode = proc.wait()
        logger.debug("Package: %s, returncode: %s", package, returncode)
        return returncode


class Tips():
    """ Display installation Tips """
    @classmethod
    def docker_no_cuda(cls) -> None:
        """ Output Tips for Docker without Cuda """
        path = os.path.dirname(os.path.realpath(__file__))
        logger.info(
            "1. Install Docker\n"
            "https://www.docker.com/community-edition\n\n"
            "2. Build Docker Image For Faceswap\n"
            "docker build -t deepfakes-cpu -f Dockerfile.cpu .\n\n"
            "3. Mount faceswap volume and Run it\n"
            "# without GUI\n"
            "docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-cpu --name deepfakes-cpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\tdeepfakes-cpu\n\n"
            "# with gui. tools.py gui working.\n"
            "## enable local access to X11 server\n"
            "xhost +local:\n"
            "## create container\n"
            "nvidia-docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-cpu --name deepfakes-cpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\t-v /tmp/.X11-unix:/tmp/.X11-unix \\ \n"
            "\t-e DISPLAY=unix$DISPLAY \\ \n"
            "\t-e AUDIO_GID=`getent group audio | cut -d: -f3` \\ \n"
            "\t-e VIDEO_GID=`getent group video | cut -d: -f3` \\ \n"
            "\t-e GID=`id -g` \\ \n"
            "\t-e UID=`id -u` \\ \n"
            "\tdeepfakes-cpu \n\n"
            "4. Open a new terminal to run faceswap.py in /srv\n"
            "docker exec -it deepfakes-cpu bash", path, path)
        logger.info("That's all you need to do with a docker. Have fun.")

    @classmethod
    def docker_cuda(cls) -> None:
        """ Output Tips for Docker with Cuda"""
        path = os.path.dirname(os.path.realpath(__file__))
        logger.info(
            "1. Install Docker\n"
            "https://www.docker.com/community-edition\n\n"
            "2. Install latest CUDA\n"
            "CUDA: https://developer.nvidia.com/cuda-downloads\n\n"
            "3. Install Nvidia-Docker & Restart Docker Service\n"
            "https://github.com/NVIDIA/nvidia-docker\n\n"
            "4. Build Docker Image For Faceswap\n"
            "docker build -t deepfakes-gpu -f Dockerfile.gpu .\n\n"
            "5. Mount faceswap volume and Run it\n"
            "# without gui \n"
            "docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-gpu --name deepfakes-gpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\tdeepfakes-gpu\n\n"
            "# with gui.\n"
            "## enable local access to X11 server\n"
            "xhost +local:\n"
            "## enable nvidia device if working under bumblebee\n"
            "echo ON > /proc/acpi/bbswitch\n"
            "## create container\n"
            "nvidia-docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-gpu --name deepfakes-gpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\t-v /tmp/.X11-unix:/tmp/.X11-unix \\ \n"
            "\t-e DISPLAY=unix$DISPLAY \\ \n"
            "\t-e AUDIO_GID=`getent group audio | cut -d: -f3` \\ \n"
            "\t-e VIDEO_GID=`getent group video | cut -d: -f3` \\ \n"
            "\t-e GID=`id -g` \\ \n"
            "\t-e UID=`id -u` \\ \n"
            "\tdeepfakes-gpu\n\n"
            "6. Open a new terminal to interact with the project\n"
            "docker exec deepfakes-gpu python /srv/faceswap.py gui\n",
            path, path)

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
