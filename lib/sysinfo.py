#!/usr/bin python3
""" Obtain information about the running system, environment and GPU. """

import json
import locale
import os
import platform
import sys

from subprocess import PIPE, Popen

import psutil

from lib.git import git
from lib.gpu_stats import GPUStats, GPUInfo
from lib.utils import get_backend
from setup import CudaCheck


class _SysInfo():
    """ Obtain information about the System, Python and GPU """
    def __init__(self) -> None:
        self._state_file = _State().state_file
        self._configs = _Configs().configs
        self._system = {"platform": platform.platform(),
                        "system": platform.system().lower(),
                        "machine": platform.machine(),
                        "release": platform.release(),
                        "processor": platform.processor(),
                        "cpu_count": os.cpu_count()}
        self._python = {"implementation": platform.python_implementation(),
                        "version": platform.python_version()}
        self._gpu = self._get_gpu_info()
        self._cuda_check = CudaCheck()

    @property
    def _encoding(self) -> str:
        """ str: The system preferred encoding """
        return locale.getpreferredencoding()

    @property
    def _is_conda(self) -> bool:
        """ bool: `True` if running in a Conda environment otherwise ``False``. """
        return ("conda" in sys.version.lower() or
                os.path.exists(os.path.join(sys.prefix, 'conda-meta')))

    @property
    def _is_linux(self) -> bool:
        """ bool: `True` if running on a Linux system otherwise ``False``. """
        return self._system["system"] == "linux"

    @property
    def _is_macos(self) -> bool:
        """ bool: `True` if running on a macOS system otherwise ``False``. """
        return self._system["system"] == "darwin"

    @property
    def _is_windows(self) -> bool:
        """ bool: `True` if running on a Windows system otherwise ``False``. """
        return self._system["system"] == "windows"

    @property
    def _is_virtual_env(self) -> bool:
        """ bool: `True` if running inside a virtual environment otherwise ``False``. """
        if not self._is_conda:
            retval = (hasattr(sys, "real_prefix") or
                      (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
        else:
            prefix = os.path.dirname(sys.prefix)
            retval = os.path.basename(prefix) == "envs"
        return retval

    @property
    def _ram_free(self) -> int:
        """ int: The amount of free RAM in bytes. """
        return psutil.virtual_memory().free

    @property
    def _ram_total(self) -> int:
        """ int: The amount of total RAM in bytes. """
        return psutil.virtual_memory().total

    @property
    def _ram_available(self) -> int:
        """ int: The amount of available RAM in bytes. """
        return psutil.virtual_memory().available

    @property
    def _ram_used(self) -> int:
        """ int: The amount of used RAM in bytes. """
        return psutil.virtual_memory().used

    @property
    def _fs_command(self) -> str:
        """ str: The command line command used to execute faceswap. """
        return " ".join(sys.argv)

    @property
    def _installed_pip(self) -> str:
        """ str: The list of installed pip packages within Faceswap's scope. """
        with Popen(f"{sys.executable} -m pip freeze", shell=True, stdout=PIPE) as pip:
            installed = pip.communicate()[0].decode(self._encoding, errors="replace").splitlines()
        return "\n".join(installed)

    @property
    def _installed_conda(self) -> str:
        """ str: The list of installed Conda packages within Faceswap's scope. """
        if not self._is_conda:
            return ""
        with Popen("conda list", shell=True, stdout=PIPE, stderr=PIPE) as conda:
            stdout, stderr = conda.communicate()
        if stderr:
            return "Could not get package list"
        installed = stdout.decode(self._encoding, errors="replace").splitlines()
        return "\n".join(installed)

    @property
    def _conda_version(self) -> str:
        """ str: The installed version of Conda, or `N/A` if Conda is not installed. """
        if not self._is_conda:
            return "N/A"
        with Popen("conda --version", shell=True, stdout=PIPE, stderr=PIPE) as conda:
            stdout, stderr = conda.communicate()
        if stderr:
            return "Conda is used, but version not found"
        version = stdout.decode(self._encoding, errors="replace").splitlines()
        return "\n".join(version)

    @property
    def _git_commits(self) -> str:
        """ str: The last 5 git commits for the currently running Faceswap. """
        commits = git.get_commits(3)
        if not commits:
            return "Not Found"
        return " | ".join(commits)

    @property
    def _cuda_version(self) -> str:
        """ str: The installed CUDA version. """
        # TODO Handle multiple CUDA installs
        retval = self._cuda_check.cuda_version
        if not retval:
            retval = "No global version found"
            if self._is_conda:
                retval += ". Check Conda packages for Conda Cuda"
        return retval

    @property
    def _cudnn_version(self) -> str:
        """ str: The installed cuDNN version. """
        retval = self._cuda_check.cudnn_version
        if not retval:
            retval = "No global version found"
            if self._is_conda:
                retval += ". Check Conda packages for Conda cuDNN"
        return retval

    def _get_gpu_info(self) -> GPUInfo:
        """ Obtain GPU Stats. If an error is raised, swallow the error, and add to GPUInfo output

        Returns
        -------
        :class:`~lib.gpu_stats.GPUInfo`
            The information on connected GPUs
        """
        try:
            retval = GPUStats(log=False).sys_info
        except Exception as err:  # pylint:disable=broad-except
            err_string = f"{type(err)}: {err}"
            retval = GPUInfo(vram=[],
                             vram_free=[],
                             driver="N/A",
                             devices=[f"Error obtaining GPU Stats: '{err_string}'"],
                             devices_active=[])
        return retval

    def full_info(self) -> str:
        """ Obtain extensive system information stats, formatted into a human readable format.

        Returns
        -------
        str
            The system information for the currently running system, formatted for output to
            console or a log file.
        """
        retval = "\n============ System Information ============\n"
        sys_info = {"backend": get_backend(),
                    "os_platform": self._system["platform"],
                    "os_machine": self._system["machine"],
                    "os_release": self._system["release"],
                    "py_conda_version": self._conda_version,
                    "py_implementation": self._python["implementation"],
                    "py_version": self._python["version"],
                    "py_command": self._fs_command,
                    "py_virtual_env": self._is_virtual_env,
                    "sys_cores": self._system["cpu_count"],
                    "sys_processor": self._system["processor"],
                    "sys_ram": self._format_ram(),
                    "encoding": self._encoding,
                    "git_branch": git.branch,
                    "git_commits": self._git_commits,
                    "gpu_cuda": self._cuda_version,
                    "gpu_cudnn": self._cudnn_version,
                    "gpu_driver": self._gpu.driver,
                    "gpu_devices": ", ".join([f"GPU_{idx}: {device}"
                                              for idx, device in enumerate(self._gpu.devices)]),
                    "gpu_vram": ", ".join(
                        f"GPU_{idx}: {int(vram)}MB ({int(vram_free)}MB free)"
                        for idx, (vram, vram_free) in enumerate(zip(self._gpu.vram,
                                                                    self._gpu.vram_free))),
                    "gpu_devices_active": ", ".join([f"GPU_{idx}"
                                                     for idx in self._gpu.devices_active])}
        for key in sorted(sys_info.keys()):
            retval += (f"{key + ':':<20} {sys_info[key]}\n")
        retval += "\n=============== Pip Packages ===============\n"
        retval += self._installed_pip
        if self._is_conda:
            retval += "\n\n============== Conda Packages ==============\n"
            retval += self._installed_conda
        retval += self._state_file
        retval += "\n\n================= Configs =================="
        retval += self._configs
        return retval

    def _format_ram(self) -> str:
        """ Format the RAM stats into Megabytes to make it more readable.

        Returns
        -------
        str
            The total, available, used and free RAM displayed in Megabytes
        """
        retval = []
        for name in ("total", "available", "used", "free"):
            value = getattr(self, f"_ram_{name}")
            value = int(value / (1024 * 1024))
            retval.append(f"{name.capitalize()}: {value}MB")
        return ", ".join(retval)


def get_sysinfo() -> str:
    """ Obtain extensive system information stats, formatted into a human readable format.
    If an error occurs obtaining the system information, then the error message is returned
    instead.

    Returns
    -------
    str
        The system information for the currently running system, formatted for output to
        console or a log file.
    """
    try:
        retval = _SysInfo().full_info()
    except Exception as err:  # pylint:disable=broad-except
        retval = f"Exception occured trying to retrieve sysinfo: {str(err)}"
        raise
    return retval


class _Configs():  # pylint:disable=too-few-public-methods
    """ Parses the config files in /faceswap/config and outputs the information stored within them
    in a human readable format. """

    def __init__(self) -> None:
        self.config_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "config")
        self.configs = self._get_configs()

    def _get_configs(self) -> str:
        """ Obtain the formatted configurations from the config folder.

        Returns
        -------
        str
            The current configuration in the config files formatted in a human readable format
        """
        try:
            config_files = [os.path.join(self.config_dir, cfile)
                            for cfile in os.listdir(self.config_dir)
                            if os.path.basename(cfile) == ".faceswap"
                            or os.path.splitext(cfile)[1] == ".ini"]
            return self._parse_configs(config_files)
        except FileNotFoundError:
            return ""

    def _parse_configs(self, config_files: list[str]) -> str:
        """ Parse the given list of config files into a human readable format.

        Parameters
        ----------
        config_files: list
            A list of paths to the faceswap config files

        Returns
        -------
        str
            The current configuration in the config files formatted in a human readable format
        """
        formatted = ""
        for cfile in config_files:
            fname = os.path.basename(cfile)
            ext = os.path.splitext(cfile)[1]
            formatted += f"\n--------- {fname} ---------\n"
            if ext == ".ini":
                formatted += self._parse_ini(cfile)
            elif fname == ".faceswap":
                formatted += self._parse_json(cfile)
        return formatted

    def _parse_ini(self, config_file: str) -> str:
        """ Parse an ``.ini`` formatted config file into a human readable format.

        Parameters
        ----------
        config_file: str
            The path to the config.ini file

        Returns
        -------
        str
            The current configuration in the config file formatted in a human readable format
        """
        formatted = ""
        with open(config_file, "r", encoding="utf-8", errors="replace") as cfile:
            for line in cfile.readlines():
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                item = line.split("=")
                if len(item) == 1:
                    formatted += f"\n{item[0].strip()}\n"
                else:
                    formatted += self._format_text(item[0], item[1])
        return formatted

    def _parse_json(self, config_file: str) -> str:
        """ Parse an ``.json`` formatted config file into a formatted string.

        Parameters
        ----------
        config_file: str
            The path to the config.json file

        Returns
        -------
        dict
            The current configuration in the config file formatted as a python dictionary
        """
        formatted: str = ""
        with open(config_file, "r", encoding="utf-8", errors="replace") as cfile:
            conf_dict = json.load(cfile)
            for key in sorted(conf_dict.keys()):
                formatted += self._format_text(key, conf_dict[key])
        return formatted

    @staticmethod
    def _format_text(key: str, value: str) -> str:
        """Format a key value pair into a consistently spaced string output for display.

        Parameters
        ----------
        key: str
            The label for this display item
        value: str
            The value for this display item

        Returns
        -------
        str
            The formatted key value pair for display
        """
        return f"{key.strip() + ':':<25} {value.strip()}\n"


class _State():  # pylint:disable=too-few-public-methods
    """ Parses the state file in the current model directory, if the model is training, and
    formats the content into a human readable format. """
    def __init__(self) -> None:
        self._model_dir = self._get_arg("-m", "--model-dir")
        self._trainer = self._get_arg("-t", "--trainer")
        self.state_file = self._get_state_file()

    @property
    def _is_training(self) -> bool:
        """ bool: ``True`` if this function has been called during a training session
        otherwise ``False``. """
        return len(sys.argv) > 1 and sys.argv[1].lower() == "train"

    @staticmethod
    def _get_arg(*args: str) -> str | None:
        """ Obtain the value for a given command line option from sys.argv.

        Returns
        -------
        str or ``None``
            The value of the given command line option, if it exists, otherwise ``None``
        """
        cmd = sys.argv
        for opt in args:
            if opt in cmd:
                return cmd[cmd.index(opt) + 1]
        return None

    def _get_state_file(self) -> str:
        """ Parses the model's state file and compiles the contents into a human readable string.

        Returns
        -------
        str
            The state file formatted into a human readable format
        """
        if not self._is_training or self._model_dir is None or self._trainer is None:
            return ""
        fname = os.path.join(self._model_dir, f"{self._trainer}_state.json")
        if not os.path.isfile(fname):
            return ""

        retval = "\n\n=============== State File =================\n"
        with open(fname, "r", encoding="utf-8", errors="replace") as sfile:
            retval += sfile.read()
        return retval


sysinfo = get_sysinfo()  # pylint:disable=invalid-name
