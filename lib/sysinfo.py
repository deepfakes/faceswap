#!/usr/bin python3
""" Obtain information about the running system, environment and gpu """

import json
import locale
import os
import platform
import re
import sys
from subprocess import PIPE, Popen

import psutil

from lib.gpu_stats import GPUStats


class SysInfo():
    """ System and Python Information """
    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    def __init__(self):
        gpu_stats = GPUStats(log=False)
        self.state_file = State().state_file
        self.configs = Configs().configs
        self.platform = platform.platform()
        self.system = platform.system()
        self.machine = platform.machine()
        self.release = platform.release()
        self.processor = platform.processor()
        self.cpu_count = os.cpu_count()
        self.py_implementation = platform.python_implementation()
        self.py_version = platform.python_version()
        self._cuda_path = self.get_cuda_path()
        self.vram = gpu_stats.vram
        self.gfx_driver = gpu_stats.driver
        self.gfx_devices = gpu_stats.devices
        self.gfx_devices_active = gpu_stats.active_devices

    @property
    def encoding(self):
        """ Return system preferred encoding """
        return locale.getpreferredencoding()

    @property
    def is_conda(self):
        """ Boolean for whether in a conda environment """
        return ("conda" in sys.version.lower() or
                os.path.exists(os.path.join(sys.prefix, 'conda-meta')))

    @property
    def is_linux(self):
        """ Boolean for whether system is Linux """
        return self.system.lower() == "linux"

    @property
    def is_macos(self):
        """ Boolean for whether system is macOS """
        return self.system.lower() == "darwin"

    @property
    def is_windows(self):
        """ Boolean for whether system is Windows """
        return self.system.lower() == "windows"

    @property
    def is_virtual_env(self):
        """ Boolean for whether running in a virtual environment """
        if not self.is_conda:
            retval = (hasattr(sys, "real_prefix") or
                      (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
        else:
            prefix = os.path.dirname(sys.prefix)
            retval = (os.path.basename(prefix) == "envs")
        return retval

    @property
    def ram(self):
        """ Return RAM stats """
        return psutil.virtual_memory()

    @property
    def ram_free(self):
        """ return free RAM """
        return getattr(self.ram, "free")

    @property
    def ram_total(self):
        """ return total RAM """
        return getattr(self.ram, "total")

    @property
    def ram_available(self):
        """ return available RAM """
        return getattr(self.ram, "available")

    @property
    def ram_used(self):
        """ return used RAM """
        return getattr(self.ram, "used")

    @property
    def fs_command(self):
        """ Return the executed faceswap command """
        return " ".join(sys.argv)

    @property
    def installed_pip(self):
        """ Installed pip packages """
        pip = Popen("{} -m pip freeze".format(sys.executable),
                    shell=True, stdout=PIPE)
        installed = pip.communicate()[0].decode().splitlines()
        return "\n".join(installed)

    @property
    def installed_conda(self):
        """ Installed Conda packages """
        if not self.is_conda:
            return None
        conda = Popen("conda list", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = conda.communicate()
        if stderr:
            return "Could not get package list"
        installed = stdout.decode().splitlines()
        return "\n".join(installed)

    @property
    def conda_version(self):
        """ Get conda version """
        if not self.is_conda:
            return "N/A"
        conda = Popen("conda --version", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = conda.communicate()
        if stderr:
            return "Conda is used, but version not found"
        version = stdout.decode().splitlines()
        return "\n".join(version)

    @property
    def git_branch(self):
        """ Get the current git branch """
        git = Popen("git status", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = git.communicate()
        if stderr:
            return "Not Found"
        branch = stdout.decode().splitlines()[0].replace("On branch ", "")
        return branch

    @property
    def git_commits(self):
        """ Get last 5 git commits """
        git = Popen("git log --pretty=oneline --abbrev-commit -n 5",
                    shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = git.communicate()
        if stderr:
            return "Not Found"
        commits = stdout.decode().splitlines()
        return ". ".join(commits)

    @property
    def cuda_keys_windows(self):
        """ Return the OS Environ CUDA Keys for Windows """
        return [key for key in os.environ.keys() if key.lower().startswith("cuda_path_v")]

    @property
    def cuda_version(self):
        """ Get the installed CUDA version """
        chk = Popen("nvcc -V", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = chk.communicate()
        if not stderr:
            version = re.search(r".*release (?P<cuda>\d+\.\d+)", stdout.decode(self.encoding))
            version = version.groupdict().get("cuda", None)
            if version:
                return version
        # Failed to load nvcc
        if self.is_linux:
            version = self.cuda_version_linux()
        elif self.is_windows:
            version = self.cuda_version_windows()
        else:
            version = "Unsupported OS"
            if self.is_conda:
                version += ". Check Conda packages for Conda Cuda"
        return version

    @property
    def cudnn_version(self):
        """ Get the installed cuDNN version """
        if self.is_linux:
            cudnn_checkfiles = self.cudnn_checkfiles_linux()
        elif self.is_windows:
            cudnn_checkfiles = self.cudnn_checkfiles_windows()
        else:
            retval = "Unsupported OS"
            if self.is_conda:
                retval += ". Check Conda packages for Conda cuDNN"
            return retval

        cudnn_checkfile = None
        for checkfile in cudnn_checkfiles:
            if os.path.isfile(checkfile):
                cudnn_checkfile = checkfile
                break

        if not cudnn_checkfile:
            retval = "No global version found"
            if self.is_conda:
                retval += ". Check Conda packages for Conda cuDNN"
            return retval

        found = 0
        with open(cudnn_checkfile, "r") as ofile:
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
        if found != 3:
            retval = "No global version found"
            if self.is_conda:
                retval += ". Check Conda packages for Conda cuDNN"
            return retval
        return "{}.{}.{}".format(major, minor, patchlevel)

    @staticmethod
    def cudnn_checkfiles_linux():
        """ Return the checkfile locations for linux """
        chk = os.popen("ldconfig -p | grep -P \"libcudnn.so.\\d+\" | head -n 1").read()
        if "libcudnn.so." not in chk:
            return list()
        chk = chk.strip().replace("libcudnn.so.", "")
        cudnn_vers = chk[0]
        cudnn_path = chk[chk.find("=>") + 3:chk.find("libcudnn") - 1]
        cudnn_path = cudnn_path.replace("lib", "include")
        cudnn_checkfiles = [os.path.join(cudnn_path, "cudnn_v{}.h".format(cudnn_vers)),
                            os.path.join(cudnn_path, "cudnn.h")]
        return cudnn_checkfiles

    def cudnn_checkfiles_windows(self):
        """ Return the checkfile locations for windows """
        # TODO A more reliable way of getting the windows location
        if not self._cuda_path and not self.cuda_keys_windows:
            return list()
        if not self._cuda_path:
            self._cuda_path = os.environ[self.cuda_keys_windows[0]]

        cudnn_checkfile = os.path.join(self._cuda_path, "include", "cudnn.h")
        return [cudnn_checkfile]

    def get_cuda_path(self):
        """ Return the correct CUDA Path """
        if self.is_linux:
            path = self.cuda_path_linux()
        elif self.is_windows:
            path = self.cuda_path_windows()
        else:
            path = None
        return path

    @staticmethod
    def cuda_path_linux():
        """ Get the path to Cuda on linux systems """
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", None)
        chk = os.popen("ldconfig -p | grep -P \"libcudart.so.\\d+.\\d+\" | head -n 1").read()
        if ld_library_path and not chk:
            paths = ld_library_path.split(":")
            for path in paths:
                chk = os.popen("ls {} | grep -P -o \"libcudart.so.\\d+.\\d+\" | "
                               "head -n 1".format(path)).read()
                if chk:
                    break
        if not chk:
            return None
        return chk[chk.find("=>") + 3:chk.find("targets") - 1]

    @staticmethod
    def cuda_path_windows():
        """ Get the path to Cuda on Windows systems """
        cuda_path = os.environ.get("CUDA_PATH", None)
        return cuda_path

    def cuda_version_linux(self):
        """ Get CUDA version for linux systems """
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", None)
        chk = os.popen("ldconfig -p | grep -P \"libcudart.so.\\d+.\\d+\" | head -n 1").read()
        if ld_library_path and not chk:
            paths = ld_library_path.split(":")
            for path in paths:
                chk = os.popen("ls {} | grep -P -o \"libcudart.so.\\d+.\\d+\" | "
                               "head -n 1".format(path)).read()
                if chk:
                    break
        if not chk:
            retval = "No global version found"
            if self.is_conda:
                retval += ". Check Conda packages for Conda Cuda"
            return retval
        cudavers = chk.strip().replace("libcudart.so.", "")
        return cudavers[:cudavers.find(" ")]

    def cuda_version_windows(self):
        """ Get CUDA version for Windows systems """
        cuda_keys = self.cuda_keys_windows
        if not cuda_keys:
            retval = "No global version found"
            if self.is_conda:
                retval += ". Check Conda packages for Conda Cuda"
            return retval
        cudavers = [key.lower().replace("cuda_path_v", "").replace("_", ".") for key in cuda_keys]
        return " ".join(cudavers)

    def full_info(self):
        """ Format system info human readable """
        retval = "\n============ System Information ============\n"
        sys_info = {"os_platform": self.platform,
                    "os_machine": self.machine,
                    "os_release": self.release,
                    "py_conda_version": self.conda_version,
                    "py_implementation": self.py_implementation,
                    "py_version": self.py_version,
                    "py_command": self.fs_command,
                    "py_virtual_env": self.is_virtual_env,
                    "sys_cores": self.cpu_count,
                    "sys_processor": self.processor,
                    "sys_ram": self.format_ram(),
                    "encoding": self.encoding,
                    "git_branch": self.git_branch,
                    "git_commits": self.git_commits,
                    "gpu_cuda": self.cuda_version,
                    "gpu_cudnn": self.cudnn_version,
                    "gpu_driver": self.gfx_driver,
                    "gpu_devices": ", ".join(["GPU_{}: {}".format(idx, device)
                                              for idx, device in enumerate(self.gfx_devices)]),
                    "gpu_vram": ", ".join(["GPU_{}: {}MB".format(idx, int(vram))
                                           for idx, vram in enumerate(self.vram)]),
                    "gpu_devices_active": ", ".join(["GPU_{}".format(idx)
                                                     for idx in self.gfx_devices_active])}
        for key in sorted(sys_info.keys()):
            retval += ("{0: <20} {1}\n".format(key + ":", sys_info[key]))
        retval += "\n=============== Pip Packages ===============\n"
        retval += self.installed_pip
        if self.is_conda:
            retval += "\n\n============== Conda Packages ==============\n"
            retval += self.installed_conda
        retval += self.state_file
        retval += "\n\n================= Configs =================="
        retval += self.configs
        return retval

    def format_ram(self):
        """ Format the RAM stats for human output """
        retval = list()
        for name in ("total", "available", "used", "free"):
            value = getattr(self, "ram_{}".format(name))
            value = int(value / (1024 * 1024))
            retval.append("{}: {}MB".format(name.capitalize(), value))
        return ", ".join(retval)


def get_sysinfo():
    """ Return sys info or error message if there is an error """
    try:
        retval = SysInfo().full_info()
    except Exception as err:  # pylint: disable=broad-except
        retval = "Exception occured trying to retrieve sysinfo: {}".format(err)
    return retval


class Configs():
    """ Parses the config files in /config and outputs the information """

    def __init__(self):
        self.config_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "config")
        self.configs = self.get_configs()

    def get_configs(self):
        """ Return the configs from the config dir """
        config_files = [os.path.join(self.config_dir, cfile)
                        for cfile in os.listdir(self.config_dir)
                        if os.path.basename(cfile) == ".faceswap"
                        or os.path.splitext(cfile)[1] == ".ini"]
        return self.parse_configs(config_files)

    def parse_configs(self, config_files):
        """ Parse the config files into the output format """
        formatted = ""
        for cfile in config_files:
            fname = os.path.basename(cfile)
            ext = os.path.splitext(cfile)[1]
            formatted += "\n--------- {} ---------\n".format(fname)
            if ext == ".ini":
                formatted += self.parse_ini(cfile)
            elif fname == ".faceswap":
                formatted += self.parse_json(cfile)
        return formatted

    def parse_ini(self, config_file):
        """ Parse an INI file converting it to a dict """
        formatted = ""
        with open(config_file, "r") as cfile:
            for line in cfile.readlines():
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                item = line.split("=")
                if len(item) == 1:
                    formatted += "\n{}\n".format(item[0].strip())
                else:
                    formatted += self.format_text(item[0], item[1])
        return formatted

    def parse_json(self, config_file):
        """ Parse a Json File converting it to a dict """
        formatted = ""
        with open(config_file, "r") as cfile:
            conf_dict = json.load(cfile)
            for key in sorted(conf_dict.keys()):
                formatted += self.format_text(key, conf_dict[key])
        return formatted

    @staticmethod
    def format_text(key, val):
        """Format the text for output """
        return "{0: <25} {1}\n".format(key.strip() + ":", val.strip())


class State():
    """ State file for training command """
    def __init__(self):
        self.model_dir = self.get_arg("-m", "--model-dir")
        self.trainer = self.get_arg("-t", "--trainer")
        self.state_file = self.get_state_file()

    @property
    def is_training(self):
        """ Return whether this has been called during training """
        return len(sys.argv) > 1 and sys.argv[1].lower() == "train"

    @staticmethod
    def get_arg(*args):
        """ Return the value for a given option from sys.argv. Returns None if not found """
        cmd = sys.argv
        for opt in args:
            if opt in cmd:
                return cmd[cmd.index(opt) + 1]
        return None

    def get_state_file(self):
        """ Return the state file in a string """
        if not self.is_training or self.model_dir is None or self.trainer is None:
            return ""
        fname = os.path.join(self.model_dir, "{}_state.json".format(self.trainer))
        if not os.path.isfile(fname):
            return ""

        retval = "\n\n=============== State File =================\n"
        with open(fname, "r") as sfile:
            retval += sfile.read()
        return retval


sysinfo = get_sysinfo()  # pylint: disable=invalid-name
