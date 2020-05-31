#!/usr/bin/env python3
""" Install packages for faceswap.py """

# >>> ENV
import ctypes
import json
import locale
import platform
import operator
import os
import re
import sys
from subprocess import CalledProcessError, run, PIPE, Popen

from pkg_resources import parse_requirements

INSTALL_FAILED = False
# Revisions of tensorflow-gpu and cuda/cudnn requirements
TENSORFLOW_REQUIREMENTS = {"==1.12.0": ["9.0", "7.2"],
                           ">=1.13.1,<1.16": ["10.0", "7.4"]}  # TF 2.0 Not currently supported
# Mapping of Python packages to their conda names if different from pypi or in non-default channel
CONDA_MAPPING = {
    # "opencv-python": ("opencv", "conda-forge"),  # Periodic issues with conda-forge opencv
    "fastcluster": ("fastcluster", "conda-forge"),
    "toposort": ("toposort", "conda-forge"),
    "imageio-ffmpeg": ("imageio-ffmpeg", "conda-forge")}


class Environment():
    """ The current install environment """
    def __init__(self, logger=None, updater=False):
        """ logger will override built in Output() function if passed in
            updater indicates that this is being run from update_deps.py
            so certain steps can be skipped/output limited """
        self.conda_required_packages = [("tk", )]
        self.output = logger if logger else Output()
        self.updater = updater
        # Flag that setup is being run by installer so steps can be skipped
        self.is_installer = False
        self.cuda_path = ""
        self.cuda_version = ""
        self.cudnn_version = ""
        self.enable_amd = False
        self.enable_docker = False
        self.enable_cuda = False
        self.required_packages = list()
        self.missing_packages = list()
        self.conda_missing_packages = list()

        self.process_arguments()
        self.check_permission()
        self.check_system()
        self.check_python()
        self.output_runtime_info()
        self.check_pip()
        self.upgrade_pip()

        self.installed_packages = self.get_installed_packages()
        self.get_installed_conda_packages()

    @property
    def encoding(self):
        """ Get system encoding """
        return locale.getpreferredencoding()

    @property
    def os_version(self):
        """ Get OS Verion """
        return platform.system(), platform.release()

    @property
    def py_version(self):
        """ Get Python Verion """
        return platform.python_version(), platform.architecture()[0]

    @property
    def is_conda(self):
        """ Check whether using Conda """
        return ("conda" in sys.version.lower() or
                os.path.exists(os.path.join(sys.prefix, 'conda-meta')))

    @property
    def ld_library_path(self):
        """ Get the ld library path """
        return os.environ.get("LD_LIBRARY_PATH", None)

    @property
    def is_admin(self):
        """ Check whether user is admin """
        try:
            retval = os.getuid() == 0
        except AttributeError:
            retval = ctypes.windll.shell32.IsUserAnAdmin() != 0
        return retval

    @property
    def is_virtualenv(self):
        """ Check whether this is a virtual environment """
        if not self.is_conda:
            retval = (hasattr(sys, "real_prefix") or
                      (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
        else:
            prefix = os.path.dirname(sys.prefix)
            retval = (os.path.basename(prefix) == "envs")
        return retval

    def process_arguments(self):
        """ Process any cli arguments """
        for arg in sys.argv:
            if arg == "--installer":
                self.is_installer = True
            if arg == "--nvidia":
                self.enable_cuda = True
            if arg == "--amd":
                self.enable_amd = True

    def get_required_packages(self):
        """ Load requirements list """
        if self.enable_amd:
            suffix = "amd.txt"
        elif self.enable_cuda:
            suffix = "nvidia.txt"
        else:
            suffix = "cpu.txt"
        req_files = ["_requirements_base.txt", f"requirements_{suffix}"]
        pypath = os.path.dirname(os.path.realpath(__file__))
        requirements = list()
        git_requirements = list()
        for req_file in req_files:
            requirements_file = os.path.join(pypath, req_file)
            with open(requirements_file) as req:
                for package in req.readlines():
                    package = package.strip()
                    # parse_requirements can't handle git dependencies, so extract and then
                    # manually add to final list
                    if package and package.startswith("git+"):
                        git_requirements.append((package, []))
                        continue
                    if package and (not package.startswith(("#", "-r"))):
                        requirements.append(package)
        self.required_packages = [(pkg.name, pkg.specs)
                                  for pkg in parse_requirements(requirements)
                                  if pkg.marker is None or pkg.marker.evaluate()]
        self.required_packages.extend(git_requirements)

    def check_permission(self):
        """ Check for Admin permissions """
        if self.updater:
            return
        if self.is_admin:
            self.output.info("Running as Root/Admin")
        else:
            self.output.info("Running without root/admin privileges")

    def check_system(self):
        """ Check the system """
        if not self.updater:
            self.output.info("The tool provides tips for installation\n"
                             "and installs required python packages")
        self.output.info("Setup in %s %s" % (self.os_version[0], self.os_version[1]))
        if not self.updater and not self.os_version[0] in ["Windows", "Linux", "Darwin"]:
            self.output.error("Your system %s is not supported!" % self.os_version[0])
            sys.exit(1)

    def check_python(self):
        """ Check python and virtual environment status """
        self.output.info("Installed Python: {0} {1}".format(self.py_version[0],
                                                            self.py_version[1]))
        if not (self.py_version[0].split(".")[0] == "3"
                and self.py_version[0].split(".")[1] in ("3", "4", "5", "6", "7")
                and self.py_version[1] == "64bit") and not self.updater:
            self.output.error("Please run this script with Python version 3.3, 3.4, 3.5, 3.6 or "
                              "3.7 64bit and try again.")
            sys.exit(1)

    def output_runtime_info(self):
        """ Output runtime info """
        if self.is_conda:
            self.output.info("Running in Conda")
        if self.is_virtualenv:
            self.output.info("Running in a Virtual Environment")
        self.output.info("Encoding: {}".format(self.encoding))

    def check_pip(self):
        """ Check installed pip version """
        if self.updater:
            return
        try:
            import pip  # noqa pylint:disable=unused-import
        except ImportError:
            self.output.error("Import pip failed. Please Install python3-pip and try again")
            sys.exit(1)

    def upgrade_pip(self):
        """ Upgrade pip to latest version """
        if not self.is_conda:
            # Don't do this with Conda, as we must use conda's pip
            self.output.info("Upgrading pip...")
            pipexe = [sys.executable, "-m", "pip"]
            pipexe.extend(["install", "--no-cache-dir", "-qq", "--upgrade"])
            if not self.is_admin and not self.is_virtualenv:
                pipexe.append("--user")
            pipexe.append("pip")
            run(pipexe)
        import pip
        pip_version = pip.__version__
        self.output.info("Installed pip: {}".format(pip_version))

    def get_installed_packages(self):
        """ Get currently installed packages """
        installed_packages = dict()
        chk = Popen("\"{}\" -m pip freeze".format(sys.executable),
                    shell=True, stdout=PIPE)
        installed = chk.communicate()[0].decode(self.encoding).splitlines()

        for pkg in installed:
            if "==" not in pkg:
                continue
            item = pkg.split("==")
            installed_packages[item[0]] = item[1]
        return installed_packages

    def get_installed_conda_packages(self):
        """ Get currently installed conda packages """
        if not self.is_conda:
            return
        chk = os.popen("conda list").read()
        installed = [re.sub(" +", " ", line.strip())
                     for line in chk.splitlines() if not line.startswith("#")]
        for pkg in installed:
            item = pkg.split(" ")
            self.installed_packages[item[0]] = item[1]

    def update_tf_dep(self):
        """ Update Tensorflow Dependency """
        if self.is_conda or not self.enable_cuda:
            # CPU/AMD doesn't need Cuda and Conda handles Cuda and cuDNN so nothing to do here
            return

        tf_ver = None
        cudnn_inst = self.cudnn_version.split(".")
        for key, val in TENSORFLOW_REQUIREMENTS.items():
            cuda_req = val[0]
            cudnn_req = val[1].split(".")
            if cuda_req == self.cuda_version and (cudnn_req[0] == cudnn_inst[0] and
                                                  cudnn_req[1] <= cudnn_inst[1]):
                tf_ver = key
                break
        if tf_ver:
            # Remove the version of tensorflow in requirements.txt and add the correct version that
            # corresponds to the installed Cuda/cuDNN versions
            self.required_packages = [pkg for pkg in self.required_packages
                                      if not pkg.startswith("tensorflow-gpu")]
            tf_ver = "tensorflow-gpu{}".format(tf_ver)
            self.required_packages.append(tf_ver)
            return

        self.output.warning(
            "The minimum Tensorflow requirement is 1.12. \n"
            "Tensorflow currently has no official prebuild for your CUDA, cuDNN "
            "combination.\nEither install a combination that Tensorflow supports or "
            "build and install your own tensorflow-gpu.\r\n"
            "CUDA Version: {}\r\n"
            "cuDNN Version: {}\r\n"
            "Help:\n"
            "Building Tensorflow: https://www.tensorflow.org/install/install_sources\r\n"
            "Tensorflow supported versions: "
            "https://www.tensorflow.org/install/source#tested_build_configurations".format(
                self.cuda_version, self.cudnn_version))

        custom_tf = input("Location of custom tensorflow-gpu wheel (leave "
                          "blank to manually install): ")
        if not custom_tf:
            return

        custom_tf = os.path.realpath(os.path.expanduser(custom_tf))
        if not os.path.isfile(custom_tf):
            self.output.error("{} not found".format(custom_tf))
        elif os.path.splitext(custom_tf)[1] != ".whl":
            self.output.error("{} is not a valid pip wheel".format(custom_tf))
        elif custom_tf:
            self.required_packages.append(custom_tf)

    def set_config(self):
        """ Set the backend in the faceswap config file """
        if self.enable_amd:
            backend = "amd"
        elif self.enable_cuda:
            backend = "nvidia"
        else:
            backend = "cpu"
        config = {"backend": backend}
        pypath = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(pypath, "config", ".faceswap")
        with open(config_file, "w") as cnf:
            json.dump(config, cnf)
        self.output.info("Faceswap config written to: {}".format(config_file))


class Output():
    """ Format and display output """
    def __init__(self):
        self.red = "\033[31m"
        self.green = "\033[32m"
        self.yellow = "\033[33m"
        self.default_color = "\033[0m"
        self.term_support_color = platform.system() in ("Linux", "Darwin")

    @staticmethod
    def __indent_text_block(text):
        """ Indent a text block """
        lines = text.splitlines()
        if len(lines) > 1:
            out = lines[0] + "\r\n"
            for i in range(1, len(lines)-1):
                out = out + "        " + lines[i] + "\r\n"
            out = out + "        " + lines[-1]
            return out
        return text

    def info(self, text):
        """ Format INFO Text """
        trm = "INFO    "
        if self.term_support_color:
            trm = "{}INFO   {} ".format(self.green, self.default_color)
        print(trm + self.__indent_text_block(text))

    def warning(self, text):
        """ Format WARNING Text """
        trm = "WARNING "
        if self.term_support_color:
            trm = "{}WARNING{} ".format(self.yellow, self.default_color)
        print(trm + self.__indent_text_block(text))

    def error(self, text):
        """ Format ERROR Text """
        global INSTALL_FAILED  # pylint:disable=global-statement
        trm = "ERROR   "
        if self.term_support_color:
            trm = "{}ERROR  {} ".format(self.red, self.default_color)
        print(trm + self.__indent_text_block(text))
        INSTALL_FAILED = True


class Checks():
    """ Pre-installation checks """
    def __init__(self, environment):
        self.env = environment
        self.output = Output()
        self.tips = Tips()

    # Checks not required for installer
        if self.env.is_installer:
            return

    # Ask AMD/Docker/Cuda
        self.amd_ask_enable()
        if not self.env.enable_amd:
            self.docker_ask_enable()
            self.cuda_ask_enable()
        if self.env.os_version[0] != "Linux" and self.env.enable_docker and self.env.enable_cuda:
            self.docker_confirm()
        if self.env.enable_docker:
            self.docker_tips()
            self.env.set_config()
            sys.exit(0)

    # Check for CUDA and cuDNN
        if self.env.enable_cuda and self.env.is_conda:
            self.output.info("Skipping Cuda/cuDNN checks for Conda install")
        elif self.env.enable_cuda and self.env.os_version[0] in ("Linux", "Windows"):
            self.cuda_check()
            self.cudnn_check()
        elif self.env.enable_cuda and self.env.os_version[0] not in ("Linux", "Windows"):
            self.tips.macos()
            self.output.warning("Cannot find CUDA on macOS")
            self.env.cuda_version = input("Manually specify CUDA version: ")

        self.env.update_tf_dep()
        if self.env.os_version[0] == "Windows":
            self.tips.pip()

    @property
    def cuda_keys_windows(self):
        """ Return the OS Environ CUDA Keys for Windows """
        return [key for key in os.environ.keys() if key.lower().startswith("cuda_path_v")]

    def amd_ask_enable(self):
        """ Enable or disable Plaidml for AMD"""
        self.output.info("AMD Support: AMD GPU support is currently limited.\r\n"
                         "Nvidia Users MUST answer 'no' to this option.")
        i = input("Enable AMD Support? [y/N] ")
        if i in ("Y", "y"):
            self.output.info("AMD Support Enabled")
            self.env.enable_amd = True
        else:
            self.output.info("AMD Support Disabled")
            self.env.enable_amd = False

    def docker_ask_enable(self):
        """ Enable or disable Docker """
        i = input("Enable  Docker? [y/N] ")
        if i in ("Y", "y"):
            self.output.info("Docker Enabled")
            self.env.enable_docker = True
        else:
            self.output.info("Docker Disabled")
            self.env.enable_docker = False

    def docker_confirm(self):
        """ Warn if nvidia-docker on non-linux system """
        self.output.warning("Nvidia-Docker is only supported on Linux.\r\n"
                            "Only CPU is supported in Docker for your system")
        self.docker_ask_enable()
        if self.env.enable_docker:
            self.output.warning("CUDA Disabled")
            self.env.enable_cuda = False

    def docker_tips(self):
        """ Provide tips for Docker use """
        if not self.env.enable_cuda:
            self.tips.docker_no_cuda()
        else:
            self.tips.docker_cuda()

    def cuda_ask_enable(self):
        """ Enable or disable CUDA """
        i = input("Enable  CUDA? [Y/n] ")
        if i in ("", "Y", "y"):
            self.output.info("CUDA Enabled")
            self.env.enable_cuda = True
        else:
            self.output.info("CUDA Disabled")
            self.env.enable_cuda = False

    def cuda_check(self):
        """ Check Cuda for Linux or Windows """
        chk = Popen("nvcc -V", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = chk.communicate()
        if not stderr:
            version = re.search(r".*release (?P<cuda>\d+\.\d+)", stdout.decode(self.env.encoding))
            self.env.cuda_version = version.groupdict().get("cuda", None)
            if self.env.cuda_version:
                self.output.info("CUDA version: " + self.env.cuda_version)
                return
        # Failed to load nvcc
        if self.env.os_version[0] == "Linux":
            self.cuda_check_linux()
        elif self.env.os_version[0] == "Windows":
            self.cuda_check_windows()

    def cuda_check_linux(self):
        """ Check Linux CUDA Version """
        chk = os.popen("ldconfig -p | grep -P \"libcudart.so.\\d+.\\d+\" | head -n 1").read()
        if self.env.ld_library_path and not chk:
            paths = self.env.ld_library_path.split(":")
            for path in paths:
                chk = os.popen("ls {} | grep -P -o \"libcudart.so.\\d+.\\d+\" | "
                               "head -n 1".format(path)).read()
                if chk:
                    break
        if not chk:
            self.output.error("CUDA not found. Install and try again.\n"
                              "Recommended version:      CUDA 9.0     cuDNN 7.1.3\n"
                              "CUDA: https://developer.nvidia.com/cuda-downloads\n"
                              "cuDNN: https://developer.nvidia.com/rdp/cudnn-download")
            return
        cudavers = chk.strip().replace("libcudart.so.", "")
        self.env.cuda_version = cudavers[:cudavers.find(" ")]
        if self.env.cuda_version:
            self.output.info("CUDA version: " + self.env.cuda_version)
            self.env.cuda_path = chk[chk.find("=>") + 3:chk.find("targets") - 1]

    def cuda_check_windows(self):
        """ Check Windows CUDA Version """
        cuda_keys = self.cuda_keys_windows
        if not cuda_keys:
            self.output.error("CUDA not found. See "
                              "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#cuda "
                              "for instructions")
            return

        self.env.cuda_version = cuda_keys[0].lower().replace("cuda_path_v", "").replace("_", ".")
        self.env.cuda_path = os.environ[cuda_keys[0]]
        self.output.info("CUDA version: " + self.env.cuda_version)

    def cudnn_check(self):
        """ Check Linux or Windows cuDNN Version from cudnn.h """
        if self.env.os_version[0] == "Linux":
            cudnn_checkfiles = self.cudnn_checkfiles_linux()
        elif self.env.os_version[0] == "Windows":
            if not self.env.cuda_path and not self.cuda_keys_windows:
                self.output.error(
                    "CUDA not found. See "
                    "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#cuda "
                    "for instructions")
                return
            if not self.env.cuda_path:
                self.env.cuda_path = os.environ[self.cuda_keys_windows[0]]
            cudnn_checkfiles = self.cudnn_checkfiles_windows()

        cudnn_checkfile = None
        for checkfile in cudnn_checkfiles:
            if os.path.isfile(checkfile):
                cudnn_checkfile = checkfile
                break
        if not cudnn_checkfile:
            self.output.error("cuDNN not found. See "
                              "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#cudnn "
                              "for instructions")
            return
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
            self.output.error("cuDNN version could not be determined. See "
                              "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#cudnn "
                              "for instructions")
            return

        self.env.cudnn_version = "{}.{}".format(major, minor)
        self.output.info("cuDNN version: {}.{}".format(self.env.cudnn_version, patchlevel))

    @staticmethod
    def cudnn_checkfiles_linux():
        """ Return the checkfile locations for linux """
        chk = os.popen("ldconfig -p | grep -P \"libcudnn.so.\\d+\" | head -n 1").read()
        chk = chk.strip().replace("libcudnn.so.", "")
        if not chk:
            return list()
        cudnn_vers = chk[0]
        cudnn_path = chk[chk.find("=>") + 3:chk.find("libcudnn") - 1]
        cudnn_path = cudnn_path.replace("lib", "include")
        cudnn_checkfiles = [os.path.join(cudnn_path, "cudnn_v{}.h".format(cudnn_vers)),
                            os.path.join(cudnn_path, "cudnn.h")]
        return cudnn_checkfiles

    def cudnn_checkfiles_windows(self):
        """ Return the checkfile locations for windows """
        # TODO A more reliable way of getting the windows location
        if not self.env.cuda_path:
            return list()
        cudnn_checkfile = os.path.join(self.env.cuda_path, "include", "cudnn.h")
        return [cudnn_checkfile]


class Install():
    """ Install the requirements """
    def __init__(self, environment):
        self._operators = {"==": operator.eq,
                           ">=": operator.ge,
                           "<=": operator.le,
                           ">": operator.gt,
                           "<": operator.lt}
        self.output = environment.output
        self.env = environment

        if not self.env.is_installer and not self.env.updater:
            self.ask_continue()
        self.env.get_required_packages()
        self.check_missing_dep()
        self.check_conda_missing_dep()
        if (self.env.updater and
                not self.env.missing_packages and not self.env.conda_missing_packages):
            self.output.info("All Dependencies are up to date")
            return
        self.install_missing_dep()
        if self.env.updater:
            return
        self.output.info("All python3 dependencies are met.\r\nYou are good to go.\r\n\r\n"
                         "Enter:  'python faceswap.py -h' to see the options\r\n"
                         "        'python faceswap.py gui' to launch the GUI")

    def ask_continue(self):
        """ Ask Continue with Install """
        inp = input("Please ensure your System Dependencies are met. Continue? [y/N] ")
        if inp in ("", "N", "n"):
            self.output.error("Please install system dependencies to continue")
            sys.exit(1)

    def check_missing_dep(self):
        """ Check for missing dependencies """
        for key, specs in self.env.required_packages:
            if self.env.is_conda:
                # Get Conda alias for Key
                key = CONDA_MAPPING.get(key, (key, None))[0]
            if (key == "git+https://github.com/deepfakes/nvidia-ml-py3.git" and
                    self.env.installed_packages.get("nvidia-ml-py3", "") == "7.352.1"):
                # Annoying explicit hack to get around our custom version of nvidia-ml=py3 being
                # constantly re-downloaded
                continue
            if key not in self.env.installed_packages:
                self.env.missing_packages.append((key, specs))
                continue
            installed_vers = self.env.installed_packages.get(key, "")
            if specs and not all(self._operators[spec[0]](installed_vers, spec[1])
                                 for spec in specs):
                self.env.missing_packages.append((key, specs))

    def check_conda_missing_dep(self):
        """ Check for conda missing dependencies """
        if not self.env.is_conda:
            return
        for pkg in self.env.conda_required_packages:
            key = pkg[0].split("==")[0]
            if key not in self.env.installed_packages:
                self.env.conda_missing_packages.append(pkg)
                continue
            if len(pkg[0].split("==")) > 1:
                if pkg[0].split("==")[1] != self.env.installed_conda_packages.get(key):
                    self.env.conda_missing_packages.append(pkg)
                    continue

    def install_missing_dep(self):
        """ Install missing dependencies """
        # Install conda packages first
        if self.env.conda_missing_packages:
            self.install_conda_packages()
        if self.env.missing_packages:
            self.install_python_packages()

    def install_python_packages(self):
        """ Install required pip packages """
        self.output.info("Installing Required Python Packages. This may take some time...")
        for pkg, version in self.env.missing_packages:
            if version:
                pkg = "{}{}".format(pkg, ",".join("".join(spec) for spec in version))
            if self.env.is_conda and not pkg.startswith("git"):
                verbose = pkg.startswith("tensorflow") or self.env.updater
                pkg = CONDA_MAPPING.get(pkg, (pkg, None))
                channel = None if len(pkg) != 2 else pkg[1]
                pkg = pkg[0]
                if self.conda_installer(pkg, verbose=verbose, channel=channel, conda_only=False):
                    continue
            self.pip_installer(pkg)

    def install_conda_packages(self):
        """ Install required conda packages """
        self.output.info("Installing Required Conda Packages. This may take some time...")
        for pkg in self.env.conda_missing_packages:
            channel = None if len(pkg) != 2 else pkg[1]
            self.conda_installer(pkg[0], channel=channel, conda_only=True)

    def conda_installer(self, package, channel=None, verbose=False, conda_only=False):
        """ Install a conda package """
        #  Packages with special characters need to be enclosed in double quotes
        if any(char in package for char in (" ", "<", ">", "*", "|")):
            package = "\"{}\"".format(package)
        success = True
        condaexe = ["conda", "install", "-y"]
        if not verbose or self.env.updater:
            condaexe.append("-q")
        if channel:
            condaexe.extend(["-c", channel])
        condaexe.append(package)
        self.output.info("Installing {}".format(package))
        shell = self.env.os_version[0] == "Windows"
        try:
            if verbose:
                run(condaexe, check=True, shell=shell)
            else:
                with open(os.devnull, "w") as devnull:
                    run(condaexe, stdout=devnull, stderr=devnull, check=True, shell=shell)
        except CalledProcessError:
            if not conda_only:
                self.output.info("{} not available in Conda. Installing with pip".format(package))
            else:
                self.output.warning("Couldn't install {} with Conda. "
                                    "Please install this package manually".format(package))
            success = False
        return success

    def pip_installer(self, package):
        """ Install a pip package """
        pipexe = [sys.executable, "-m", "pip"]
        # hide info/warning and fix cache hang
        pipexe.extend(["install", "--no-cache-dir"])
        if not self.env.updater:
            pipexe.append("-qq")
        # install as user to solve perm restriction
        if not self.env.is_admin and not self.env.is_virtualenv:
            pipexe.append("--user")
        msg = "Installing {}".format(package)
        self.output.info(msg)
        pipexe.append(package)
        try:
            run(pipexe, check=True)
        except CalledProcessError:
            self.output.warning("Couldn't install {} with pip. "
                                "Please install this package manually".format(package))


class Tips():
    """ Display installation Tips """
    def __init__(self):
        self.output = Output()

    def docker_no_cuda(self):
        """ Output Tips for Docker without Cuda """
        self.output.info(
            "1. Install Docker\n"
            "https://www.docker.com/community-edition\n\n"
            "2. Build Docker Image For Faceswap\n"
            "docker build -t deepfakes-cpu -f Dockerfile.cpu .\n\n"
            "3. Mount faceswap volume and Run it\n"
            "# without GUI\n"
            "docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-cpu --name deepfakes-cpu \\ \n"
            "\t-v {path}:/srv \\ \n"
            "\tdeepfakes-cpu\n\n"
            "# with gui. tools.py gui working.\n"
            "## enable local access to X11 server\n"
            "xhost +local:\n"
            "## create container\n"
            "nvidia-docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-cpu --name deepfakes-cpu \\ \n"
            "\t-v {path}:/srv \\ \n"
            "\t-v /tmp/.X11-unix:/tmp/.X11-unix \\ \n"
            "\t-e DISPLAY=unix$DISPLAY \\ \n"
            "\t-e AUDIO_GID=`getent group audio | cut -d: -f3` \\ \n"
            "\t-e VIDEO_GID=`getent group video | cut -d: -f3` \\ \n"
            "\t-e GID=`id -g` \\ \n"
            "\t-e UID=`id -u` \\ \n"
            "\tdeepfakes-cpu \n\n"
            "4. Open a new terminal to run faceswap.py in /srv\n"
            "docker exec -it deepfakes-cpu bash".format(
                path=os.path.dirname(os.path.realpath(__file__))))
        self.output.info("That's all you need to do with a docker. Have fun.")

    def docker_cuda(self):
        """ Output Tips for Docker wit Cuda"""
        self.output.info(
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
            "\t-v {path}:/srv \\ \n"
            "\tdeepfakes-gpu\n\n"
            "# with gui.\n"
            "## enable local access to X11 server\n"
            "xhost +local:\n"
            "## enable nvidia device if working under bumblebee\n"
            "echo ON > /proc/acpi/bbswitch\n"
            "## create container\n"
            "nvidia-docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-gpu --name deepfakes-gpu \\ \n"
            "\t-v {path}:/srv \\ \n"
            "\t-v /tmp/.X11-unix:/tmp/.X11-unix \\ \n"
            "\t-e DISPLAY=unix$DISPLAY \\ \n"
            "\t-e AUDIO_GID=`getent group audio | cut -d: -f3` \\ \n"
            "\t-e VIDEO_GID=`getent group video | cut -d: -f3` \\ \n"
            "\t-e GID=`id -g` \\ \n"
            "\t-e UID=`id -u` \\ \n"
            "\tdeepfakes-gpu\n\n"
            "6. Open a new terminal to interact with the project\n"
            "docker exec deepfakes-gpu python /srv/faceswap.py gui\n".format(
                path=os.path.dirname(os.path.realpath(__file__))))

    def macos(self):
        """ Output Tips for macOS"""
        self.output.info(
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

    def pip(self):
        """ Pip Tips """
        self.output.info("1. Install PIP requirements\n"
                         "You may want to execute `chcp 65001` in cmd line\n"
                         "to fix Unicode issues on Windows when installing dependencies")


if __name__ == "__main__":
    ENV = Environment()
    Checks(ENV)
    ENV.set_config()
    if INSTALL_FAILED:
        sys.exit(1)
    Install(ENV)
