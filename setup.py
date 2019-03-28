#!/usr/bin/env python3
""" Install packages for faceswap.py """

# >>> ENV
import ctypes
import locale
import os
import re
import sys
import platform

from subprocess import CalledProcessError, run, PIPE, Popen

INSTALL_FAILED = False
# Revisions of tensorflow-gpu and cuda/cudnn requirements
TENSORFLOW_REQUIREMENTS = {"==1.12.0": ["9.0", "7.2"],
                           ">=1.13.1": ["10.0", "7.4"]}


class Environment():
    """ The current install environment """
    def __init__(self):
        self.macos_required_packages = ["pynvx==0.0.4"]
        self.conda_required_packages = [("ffmpeg", "conda-forge"), ("tk", )]
        self.output = Output()
        # Flag that setup is being run by installer so steps can be skipped
        self.is_installer = False
        self.cuda_path = ""
        self.cuda_version = ""
        self.cudnn_version = ""
        self.enable_docker = False
        self.enable_cuda = False
        self.required_packages = self.get_required_packages()
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
    def is_macos(self):
        """ Check whether MacOS """
        return bool(platform.system() == "Darwin")

    @property
    def is_conda(self):
        """ Check whether using Conda """
        return bool("conda" in sys.version.lower())

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
        argv = [arg for arg in sys.argv]
        for arg in argv:
            if arg == "--installer":
                self.is_installer = True
            if arg == "--gpu":
                self.enable_cuda = True

    @staticmethod
    def get_required_packages():
        """ Load requirements list """
        packages = list()
        pypath = os.path.dirname(os.path.realpath(__file__))
        requirements_file = os.path.join(pypath, "requirements.txt")
        with open(requirements_file) as req:
            for package in req.readlines():
                package = package.strip()
                if package and (not package.startswith("#")):
                    packages.append(package)
        return packages

    def check_permission(self):
        """ Check for Admin permissions """
        if self.is_admin:
            self.output.info("Running as Root/Admin")
        else:
            self.output.warning("Running without root/admin privileges")

    def check_system(self):
        """ Check the system """
        self.output.info("The tool provides tips for installation\n"
                         "and installs required python packages")
        self.output.info("Setup in %s %s" % (self.os_version[0], self.os_version[1]))
        if not self.os_version[0] in ["Windows", "Linux", "Darwin"]:
            self.output.error("Your system %s is not supported!" % self.os_version[0])
            exit(1)

    def check_python(self):
        """ Check python and virtual environment status """
        self.output.info("Installed Python: {0} {1}".format(self.py_version[0],
                                                            self.py_version[1]))
        if not (self.py_version[0].split(".")[0] == "3"
                and self.py_version[0].split(".")[1] in ("3", "4", "5", "6")
                and self.py_version[1] == "64bit"):
            self.output.error("Please run this script with Python version 3.3, 3.4, 3.5 or 3.6 "
                              "64bit and try again.")
            exit(1)

    def output_runtime_info(self):
        """ Output runtime info """
        if self.is_conda:
            self.output.info("Running in Conda")
        if self.is_virtualenv:
            self.output.info("Running in a Virtual Environment")
        self.output.info("Encoding: {}".format(self.encoding))

    def check_pip(self):
        """ Check installed pip version """
        try:
            import pip
        except ImportError:
            self.output.error("Import pip failed. Please Install python3-pip and try again")
            exit(1)

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
        if self.is_conda:
            self.update_tf_dep_conda()
            return

        if not self.enable_cuda:
            self.required_packages.append("tensorflow")
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

    def update_tf_dep_conda(self):
        """ Update Conda TF Dependency """
        if not self.enable_cuda:
            self.required_packages.append("tensorflow==1.12.0")
        else:
            self.required_packages.append("tensorflow-gpu==1.12.0")


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
        global INSTALL_FAILED
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
            self.env.update_tf_dep()
            return

    # Ask Docker/Cuda
        self.docker_ask_enable()
        self.cuda_ask_enable()
        if self.env.os_version[0] != "Linux" and self.env.enable_docker and self.env.enable_cuda:
            self.docker_confirm()
        if self.env.enable_docker:
            self.docker_tips()
            exit(0)

    # Check for CUDA and cuDNN
        if self.env.enable_cuda and self.env.os_version[0] in ("Linux", "Windows"):
            self.cuda_check()
            self.cudnn_check()
        elif self.env.enable_cuda and self.env.os_version[0] not in ("Linux", "Windows"):
            self.tips.macos()
            self.output.warning("Cannot find CUDA on macOS")
            self.env.cuda_version = input("Manually specify CUDA version: ")

        self.env.update_tf_dep()
        self.check_system_dependencies()
        if self.env.os_version[0] == "Windows":
            self.tips.pip()

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
        cuda_keys = [key
                     for key in os.environ.keys()
                     if key.lower().startswith("cuda_path_v")]
        if not cuda_keys:
            self.output.error("CUDA not found. See "
                              "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#cuda "
                              "for instructions")
            return

        self.env.cuda_version = cuda_keys[0].replace("CUDA_PATH_V", "").replace("_", ".")
        self.env.cuda_path = os.environ[cuda_keys[0]]
        self.output.info("CUDA version: " + self.env.cuda_version)

    def cudnn_check(self):
        """ Check Linux or Windows cuDNN Version from cudnn.h """
        cudnn_checkfile = os.path.join(self.env.cuda_path, "include", "cudnn.h")
        if not os.path.isfile(cudnn_checkfile):
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

    def check_system_dependencies(self):
        """ Check that system applications are installed """
        self.output.info("Checking System Dependencies...")
        self.cmake_check()
        if self.env.os_version[0] == "Windows":
            self.visual_studio_check()
            self.check_cplus_plus()
        if self.env.os_version[0] == "Linux":
            self.gcc_check()
            self.gpp_check()

    def gcc_check(self):
        """ Check installed gcc version for linux """
        chk = Popen("gcc --version", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = chk.communicate()
        if stderr:
            self.output.error("gcc not installed. Please install gcc for your distribution")
            return
        gcc = [re.sub(" +", " ", line.strip())
               for line in stdout.decode(self.env.encoding).splitlines()
               if line.lower().strip().startswith("gcc")][0]
        version = gcc[gcc.rfind(" ") + 1:]
        self.output.info("gcc version: {}".format(version))

    def gpp_check(self):
        """ Check installed g++ version for linux """
        chk = Popen("g++ --version", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = chk.communicate()
        if stderr:
            self.output.error("g++ not installed. Please install g++ for your distribution")
            return
        gpp = [re.sub(" +", " ", line.strip())
               for line in stdout.decode(self.env.encoding).splitlines()
               if line.lower().strip().startswith("g++")][0]
        version = gpp[gpp.rfind(" ") + 1:]
        self.output.info("g++ version: {}".format(version))

    def cmake_check(self):
        """ Check CMake is installed """
        chk = Popen("cmake --version", shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = chk.communicate()
        stdout = stdout.decode(self.env.encoding)
        if stderr and self.env.os_version[0] == "Windows":
            stdout, stderr = self.cmake_check_windows()
        if stderr:
            self.output.error("CMake could not be found. See "
                              "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#cmake "
                              "for instructions")
            return
        cmake = [re.sub(" +", " ", line.strip())
                 for line in stdout.splitlines()
                 if line.lower().strip().startswith("cmake")][0]
        version = cmake[cmake.rfind(" ") + 1:]
        self.output.info("CMake version: {}".format(version))

    def cmake_check_windows(self):
        """ Additional checks for cmake on Windows """
        chk = Popen("wmic product where \"name = 'cmake'\" get installlocation,version",
                    shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = chk.communicate()
        if stderr:
            return False, stderr
        lines = [re.sub(" +", " ", line.strip())
                 for line in stdout.decode(self.env.encoding).splitlines()
                 if line.strip()]
        stdout = lines[1]
        location = stdout[:stdout.rfind(" ")] + "bin"
        self.output.info("CMake not found in %PATH%. Temporarily adding: \"{}\"".format(location))
        os.environ["PATH"] += ";{}".format(location)
        stdout = "cmake {}".format(stdout)
        return stdout, False

    def visual_studio_check(self):
        """ Check Visual Studio 2015 is installed for Windows

            Somewhat hacky solution which checks for the existence
            of the VS2015 Performance Report
        """
        chk = Popen("reg query HKLM\\SOFTWARE\\Microsoft\\VisualStudio\\14.0\\VSPerf",
                    shell=True, stdout=PIPE, stderr=PIPE)
        _, stderr = chk.communicate()
        if stderr:
            self.output.error("Visual Studio 2015 could not be found. See "
                              "https://github.com/deepfakes/faceswap/blob/master/"
                              "INSTALL.md#microsoft-visual-studio-2015 for instructions")
            return
        self.output.info("Visual Studio 2015 version: 14.0")

    def check_cplus_plus(self):
        """ Check Visual C++ Redistributable 2015 is instlled for Windows """
        keys = (
            "HKLM\\SOFTWARE\\Classes\\Installer\\Dependencies\\{d992c12e-cab2-426f-bde3-fb8c53950b0d}",
            "HKLM\\SOFTWARE\\WOW6432Node\\Microsoft\\VisualStudio\\14.0\\VC\\Runtimes\\x64")
        for key in keys:
            chk = Popen("reg query {}".format(key), shell=True, stdout=PIPE, stderr=PIPE)
            stdout, stderr = chk.communicate()
            if stdout:
                break
        if stderr:
            self.output.error("Visual C++ 2015 could not be found. Make sure you have selected "
                              "'Visual C++' in Visual Studio 2015 Configuration or download the "
                              "Visual C++ 2015 Redistributable from: "
                              "https://www.microsoft.com/en-us/download/details.aspx?id=48145")
            return
        vscpp = [re.sub(" +", " ", line.strip())
                 for line in stdout.decode(self.env.encoding).splitlines()
                 if line.lower().strip().startswith(("displayname", "version"))][0]
        version = vscpp[vscpp.find("REG_SZ") + 7:]
        self.output.info("Visual Studio C++ version: {}".format(version))


class Install():
    """ Install the requirements """
    def __init__(self, environment):
        self.output = Output()
        self.env = environment

        if not self.env.is_installer:
            self.ask_continue()
        self.check_missing_dep()
        self.check_conda_missing_dep()
        self.install_missing_dep()
        self.output.info("All python3 dependencies are met.\r\nYou are good to go.\r\n\r\n"
                         "Enter:  'python faceswap.py -h' to see the options\r\n"
                         "        'python faceswap.py gui' to launch the GUI")

    def ask_continue(self):
        """ Ask Continue with Install """
        inp = input("Please ensure your System Dependencies are met. Continue? [y/N] ")
        if inp in ("", "N", "n"):
            self.output.error("Please install system dependencies to continue")
            exit(1)

    def check_missing_dep(self):
        """ Check for missing dependencies """
        if self.env.enable_cuda and self.env.is_macos:
            self.env.required_packages.extend(self.env.macos_required_packages)
        for pkg in self.env.required_packages:
            key = pkg.split("==")[0]
            if key not in self.env.installed_packages:
                self.env.missing_packages.append(pkg)
                continue
            else:
                if len(pkg.split("==")) > 1:
                    if pkg.split("==")[1] != self.env.installed_packages.get(key):
                        self.env.missing_packages.append(pkg)
                        continue

    def check_conda_missing_dep(self):
        """ Check for conda missing dependencies """
        if not self.env.is_conda:
            return
        for pkg in self.env.conda_required_packages:
            key = pkg[0].split("==")[0]
            if key not in self.env.installed_packages:
                self.env.conda_missing_packages.append(pkg)
                continue
            else:
                if len(pkg[0].split("==")) > 1:
                    if pkg[0].split("==")[1] != self.env.installed_conda_packages.get(key):
                        self.env.conda_missing_packages.append(pkg)
                        continue

    def install_missing_dep(self):
        """ Install missing dependencies """
        if self.env.missing_packages:
            self.install_python_packages()
        if self.env.conda_missing_packages:
            self.install_conda_packages()

    def install_python_packages(self):
        """ Install required pip packages """
        self.output.info("Installing Required Python Packages. This may take some time...")
        for pkg in self.env.missing_packages:
            if self.env.is_conda:
                verbose = pkg.startswith("tensorflow")
                if self.conda_installer(pkg, verbose=verbose):
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
        success = True
        condaexe = ["conda", "install", "-y"]
        if not verbose:
            condaexe.append("-q")
        if channel:
            condaexe.extend(["-c", channel])
        condaexe.append(package)
        self.output.info("Installing {}".format(package))
        try:
            if verbose:
                run(condaexe, check=True)
            else:
                with open(os.devnull, "w") as devnull:
                    run(condaexe, stdout=devnull, stderr=devnull, check=True)
        except CalledProcessError:
            if not conda_only:
                self.output.info("Couldn't install {} with Conda. Trying pip".format(package))
            else:
                self.output.warning("Couldn't install {} with Conda. "
                                    "Please install this package manually".format(package))
            success = False
        return success

    def pip_installer(self, package):
        """ Install a pip package """
        pipexe = [sys.executable, "-m", "pip"]
        # hide info/warning and fix cache hang
        pipexe.extend(["install", "-qq", "--no-cache-dir"])
        # install as user to solve perm restriction
        if not self.env.is_admin and not self.env.is_virtualenv:
            pipexe.append("--user")
        if package.startswith("dlib"):
            if not self.env.enable_cuda:
                pipexe.extend(["--install-option=--no", "--install-option=DLIB_USE_CUDA"])
            if self.env.os_version[0] == "Windows":
                pipexe.extend(["--global-option=-G", "--global-option=Visual Studio 14 2015"])
            msg = ("Compiling {}. This will take a while...\n"
                   "Please ignore the following UserWarning: "
                   "'Disabling all use of wheels...'".format(package))
        else:
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
            "docker exec -it deepfakes-cpu bash".format(path=os.path.dirname(os.path.realpath(__file__))))
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
            "docker exec deepfakes-gpu python /srv/tools.py gui\n".format(path=os.path.dirname(os.path.realpath(__file__))))

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

            "2b. If you do not want to use Anaconda, or if you wish to compile DLIB with GPU\n"
            "support you will need to manually install CUDA and cuDNN:\n"
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
    if INSTALL_FAILED:
        exit(1)
    Install(ENV)
