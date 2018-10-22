#!/usr/bin/env python3
""" Install packages for faceswap.py """

# >>> ENV
import os
import sys
import platform
OS_VERSION = (platform.system(), platform.release())
PY_VERSION = (platform.python_version(), platform.architecture()[0])
IS_MACOS = (platform.system() == 'Darwin')
LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", None)
IS_ADMIN = False
IS_VIRTUALENV = False
CUDA_VERSION = ""
ENABLE_DOCKER = True
ENABLE_CUDA = True
COMPILE_DLIB_WITH_AVX_CUDA = True
REQUIRED_PACKAGES = [
    "tensorflow"
    ]
MACOS_REQUIRED_PACKAGES = [
    "pynvx==0.0.4"
    ]
INSTALLED_PACKAGES = {}
MISSING_PACKAGES = []

# load requirements list
with open("requirements.txt") as req:
    for r in req.readlines():
        r = r.strip()
        if r and (not r.startswith("#")):
            REQUIRED_PACKAGES.append(r)

# <<< ENV

# >>> OUTPUT
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_DEFAULT = "\033[0m"


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


def term_support_color():
    """ Set whether OS Support terminal colour """
    global OS_VERSION
    return OS_VERSION[0] == "Linux" or OS_VERSION[0] == "Darwin"


def out_info(text):
    """ Format INFO Text """
    trm = "INFO    "
    if term_support_color():
        trm = "{}INFO   {} ".format(COLOR_GREEN, COLOR_DEFAULT)
    print(trm + __indent_text_block(text))


def out_warning(text):
    """ Format WARNING Text """
    trm = "WARNING "
    if term_support_color():
        trm = "{}WARNING{} ".format(COLOR_YELLOW, COLOR_DEFAULT)
    print(trm + __indent_text_block(text))


def out_error(text):
    """ Format ERROR Text """
    trm = "ERROR   "
    if term_support_color():
        trm = "{}ERROR  {} ".format(COLOR_RED, COLOR_DEFAULT)
    print(trm + __indent_text_block(text))
    exit(1)

# <<< OUTPUT


def check_permission():
    """ Check for Admin permissions """
    import ctypes
    global IS_ADMIN
    try:
        IS_ADMIN = os.getuid() == 0
    except AttributeError:
        IS_ADMIN = ctypes.windll.shell32.IsUserAnAdmin() != 0
    if IS_ADMIN:
        out_info("Running as Root/Admin")
    else:
        out_warning("Running without root/admin privileges")


def check_system():
    """ Check the system """
    global OS_VERSION
    out_info("The tool provides tips for installation\n"
             "and installs required python packages")
    out_info("Setup in %s %s" % (OS_VERSION[0], OS_VERSION[1]))
    if not OS_VERSION[0] in ["Windows", "Linux", "Darwin"]:
        out_error("Your system %s is not supported!" % OS_VERSION[0])


def ask_enable_cuda():
    """ Enable or disable CUDA """
    global ENABLE_CUDA
    i = input("Enable  CUDA? [Y/n] ")
    if i in ("", "Y", "y"):
        out_info("CUDA Enabled")
        ENABLE_CUDA = True
    else:
        out_info("CUDA Disabled")
        ENABLE_CUDA = False


def ask_enable_docker():
    """ Enable or disable Docker """
    global ENABLE_DOCKER
    i = input("Enable  Docker? [Y/n] ")
    if i in ("", "Y", "y"):
        out_info("Docker Enabled")
        ENABLE_DOCKER = True
    else:
        out_info("Docker Disabled")
        ENABLE_DOCKER = False


def check_python():
    """ Check python and virtual environment status """
    global PY_VERSION, IS_VIRTUALENV
    # check if in virtualenv
    IS_VIRTUALENV = (hasattr(sys, "real_prefix")
                     or (hasattr(sys, "base_prefix") and
                         sys.base_prefix != sys.prefix))
    if PY_VERSION[0].split(".")[0] == "3" and PY_VERSION[1] == "64bit":
        out_info("Installed Python: {0} {1}".format(PY_VERSION[0],
                                                    PY_VERSION[1]))
        return True

    out_error("Please run this script with Python3 64bit and try again.")
    return False


def check_pip():
    """ Check installed pip version """
    try:
        try:  # for pip >= 10
            from pip._internal.utils.misc import (get_installed_distributions,
                                                  get_installed_version)
        except ImportError:  # for pip <= 9.0.3
            from pip.utils import (get_installed_distributions,
                                   get_installed_version)
        global INSTALLED_PACKAGES
        INSTALLED_PACKAGES = {pkg.project_name: pkg.version
                              for pkg in get_installed_distributions()}
        out_info("Installed PIP: " + get_installed_version("pip"))
        return True
    except ImportError:
        out_error("Import pip failed. Please Install python3-pip "
                  "and try again")
        return False


# only invoked in linux
def check_cuda():
    """ Check CUDA Version """
    global CUDA_VERSION
    chk = os.popen("ldconfig -p | grep -P -o \"libcudart.so.\d+.\d+\" | "
                   "head -n 1")
    libcudart = chk.read()
    if LD_LIBRARY_PATH and not libcudart:
        paths = LD_LIBRARY_PATH.split(":")
        for path in paths:
            chk = os.popen("ls {} | grep -P -o \"libcudart.so.\d+.\d+\" | "
                           "head -n 1".format(path))
            libcudart = chk.read()
            if libcudart:
                break
    if libcudart:
        CUDA_VERSION = libcudart[13:].rstrip()
        if CUDA_VERSION:
            out_info("CUDA version: " + CUDA_VERSION)
    else:
        out_error("""CUDA not found. Install and try again.
Recommended version:      CUDA 9.0     cuDNN 7.1.3
CUDA: https://developer.nvidia.com/cuda-downloads
cuDNN: https://developer.nvidia.com/rdp/cudnn-download
""")


# only invoked in linux
def check_cudnn():
    """ Check cuDNN Version """
    chk = os.popen("ldconfig -p | grep -P -o \"libcudnn.so.\d\" | head -n 1")
    libcudnn = chk.read()
    if LD_LIBRARY_PATH and not libcudnn:
        paths = LD_LIBRARY_PATH.split(":")
        for path in paths:
            chk = os.popen("ls {} | grep -P -o \"libcudnn.so.\d\" | "
                           "head -n 1".format(path))
            libcudnn = chk.read()
            if libcudnn:
                break
    if libcudnn:
        cudnn_version = libcudnn[12:].rstrip()
        if cudnn_version:
            out_info("cuDNN version: " + cudnn_version)
    else:
        out_error("""cuDNN not found. Install and try again.
Recommended version:      CUDA 9.0     cuDNN 7.1.3
CUDA: https://developer.nvidia.com/cuda-downloads
cuDNN: https://developer.nvidia.com/rdp/cudnn-download
""")


def ask_continue():
    """ Ask Continue with Install """
    i = input("Are System Dependencies met? [y/N] ")
    if i in ("", "N", "n"):
        out_error('Please install system dependencies to continue')


def check_missing_dep():
    """ Check for missing dependencies """
    global MISSING_PACKAGES, INSTALLED_PACKAGES, ENABLE_CUDA, IS_MACOS
    if ENABLE_CUDA and IS_MACOS:
        REQUIRED_PACKAGES.extend(MACOS_REQUIRED_PACKAGES)
    MISSING_PACKAGES = []
    for pkg in REQUIRED_PACKAGES:
        key = pkg.split("==")[0]
        if key not in INSTALLED_PACKAGES:
            MISSING_PACKAGES.append(pkg)
            continue
        else:
            if len(pkg.split("==")) > 1:
                if pkg.split("==")[1] != INSTALLED_PACKAGES.get(key):
                    MISSING_PACKAGES.append(pkg)
                    continue


def check_dlib():
    """ Check dlib install requirements """
    global MISSING_PACKAGES, COMPILE_DLIB_WITH_AVX_CUDA
    if "dlib" in MISSING_PACKAGES:
        i = input("Compile dlib with AVX (and CUDA if enabled)? [Y/n] ")
        if i in ("", "Y", "y"):
            out_info("dlib Configured")
            out_warning("Make sure you are using gcc-5/g++-5 "
                        "and CUDA bin/lib in path")
            COMPILE_DLIB_WITH_AVX_CUDA = True
        else:
            COMPILE_DLIB_WITH_AVX_CUDA = False


def install_missing_dep():
    """ Install missing dependencies """
    global MISSING_PACKAGES
    if MISSING_PACKAGES:
        out_info("Installing Required Python Packages. "
                 "This may take some time...")
        try:
            from pip._internal import main as pipmain
        except:
            from pip import main as pipmain
        for pkg in MISSING_PACKAGES:
            msg = "Installing {}".format(pkg)
            out_info(msg)
            # hide info/warning and fix cache hang
            pipargs = ["install", "-qq", "--no-cache-dir"]
            # install as user to solve perm restriction
            if not IS_ADMIN and not IS_VIRTUALENV:
                pipargs.append("--user")
            # compile dlib with AVX ins and CUDA
            if pkg.startswith("dlib") and COMPILE_DLIB_WITH_AVX_CUDA:
                pipargs.extend(["--install-option=--yes",
                                "--install-option=USE_AVX_INSTRUCTIONS"])
            pipargs.append(pkg)
            # pip install -qq (--user) (--install-options) pkg
            pipmain(pipargs)


def update_tf_dep():
    """ Update Tensorflow Dependency """
    global CUDA_VERSION
    REQUIRED_PACKAGES[0] = "tensorflow-gpu"
    if CUDA_VERSION.startswith("8.0"):
        REQUIRED_PACKAGES[0] += "==1.4.0"
    elif not CUDA_VERSION.startswith("9.0"):
        out_warning("Tensorflow has currently no official prebuild for CUDA "
                    "versions above 9.0.\r\nTo continue, You have to build "
                    "and install your own tensorflow-gpu.\r\n"
                    "Help: "
                    "https://www.tensorflow.org/install/install_sources")
        custom_tf = input("Location of custom tensorflow-gpu wheel (leave "
                          "blank to manually install): ")
        if not custom_tf:
            del REQUIRED_PACKAGES[0]
            return
        if os.path.isfile(custom_tf):
            REQUIRED_PACKAGES[0] = custom_tf
        else:
            out_error("{} not found".format(custom_tf))


def tips_1_1():
    """ Output Tips """
    out_info("""1. Install Docker
https://www.docker.com/community-edition

2. Build Docker Image For Faceswap
docker build -t deepfakes-cpu -f Dockerfile.cpu .

3. Mount faceswap volume and Run it
# without gui. tools.py gui not working.
docker run -p 8888:8888 \
    --hostname deepfakes-cpu --name deepfakes-cpu \
    -v {path}:/srv \
    deepfakes-cpu

# with gui. tools.py gui working.
## enable local access to X11 server
xhost +local:
## create container
nvidia-docker run -p 8888:8888 \\
    --hostname deepfakes-cpu --name deepfakes-cpu \\
    -v {path}:/srv \\
    -v /tmp/.X11-unix:/tmp/.X11-unix \\
    -e DISPLAY=unix$DISPLAY \\
    -e AUDIO_GID=`getent group audio | cut -d: -f3` \\
    -e VIDEO_GID=`getent group video | cut -d: -f3` \\
    -e GID=`id -g` \\
    -e UID=`id -u` \\
    deepfakes-cpu


4. Open a new terminal to run faceswap.py in /srv
docker exec -it deepfakes-cpu bash
""".format(path=sys.path[0]))
    out_info("That's all you need to do with a docker. Have fun.")


def tips_1_2():
    """ Output Tips """
    out_info("""1. Install Docker
https://www.docker.com/community-edition

2. Install latest CUDA
CUDA: https://developer.nvidia.com/cuda-downloads

3. Install Nvidia-Docker & Restart Docker Service
https://github.com/NVIDIA/nvidia-docker

4. Build Docker Image For Faceswap
docker build -t deepfakes-gpu -f Dockerfile.gpu .

5. Mount faceswap volume and Run it
# without gui. tools.py gui not working.
docker run -p 8888:8888 \
    --hostname deepfakes-gpu --name deepfakes-gpu \
    -v {path}:/srv \
    deepfakes-gpu

# with gui. tools.py gui working.
## enable local access to X11 server
xhost +local:
## enable nvidia device if working under bumblebee
echo ON > /proc/acpi/bbswitch
## create container
nvidia-docker run -p 8888:8888 \\
    --hostname deepfakes-gpu --name deepfakes-gpu \\
    -v {path}:/srv \\
    -v /tmp/.X11-unix:/tmp/.X11-unix \\
    -e DISPLAY=unix$DISPLAY \\
    -e AUDIO_GID=`getent group audio | cut -d: -f3` \\
    -e VIDEO_GID=`getent group video | cut -d: -f3` \\
    -e GID=`id -g` \\
    -e UID=`id -u` \\
    deepfakes-gpu

6. Open a new terminal to interact with the project
docker exec deepfakes-gpu python /srv/tools.py gui
""".format(path=sys.path[0]))


def tips_2_1():
    """ Output Tips """
    out_info("""Tensorflow has no official prebuilts for CUDA 9.1 currently.

1. Install CUDA 9.0 and cuDNN
CUDA: https://developer.nvidia.com/cuda-downloads
cuDNN: https://developer.nvidia.com/rdp/cudnn-download (Add DLL to "
"%PATH% in Windows)

2. Install System Dependencies.
In Windows:
Install CMake x64: https://cmake.org/download/

In Debian/Ubuntu, try:
apt-get install -y cmake libsm6 libxrender1 libxext-dev python3-tk

3. Install PIP requirements
You may want to execute `chcp 866` in cmd line
to fix Unicode issues on Windows when installing dependencies
""")


def tips_2_2():
    """ Output Tips """
    out_info("""1. Install System Dependencies.
In Windows:
Install CMake x64: https://cmake.org/download/

In Debian/Ubuntu, try:
apt-get install -y cmake libsm6 libxrender1 libxext-dev python3-tk

2. Install PIP requirements
You may want to execute `chcp 866` in cmd line
to fix Unicode issues on Windows when installing dependencies
""")


def main():
    """" Run Setup """
    global ENABLE_DOCKER, ENABLE_CUDA, CUDA_VERSION, OS_VERSION
    check_system()
    check_python()
    check_pip()
    # ask questions
    ask_enable_docker()
    ask_enable_cuda()
    # warn if nvidia-docker on non-linux system
    if OS_VERSION[0] != "Linux" and ENABLE_DOCKER and ENABLE_CUDA:
        out_warning("Nvidia-Docker is only supported on Linux.\r\n"
                    "Only CPU is supported in Docker for your system")
        ask_enable_docker()
        if ENABLE_DOCKER:
            out_warning("CUDA Disabled")
            ENABLE_CUDA = False

    # provide tips
    if ENABLE_DOCKER:
        # docker, quick help
        if not ENABLE_CUDA:
            tips_1_1()
        else:
            tips_1_2()
    else:
        if ENABLE_CUDA:
            # update dep info if cuda enabled
            if OS_VERSION[0] == "Linux":
                check_cuda()
                check_cudnn()
            else:
                tips_2_1()
                out_warning("Cannot find CUDA on non-Linux system")
                CUDA_VERSION = input("Manually specify CUDA version: ")
            update_tf_dep()
        else:
            tips_2_2()
        # finally check dep
        ask_continue()
        check_missing_dep()
        check_dlib()
        install_missing_dep()
        out_info("All python3 dependencies are met.\r\nYou are good to go.")


if __name__ == "__main__":
    main()
