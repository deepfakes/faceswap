#!/usr/bin/env python3

### >>> ENV
import os
import sys
import platform
OS_Version = (platform.system(), platform.release())
Py_Version = (platform.python_version(), platform.architecture()[0])
LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", None)
IS_ADMIN = False
IS_VIRTUALENV = False
CUDA_Version = ""
ENABLE_DOCKER = True
ENABLE_CUDA = True
COMPILE_DLIB_WITH_AVX_CUDA = True
Required_Packages = [
"tensorflow"
]
Installed_Packages = {}
Missing_Packages = []

# load requirements list
with open("requirements.txt") as req:
    for r in req.readlines():
        r = r.strip()
        if r and (not r.startswith("#")):
            Required_Packages.append(r)

### <<< ENV

### >>> OUTPUT
color_red = "\033[31m"
color_green = "\033[32m"
color_yellow = "\033[33m"
color_default = "\033[0m"

def __indent_text_block(text):
    a = text.splitlines()
    if len(a)>1:
        b = a[0] + "\r\n"
        for i in range(1, len(a)-1):
            b = b + "        " + a[i] + "\r\n"
        b = b +  "        " + a[-1]
        return b
    else:
        return text

def Term_Support_Color():
    global OS_Version
    return (OS_Version[0] == "Linux" or OS_Version[0] == "Darwin")

def INFO(text):
    t = "%sINFO   %s " % (color_green, color_default) if Term_Support_Color() else "INFO    "
    print(t + __indent_text_block(text))

def WARNING(text):
    t = "%sWARNING%s " % (color_yellow, color_default) if Term_Support_Color() else "WARNING "
    print(t + __indent_text_block(text))

def ERROR(text):
    t = "%sERROR  %s " % (color_red, color_default) if Term_Support_Color() else "ERROR   "
    print(t + __indent_text_block(text))
    exit(1)

### <<< OUTPUT

def Check_Permission():
    import ctypes, os
    global IS_ADMIN
    try:
        IS_ADMIN = os.getuid() == 0
    except AttributeError:
        IS_ADMIN = ctypes.windll.shell32.IsUserAnAdmin() != 0
    if IS_ADMIN:
        INFO("Running as Root/Admin")
    else:
        WARNING("Running without root/admin privileges")

def Check_System():
    global OS_Version
    INFO("The tool provides tips for installation\nand installs required python packages")
    INFO("Setup in %s %s" % (OS_Version[0], OS_Version[1]))
    if not OS_Version[0] in ["Windows", "Linux", "Darwin"]:
        ERROR("Your system %s is not supported!" % OS_Version[0])

def Enable_CUDA():
    global ENABLE_CUDA
    i = input("Enable  CUDA? [Y/n] ")
    if i == "" or i == "Y" or i == "y":
        INFO("CUDA Enabled")
        ENABLE_CUDA = True
    else:
        INFO("CUDA Disabled")
        ENABLE_CUDA = False

def Enable_Docker():
    global ENABLE_DOCKER
    i = input("Enable  Docker? [Y/n] ")
    if i == "" or i == "Y" or i == "y":
        INFO("Docker Enabled")
        ENABLE_DOCKER = True
    else:
        INFO("Docker Disabled")
        ENABLE_DOCKER = False

def Check_Python():
    global Py_Version, IS_VIRTUALENV
    # check if in virtualenv
    IS_VIRTUALENV = (hasattr(sys, "real_prefix")
                     or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
    if Py_Version[0].split(".")[0] == "3" and  Py_Version[1] == "64bit":
        INFO("Installed Python: {0} {1}".format(Py_Version[0],Py_Version[1]))
        return True
    else:
        ERROR("Please run this script with Python3 64bit and try again.")
        return False

def Check_PIP():
    try:
        try: # for pip >= 10
            from pip._internal.utils.misc import get_installed_distributions, get_installed_version
        except ImportError: # for pip <= 9.0.3
            from pip.utils import get_installed_distributions, get_installed_version
        global Installed_Packages
        Installed_Packages = {pkg.project_name:pkg.version for pkg in get_installed_distributions()}
        INFO("Installed PIP: " + get_installed_version("pip"))
        return True
    except ImportError:
        ERROR("Import pip failed. Please Install python3-pip and try again")
        return False

# only invoked in linux
def Check_CUDA():
    global CUDA_Version
    a=os.popen("ldconfig -p | grep -P -o \"libcudart.so.\d.\d\" | head -n 1")
    libcudart = a.read()
    if LD_LIBRARY_PATH and not libcudart:
        paths = LD_LIBRARY_PATH.split(":")
        for path in paths:
            a = os.popen("ls {} | grep -P -o \"libcudart.so.\d.\d\" | head -n 1".format(path))
            libcudart = a.read()
            if libcudart:
                break
    if libcudart:
        CUDA_Version = libcudart[13:].rstrip()
        if CUDA_Version:
            INFO("CUDA version: " + CUDA_Version)
    else:
        ERROR("""CUDA not found. Install and try again.
Recommended version:      CUDA 9.0     cuDNN 7.1.3
CUDA: https://developer.nvidia.com/cuda-downloads
cuDNN: https://developer.nvidia.com/rdp/cudnn-download
""")

# only invoked in linux
def Check_cuDNN():
    a=os.popen("ldconfig -p | grep -P -o \"libcudnn.so.\d\" | head -n 1")
    libcudnn = a.read()
    if LD_LIBRARY_PATH and not libcudnn:
        paths = LD_LIBRARY_PATH.split(":")
        for path in paths:
            a = os.popen("ls {} | grep -P -o \"libcudnn.so.\d\" | head -n 1".format(path))
            libcudnn = a.read()
            if libcudnn:
                break
    if libcudnn:
        cudnn_version = libcudnn[12:].rstrip()
        if cudnn_version:
            INFO("cuDNN version: " + cudnn_version)
    else:
        ERROR("""cuDNN not found. Install and try again.
Recommended version:      CUDA 9.0     cuDNN 7.1.3
CUDA: https://developer.nvidia.com/cuda-downloads
cuDNN: https://developer.nvidia.com/rdp/cudnn-download
""")

def Continue():
    i = input("Are System Dependencies met? [y/N] ")
    if i == "" or i == "N" or i == "n":
        ERROR('Please install system dependencies to continue')

def Check_Missing_Dep():
    global Missing_Packages, Installed_Packages
    Missing_Packages = []
    for pkg in Required_Packages:
        key = pkg.split("==")[0]
        if not key in Installed_Packages:
            Missing_Packages.append(pkg)
            continue
        else:
            if len(pkg.split("=="))>1:
                if pkg.split("==")[1] != Installed_Packages.get(key):
                    Missing_Packages.append(pkg)
                    continue

def Check_dlib():
    global Missing_Packages, COMPILE_DLIB_WITH_AVX_CUDA
    if "dlib" in Missing_Packages:
        i = input("Compile dlib with AVX (and CUDA if enabled)? [Y/n] ")
        if i == "" or i == "Y" or i == "y":
            INFO("dlib Configured")
            WARNING("Make sure you are using gcc-5/g++-5 and CUDA bin/lib in path")
            COMPILE_DLIB_WITH_AVX_CUDA = True
        else:
            COMPILE_DLIB_WITH_AVX_CUDA = False

def Install_Missing_Dep():
    global Missing_Packages
    if len(Missing_Packages):
        INFO("""Installing Required Python Packages. This may take some time...""")
        try:
            from pip._internal import main as pipmain
        except:
            from pip import main as pipmain
        for m in Missing_Packages:
            msg = "Installing {}".format(m)
            INFO(msg)
            # hide info/warning and fix cache hang
            pipargs = ["install", "-qq", "--no-cache-dir"]
            # install as user to solve perm restriction
            if not IS_ADMIN and not IS_VIRTUALENV:
                pipargs.append("--user")
            # compile dlib with AVX ins and CUDA
            if m.startswith("dlib") and COMPILE_DLIB_WITH_AVX_CUDA:
                pipargs.extend(["--install-option=--yes", "--install-option=USE_AVX_INSTRUCTIONS"])
            pipargs.append(m)
            # pip install -qq (--user) (--install-options) m
            pipmain(pipargs)

def Update_TF_Dep():
    global CUDA_Version
    Required_Packages[0] = "tensorflow-gpu"
    if CUDA_Version.startswith("8.0"):
        Required_Packages[0] += "==1.4.0"
    elif not CUDA_Version.startswith("9.0"):
            WARNING("Tensorflow has no official prebuild for CUDA 9.1 currently.\r\n"
                    "To continue, You have to build and install your own tensorflow-gpu.\r\n"
                    "Help: https://www.tensorflow.org/install/install_sources")
            custom_tf = input("Location of custom tensorflow-gpu wheel (leave blank to manually install): ")
            if not custom_tf:
                del Required_Packages[0]
                return
            if os.path.isfile(custom_tf):
                Required_Packages[0] = custom_tf
            else:
                ERROR("{} not found".format(custom_tf))


def Tips_1_1():
    INFO("""1. Install Docker
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
    INFO("That's all you need to do with a docker. Have fun.")

def Tips_1_2():
    INFO("""1. Install Docker
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

def Tips_2_1():
    INFO("""Tensorflow has no official prebuilts for CUDA 9.1 currently.

1. Install CUDA 9.0 and cuDNN
CUDA: https://developer.nvidia.com/cuda-downloads
cuDNN: https://developer.nvidia.com/rdp/cudnn-download (Add DLL to %PATH% in Windows)

2. Install System Dependencies.
In Windows:
Install CMake x64: https://cmake.org/download/

In Debian/Ubuntu, try:
apt-get install -y cmake libsm6 libxrender1 libxext-dev python3-tk

3. Install PIP requirements
You may want to execute `chcp 866` in cmd line
to fix Unicode issues on Windows when installing dependencies
""")


def Tips_2_2():
    INFO("""1. Install System Dependencies.
In Windows:
Install CMake x64: https://cmake.org/download/

In Debian/Ubuntu, try:
apt-get install -y cmake libsm6 libxrender1 libxext-dev python3-tk

2. Install PIP requirements
You may want to execute `chcp 866` in cmd line
to fix Unicode issues on Windows when installing dependencies
""")


def Main():
    global ENABLE_DOCKER, ENABLE_CUDA, CUDA_Version, OS_Version
    Check_System()
    Check_Python()
    Check_PIP()
    # ask questions
    Enable_Docker()
    Enable_CUDA()
    # warn if nvidia-docker on non-linux system
    if OS_Version[0] != "Linux" and ENABLE_DOCKER and ENABLE_CUDA:
        WARNING("Nvidia-Docker is only supported on Linux.\r\nOnly CPU is supported in Docker for your system")
        Enable_Docker()
        if ENABLE_DOCKER:
            WARNING("CUDA Disabled")
            ENABLE_CUDA = False
    
    # provide tips
    if ENABLE_DOCKER:
        # docker, quick help
        if not ENABLE_CUDA:
            Tips_1_1()
        else:
            Tips_1_2()
    else:
        if ENABLE_CUDA:
            # update dep info if cuda enabled
            if OS_Version[0] == "Linux":
                Check_CUDA()
                Check_cuDNN()
            else:
                Tips_2_1()
                WARNING("Cannot find CUDA on non-Linux system")
                CUDA_Version = input("Manually specify CUDA version: ")
            Update_TF_Dep()
        else:
            Tips_2_2()
        # finally check dep
        Continue()
        Check_Missing_Dep()
        Check_dlib()
        Install_Missing_Dep()
        INFO("All python3 dependencies are met.\r\nYou are good to go.")

if __name__ == "__main__":
    Main()

