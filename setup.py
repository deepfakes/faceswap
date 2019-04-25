#!/usr/bin/env python3
""" Install packages for faceswap.py """

# >>> ENV
import ctypes
import importlib
import locale
import os
import re
import sys
import platform

from subprocess import run, PIPE, Popen

ENCODING = locale.getpreferredencoding()
# Revisions of tensorflow-gpu and cuda/cudnn requirements
TENSORFLOW_REQUIREMENTS = {"1.2": ["8.0", "5.1"],
                           "1.4": ["8.0", "6.0"],
                           "1.12": ["9.0", "7.2"]}
OS_VERSION = (platform.system(), platform.release())
PY_VERSION = (platform.python_version(), platform.architecture()[0])
IS_MACOS = (platform.system() == "Darwin")
IS_CONDA = ("conda" in sys.version.lower())
LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", None)
try:
    IS_ADMIN = os.getuid() == 0
except AttributeError:
    IS_ADMIN = ctypes.windll.shell32.IsUserAnAdmin() != 0
IS_VIRTUALENV = (hasattr(sys, "real_prefix")
                 or (hasattr(sys, "base_prefix")
                     and sys.base_prefix != sys.prefix))

CUDA_PATH = ""
CUDA_VERSION = ""
CUDNN_VERSION = ""
ENABLE_DOCKER = False
ENABLE_CUDA = True
COMPILE_DLIB_WITH_AVX = True
REQUIRED_PACKAGES = list()
MACOS_REQUIRED_PACKAGES = [
    "pynvx==0.0.4"
    ]
INSTALLED_PACKAGES = dict()
MISSING_PACKAGES = list()
FAIL = False


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
    global FAIL
    trm = "ERROR   "
    if term_support_color():
        trm = "{}ERROR  {} ".format(COLOR_RED, COLOR_DEFAULT)
    print(trm + __indent_text_block(text))
    FAIL = True

# <<< OUTPUT


def check_permission():
    """ Check for Admin permissions """
    if IS_ADMIN:
        out_info("Running as Root/Admin")
    else:
        out_warning("Running without root/admin privileges")


def check_system():
    """ Check the system """
    out_info("The tool provides tips for installation\n"
             "and installs required python packages")
    out_info("Setup in %s %s" % (OS_VERSION[0], OS_VERSION[1]))
    if not OS_VERSION[0] in ["Windows", "Linux", "Darwin"]:
        out_error("Your system %s is not supported!" % OS_VERSION[0])
        exit(1)


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
    i = input("Enable  Docker? [y/N] ")
    if i in ("Y", "y"):
        out_info("Docker Enabled")
        ENABLE_DOCKER = True
    else:
        out_info("Docker Disabled")
        ENABLE_DOCKER = False


def check_python():
    """ Check python and virtual environment status """
    out_info("Installed Python: {0} {1}".format(PY_VERSION[0],
                                                PY_VERSION[1]))
    if not (PY_VERSION[0].split(".")[0] == "3"
            and PY_VERSION[0].split(".")[1] in ("3", "4", "5", "6")
            and PY_VERSION[1] == "64bit"):
        out_error("Please run this script with Python version 3.3, 3.4, 3.5 or 3.6 "
                  "64bit and try again.")
        exit(1)


def check_pip():
    """ Check installed pip version """
    try:
        import pip
    except ImportError:
        out_error("Import pip failed. Please Install python3-pip "
                  "and try again")
        exit(1)
    upgrade_pip()
    importlib.reload(pip)
    pip_version = pip.__version__
    del pip

    get_installed_packages()
    out_info("Installed pip: {}".format(pip_version))


def upgrade_pip():
    """ Upgrade pip to latest version """
    out_info("Upgrading pip...")
    pipexe = [sys.executable, "-m", "pip"]
    pipexe.extend(["install", "--no-cache-dir", "-qq", "--upgrade"])
    if not IS_ADMIN and not IS_VIRTUALENV:
        pipexe.append("--user")
    pipexe.append("pip")
    run(pipexe)


def get_installed_packages():
    """ Get currently installed packages """
    global INSTALLED_PACKAGES
    chk = Popen("{} -m pip freeze".format(sys.executable),
                shell=True, stdout=PIPE)
    installed = chk.communicate()[0].decode(ENCODING).splitlines()
    for pkg in installed:
        item = pkg.split("==")
        INSTALLED_PACKAGES[item[0]] = item[1]


def check_system_dependencies():
    """ Check that system applications are installed """
    out_info("Checking System Dependencies...")
    check_cmake()
    if OS_VERSION[0] == "Windows":
        check_visual_studio()
        check_cplus_plus()
    if OS_VERSION[0] == "Linux":
        check_gcc()
        check_gpp()


def check_gcc():
    """ Check installed gcc version for linux """
    chk = Popen("gcc --version", shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = chk.communicate()
    if stderr:
        out_error("gcc not installed. Please install gcc for your distribution")
        return
    gcc = [re.sub(" +", " ", line.strip())
           for line in stdout.decode(ENCODING).splitlines()
           if line.lower().strip().startswith("gcc")][0]
    version = gcc[gcc.rfind(" ") + 1:]
    out_info("gcc version: {}".format(version))


def check_gpp():
    """ Check installed g++ version for linux """
    chk = Popen("g++ --version", shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = chk.communicate()
    if stderr:
        out_error("g++ not installed. Please install g++ for your distribution")
        return
    gpp = [re.sub(" +", " ", line.strip())
           for line in stdout.decode(ENCODING).splitlines()
           if line.lower().strip().startswith("g++")][0]
    version = gpp[gpp.rfind(" ") + 1:]
    out_info("g++ version: {}".format(version))


def check_cmake():
    """ Check CMake is installed for Windows """
    chk = Popen("cmake --version", shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = chk.communicate()
    stdout = stdout.decode(ENCODING)
    if stderr and OS_VERSION[0] == "Windows":
        stdout, stderr = check_cmake_windows()
    if stderr:
        out_error("CMake could not be found. See "
                  "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#cmake "
                  "for instructions")
        return
    cmake = [re.sub(" +", " ", line.strip())
             for line in stdout.splitlines()
             if line.lower().strip().startswith("cmake")][0]
    version = cmake[cmake.rfind(" ") + 1:]
    out_info("CMake version: {}".format(version))


def check_cmake_windows():
    """ Additional checks for cmake on Windows """
    chk = Popen("wmic product where \"name = 'cmake'\" get installlocation,version",
                shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = chk.communicate()
    if stderr:
        return False, stderr
    lines = [re.sub(" +", " ", line.strip())
             for line in stdout.decode(ENCODING).splitlines()
             if line.strip()]
    stdout = lines[1]
    location = stdout[:stdout.rfind(" ")] + "bin"
    out_info("CMake not found in %PATH%. Temporarily adding: \"{}\"".format(location))
    os.environ["PATH"] += ";{}".format(location)
    stdout = "cmake {}".format(stdout)
    return stdout, False


def check_visual_studio():
    """ Check Visual Studio 2015 is installed for Windows

        Somewhat hacky solution which checks for the existence
        of the VS2015 Performance Report
    """
    chk = Popen("reg query HKLM\\SOFTWARE\\Microsoft\\VisualStudio\\14.0\\VSPerf",
                shell=True, stdout=PIPE, stderr=PIPE)
    _, stderr = chk.communicate()
    if stderr:
        out_error("Visual Studio 2015 could not be found. See "
                  "https://github.com/deepfakes/faceswap/blob/master/"
                  "INSTALL.md#microsoft-visual-studio-2015 for instructions")
        return
    out_info("Visual Studio 2015 version: 14.0")


def check_cplus_plus():
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
        out_error("Visual C++ 2015 could not be found. Make sure you have selected 'Visual C++' "
                  "in Visual Studio 2015 Configuration or download the Visual C++ 2015 "
                  "Redistributable from: "
                  "https://www.microsoft.com/en-us/download/details.aspx?id=48145")
        return
    vscpp = [re.sub(" +", " ", line.strip())
             for line in stdout.decode(ENCODING).splitlines()
             if line.lower().strip().startswith(("displayname", "version"))][0]
    version = vscpp[vscpp.find("REG_SZ") + 7:]
    out_info("Visual Studio C++ version: {}".format(version))


def check_cuda():
    """ Check Cuda for Linux or Windows """
    if OS_VERSION[0] == "Linux":
        check_cuda_linux()
    elif OS_VERSION[0] == "Windows":
        check_cuda_windows()


def check_cuda_linux():
    """ Check Linux CUDA Version """
    global CUDA_VERSION, CUDA_PATH
    chk = os.popen("ldconfig -p | grep -P \"libcudart.so.\\d+.\\d+\" | head -n 1").read()
    if LD_LIBRARY_PATH and not chk:
        paths = LD_LIBRARY_PATH.split(":")
        for path in paths:
            chk = os.popen("ls {} | grep -P -o \"libcudart.so.\\d+.\\d+\" | "
                           "head -n 1".format(path)).read()
            if chk:
                break

    if not chk:
        out_error("CUDA not found. Install and try again.\n"
                  "Recommended version:      CUDA 9.0     cuDNN 7.1.3\n"
                  "CUDA: https://developer.nvidia.com/cuda-downloads\n"
                  "cuDNN: https://developer.nvidia.com/rdp/cudnn-download")
        return
    cudavers = chk.strip().replace("libcudart.so.", "")
    CUDA_VERSION = cudavers[:cudavers.find(" ")]
    if CUDA_VERSION:
        out_info("CUDA version: " + CUDA_VERSION)
        CUDA_PATH = chk[chk.find("=>") + 3:chk.find("targets") - 1]


def check_cuda_windows():
    """ Check Windows CUDA Version """
    global CUDA_VERSION, CUDA_PATH
    cuda_keys = [key
                 for key in os.environ.keys()
                 if key.lower().startswith("cuda") and key.lower() != "cuda_path"]
    if not cuda_keys:
        out_error("CUDA not found. See "
                  "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#cuda "
                  "for instructions")
        return

    CUDA_VERSION = cuda_keys[0].replace("CUDA_PATH_V", "").replace("_", ".")
    CUDA_PATH = os.environ[cuda_keys[0]]
    out_info("CUDA version: " + CUDA_VERSION)


def check_cudnn():
    """ Check Linux or Windows cuDNN Version from cudnn.h """
    global CUDNN_VERSION
    cudnn_checkfile = os.path.join(CUDA_PATH, "include", "cudnn.h")
    if not os.path.isfile(cudnn_checkfile):
        out_error("cuDNN not found. See "
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
        out_error("cuDNN version could not be determined. See "
                  "https://github.com/deepfakes/faceswap/blob/master/INSTALL.md#cudnn "
                  "for instructions")
        return

    CUDNN_VERSION = "{}.{}".format(major, minor)
    out_info("cuDNN version: {}.{}".format(CUDNN_VERSION, patchlevel))


def ask_continue():
    """ Ask Continue with Install """
    i = input("Please ensure your System Dependencies are met. Continue? [y/N] ")
    if i in ("", "N", "n"):
        out_error("Please install system dependencies to continue")
        exit(1)


def check_missing_dep():
    """ Check for missing dependencies """
    global MISSING_PACKAGES, INSTALLED_PACKAGES, ENABLE_CUDA
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
    global MISSING_PACKAGES, COMPILE_DLIB_WITH_AVX
    if "dlib" in MISSING_PACKAGES:
        i = input("Compile dlib with AVX? [Y/n] ")
        if i in ("", "Y", "y"):
            out_info("dlib Configured")
            COMPILE_DLIB_WITH_AVX = True
        else:
            COMPILE_DLIB_WITH_AVX = False


def update_tf_dep(cpu_only):
    """ Update Tensorflow Dependency """
    global CUDA_VERSION, CUDNN_VERSION

    if cpu_only:
        REQUIRED_PACKAGES.append("tensorflow")
        return

    tf_ver = None
    cudnn_inst = CUDNN_VERSION.split(".")
    for key, val in TENSORFLOW_REQUIREMENTS.items():
        cuda_req = val[0]
        cudnn_req = val[1].split(".")
        if cuda_req == CUDA_VERSION and (cudnn_req[0] == cudnn_inst[0] and
                                         cudnn_req[1] <= cudnn_inst[1]):
            tf_ver = key
            break
    if tf_ver:
        tf_ver = "tensorflow-gpu=={}.0".format(tf_ver)
        REQUIRED_PACKAGES.append(tf_ver)
        return

    out_warning("Tensorflow currently has no official prebuild for your CUDA, cuDNN "
                "combination.\nEither install a combination that Tensorflow supports or "
                "build and install your own tensorflow-gpu.\r\n"
                "CUDA Version: {}\r\n"
                "cuDNN Version: {}\r\n"
                "Help:\n"
                "Building Tensorflow: https://www.tensorflow.org/install/install_sources\r\n"
                "Tensorflow supported versions: "
                "https://www.tensorflow.org/install/source#tested_build_configurations".format(
                    CUDA_VERSION, CUDNN_VERSION))

    custom_tf = input("Location of custom tensorflow-gpu wheel (leave "
                      "blank to manually install): ")
    if not custom_tf:
        return

    custom_tf = os.path.realpath(os.path.expanduser(custom_tf))
    if not os.path.isfile(custom_tf):
        out_error("{} not found".format(custom_tf))
    elif os.path.splitext(custom_tf)[1] != ".whl":
        out_error("{} is not a valid pip wheel".format(custom_tf))
    elif custom_tf:
        REQUIRED_PACKAGES.append(custom_tf)


def install_missing_dep():
    """ Install missing dependencies """
    global MISSING_PACKAGES, ENABLE_CUDA
    install_tkinter()
    install_ffmpeg()
    if MISSING_PACKAGES:
        out_info("Installing Required Python Packages. "
                 "This may take some time...")
        for pkg in MISSING_PACKAGES:
            if pkg.startswith("dlib"):
                msg = ("Compiling {}. This will take a while...\n"
                       "Please ignore the following UserWarning: "
                       "'Disabling all use of wheels...'".format(pkg))
            else:
                msg = "Installing {}".format(pkg)
            out_info(msg)
            pipexe = [sys.executable, "-m", "pip"]
            # hide info/warning and fix cache hang
            pipexe.extend(["install", "-qq", "--no-cache-dir"])
            # install as user to solve perm restriction
            if not IS_ADMIN and not IS_VIRTUALENV:
                pipexe.append("--user")
            # compile dlib with AVX and CUDA
            if pkg.startswith("dlib"):
                if OS_VERSION[0] == "Windows":
                    pipexe.extend(["--global-option=-G",
                                   "--global-option=Visual Studio 14 2015"])
                opt = "yes" if COMPILE_DLIB_WITH_AVX else "no"
                pipexe.extend(["--install-option=--{}".format(opt),
                               "--install-option=USE_AVX_INSTRUCTIONS"])
                opt = "yes" if ENABLE_CUDA else "no"
                pipexe.extend(["--install-option=--{}".format(opt),
                               "--install-option=DLIB_USE_CUDA"])

            pipexe.append(pkg)
            run(pipexe)


def install_tkinter():
    """ Install tkInter on Conda Environments """
    if not IS_CONDA:
        return
    pkgs = os.popen("conda list").read()
    tki = [re.sub(" +", " ", line.strip())
           for line in pkgs.splitlines()
           if line.lower().strip().startswith("tk")]
    if tki:
        return
    out_info("Installing tkInter")
    with open(os.devnull, "w") as devnull:
        run(["conda", "install", "-q", "-y", "tk"], stdout=devnull)


def install_ffmpeg():
    """ Install ffmpeg on Conda Environments """
    if not IS_CONDA:
        return
    pkgs = os.popen("conda list").read()
    ffm = [re.sub(" +", " ", line.strip())
           for line in pkgs.splitlines()
           if line.lower().strip().startswith("ffmpeg")]
    if ffm:
        return
    out_info("Installing ffmpeg")
    with open(os.devnull, "w") as devnull:
        run(["conda", "install", "-q", "-y", "-c", "conda-forge", "ffmpeg"], stdout=devnull)


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
    """ Pip Tips """
    out_info("1. Install PIP requirements\n"
             "You may want to execute `chcp 866` in cmd line\n"
             "to fix Unicode issues on Windows when installing dependencies")


def main():
    """" Run Setup """
    global ENABLE_DOCKER, ENABLE_CUDA, CUDA_VERSION
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
        return

    if ENABLE_CUDA:
        # update dep info if cuda enabled
        if OS_VERSION[0] in ("Linux", "Windows"):
            check_cuda()
            check_cudnn()
        else:
            tips_2_1()
            out_warning("Cannot find CUDA on macOS")
            CUDA_VERSION = input("Manually specify CUDA version: ")
        update_tf_dep(cpu_only=False)
    else:
        update_tf_dep(cpu_only=True)
    check_system_dependencies()
    if FAIL:
        exit(1)
    if OS_VERSION[0] == "Windows":
        tips_2_2()
    # finally check dep
    ask_continue()
    check_missing_dep()
    check_dlib()
    install_missing_dep()
    out_info("All python3 dependencies are met.\r\nYou are good to go.\r\n\r\n"
             "Enter:  'python faceswap.py -h' to see the options\r\n"
             "        'python faceswap.py gui' to launch the GUI")


if __name__ == "__main__":
    main()
