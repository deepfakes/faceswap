# Installing Faceswap
- [Installing Faceswap](#installing-faceswap)
- [Prerequisites](#prerequisites)
  - [Hardware Requirements](#hardware-requirements)
  - [Supported operating systems](#supported-operating-systems)
- [Important before you proceed](#important-before-you-proceed)
- [Windows Install Guide](#windows-install-guide)
  - [Installer](#installer)
  - [Manual Install](#manual-install)
  - [Prerequisites](#prerequisites-1)
    - [Anaconda](#anaconda)
    - [Git](#git)
  - [Setup](#setup)
    - [Anaconda](#anaconda-1)
      - [Set up a virtual environment](#set-up-a-virtual-environment)
      - [Entering your virtual environment](#entering-your-virtual-environment)
    - [Faceswap](#faceswap)
      - [Easy install](#easy-install)
      - [Manual install](#manual-install)
  - [Running Faceswap](#running-faceswap)
  - [Create a desktop shortcut](#create-a-desktop-shortcut)
  - [Updating faceswap](#updating-faceswap)
- [General Install Guide](#general-install-guide)
  - [Installing dependencies](#installing-dependencies)
  - [Getting the faceswap code](#getting-the-faceswap-code)
  - [Setup](#setup-1)
    - [About some of the options](#about-some-of-the-options)
  - [Run the project](#run-the-project)
  - [Notes](#notes)

# Prerequisites
Machine learning essentially involves a ton of trial and error. You're letting a program try millions of different settings to land on an algorithm that sort of does what you want it to do. This process is really really slow unless you have the hardware required to speed this up. 

The type of computations that the process does are well suited for graphics cards, rather than regular processors. **It is pretty much required that you run the training process on a desktop or server capable GPU.** Running this on your CPU means it can take weeks to train your model, compared to several hours on a GPU.

## Hardware Requirements
**TL;DR: you need at least one of the following:**

- **A powerful CPU**
    - Laptop CPUs can often run the software, but will not be fast enough to train at reasonable speeds
- **A powerful GPU**
    - Currently, only Nvidia GPUs are supported. AMD graphics cards are not supported.
      This is not something that we have control over. It is a requirement of the Tensorflow library.
    - The GPU needs to support at least CUDA Compute Capability 3.0 or higher.
      To see which version your GPU supports, consult this list: https://developer.nvidia.com/cuda-gpus
      Desktop cards later than the 7xx series are most likely supported.
- **A lot of patience**

## Supported operating systems
- **Windows 10**
  Windows 7 and 8 might work. Your mileage may vary. Windows has an installer which will set up everything you need. See: https://github.com/deepfakes/faceswap/releases
- **Linux**
  Most Ubuntu/Debian or CentOS based Linux distributions will work.
- **macOS**
  GPU support on macOS is limited due to lack of drivers/libraries from Nvidia.
- All operating systems must be 64-bit for Tensorflow to run.

Alternatively, there is a docker image that is based on Debian.

# Important before you proceed
**In its current iteration, the project relies heavily on the use of the command line, although a gui is available. if you are unfamiliar with command line tools, you may have difficulty setting up the environment and should perhaps not attempt any of the steps described in this guide.** This guide assumes you have intermediate knowledge of the command line. 

The developers are also not responsible for any damage you might cause to your own computer.

# Windows Install Guide

## Installer
Windows now has an installer which installs everything for you and creates a desktop shortcut to launch straight into the GUI. You can download the installer from https://github.com/deepfakes/faceswap/releases.

If you have issues with the installer then read on for the more manual way to install Faceswap on Windows.

## Manual Install

Setting up Faceswap can seem a little intimidating to new users, but it isn't that complicated, although a little time consuming. It is recommended to use Linux where possible as Windows will hog about 20% of your GPU Memory, making Faceswap run a little slower, however using Windows is perfectly fine and 100% supported.

## Prerequisites

### Anaconda
Download and install the latest Python 3 Anaconda from: https://www.anaconda.com/download/. Unless you know what you are doing, you can leave all the options at default.

### Git
Download and install Git for Windows: https://git-scm.com/download/win. Unless you know what you are doing, you can leave all the options at default.

## Setup
Reboot your PC, so that everything you have just installed gets registered.

### Anaconda
#### Set up a virtual environment
- Open up Anaconda Navigator
- Select "Environments" on the left hand side
- Select "Create" at the bottom
- In the pop up:
    - Give it the name: faceswap
    - **IMPORTANT**: Select python version 3.6
    - Hit "Create" (NB: This may take a while as it will need to download Python 3.6)
![Anaconda virtual env setup](https://i.imgur.com/Tl5tyVq.png)

#### Entering your virtual environment
To enter the virtual environment:
- Open up Anaconda Navigator
- Select "Environments" on the left hand side
- Hit the ">" arrow next to your faceswap environment and select "Open Terminal"
![Anaconda enter virtual env](https://i.imgur.com/rKSq2Pd.png)

### Faceswap
- If you are not already in your virtual environment follow [these steps](#entering-your-virtual-environment)
- Get the Faceswap repo by typing: `git clone --depth 1 https://github.com/deepfakes/faceswap.git`
- Enter the faceswap folder: `cd faceswap`

#### Easy install
- Enter the command `python setup.py` and follow the prompts:
- If you have issues/errors follow the Manual install steps below.

#### Manual install
- Install tkinter (required for the GUI) by typing: `conda install tk`
- Install requirements: `pip install -r requirements.txt`
- Install Tensorflow (either GPU or CPU version depending on your setup):
    - GPU Version: `pip install tensorflow-gpu`
    - Non GPU Version: `pip install tensorflow`

## Running Faceswap
- If you are not already in your virtual environment follow [these steps](#entering-your-virtual-environment)
- Enter the faceswap folder: `cd faceswap`
- Enter the following to see the list of commands: `python faceswap.py -h` or enter `python faceswap.py gui` to launch the GUI

## Create a desktop shortcut
A desktop shortcut can be added to easily launch straight into the faceswap GUI:

- Open Notepad
- Paste the following:
```
%USERPROFILE%\Anaconda3\envs\faceswap\python.exe %USERPROFILE%/faceswap/faceswap.py gui
```
- Save the file to your desktop as "faceswap.bat"

## Updating faceswap
It's good to keep faceswap up to date as new features are added and bugs are fixed. To do so:
- If you are not already in your virtual environment follow [these steps](#entering-your-virtual-environment)
- Enter the faceswap folder: `cd faceswap`
- Enter the following `git pull --all`
- Once the latest version has downloaded, make sure your requirements are up to date: `pip install --upgrade -r requirements.txt`

# General Install Guide
## Installing dependencies
- Python >= 3.2-3.6 64-bit (cannot be 3.7.x as Tensorflow has not been updated to provide support)
  - apt/yum install python3 (Linux)
  - [Installer](https://www.python.org/downloads/release/python-368/) (Windows)
  - [brew](https://brew.sh/) install python3 (macOS)

- [virtualenv](https://github.com/pypa/virtualenv) and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io) may help when you are not using docker.
- If you are using an Nvidia graphics card You should install CUDA (https://developer.nvidia.com/cuda-zone) and CUDNN (https://developer.nvidia.com/cudnn). If you do not plan to build Tensorflow yourself, make sure you install no higher than version 10.0 of CUDA and 7.4.x of CUDNN

## Getting the faceswap code
Simply download the code from http://github.com/deepfakes/faceswap - For development, it is recommended to use git instead of downloading the code and extracting it.

For now, extract the code to a directory where you're comfortable working with it. Navigate to it with the command line. For our example, we will use `~/faceswap/` as our project directory.

## Setup
Enter the folder that faceswap has been downloaded to and run:
```bash
python setup.py
```
If setup fails for any reason you can still manually install the packages listed within requirements.txt

### About some of the options
   - CUDA: For acceleration. Requires a good nVidia Graphics Card (which supports CUDA inside)
   - Docker: Provide a ready-made image. Hide trivial details. Get you straight to the project.
   - nVidia-Docker: Access to the nVidia GPU on host machine from inside container.

CUDA with Docker in 20 minutes.
```
INFO    The tool provides tips for installation
        and installs required python packages
INFO    Setup in Linux 4.14.39-1-MANJARO
INFO    Installed Python: 3.6.5 64bit
INFO    Installed PIP: 10.0.1
Enable  Docker? [Y/n] 
INFO    Docker Enabled
Enable  CUDA? [Y/n] 
INFO    CUDA Enabled
INFO    1. Install Docker
        https://www.docker.com/community-edition
        
        2. Install Nvidia-Docker & Restart Docker Service
        https://github.com/NVIDIA/nvidia-docker
        
        3. Build Docker Image For Faceswap
        docker build -t deepfakes-gpu -f Dockerfile.gpu .
        
        4. Mount faceswap volume and Run it
        # without gui. tools.py gui not working.
        nvidia-docker run --rm -it -p 8888:8888 \
            --hostname faceswap-gpu --name faceswap-gpu \
            -v /opt/faceswap:/srv \
            deepfakes-gpu
        
        # with gui. tools.py gui working.
        ## enable local access to X11 server
        xhost +local:
        ## enable nvidia device if working under bumblebee
        echo ON > /proc/acpi/bbswitch
        ## create container 
        nvidia-docker run -p 8888:8888 \
            --hostname faceswap-gpu --name faceswap-gpu \
            -v /opt/faceswap:/srv \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -e DISPLAY=unix$DISPLAY \
            -e AUDIO_GID=`getent group audio | cut -d: -f3` \
            -e VIDEO_GID=`getent group video | cut -d: -f3` \
            -e GID=`id -g` \
            -e UID=`id -u` \
            deepfakes-gpu
        
        5. Open a new terminal to interact with the project
        docker exec faceswap-gpu python /srv/tools.py gui
```

A successful setup log, without docker.
```
INFO    The tool provides tips for installation
        and installs required python packages
INFO    Setup in Linux 4.14.39-1-MANJARO
INFO    Installed Python: 3.6.5 64bit
INFO    Installed PIP: 10.0.1
Enable  Docker? [Y/n] n
INFO    Docker Disabled
Enable  CUDA? [Y/n] 
INFO    CUDA Enabled
INFO    CUDA version: 9.1
INFO    cuDNN version: 7
WARNING Tensorflow has no official prebuild for CUDA 9.1 currently.
        To continue, You have to build your own tensorflow-gpu.
        Help: https://www.tensorflow.org/install/install_sources
Are System Dependencies met? [y/N] y
INFO    Installing Missing Python Packages...
INFO    Installing tensorflow-gpu
INFO    Installing pathlib==1.0.1
......
INFO    Installing tqdm
INFO    Installing matplotlib
INFO    All python3 dependencies are met.
        You are good to go.
```

## Run the project
Once all these requirements are installed, you can attempt to run the faceswap tools. Use the `-h` or `--help` options for a list of options.

```bash
python faceswap.py -h
```

or run with `gui` to launch the GUI
```bash
python faceswap.py gui
```


Proceed to [../blob/master/USAGE.md](USAGE.md)

## Notes
This guide is far from complete. Functionality may change over time, and new dependencies are added and removed as time goes on. 

If you are experiencing issues, please raise them in the [faceswap-playground](https://github.com/deepfakes/faceswap-playground) repository instead of the main repo.