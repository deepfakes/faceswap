# Installing faceswap
- [Installing faceswap](#installing-faceswap)
- [Prerequisites](#prerequisites)
  - [Hardware Requirements](#hardware-requirements)
  - [Supported operating systems](#supported-operating-systems)
- [Important before you proceed](#important-before-you-proceed)
- [Linux and Windows Install Guide](#linux-and-windows-install-guide)
  - [Installer](#installer)
  - [Manual Install](#manual-install)
  - [Prerequisites](#prerequisites-1)
    - [Anaconda](#anaconda)
    - [Git](#git)
  - [Setup](#setup)
    - [Anaconda](#anaconda-1)
      - [Set up a virtual environment](#set-up-a-virtual-environment)
      - [Entering your virtual environment](#entering-your-virtual-environment)
    - [faceswap](#faceswap)
      - [Easy install](#easy-install)
      - [Manual install](#manual-install-1)
  - [Running faceswap](#running-faceswap)
  - [Create a desktop shortcut](#create-a-desktop-shortcut)
  - [Updating faceswap](#updating-faceswap)
- [General Install Guide](#general-install-guide)
  - [Installing dependencies](#installing-dependencies)
    - [Git](#git-1)
    - [Python](#python)
    - [Virtual Environment](#virtual-environment)
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
    - Currently, Nvidia GPUs are fully supported. and AMD graphics cards are partially supported through plaidML.
    - If using an Nvidia GPU, then it needs to support at least CUDA Compute Capability 3.0 or higher.
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

# Linux and Windows Install Guide

## Installer
Windows and Linux now both have an installer which installs everything for you and creates a desktop shortcut to launch straight into the GUI. You can download the installer from https://github.com/deepfakes/faceswap/releases.

If you have issues with the installer then read on for the more manual way to install faceswap on Windows.

## Manual Install

Setting up faceswap can seem a little intimidating to new users, but it isn't that complicated, although a little time consuming. It is recommended to use Linux where possible as Windows will hog about 20% of your GPU Memory, making faceswap run a little slower, however using Windows is perfectly fine and 100% supported.

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
    - **IMPORTANT**: Select python version 3.7
    - Hit "Create" (NB: This may take a while as it will need to download Python 3.7)
![Anaconda virtual env setup](https://i.imgur.com/59RHnLs.png)

#### Entering your virtual environment
To enter the virtual environment:
- Open up Anaconda Navigator
- Select "Environments" on the left hand side
- Hit the ">" arrow next to your faceswap environment and select "Open Terminal"
![Anaconda enter virtual env](https://i.imgur.com/rKSq2Pd.png)

### faceswap
- If you are not already in your virtual environment follow [these steps](#entering-your-virtual-environment)
- Get the faceswap repo by typing: `git clone --depth 1 https://github.com/deepfakes/faceswap.git`
- Enter the faceswap folder: `cd faceswap`

#### Easy install
- Enter the command `python setup.py` and follow the prompts:
- If you have issues/errors follow the Manual install steps below.

#### Manual install
Do not follow these steps if the Easy Install above completed succesfully.
If you are using an Nvidia card make sure you have the correct versions of Cuda/cuDNN installed for the required version of Tensorflow
- Install tkinter (required for the GUI) by typing: `conda install tk`
- Install requirements:
  - For Nvidia GPU users: `pip install -r requirements_nvidia.txt`
  - For AMD GPU users: `pip install -r requirements_amd.txt`
  - For CPU users: `pip install -r requirements_cpu.txt`

## Running faceswap
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
- If using the GUI you can go to the Help menu and select "Check for Updates...". If updates are available go to the Help menu and select "Update Faceswap". Restart Faceswap to complete the update.
- If you are not already in your virtual environment follow [these steps](#entering-your-virtual-environment)
- Enter the faceswap folder: `cd faceswap`
- Enter the following `git pull --all`
- Once the latest version has downloaded, make sure your dependencies are up to date. There is a script to help with this: `python update_deps.py`

# General Install Guide
## Installing dependencies
### Git
Git is required for obtaining the code and keeping your codebase up to date.
Obtain git for your distribution from the [git website](https://git-scm.com/downloads).

### Python
The recommended install method is to use a Conda3 Environment as this will handle the installation of Nvidia's CUDA and cuDNN straight into your Conda Environment. This is by far the easiest and most reliable way to setup the project.
  - MiniConda3 is recommended: [MiniConda3](https://docs.conda.io/en/latest/miniconda.html)
  
Alternatively you can install Python (>= 3.6-3.7 64-bit) for your distribution (links below.) If you go down this route and are using an Nvidia GPU you should install CUDA (https://developer.nvidia.com/cuda-zone) and cuDNN (https://developer.nvidia.com/cudnn). for your system. If you do not plan to build Tensorflow yourself, make sure you install no higher than version 10.0 of CUDA and 7.5.x of CUDNN.
  - Python distributions:
    - apt/yum install python3 (Linux)
    - [Installer](https://www.python.org/downloads/release/python-368/) (Windows)
    - [brew](https://brew.sh/) install python3 (macOS)

### Virtual Environment
  It is highly recommended that you setup faceswap inside a virtual environment. In fact we will not generally support installations that are not within a virtual environment as troubleshooting package conflicts can be next to impossible.

  If using Conda3 then setting up virtual environments is relatively straight forward. More information can be found at [Conda Docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

  If using a default Python distribution then [virtualenv](https://github.com/pypa/virtualenv) and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io) may help when you are not using docker.


## Getting the faceswap code
It is recommended to clone the repo with git instead of downloading the code from http://github.com/deepfakes/faceswap and extracting it as this will make it far easier to get the latest code (which can be done from the GUI). To clone a repo you can either use the Git GUI for your distribution or open up a command prompt, enter the folder where you want to store faceswap and enter:
```bash
git clone https://github.com/deepfakes/faceswap.git
```


## Setup
Enter your virtual environment and then enter the folder that faceswap has been downloaded to and run:
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
        
        1. Install Nvidia-Docker & Restart Docker Service
        https://github.com/NVIDIA/nvidia-docker
        
        1. Build Docker Image For faceswap
        docker build -t deepfakes-gpu -f Dockerfile.gpu .
        
        1. Mount faceswap volume and Run it
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
        
        1. Open a new terminal to interact with the project
        docker exec faceswap-gpu python /srv/faceswap.py gui
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

If you are experiencing issues, please raise them in the [faceswap Forum](https://faceswap.dev/forum) instead of the main repo. Usage questions raised in the issues within this repo are liable to be closed without response.
