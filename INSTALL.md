# Installing faceswap
- [Installing faceswap](#installing-faceswap)
- [Prerequisites](#prerequisites)
  - [Hardware Requirements](#hardware-requirements)
  - [Supported operating systems](#supported-operating-systems)
- [Important before you proceed](#important-before-you-proceed)
- [Linux, Windows and macOS Install Guide](#linux-windows-and-macos-install-guide)
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
- [macOS (Apple Silicon) Install Guide](#macos-apple-silicon-install-guide)
  - [Prerequisites](#prerequisites-2)
    - [OS](#os)
    - [XCode Tools](#xcode-tools)
    - [XQuartz](#xquartz)
    - [Conda](#conda)
  - [Setup](#setup-1)
    - [Create and Activate the Environment](#create-and-activate-the-environment)
    - [faceswap](#faceswap-1)
      - [Easy install](#easy-install-1)
- [General Install Guide](#general-install-guide)
  - [Installing dependencies](#installing-dependencies)
    - [Git](#git-1)
    - [Python](#python)
    - [Virtual Environment](#virtual-environment)
  - [Getting the faceswap code](#getting-the-faceswap-code)
  - [Setup](#setup-2)
    - [About some of the options](#about-some-of-the-options)
- [Docker Install Guide](#docker-install-guide)
  - [Docker CPU](#docker-cpu)
  - [Docker Nvidia](#docker-nvidia)
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
    - Currently, Nvidia GPUs are fully supported
    - DirectX 12 AMD GPUs are supported on Windows through DirectML.
    - More modern AMD GPUs are supported on Linux through ROCm.
    - M-series Macs are supported through Tensorflow-Metal
    - If using an Nvidia GPU, then it needs to support at least CUDA Compute Capability 3.5. (Release 1.0 will work on Compute Capability 3.0)
      To see which version your GPU supports, consult this list: https://developer.nvidia.com/cuda-gpus
      Desktop cards later than the 7xx series are most likely supported.
- **A lot of patience**

## Supported operating systems
- **Windows 10/11**
  Windows 7 and 8 might work for Nvidia. Your mileage may vary.
  DirectML support is only available in Windows 10 onwards.
  Windows has an installer which will set up everything you need. See: https://github.com/deepfakes/faceswap/releases
- **Linux**
  Most Ubuntu/Debian or CentOS based Linux distributions will work. There is a Linux install script that will install and set up everything you need. See: https://github.com/deepfakes/faceswap/releases
- **macOS**
  Experimental support for GPU-accelerated, native Apple Silicon processing (e.g. Apple M1 chips). Installation instructions can be found [further down this page](#macos-apple-silicon-install-guide).
  Intel based macOS systems should work, but you will need to follow the [Manual Install](#manual-install) instructions.
- All operating systems must be 64-bit for Tensorflow to run.

Alternatively, there is a docker image that is based on Debian.

# Important before you proceed
**In its current iteration, the project relies heavily on the use of the command line, although a gui is available. if you are unfamiliar with command line tools, you may have difficulty setting up the environment and should perhaps not attempt any of the steps described in this guide.** This guide assumes you have intermediate knowledge of the command line.

The developers are also not responsible for any damage you might cause to your own computer.

# Linux, Windows and macOS Install Guide

## Installer
Windows, Linux and macOS all have installers which set up everything for you. You can download the installer from https://github.com/deepfakes/faceswap/releases.

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
    - **IMPORTANT**: Select python version 3.10
    - Hit "Create" (NB: This may take a while as it will need to download Python)
![Anaconda virtual env setup](https://i.imgur.com/CLIDDfa.png)

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
  - For Nvidia GPU users: `pip install -r ./requirements/requirements_nvidia.txt`
  - For CPU users: `pip install -r ./requirements/requirements_cpu.txt`

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

# macOS (Apple Silicon) Install Guide

macOS now has [an installer](#linux-windows-and-macos-install-guide) which sets everything up for you, but if you run into difficulties and need to set things up manually, the steps are as follows:

## Prerequisites

### OS
macOS 12.0+

### XCode Tools
```sh
xcode-select --install
```

### XQuartz
Download and install from:
- https://www.xquartz.org/

### Conda
Download and install the latest Conda env from:
- https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

Install Conda:
```sh
$ chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
$ sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
$ source ~/miniforge3/bin/activate
```
## Setup
### Create and Activate the Environment
```sh
$ conda create --name faceswap python=3.10
$ conda activate faceswap
```

### faceswap
- Download the faceswap repo and enter the faceswap folder:
```sh
$ git clone --depth 1 https://github.com/deepfakes/faceswap.git
$ cd faceswap
```

#### Easy install
```sh
$ python setup.py
```

- If you have issues/errors follow the Manual install steps below.


# General Install Guide

## Installing dependencies
### Git
Git is required for obtaining the code and keeping your codebase up to date.
Obtain git for your distribution from the [git website](https://git-scm.com/downloads).

### Python
The recommended install method is to use a Conda3 Environment as this will handle the installation of Nvidia's CUDA and cuDNN straight into your Conda Environment. This is by far the easiest and most reliable way to setup the project.
  - MiniConda3 is recommended: [MiniConda3](https://docs.conda.io/en/latest/miniconda.html)

Alternatively you can install Python (3.10 64-bit) for your distribution (links below.) If you go down this route and are using an Nvidia GPU you should install CUDA (https://developer.nvidia.com/cuda-zone) and cuDNN (https://developer.nvidia.com/cudnn). for your system. If you do not plan to build Tensorflow yourself, make sure you install the correct Cuda and cuDNN package for the currently installed version of Tensorflow (Current release: Tensorflow 2.9. Release v1.0: Tensorflow 1.15). You can check for the compatible versions here: (https://www.tensorflow.org/install/source#gpu).
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
If setup fails for any reason you can still manually install the packages listed within the files in the requirements folder.

### About some of the options
   - CUDA: For acceleration. Requires a good nVidia Graphics Card (which supports CUDA inside)
   - Docker: Provide a ready-made image. Hide trivial details. Get you straight to the project.
   - nVidia-Docker: Access to the nVidia GPU on host machine from inside container.

# Docker Install Guide

This Faceswap repo contains Docker build scripts for CPU and Nvidia backends. The scripts will set up a Docker container for you and install the latest version of the Faceswap software.

You must first ensure that Docker is installed and running on your system. Follow the guide for downloading and installing Docker from their website:

  - https://www.docker.com/get-started

Once Docker is installed and running, follow the relevant steps for your chosen backend
## Docker CPU
To run the CPU version of Faceswap follow these steps:

1. Build the Docker image For faceswap:
```
docker build \
-t faceswap-cpu \
https://raw.githubusercontent.com/deepfakes/faceswap/master/Dockerfile.cpu
```
2. Launch and enter the Faceswap container:

    a. For the **headless/command line** version of Faceswap run:
    ```
    docker run --rm -it faceswap-cpu
    ```
    You can then execute faceswap the standard way:
    ```
    python faceswap.py --help
    ```
    b. For the **GUI** version of Faceswap run:
    ```
    xhost +local: && \
    docker run --rm -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=${DISPLAY} \
    faceswap-cpu
    ```
    You can then launch the GUI with
    ```
    python faceswap.py gui
    ```
  ## Docker Nvidia
To build the NVIDIA GPU version of Faceswap, follow these steps:

1. Nvidia Docker builds need extra resources to provide the Docker container with access to your GPU.

    a. Follow the instructions to install and apply the `Nvidia Container Toolkit` for your distribution from:
    -  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

    b. If Docker is already running, restart it to pick up the changes made by the Nvidia Container Toolkit.

2. Build the Docker image For faceswap
```
docker build \
-t faceswap-gpu \
https://raw.githubusercontent.com/deepfakes/faceswap/master/Dockerfile.gpu
```
1. Launch and enter the Faceswap container:

    a. For the **headless/command line** version of Faceswap run:
    ```
    docker run --runtime=nvidia --rm -it faceswap-gpu
    ```
    You can then execute faceswap the standard way:
    ```
    python faceswap.py --help
    ```
    b. For the **GUI** version of Faceswap run:
    ```
    xhost +local: && \
    docker run --runtime=nvidia --rm -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=${DISPLAY} \
    faceswap-gpu
    ```
    You can then launch the GUI with
    ```
    python faceswap.py gui
    ```
# Run the project
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
