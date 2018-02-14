# Prerequisites
Machine learning essentially involves a ton of trial and error. You're letting a program try millions of different settings to land on an algorithm that sort of does what you want it to do. This process is really really slow unless you have the hardware required to speed this up. 

The type of computations that the process does are well suited for graphics cards, rather than regular processors. **It is pretty much required that you run the training process on a desktop or server capable GPU.** Running this on your CPU means it can take weeks to train your model, compared to several hours on a GPU.

## Hardware Requirements
**TL;DR: you need at least one of the following:**

- **A powerful CPU**
    - Laptop CPUs can often run the software, but will not be fast enough to train at reasonable speeds
- **A powerful GPU**
    - Currently only Nvidia GPUs are supported. AMD graphics cards are not supported.
      This is not something that we have control over. It is a requirement of the Tensorflow library.
    - The GPU needs to support at least CUDA Compute Capability 3.0 or higher.
      To see which version your GPU supports, consult this list: https://developer.nvidia.com/cuda-gpus
      Desktop cards later than the 7xx series are most likely supported.
- **A lot of patience**

## Supported operating systems:
- **Windows 10**
  Windows 7 and 8 might work. Your milage may vary
- **Linux**
  Most Ubuntu/Debian or CentOS based Linux distributions will work.
- **macOS**
  GPU support on macOS is limited due to lack of drivers/libraries from Nvidia.

Alternatively there is a docker image that is based on Debian.

# Important before you proceed
**In its current iteration, the project relies heavily on the use of the command line. If you are unfamiliar with command line tools, you should not attempt any of the steps described in this guide.** Wait instead for this tool to become usable, or start learning more about working with the command line. This guide assumes you have intermediate knowledge of the command line. 

The developers are also not responsible for any damage you might cause to your own computer.

# Installation Instructions

## Installing dependencies


### Python >= 3.2

Note that you will need the 64bit version of Python, especially to setup the GPU version!

#### Windows
Download the latest version of Python 3 from Python.org: https://www.python.org/downloads/release/python-364

#### macOS

By default, macOS comes with Python 2.7. For best usage, need at least Python 3.2.  The easiest way to do so is to install it through `Homebrew`. If you are not familiar with `homebrew`, read more about it here: https://brew.sh/

To install Python 3.2 or higher:

```
brew install python3
```

#### Linux
You know how this works, don't you?

### Virtualenv
Install virtualenv next. Virtualenv helps us make a containing environment for our project. This means that any python packages we install for this project will be compartmentalized to this specific environment. We'll install virtualenv with `pip` which is Python's package/dependency manager.

```pip install virtualenv```

or

```pip3 install virtualenv```

Alternative, if your Linux distribution provides its own virtualenv through apt or yum, you can use that as well.

#### Windows specific:
`virtualenvwrapper-win` is a package that makes virtualenvs easier to manage on Windows.

```pip install virtualenvwrapper-win```

## Getting the faceswap code
Simply download the code from http://github.com/deepfakes/faceswap - For development it is recommended to use git instead of downloading the code and extracting it.

For now, extract the code to a directory where you're comfortable working with it. Navigate to it with the command line. For our example we will use `~/faceswap/` as our project directory.

## Setting up our virtualenv
### First steps
We will now initialize our virtualenv:

```
virtualenv faceswap_env/
```

On Windows you can use: 

```
mkvirtualenv faceswap
setprojectdir .
```

This will create a folder with python, pip, and setuptools all ready to go in its own little environment. It will also activate the Virtual Environment which is indicated with the (faceswap) on the left side of the prompt. Anything we install now will be specific to this project. And available to the projects we connect to this environment. 

Let's say you’re content with the work you’ve contributed to this project and you want to move onto something else in the command line. Simply type `deactivate` to deactivate your environment. 

To reactive your environment on Windows, you can use `workon faceswap`. On Mac and Linux, you can use `source ./faceswap_env/bin/activate`. Note that the Mac/Linux command is relative to the project and virtualenv directory.

### Setting up for our project
With your virtualenv activated, install the dependencies from the requirements files. Like so:

```bash
pip install -r requirements.txt
```

If you want to use your GPU instead of your CPU, substitute `requirements.txt` with `requirements-gpu.txt`:

```bash
pip install -r requirements-gpu.txt
```

Should you choose the GPU version, Tensorflow might ask you to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-zone) and the [cuDNN libraries](https://developer.nvidia.com/cudnn). Instructions on installing those can be found on Nvidia's website. (For Ubuntu, maybe all Linux, see: https://yangcha.github.io/Install-CUDA8)

Once all these requirements are installed, you can attempt to run the faceswap tools. Use the `-h` or `--help` options for a list of options.

```bash
python faceswap.py -h
```

Proceed to [../blob/master/USAGE.md](USAGE.md)

## Notes
This guide is far from complete. Functionality may change over time, and new dependencies are added and removed as time goes on. 

If you are experiencing issues, please raise them in the [faceswap-playground](https://github.com/deepfakes/faceswap-playground) repository instead of the main repo.
