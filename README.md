**Notice:** This repository is not operated or maintained by [/u/deepfakes](https://www.reddit.com/user/deepfakes/). Please read the explanation below for details.

---

# deepfakes_faceswap
Faceswap is a tool that utilizes deep learning to recognize and swap faces in pictures and videos.

## Overview
The project has multiple entry points. You will have to:
 - Gather photos (or use the one provided in the training data provided below)
 - **Extract** faces from your raw photos
 - **Train** a model on your photos (or use the one provided in the training data provided below)
 - **Convert** your sources with the model

### Extract
From your setup folder, run `python faceswap.py extract`. This will take photos from `src` folder and extract faces into `extract` folder.

### Train
From your setup folder, run `python faceswap.py train`. This will take photos from two folders containing pictures of both faces and train a model that will be saved inside the `models` folder.

### Convert
From your setup folder, run `python faceswap.py convert`. This will take photos from `original` folder and apply new faces into `modified` folder.

#### General notes:
- All of the scripts mentioned have `-h`/`--help` options with arguments that they will accept. You're smart, you can figure out how this works, right?!

Note: there is no conversion for video yet. You can use [ffmpeg](https://www.ffmpeg.org) to convert video into photos, process images, and convert images back to video.

## Training Data
**Whole project with training images and trained model (~300MB):**
https://anonfile.com/p7w3m0d5be/face-swap.zip or [click here to download](https://anonfile.com/p7w3m0d5be/face-swap.zip)

## How To setup and run the project

### Setup
Clone the repo and setup you environment. There is a Dockerfile that should kickstart you. Otherwise you can setup things manually, see in the Dockerfiles for dependencies.

Check out [../blob/master/INSTALL.md](INSTALL.md) and [../blob/master/USAGE.md](USAGE.md) for basic information on how to configure virtualenv and use the program.

You also need a modern GPU with CUDA support for best performance

**Some tips:**

Reusing existing models will train much faster than starting from nothing.  
If there is not enough training data, start with someone who looks similar, then switch the data.

#### Docker
If you prefer using Docker, You can start the project with:
 - GPU:
   - Prerequiste: Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and a CUDA driver on the host machine.
   - Build: `docker build -t deepfakes-gpu -f Dockerfile.gpu .`
   - Run: `nvidia-docker run --name deepfakes-gpu -p 8888:8888 -v [src_folder]:/src -it deepfakes-gpu`
   - Execute: `docker exec -it deepfakes bash`
     - Tested working on training.
 - CPU: 
   - Build: `docker build -t deepfakes -f Dockerfile.cpu .`                                                                     
   - Run: `docker run --rm --name deepfakes -v [src_folder]:/srv -it deepfakes bash` . `bash` can be replaced by your command line
     - Note that Dockerfile.cpu does not have all good requirments, so it will fail on some python 3 commands.
     - Also note that it does not have a GUI output, so the train.py will fail on showing image. You can comment this, or save it as a file.

## How to contribute

### For people interested in the generative models
 - Go to the 'faceswap-model' to discuss/suggest/commit alternatives to the current algorithm.

### For devs
 - Read this README entirely
 - Fork the repo
 - Download the data with the link provided below
 - Play with it
 - Check issues with the 'dev' tag
 - For devs more interested in computer vision and openCV, look at issues with the 'opencv' tag. Also feel free to add your own alternatives/improvments
 
### For non-dev advanced users
 - Read this README entirely
 - Clone the repo
 - Download the data with the link provided below
 - Play with it
 - Check issues with the 'advuser' tag
 - Also go to the 'faceswap-playground' repo and help others.

### For end-users
 - Get the code here and play with it if you can
 - You can also go to the 'faceswap-playground' repo and help or get help from others.
 - Be patient. This is relatively new technology for developers as well. Much effort is already being put into making this program easy to use for the average user. It just takes time!
 - **Notice** Any issue related to running the code has to be open in the 'faceswap-playground' project!

### For haters
Sorry, no time for that.

# About github.com/deepfakes

## What is this repo?
It is a community repository for active users.

## Why this repo?
The joshua-wu repo seems not active. Simple bugs like missing _http://_ in front of urls have not been solved since days.

## Why is it named 'deepfakes' if it is not /u/deepfakes?
 1. Because a typosquat would have happened sooner or later as project grows
 2. Because all glory go to /u/deepfakes
 3. Because it will better federate contributors and users
 
## What if /u/deepfakes feels bad about that?
This is a friendly typosquat, and it is fully dedicated to the project. If /u/deepfakes wants to take over this repo/user and drive the project, he is welcomed to do so (Raise an issue, and he will be contacted on Reddit). Please do not send /u/deepfakes messages for help with the code you find here.

# About machine learning

## How does a computer know how to recognise/shape a faces? How does machine learning work? What is a neural network?
It's complicated. Here's a good video that makes the process understandable:
[![How Machines Learn](https://img.youtube.com/vi/R9OHn5ZF4Uo/0.jpg)](https://www.youtube.com/watch?v=R9OHn5ZF4Uo)

Here's a slightly more in depth video that tries to explain the basic functioning of a neural network:
[![How Machines Learn](https://img.youtube.com/vi/aircAruvnKk/0.jpg)](https://www.youtube.com/watch?v=aircAruvnKk)

tl;dr: training data + trial and error
