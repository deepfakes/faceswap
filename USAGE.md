# Workflow

**Before attempting any of this, please make sure you have read, understood and completed the [installation instructions](../master/INSTALL.md). If you are experiencing issues, please raise them in the [faceswap Forum](https://faceswap.dev/forum) or the [FaceSwap Discord server](https://discord.gg/FdEwxXd) instead of the main repo.**

- [Workflow](#workflow)
- [Introduction](#introduction)
  - [Disclaimer](#disclaimer)
  - [Getting Started](#getting-started)
- [Extract](#extract)
  - [Gathering raw data](#gathering-raw-data)
  - [Extracting Faces](#extracting-faces)
  - [General Tips](#general-tips)
- [Training a model](#training-a-model)
  - [General Tips](#general-tips-1)
- [Converting a video](#converting-a-video)
  - [General Tips](#general-tips-2)
- [GUI](#gui)
- [Video's](#videos)
- [EFFMPEG](#effmpeg)
- [Extracting video frames with FFMPEG](#extracting-video-frames-with-ffmpeg)
- [Generating a video](#generating-a-video)
- [Notes](#notes)
  
# Introduction

## Disclaimer
This guide provides a high level overview of the faceswapping process. It does not aim to go into every available option, but will provide a useful entry point to using the software. There are many more options available that are not covered by this guide. These can be found, and explained, by passing the `-h` flag to the command line (eg: `python faceswap.py extract -h`) or by hovering over the options within the GUI.

## Getting Started
So, you want to swap faces in pictures and videos? Well hold up, because first you gotta understand what this application will do, how it does it and what it can't currently do.

The basic operation of this script is simple. It trains a machine learning model to recognize and transform two faces based on pictures. The machine learning model is our little "bot" that we're teaching to do the actual swapping and the pictures are the "training data" that we use to train it. Note that the bot is primarily processing faces. Other objects might not work.

So here's our plan. We want to create a reality where Donald Trump lost the presidency to Nic Cage; we have his inauguration video; let's replace Trump with Cage.

# Extract
## Gathering raw data
In order to accomplish this, the bot needs to learn to recognize both face A (Trump) and face B (Nic Cage). By default, the bot doesn't know what a Trump or a Nic Cage looks like. So we need to show it lots of pictures and let it guess which is which. So we need pictures of both of these faces first.

A possible source is Google, DuckDuckGo or Bing image search. There are scripts to download large amounts of images. A better source of images are videos (from interviews, public speeches, or movies) as these will capture many more natural poses and expressions. Fortunately FaceSwap has you covered and can extract faces from both still images and video files. See [Extracting video frames](#Extracting_video_frames) for more information.

Feel free to list your image sets in the [faceswap Forum](https://faceswap.dev/forum), or add more methods to this file.

So now we have a folder full of pictures/videos of Trump and a separate folder of Nic Cage. Let's save them in our directory where we put the FaceSwap project. Example: `~/faceswap/src/trump` and `~/faceswap/src/cage`

## Extracting Faces
So here's a problem. We have a ton of pictures and videos of both our subjects, but these are just of them doing stuff or in an environment with other people. Their bodies are on there, they're on there with other people... It's a mess. We can only train our bot if the data we have is consistent and focuses on the subject we want to swap. This is where FaceSwap first comes in.

**Command Line:**
```bash
# To extract trump from photos in a folder:
python faceswap.py extract -i ~/faceswap/src/trump -o ~/faceswap/faces/trump
# To extract trump from a video file:
python faceswap.py extract -i ~/faceswap/src/trump.mp4 -o ~/faceswap/faces/trump
# To extract cage from photos in a folder:
python faceswap.py extract -i ~/faceswap/src/cage -o ~/faceswap/faces/cage
# To extract cage from a video file:
python faceswap.py extract -i ~/faceswap/src/cage.mp4 -o ~/faceswap/faces/cage
```

**GUI:**

To extract trump from photos in a folder (Right hand folder icon):
![ExtractFolder](https://i.imgur.com/H3h0k36.jpg)

To extract cage from a video file (Left hand folder icon):
![ExtractVideo](https://i.imgur.com/TK02F0u.jpg)

For input we either specify our photo directory or video file and for output we specify the folder where our extracted faces will be saved. The script will then try its best to recognize face landmarks, crop the images to a consistent size, and save the faces to the output folder. An `alignments.json` file will also be created and saved into your input folder. This file contains information about each of the faces that will be used by FaceSwap.

Note: this script will make grabbing test data much easier, but it is not perfect. It will (incorrectly) detect multiple faces in some photos and does not recognize if the face is the person whom we want to swap. Therefore: **Always check your training data before you start training.** The training data will influence how good your model will be at swapping.

## General Tips
When extracting faces for training, you are looking to gather around 500 to 5000 faces for each subject you wish to train. These should be of a high quality and contain a wide variety of angles, expressions and lighting conditions. 

You do not want to extract every single frame from a video for training as from frame to frame the faces will be very similar.

You can see the full list of arguments for extracting by hovering over the options in the GUI or passing the help flag. i.e:
```bash
python faceswap.py extract -h
```

Some of the plugins have configurable options. You can find the config options in: `<faceswap_folder>\config\extract.ini`. You will need to have run Extract or the GUI at least once for this file to be generated.

# Training a model
Ok, now you have a folder full of Trump faces and a folder full of Cage faces. What now? It's time to train our bot! This creates a 'model' that contains information about what a Cage is and what a Trump is and how to swap between the two.

The training process will take the longest, how long depends on many factors; the model used, the number of images, your GPU etc. However, a ballpark figure is 12-48 hours on GPU and weeks if training on CPU.

We specify the folders where the two faces are, and where we will save our training model.

**Command Line:**
```bash
python faceswap.py train -A ~/faceswap/faces/trump -B ~/faceswap/faces/cage -m ~/faceswap/trump_cage_model/
# or -p to show a preview
python faceswap.py train -A ~/faceswap/faces/trump -B ~/faceswap/faces/cage -m ~/faceswap/trump_cage_model/ -p 
```
**GUI:**

![Training](https://i.imgur.com/j8bjk4I.jpg)

Once you run the command, it will start hammering the training data. If you have a preview up, then you will see a load of blotches appear. These are the faces it is learning. They don't look like much, but then your model hasn't learned anything yet. Over time these will more and more start to resemble trump and cage.

You want to leave your model learning until you are happy with the images in the preview. To stop training you can:
- Command Line: press "Enter" in the preview window or in the console
- GUI: Press the Terminate button

When stopping training, the model will save and the process will exit. This can take a little while, so be patient. The model will also save every 100 iterations or so.

You can stop and resume training at any time. Just point FaceSwap at the same folders and carry on.

## General Tips
If you are training with a mask or using Warp to Landmarks, you will need to pass in an `alignments.json` file for each of the face sets. See [Extract - General Tips](#general-tips) for more information.

The model is automatically backed up at every save iteration where the overall loss has dropped (i.e. the model has improved). If your model corrupts for some reason, you can go into the model folder and remove the `.bk` extension from the backups to restore the model from backup.

You can see the full list of arguments for training by hovering over the options in the GUI or passing the help flag. i.e:

```bash
python faceswap.py train -h
```

Some of the plugins have configurable options. You can find the config options in: `<faceswap_folder>\config\train.ini`. You will need to have run Train or the GUI at least once for this file to be generated.


# Converting a video
Now that we're happy with our trained model, we can convert our video. How does it work? 

Well firstly we need to generate an `alignments.json` file for our swap. To do this, follow the steps in [Extracting Faces](#extracting-faces), only this time you want to run extract for every face in your source video. This file tells the convert process where the face is on the source frame.

You are likely going to want to cleanup your alignments file, by deleting false positives, badly aligned faces etc. These will not look good on your final convert. There are tools to help with this.

Just like extract you can convert from a series of images or from a video file.

Remember those initial pictures we had of Trump? Let's try swapping a face there. We will use that directory as our input directory, create a new folder where the output will be saved, and tell them which model to use.

**Command Line:**
```bash
python faceswap.py convert -i ~/faceswap/src/trump/ -o ~/faceswap/converted/ -m ~/faceswap/trump_cage_model/
```

**GUI:**

![convert](https://i.imgur.com/GzX1ME2.jpg)

It should now start swapping faces of all these pictures.


## General Tips
You can see the full list of arguments for Converting by hovering over the options in the GUI or passing the help flag. i.e:

```bash
python faceswap.py convert -h
```

Some of the plugins have configurable options. You can find the config options in: `<faceswap_folder>\config\convert.ini`. You will need to have run Convert or the GUI at least once for this file to be generated.

# GUI
All of the above commands and options can be run from the GUI. This is launched with:
```bash
python faceswap.py gui
```

The GUI allows a more user friendly interface into the scripts and also has some extended functionality. Hovering over options in the GUI will tell you more about what the option does.

# Video's
A video is just a series of pictures in the form of frames. Therefore you can gather the raw images from them for your dataset or combine your results into a video.

# EFFMPEG
You can perform various video processes with the built-in effmpeg tool. You can see the full list of arguments available by running:
```bash
python tools.py effmpeg -h
```

# Extracting video frames with FFMPEG
Alternatively, you can split a video into separate frames using [ffmpeg](https://www.ffmpeg.org) for instance. Below is an example command to process a video to separate frames.

```bash
ffmpeg -i /path/to/my/video.mp4 /path/to/output/video-frame-%d.png
```

# Generating a video
If you split a video, using [ffmpeg](https://www.ffmpeg.org) for example, and used them as a target for swapping faces onto you can combine these frames again. The command below stitches the png frames back into a single video again.

```bash
ffmpeg -i video-frame-%0d.png -c:v libx264 -vf "fps=25,format=yuv420p" out.mp4
```

# Notes
This guide is far from complete. Functionality may change over time, and new dependencies are added and removed as time goes on. 

If you are experiencing issues, please raise them in the [faceswap Forum](https://faceswap.dev/forum) or the [FaceSwap Discord server](https://discord.gg/FdEwxXd). Usage questions raised in this repo are likely to be closed without response.
