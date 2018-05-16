**Before attempting any of this, please make sure you have read, understood and completed the [installation instructions](../master/INSTALL.md). If you are experiencing issues, please raise them in the [faceswap-playground](https://github.com/deepfakes/faceswap-playground) repository instead of the main repo.**

# Workflow
So, you want to swap faces in pictures and videos? Well hold up, because first you gotta understand what this collection of scripts will do, how it does it and what it can't currently do.

The basic operation of this script is simple. It trains a machine learning model to recognize and transform two faces based on pictures. The machine learning model is our little "bot" that we're teaching to do the actual swapping and the pictures are the "training data" that we use to train it. Note that the bot is primarily processing faces. Other objects might not work.

So here's our plan. We want to create a reality where Donald Trump lost the presidency to Nic Cage; we have his inauguration video; let's replace Trump with Cage.

## Gathering raw data
In order to accomplish this, the bot needs to learn to recognize both face A (Trump) and face B (Nic Cage). By default, the bot doesn't know what a Trump or a Nic Cage looks like. So we need to show it some pictures and let it guess which is which. So we need pictures of both of these faces first.

A possible source is Google, DuckDuckGo or Bing image search. There are scripts to download large amounts of images. Alternatively, if you have a video of the person you're looking for (from interviews, public speeches, or movies), you can convert this video to still images and use those. see [Extracting video frames](#Extracting_video_frames) for more information.

Feel free to list your image sets in the [faceswap-playground](https://github.com/deepfakes/faceswap-playground), or add more methods to this file.

So now we have a folder full of pictures of Trump and a separate folder of Nic Cage. Let's save them in our directory where we put the faceswap project. Example: `~/faceswap/photo/trump` and `~/faceswap/photo/cage`

## EXTRACT
So here's a problem. We have a ton of pictures of both our subjects, but they're just pictures of them doing stuff or in an environment with other people. Their bodies are on there, they're on there with other people... It's a mess. We can only train our bot if the data we have is consistent and focusses on the subject we want to swap. This is where faceswap first comes in.

```bash
# To convert trump:
python faceswap.py extract -i ~/faceswap/photo/trump -o ~/faceswap/data/trump
# To convert cage:
python faceswap.py extract -i ~/faceswap/photo/cage -o ~/faceswap/data/cage
```

We specify our photo input directory and the output folder where our training data will be saved. The script will then try its best to recognize face landmarks, crop the image to that size, and save it to the output folder. Note: this script will make grabbing test data much easier, but it is not perfect. It will (incorrectly) detect multiple faces in some photos and does not recognize if the face is the person who we want to swap. Therefore: **Always check your training data before you start training.** The training data will influence how good your model will be at swapping.

You can see the full list of arguments for extracting via help flag. i.e.

```bash
python faceswap.py extract -h
```

## TRAIN
The training process will take the longest, especially on CPU. We specify the folders where the two faces are, and where we will save our training model. It will start hammering the training data once you run the command. I personally really like to go by the preview and quit the processing once I'm happy with the results.

```bash
python faceswap.py train -A ~/faceswap/data/trump -B ~/faceswap/data/cage -m ~/faceswap/models/
# or -p to show a preview
python faceswap.py train -A ~/faceswap/data/trump -B ~/faceswap/data/cage -m ~/faceswap/models/ -p 
```

If you use the preview feature, select the preview window and press ENTER to save your processed data and quit gracefully. Without the preview enabled, you might have to forcefully quit by hitting Ctrl+C to cancel the command. Note that it will save the model once it's gone through about 100 iterations, which can take quite a while. So make sure you save before stopping the process.

You can see the full list of arguments for training via help flag. i.e.

```bash
python faceswap.py train -h
```

## CONVERT
Now that we're happy with our trained model, we can convert our video. How does it work? Similarly to the extraction script, actually! The conversion script basically detects a face in a picture using the same algorithm, quickly crops the image to the right size, runs our bot on this cropped image of the face it has found, and then (crudely) pastes the processed face back into the picture.

Remember those initial pictures we had of Trump? Let's try swapping a face there. We will use that directory as our input directory, create a new folder where the output will be saved, and tell them which model to use.

```bash
python faceswap.py convert -i ~/faceswap/photo/trump/ -o ~/faceswap/output/ -m ~/faceswap/models/
```

It should now start swapping faces of all these pictures.

You can see the full list of arguments available for converting via help flag. i.e.

```bash
python faceswap.py convert -h
```

## GUI
All of the above commands and options can be run from the GUI. This is launched with:
```bash
python faceswap.py gui
```



## Video's
A video is just a series of pictures in the form of frames. Therefore you can gather the raw images from them for your dataset or combine your results into a video.

## EFFMPEG
You can perform various video processes with the built in effmpeg tool. You can see the full list of arguments available by running:
```bash
python tools.py effmpeg -h
```

## Extracting video frames with FFMPEG
Alternatively you can split a video into seperate frames using [ffmpeg](https://www.ffmpeg.org) for instance. Below is an example command to process a video to seperate frames.

```bash
ffmpeg -i /path/to/my/video.mp4 /path/to/output/video-frame-%d.png
```

## Generating a video
If you split a video, using [ffmpeg](https://www.ffmpeg.org) for example, and used them as a target for swapping faces onto you can combine these frames again. The command below stitches the png frames back into a single video again.

```bash
ffmpeg -i video-frame-%0d.png -c:v libx264 -vf "fps=25,format=yuv420p" out.mp4
```

## Notes
This guide is far from complete. Functionality may change over time, and new dependencies are added and removed as time goes on. 

If you are experiencing issues, please raise them in the [faceswap-playground](https://github.com/deepfakes/faceswap-playground) repository instead of the main repo.
