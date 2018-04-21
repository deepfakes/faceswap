## **OpenDeepFaceSwap** is global refactoring fork of FaceSwap.

**Facesets** of FaceSwap or FakeApp are **not compatible** with this repo. You should to run extract again.

### **Features**:

- new architecture, easy to experiment with models

- works on 2GB old cards , such as GT730. Example of fake trained on 2GB gtx850m notebook in 18 hours https://www.youtube.com/watch?v=EVG13JU31d8

- face data embedded to png files, so you can sort them by `sort` command, or you can delete them, then these faces will not be converted in destination images

- automatic GPU manager, which chooses best gpu(s) and supports --multi-gpu

- new preview window

- added **--debug** option for all stages

- added **MTCNN extractor** which produce less jittered aligned face than DLIBCNN, but can produce more false faces. Look at ear:
![](https://i.imgur.com/5qLiiOV.gif)

- added **Manual extractor**, video:
[![Watch the video](https://i.imgur.com/BDrPKR2.jpg)](https://webm.video/i/ogL0DL.mp4)
![Result](https://user-images.githubusercontent.com/8076202/38454756-0fa7a86c-3a7e-11e8-9065-182b4a8a7a43.gif)

- better **nn feeder**
![](https://github.com/iperov/OpenDeepFaceSwap/blob/master/doc/nnfeedprinciple.jpg)

### **Model types**:

- **H64 (2GB+)** - half face with 64 resolution. It is as original FakeApp or FaceSwap, but better due to new sample generator, nn feeder, and ConverterMasked.

- **H128 (3GB+)** - as H64, but in 128 resolution. Better face details.

- **F128 (3GB+)** - as H128, but full face + full face match warper sample generator. Fullface 128 has less details than half face 128, but allows to cover a jaw.

- **DF (4GB+)** - @dfaker model. It is as F128, but + DSSIM loss func which excludes background around face.

- **IAEF128 (5GB+)** - new model, as dfaker, but model trying to morph src face to dst, while keeping facial features of src face. Can produce strange faces.

### **Sort tool**:

`hist` groups images by similar content

`hist-dissim` places most similar to each other images to end

`face` and `face-dissim` currently useless

Best practice for gather src faceset:

1) delete first unsorted aligned groups of images what you can to delete. Dont touch target face mixed with others.
2) `blur` -> delete ~half of them
3) `hist` -> delete groups of similar and leave only target face
4) `hist-dissim` -> delete at end of list straight looking faces
5) `face-yaw` -> just finalize faceset

Best practice for dst faces:

1) delete first unsorted aligned groups of images what you can to delete. Dont touch target face mixed with others.
2) `hist` -> delete groups of similar and leave only target face
