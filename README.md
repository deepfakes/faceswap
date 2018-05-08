## **OpenDeepFaceSwap** is global refactoring fork of FaceSwap.

**Facesets** of FaceSwap or FakeApp are **not compatible** with this repo. You should to run extract again.

### **Features**:

- new architecture, easy to experiment with models

- works on 2GB old cards , such as GT730. Example of fake trained on 2GB gtx850m notebook in 18 hours https://www.youtube.com/watch?v=EVG13JU31d8

- face data embedded to png files, so you can sort them by `sort` command, or you can delete them, then these faces will not be converted in destination images

- automatic GPU manager, chooses best gpu(s) and supports --multi-gpu

- new preview window

- added **--debug** option for all stages

- added **MTCNN extractor** which produce less jittered aligned face than DLIBCNN, but can produce more false faces. Look at ear:
![](https://i.imgur.com/5qLiiOV.gif)

- added **Manual extractor**. You can fix missed faces manually or do full manual extract, video:
[![Watch the video](https://i.imgur.com/BDrPKR2.jpg)](https://webm.video/i/ogL0DL.mp4)
![Result](https://user-images.githubusercontent.com/8076202/38454756-0fa7a86c-3a7e-11e8-9065-182b4a8a7a43.gif)

- new models

### **Model types**:

- **H64 (2GB+)** - half face with 64 resolution. It is as original FakeApp or FaceSwap, but with DSSIM Loss func and separated mask decoder + better ConverterMasked.
* H64 Robert Downey Jr.:
* ![](https://github.com/iperov/OpenDeepFaceSwap/blob/master/doc/H64_Downey_0.jpg)
* ![](https://github.com/iperov/OpenDeepFaceSwap/blob/master/doc/H64_Downey_1.jpg)

- **H128 (3GB+)** - as H64, but in 128 resolution. Better face details.
* H128 example - later

- **DF (4GB+)** - @dfaker model. Fullface 128 model. Contains DSSIM loss func which excludes background around face.
* DF example - later

- **LIAEF128 (4GB+)** - new model. Result of combining DF, IAE, + experiments. Model tries to morph src face to dst, while keeping facial features of src face, but less agressive morphing.
* LIAEF128 Cage:
* ![](https://github.com/iperov/OpenDeepFaceSwap/blob/master/doc/LIAEF128_Cage_0.jpg)

- **MIAEF128 (5GB+)** - as LIAEF128, but also it tries to match brightness/color features.
* MIAEF128 model diagramm:
* ![](https://github.com/iperov/OpenDeepFaceSwap/blob/master/doc/MIAEF128_diagramm.png)
* MIAEF128 Ford success case:
* ![](https://github.com/iperov/OpenDeepFaceSwap/blob/master/doc/MIAEF128_Ford_0.jpg)
* ![](https://github.com/iperov/OpenDeepFaceSwap/blob/master/doc/MIAEF128_Ford_1.jpg)
* MIAEF128 Cage fail case:
* ![](https://github.com/iperov/OpenDeepFaceSwap/blob/master/doc/MIAEF128_Cage_fail.jpg)

### **Sort tool**:

`hist` groups images by similar content

`hist-dissim` places most similar to each other images to end.

`face` and `face-dissim` currently useless

Best practice for gather src faceset:

1) delete first unsorted aligned groups of images what you can to delete. Dont touch target face mixed with others.
2) `blur` -> delete ~half of them
3) `hist` -> delete groups of similar and leave only target face
4) `hist-dissim` -> leave only first **1000-1500 faces**, because number of src faces can affect result.
5) `face-yaw` -> just for finalize faceset

Best practice for dst faces:

1) delete first unsorted aligned groups of images what you can to delete. Dont touch target face mixed with others.
2) `hist` -> delete groups of similar and leave only target face

### **Prebuilt binary**:

Windows 7,8,8.1,10 zero dependency binary except NVidia Video Drivers can be downloaded from torrent, magnet link: 

magnet:?xt=urn:btih:0E40A1864F2D4680E1E03ABD7C833EA72FBC12E2

Torrent page: https://rutracker.org/forum/viewtopic.php?p=75318742