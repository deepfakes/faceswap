**OpenDeepFaceSwap** is global refactoring fork of FaceSwap.

**Facesets** of FaceSwap or FakeApp are **not compatible** with this repo. You should to run extract again.

**Model types**:

**H64 (2GB+)** - half face with 64 resolution. It is as original FakeApp or FaceSwap, but better due to new sample generator, nn feeder, and ConverterMasked.

**H128 (3GB+)** - as H64, but in 128 resolution. Better face details.

**F128 (3GB+)** - as H128, but full face + full face match warper sample generator. Fullface 128 has less details than half face 128, but allows to cover a jaw.

**DF (4GB+)** - @dfaker model. It is as F128, but + DSSIM loss func which excludes background around face.

**IAEF128 (5GB+)** - new model, as dfaker, but model trying to morph src face to dst, while keeping facial features of src face. Can produce strange faces.

**SIAEF128 (6GB+)** - as IAEF128 , but more deep morphing prediction.
