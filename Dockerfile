FROM gw000/keras:2.0.6-py3-tf-cpu

# install dependencies from debian packages
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    python-pip \
    python-dev

# install dependencies from python packages
RUN pip install --upgrade setuptools

RUN pip install --upgrade keras

# 'os.scandir()' does not work in utils.py. replace with 'scandir()', and modify 'import os' to 'from scandir import scandir'
RUN pip --no-cache-dir install \
    tensorflow \
    opencv-python \
    pathlib \
    scandir \
    h5py

RUN apt-get install -y \
    cmake \
    libboost-all-dev

RUN pip --no-cache-dir install \
    scikit-image \
    # boost \
    dlib
