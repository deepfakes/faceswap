FROM gw000/keras:2.0.6-py3-tf-cpu

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    python-pip \
    python-dev

COPY requirements.txt .

# install dependencies from python packages
RUN pip install --upgrade setuptools

RUN pip3 --no-cache-dir install -r ./requirements.txt

RUN apt-get install -y \
    cmake \
    libboost-all-dev

# Add these to requirements.txt
RUN pip3 --no-cache-dir install \
    scikit-image \
    # boost \
    dlib
