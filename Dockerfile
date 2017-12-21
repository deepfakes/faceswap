FROM debian:stretch

# install debian packages
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
	build-essential \ 
	g++ \ 
	git \ 
	openssh-client \ 
	# install python 2
	python \ 
	python-dev \ 
	python-pip \ 
	python-setuptools \ 
	python-virtualenv \ 
	python-wheel \ 
	pkg-config \
	# requirements for keras
	python-h5py \
	python-yaml \
	python-pydot \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# mandatory ?
RUN pip install --upgrade setuptools

# requirements for dlib
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
    cmake \
    libboost-all-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip --no-cache-dir install -r ./requirements.txt

WORKDIR /srv/
