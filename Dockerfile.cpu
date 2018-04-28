FROM debian:stretch

# install debian packages
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
	build-essential \ 
	# install python 3
	python3.5 \ 
	python3-dev \
	python3-pip \ 
	python3-wheel \ 
	# Boost for dlib
	cmake \
	libboost-all-dev \ 
	# requirements for keras
	python3-h5py \
	python3-yaml \
	python3-pydot \
	python3-setuptools \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY ./requirements-python35.txt .
RUN pip3 --no-cache-dir install -r ./requirements-python35.txt

WORKDIR /srv/
