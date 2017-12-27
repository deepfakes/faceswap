FROM debian:stretch

# install debian packages
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install essentials
	build-essential \ 
	cmake \
	g++ \ 
	git \ 
	libboost-all-dev \
	openssh-client \ 
	# install python 3
	python3.5 \ 
	python3-pip \ 
	python3-virtualenv \ 
	python3-wheel \ 
	pkg-config \
	# requirements for keras
	python3-h5py \
	python3-yaml \
	python3-pydot \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# mandatory ?
RUN pip3 install --upgrade setuptools

COPY ./requirements.txt .
RUN pip3 --no-cache-dir install -r ./requirements.txt

WORKDIR /srv/
