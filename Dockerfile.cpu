FROM tensorflow/tensorflow:1.12.0-py3

RUN add-apt-repository -y ppa:jonathonf/ffmpeg-4 \
 && apt-get update -qq -y \
 && apt-get install -y libsm6 libxrender1 libxext-dev python3-tk ffmpeg \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install -r /opt/requirements.txt && rm /opt/requirements.txt

WORKDIR "/srv"
CMD ["/bin/bash"]
