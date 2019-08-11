FROM tensorflow/tensorflow-gpu:1.13.1

RUN apt-get update -qq -y \
 && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 git wget nano unzip curl gcc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/conda/conda.git
RUN git clone https://github.com/deepfakes/faceswap.git

#COPY requirements.txt /opt/
#RUN pip3 install --upgrade pip
#RUN pip3 --no-cache-dir install -r /opt/requirements.txt && rm /opt/requirements.txt
#RUN pip3 install jupyter matplotlib
#RUN pip3 install jupyter_http_over_ws
#RUN jupyter serverextension enable --py jupyter_http_over_ws
# patch for tensorflow:latest-gpu-py3 image
#RUN cd /usr/local/cuda/lib64 \
# && mv stubs/libcuda.so ./ \
# && ln -s libcuda.so libcuda.so.1 \
# && ldconfig

#WORKDIR "/notebooks"
#CMD ["jupyter-notebook", "--allow-root" ,"--port=8888" ,"--no-browser" ,"--ip=0.0.0.0"]
