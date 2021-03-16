FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

#install python3.8
RUN apt-get update
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update
RUN apt-get install python3.8 -y 
RUN apt-get install python3.8-distutils -y
RUN apt-get install python3.8-tk -y
RUN apt-get install curl -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py
RUN rm get-pip.py

# install requirements
RUN apt-get install ffmpeg git -y
COPY _requirements_base.txt /opt/
COPY requirements_nvidia.txt /opt/
RUN python3.8 -m pip --no-cache-dir install -r /opt/requirements_nvidia.txt && rm /opt/_requirements_base.txt && rm /opt/requirements_nvidia.txt

RUN python3.8 -m pip install jupyter matplotlib
RUN python3.8 -m pip install jupyter_http_over_ws
RUN jupyter serverextension enable --py jupyter_http_over_ws
RUN alias python=python3.8
RUN echo "alias python=python3.8" >> /root/.bashrc
WORKDIR "/notebooks"
CMD ["jupyter-notebook", "--allow-root" ,"--port=8888" ,"--no-browser" ,"--ip=0.0.0.0"]
