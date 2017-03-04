FROM gcr.io/tensorflow/tensorflow:latest-gpu
MAINTAINER Fabian Chan <fabianc@stanford.edu>

RUN apt-get update
RUN apt-get install -y vim curl \
    python-numpy \
    python-scipy \
    python-h5py \
    python-yaml

RUN apt-get install git -y
    
ARG KERAS_VERSION=1.2.2
ENV KERAS_BACKEND=tensorflow
RUN pip --no-cache-dir install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

#RUN pip install h5py
#RUN pip install tflearn

RUN git clone https://www.github.com/datalogai/recurrentshop.git
WORKDIR recurrentshop
RUN python setup.py install

