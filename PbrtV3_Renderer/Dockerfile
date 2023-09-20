# FROM nvidia/cuda:10.2-devel-ubuntu18.04
FROM ubuntu:20.04

RUN apt-get update -y
# # RUN apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install wget git cmake zip -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install build-essential -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-dev

WORKDIR /opt
RUN git clone --recursive https://github.com/mmp/pbrt-v3/

# copy custom mipmap.h to avoid linear interpolation
WORKDIR /opt/pbrt-v3/src/core
COPY ./mipmap.h ./

# make
WORKDIR /opt/pbrt-v3/build
RUN cmake ../
RUN make

# Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN bash ./Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p /opt/anaconda3
ENV PATH="$PATH:/opt/anaconda3/bin"

# RUN conda install -c conda-forge jupyterlab

RUN conda install numpy -y
RUN conda install -c conda-forge opencv -y
RUN conda install tqdm -y
RUN conda install -c conda-forge matplotlib -y
RUN conda install joblib

WORKDIR /root/work/
COPY ./docker-entrypoint.sh ./
ENTRYPOINT ["./docker-entrypoint.sh"]
