FROM docker.io/nvidia/cuda:11.7.0-devel-ubuntu20.04
WORKDIR /opt

ENV DEBIAN_FRONTEND noninteractive

RUN \
    apt-get update        && \
    apt-get install --yes    \
        --no-install-recommends \
        build-essential      \
        python3-dev          \
        gfortran             \
        git             \
        wget              && \
    apt-get install -y python3-pip && \
    apt-get clean all

RUN \
    cd /home && \
    git clone https://github.com/rgayatri23/ipam-2023.git && \
    cd ipam-2023 && \
    cd container && \
    nvcc -O3 -std=c++11 -o main.ex main.cu && \
    ./main.ex
