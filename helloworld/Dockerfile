FROM docker.io/nvidia/cuda:12.1.1-devel-ubuntu22.04
WORKDIR /opt
ENV DEBIAN_FRONTEND noninteractive

ARG base=/opt
ARG threads=32
ARG lammps_install=$base/lammps
ARG mpich_install=$base/mpich/

# Basic stuff
RUN \
    apt-get update                                                                                                                  && \
    apt-get install --yes                                                                                                              \
        --no-install-recommends                                                                                                        \
        build-essential                                                                                                                \
        python3-dev                                                                                                                    \
        gfortran                                                                                                                       \
        git                                                                                                                            \
        wget                                                                                                                        && \
    apt-get install -y python3-pip                                                                                                  && \
    apt-get clean all

ARG mpich_ver=4.1.1
ARG mpich_prefix=mpich-$mpich_ver

RUN \
    wget https://www.mpich.org/static/downloads/$mpich_ver/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                               && \
    cd $mpich_prefix                                                            && \
    ./configure                                                                 && \
    make -j 4                                                                   && \
    make install                                                                && \
    make clean                                                                  && \
    cd ..                                                                       && \
    rm -rf $mpich_prefix

RUN /sbin/ldconfig


RUN \
    git clone --single-branch --branch container_build https://github.com/rgayatri23/HelloWorld.git && \
    cd HelloWorld && \
    make all
