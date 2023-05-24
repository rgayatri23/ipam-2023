FROM docker.io/nvidia/cuda:12.1.1-devel-ubuntu22.04
WORKDIR /opt
ENV DEBIAN_FRONTEND noninteractive

ARG base=/opt
ARG threads=32
ARG lammps_install=$base/lammps

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

# LAMMPS
RUN \
    wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.tar.gz                                &&      \
    tar -xvf cmake-3.26.4-linux-x86_64.tar.gz                                                                                       &&      \
    export PATH=$base/cmake-3.26.4-linux-x86_64/bin:$PATH                                                                           &&      \
    cmake --version && which cmake                                                                                                  &&      \
    wget https://github.com/lammps/lammps/archive/refs/tags/stable_23Jun2022_update4.tar.gz                                         &&      \
    tar -xvf stable_23Jun2022_update4.tar.gz                                                                                        &&      \
    cd lammps-stable_23Jun2022_update4/                                                                                             &&      \
    mkdir build && cd build                                                                                                         &&      \
    cmake \
    -D CMAKE_INSTALL_PREFIX=$lammps_install                                                                                                 \
    -D CMAKE_BUILD_TYPE=Release                                                                                                             \
    -D CMAKE_CXX_COMPILER=$PWD/../lib/kokkos/bin/nvcc_wrapper                                                                               \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON                                                                                                   \
    -D PKG_GPU=ON                                                                                                                           \
    -D PKG_ML-SNAP=ON                                                                                                                       \
    -D PKG_KOKKOS=ON -D Kokkos_ARCH_AMPERE80=ON -D Kokkos_ENABLE_CUDA=ON                                                                    \
    -D PKG_MANYBODY=ON                                                                                                                      \
    -D PKG_MOLECULE=ON                                                                                                                      \
    -D PKG_KSPACE=ON                                                                                                                        \
    -D PKG_REPLICA=ON                                                                                                                       \
    -D PKG_ASPHERE=ON                                                                                                                       \
    -D PKG_RIGID=ON                                                                                                                         \
    -D MPI_C_COMPILER=mpicc                                                                                                                 \
    -D MPI_CXX_COMPILER=mpicxx                                                                                                              \
    ../cmake                                                                                                                        &&      \
    make -j $threads                                                                                                                &&      \
    make install -j $threads                                                                                                        &&      \
    cd $base                                                                                                                        &&      \
    rm -rf stable_23Jun2022_update4.tar.gz lammps-stable_23Jun2022_update4 cmake-3.26.4-linux-x86_64.tar.gz

# SNAP input
RUN \
    cd $base                                                                                                                        &&      \
    git clone https://github.com/rgayatri23/lammps-snap-ecp-fom.git                                                                 &&      \
    cd lammps-snap-ecp-fom/

## Set paths to LAMMPS binary
ENV PATH=$PATH:$lammps_install/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lammps_install/lib64
