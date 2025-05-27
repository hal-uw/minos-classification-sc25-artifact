FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# run apt update
RUN apt update

# install git
RUN apt install -y git
# install wget
RUN apt install -y wget
# working directory
WORKDIR /workspace/
# download cmake 
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.0-rc1/cmake-3.30.0-rc1-linux-x86_64.tar.gz
# extract cmake
RUN tar -xvf cmake-3.30.0-rc1-linux-x86_64.tar.gz
# add cmake to path
ENV PATH="/workspace/cmake-3.30.0-rc1-linux-x86_64/bin:${PATH}"
ARG CACHEBUST=1
# clone the gunrock repository
RUN git clone https://github.com/austinguish/gunrock.git
# working directory
WORKDIR /workspace/gunrock
# build gunrock
RUN mkdir build
WORKDIR /workspace/gunrock/build
RUN cmake ..
RUN make -j $(nproc)
ARG target="/root/Documents/NVIDIA Nsight Compute/2022.3.0/Sections/AppClass.section"
COPY AppClass.section $target
# working directory
WORKDIR /workspace/gunrock/build/bin