FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# run apt update
RUN apt update

# install git
RUN apt install -y git
# install wget
RUN apt install -y wget
# working directory
WORKDIR /workspace/
ENV CUDA_INSTALL_PATH /usr/local/cuda
ENV PATH $CUDA_INSTALL_PATH/bin:$PATH`
RUN git clone https://github.com/austinguish/gpu_apps_collection.git
WORKDIR /workspace/gpu_apps_collection/cuda/pannotia/pagerank_mod
RUN make clean
ENV VARIANT SPMV
RUN make
RUN cp pagerank_spmv /usr/local/bin/pagerank_spmv