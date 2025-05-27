FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH
WORKDIR /workspace/repository
RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install scikit-learn==0.24.2
RUN pip install torch_geometric==2.4.0
RUN pip install --no-index  torch_scatter==2.1.1 torch_sparse==0.6.17 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
RUN pip install graphlearn-torch==0.2.2
RUN pip install pycuda
RUN apt update
RUN apt install -y git
RUN pip install git+https://github.com/mlcommons/logging.git
RUN echo "alias ncu='/opt/conda/nsight-compute/2022.3.0/ncu'" >> ~/.profile
ENV CUDA_VISIBLE_DEVICES=0
# Ensure .profile is sourced for sh
ENV ENV=~/.profile
# TF32 instead of FP32 for faster compute
ENV NVIDIA_TF32_OVERRIDE=1