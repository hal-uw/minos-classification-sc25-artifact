FROM gcyang/openfold:0.1


RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    gnupg \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Remove any existing CUDA repository configuration
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/preferences.d/cuda*.pref

# Add your CUDA repository configuration
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | apt-key add - && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    nsight-compute-2023.1.1 && \
    rm -rf /var/lib/apt/lists/*



