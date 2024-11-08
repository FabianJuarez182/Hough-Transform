FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*