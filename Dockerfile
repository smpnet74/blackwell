# Multi-stage build for vLLM with RTX 5090 multi-GPU support
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    cmake \
    libaio-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Build and install latest NCCL to fix P2P issues with RTX 5090
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j src.build && \
    make install && \
    ldconfig

# Install PyTorch with CUDA 12.4 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Clone and build vLLM from source with custom NCCL
WORKDIR /tmp
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git checkout v0.10.0 && \
    VLLM_NCCL_ROOT=/usr/local pip3 install -e .

# Production stage
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

# Set environment variables for multi-GPU support
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/local/lib:${LD_LIBRARY_PATH}
ENV NCCL_P2P_DISABLE=1
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=^docker0,lo

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libaio1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Copy NCCL libraries from builder
COPY --from=builder /usr/local/lib/libnccl* /usr/local/lib/
COPY --from=builder /usr/local/include/nccl.h /usr/local/include/

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Update library cache
RUN ldconfig

# Create working directory
WORKDIR /app

# Expose the default vLLM API port
EXPOSE 8000

# Default command to start vLLM OpenAI API server
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--host", "0.0.0.0", "--port", "8000"]