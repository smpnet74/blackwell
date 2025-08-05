# Use PyTorch container with CUDA 12.8 support (compatible with RTX 5090/Blackwell CUDA 12.9)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Set environment variables for RTX 5090
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FORCE_CUDA=1
ENV VLLM_FLASH_ATTN_VERSION=2

# NCCL environment variables to fix communication issues
ENV NCCL_DEBUG=INFO
ENV NCCL_TIMEOUT=1800
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=lo
ENV NCCL_ASYNC_ERROR_HANDLING=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM via pip
RUN pip install vllm==0.10.0

# Create working directory
WORKDIR /app

# Expose the default vLLM API port
EXPOSE 8000

# Use ENTRYPOINT so arguments can be passed to vllm
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", "--host", "0.0.0.0"]