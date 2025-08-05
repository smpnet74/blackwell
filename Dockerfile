# Use NVIDIA's PyTorch container with RTX 5090/Blackwell support
FROM nvcr.io/nvidia/pytorch:25.02-py3

# Set environment variables for RTX 5090
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FORCE_CUDA=1
ENV VLLM_FLASH_ATTN_VERSION=2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone and install vLLM from source
WORKDIR /tmp
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git checkout v0.10.0 && \
    pip install -e . --no-build-isolation

# Create working directory
WORKDIR /app

# Expose the default vLLM API port
EXPOSE 8000

# Default command to start vLLM OpenAI API server
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--host", "0.0.0.0", "--port", "8000"]