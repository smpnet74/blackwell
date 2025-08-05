# Use official vLLM OpenAI image
FROM vllm/vllm-openai:v0.10.0

# Set environment variables for RTX 5090/Blackwell compatibility
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV VLLM_FLASH_ATTN_VERSION=2

# NCCL environment variables to fix RTX 5090 multi-GPU communication issues
ENV NCCL_DEBUG=INFO
ENV NCCL_TIMEOUT=1800
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=lo
ENV NCCL_ASYNC_ERROR_HANDLING=1

# Fix NCCL version to resolve multi-GPU freezing issues
RUN pip uninstall -y nvidia-nccl-cu12 && \
    pip install nvidia-nccl-cu12==2.26.2.post1