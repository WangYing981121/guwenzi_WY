ARG PADDLE_DOCKER_PLATFORM=linux/amd64
FROM --platform=${PADDLE_DOCKER_PLATFORM} nvcr.io/nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda-12.0/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat

RUN mkdir -p /app /saisresult

RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
            /etc/apt/sources.list; \
    fi; \
    find /etc/apt/sources.list.d -type f \( -name '*.list' -o -name '*.sources' \) -exec sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|https://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        -e 's|https://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
        {} +; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        tini \
        bash \
        wget \
        ca-certificates \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxrender1 \
        libxext6; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

ENV PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN set -eux; \
    python3 -m pip install --upgrade "pip<25" setuptools wheel

# Install PyTorch with CUDA 11.8 support (compatible with CUDA 12.0 runtime)
RUN set -eux; \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining Python dependencies
RUN set -eux; \
    python3 -m pip install --prefer-binary -r /app/requirements.txt

# Copy model weights and source code
COPY src/ /app/src/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Warm up models at build time (faster container startup)
RUN python3 /app/src/warmup_models.py || echo "Warmup skipped (expected during cross-platform build)"

ENTRYPOINT ["/usr/bin/tini", "--", "bash", "/app/run.sh"]
