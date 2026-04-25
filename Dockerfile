# 3D Studio runtime image.
#
# What's baked in: system deps, Python env, server-only PyPI packages from
# requirements.txt, and this repo's source. What's NOT baked in: AniGen,
# Hunyuan3D-2, and the ~75 GB of HuggingFace weights — those live on a
# bind-mounted host volume so the image stays small and survives upgrades.
#
# First boot runs setup.sh which clones AniGen + Hunyuan3D into the volume
# and prefetches weights. Subsequent boots are instant.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip wheel && \
    pip install -r /app/requirements.txt

# Copy the rest of the repo last so iterating on code doesn't bust the deps
# cache layer.
COPY . /app

# /data is the volume for AniGen + Hunyuan3D + HF weights so they persist.
VOLUME ["/data"]

EXPOSE 9000

# entrypoint clones AniGen + Hunyuan3D into /data on first run, builds CUDA
# extensions for the current GPU's compute capability, prefetches weights,
# then execs the server.
ENTRYPOINT ["/app/docker/entrypoint.sh"]
