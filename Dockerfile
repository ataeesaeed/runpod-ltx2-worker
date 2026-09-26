# LTX-2.5 (distilled) worker for RunPod Serverless.
# Models are NOT baked in: the handler downloads them once into the network volume
# (/runpod-volume/models/ltx-2.5), so the image stays small and cold starts reuse them.
#
# Needs a host driver that supports CUDA 13.2 (torch 2.13 cu132): on the endpoint, restrict
# "Allowed CUDA Versions" to 13.x.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/runpod-volume/huggingface \
    MODEL_DIR=/runpod-volume/models/ltx-2.5

# git: fetch LTX-2. build-essential: Triton JIT-compiles small launchers at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Pinned LTX-2 commit (2026-08-26) the handler was written against; bump deliberately.
ARG LTX2_COMMIT=a95ab856bf29407b6b066ede0abe1846050db56c
RUN git clone https://github.com/Lightricks/LTX-2.git /opt/LTX-2 \
    && git -C /opt/LTX-2 checkout ${LTX2_COMMIT}

# Same package set as `uv sync --extra natten` in the LTX-2 README: torch 2.13 (cu132),
# the prebuilt natten wheel for the fast video VAE decoder, and ltx-core + ltx-pipelines.
RUN uv pip install --system --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu132 \
        --extra-index-url https://download.pytorch.org/whl/test/cu132/ \
        --find-links https://whl.natten.org \
        "torch==2.13.0" torchvision torchaudio \
        "natten==0.21.7+torch2130cu132" \
        -e /opt/LTX-2/packages/ltx-core \
        -e /opt/LTX-2/packages/ltx-pipelines \
        runpod huggingface_hub

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
