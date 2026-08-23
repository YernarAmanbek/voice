# Pin this tag and preferably its image digest after GPU smoke testing.
ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Prune in a throwaway stage. Deleting after COPY in the runtime stage would
# leave the removed bytes in an earlier image layer.
FROM alpine:3.20 AS source-prune
WORKDIR /src
COPY . /src

# Safe first-pass pruning. Do NOT delete BigVGAN, f5_tts, tools/audio_sr.py,
# tools/AP_BWE_main, or peft until the supplied lazy-import patcher is applied.
RUN rm -rf \
      /src/.git \
      /src/.github \
      /src/docs \
      /src/Docker \
      /src/logs \
      /src/logs_v2Pro \
      /src/logs_v2ProPlus \
      /src/output \
      /src/outputs \
      /src/TEMP \
      /src/tools/asr \
      /src/tools/uvr5 \
      /src/GPT_SoVITS/prepare_datasets \
      /src/patches \
    && rm -f \
      /src/webui.py \
      /src/api.py \
      /src/api_v2.py \
      /src/inference_webui.py \
      /src/inference_webui_fast.py \
      /src/GPT_SoVITS/s1_train.py \
      /src/GPT_SoVITS/s2_train.py \
      /src/GPT_SoVITS/s2_train_v3_lora.py \
      /src/GPT_SoVITS/s2_train_v3_lora_infer.py \
      /src/GPT_SoVITS/export_torch_script.py \
      /src/GPT_SoVITS/export_onnx.py \
      /src/README.md \
      /src/client.py \
      /src/voice.json.example \
      /src/Dockerfile \
      /src/.dockerignore \
      /src/requirements.infer.txt \
    && find /src -type d -name '__pycache__' -prune -exec rm -rf '{}' + \
    && find /src -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/app:/app/GPT_SoVITS \
    PORT=8000 \
    PORT_HEALTH=8000 \
    HEALTH_CHECK_PATH=/ping \
    VOLUME_ROOT=/runpod-volume/gpt-sovits \
    MODEL_ROOT=models

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt

# Build tools are removed in the same layer. Remove MeCab packages when Korean
# is disabled. This raw-PCM image expects a WAV reference and omits system ffmpeg.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 \
      libgomp1 \
      mecab \
      libmecab-dev \
      build-essential \
      cmake \
      ninja-build \
      pkg-config \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 torchaudio==2.7.0 \
    && python -m pip install -r /tmp/requirements.txt \
    && python -c "import torch, torchaudio; assert torch.__version__.startswith('2.7.0'); assert torchaudio.__version__.startswith('2.7.0'); print(torch.__version__, torchaudio.__version__)" \
    && apt-get purge -y --auto-remove \
         build-essential cmake ninja-build pkg-config libmecab-dev \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

COPY --from=source-prune /src /app

RUN chmod +x /app/entrypoint.sh \
    && python -m py_compile /app/server.py

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
