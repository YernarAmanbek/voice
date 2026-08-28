# Pin this tag and preferably its image digest after GPU smoke testing.
ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Pin the model snapshot so rebuilding the same source cannot silently pick up
# different weights. Update this revision and the checksums below together.
ARG MODEL_REPOSITORY=lj1995/GPT-SoVITS
ARG MODEL_REVISION=336b2ec4e8d4ac74740798dd40af44e74659ecaf

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
ARG MODEL_REPOSITORY
ARG MODEL_REVISION

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
    MODEL_ROOT=/app/models \
    GPT_WEIGHT=s1v3.ckpt \
    SOVITS_WEIGHT=v2Pro/s2Gv2ProPlus.pth \
    ALLOWED_LANGUAGES=en

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
    && python -m nltk.downloader -d /usr/local/share/nltk_data \
         averaged_perceptron_tagger_eng cmudict \
    && apt-get purge -y --auto-remove \
         build-essential cmake ninja-build pkg-config libmecab-dev \
    && rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

# Bake the complete v2ProPlus inference model into this image. Runtime stays in
# offline mode; network access is enabled only for this build step. Keeping this
# before the source COPY lets Docker reuse the large layer across code changes.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \
      "import os, shutil; from huggingface_hub import snapshot_download; source=snapshot_download(repo_id=os.environ['MODEL_REPOSITORY'], revision=os.environ['MODEL_REVISION'], cache_dir='/tmp/model-cache', allow_patterns=['s1v3.ckpt', 'v2Pro/s2Gv2ProPlus.pth', 'sv/pretrained_eres2netv2w24s4ep4.ckpt', 'chinese-hubert-base/*', 'chinese-roberta-wwm-ext-large/*']); shutil.copytree(source, '/app/models', symlinks=False)" \
    && printf '%s  %s\n' \
         87133414860ea14ff6620c483a3db5ed07b44be42e2c3fcdad65523a729a745a /app/models/s1v3.ckpt \
         d42a22bbbf65fb2bbdd45ad6a66841156977db45c7aabe0a6992ff378d9c7d3b /app/models/v2Pro/s2Gv2ProPlus.pth \
         4f5a0bf73c61eb41b174e1bb54e7ee3c83233892be8e0af1f187024e8e581a35 /app/models/sv/pretrained_eres2netv2w24s4ep4.ckpt \
         24164f129c66499d1346e2aa55f183250c223161ec2770c0da3d3b08cf432d3c /app/models/chinese-hubert-base/pytorch_model.bin \
         e53a693acc59ace251d143d068096ae0d7b79e4b1b503fa84c9dcf576448c1d8 /app/models/chinese-roberta-wwm-ext-large/pytorch_model.bin \
      | sha256sum --check --strict - \
    && python -c \
      "import sys; from pathlib import Path; root=Path('/app/models'); required=['chinese-hubert-base/config.json', 'chinese-hubert-base/preprocessor_config.json', 'chinese-roberta-wwm-ext-large/config.json', 'chinese-roberta-wwm-ext-large/tokenizer.json']; bad=[name for name in required if not (root/name).is_file()]; links=[str(path) for path in root.rglob('*') if path.is_symlink()]; sys.exit('Missing model metadata: ' + ', '.join(bad)) if bad else None; sys.exit('Model files must be materialized, not symlinked: ' + ', '.join(links)) if links else None" \
    && rm -rf /tmp/model-cache /root/.cache/huggingface

COPY --from=source-prune /src /app

# Resolve the complete inference import graph while the image is still building.
RUN chmod +x /app/entrypoint.sh \
    && python -m py_compile /app/server.py \
    && python -c "from GPT_SoVITS.TTS_infer_pack.TTS import TTS; print('GPT-SoVITS import ok')"

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
# RUN python /app/server.py
