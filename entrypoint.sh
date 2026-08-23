#!/usr/bin/env bash
set -euo pipefail
cd /app

export VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume/gpt-sovits}"
export MODEL_ROOT="${MODEL_ROOT:-models}"
if [[ "${MODEL_ROOT}" = /* ]]; then
  MODEL_DIR="${MODEL_ROOT}"
else
  MODEL_DIR="${VOLUME_ROOT}/${MODEL_ROOT}"
fi

# v2ProPlus initializes speaker verification through this hard-coded path.
SV_SOURCE="${SV_MODEL_DIR:-${MODEL_DIR}/sv}"
SV_TARGET="/app/GPT_SoVITS/pretrained_models/sv"
if [[ ! -f "${SV_SOURCE}/pretrained_eres2netv2w24s4ep4.ckpt" ]]; then
  echo "Missing v2ProPlus SV checkpoint: ${SV_SOURCE}/pretrained_eres2netv2w24s4ep4.ckpt" >&2
  exit 64
fi
mkdir -p "$(dirname "${SV_TARGET}")"
rm -rf "${SV_TARGET}"
ln -s "${SV_SOURCE}" "${SV_TARGET}"

# Chinese v2 frontend currently hard-codes this asset path.
G2PW_SOURCE="${G2PW_MODEL_DIR:-${MODEL_DIR}/G2PWModel}"
G2PW_TARGET="/app/GPT_SoVITS/text/G2PWModel"
if [[ -d "${G2PW_SOURCE}" ]]; then
  rm -rf "${G2PW_TARGET}"
  ln -s "${G2PW_SOURCE}" "${G2PW_TARGET}"
elif [[ "${ENABLE_CHINESE:-0}" == "1" ]]; then
  echo "ENABLE_CHINESE=1 but G2PWModel is missing at ${G2PW_SOURCE}" >&2
  exit 64
fi

# chinese2.py reads lowercase bert_path for the G2PW tokenizer.
BERT_REL="${BERT_MODEL:-chinese-roberta-wwm-ext-large}"
if [[ "${BERT_REL}" = /* ]]; then
  export bert_path="${BERT_REL}"
else
  export bert_path="${MODEL_DIR}/${BERT_REL}"
fi

# TTS_Config writes generated YAML; keep that on ephemeral container storage.
mkdir -p /app/GPT_SoVITS/configs

exec python -m uvicorn server:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers 1 \
  --no-access-log \
  --proxy-headers \
  --timeout-keep-alive 5
