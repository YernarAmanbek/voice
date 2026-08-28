# GPT-SoVITS v2ProPlus RunPod load-balanced PCM streamer

Overlay these files into the root of a **pinned** GPT-SoVITS checkout.

## Runtime volume layout

```text
/runpod-volume/gpt-sovits/
├── voice.json
└── voices/default/reference.wav
```

Copy `voice.json.example` to the volume as `voice.json` and edit it. Use a WAV
reference; this slim image intentionally omits the system ffmpeg binary. The
model weights are baked into `/app/models` by the Dockerfile and do not need a
volume mount. Current upstream checks that the reference WAV is 3–10 seconds
long.

The recommended English-only configuration below does not need G2PW. If Chinese
is enabled, `G2PWModel` is still required separately at
`${MODEL_ROOT}/G2PWModel` (or via `G2PW_MODEL_DIR`), and `ALLOWED_LANGUAGES`
must also include the desired Chinese modes.

## Build

```bash
# Run from the GPT-SoVITS repository root after copying this bundle into it.
docker build --pull -t YOUR_REGISTRY/gpt-sovits-v2proplus:COMMIT_SHA .
docker push YOUR_REGISTRY/gpt-sovits-v2proplus:COMMIT_SHA
```

The Dockerfile downloads the required v2ProPlus GPT, SoVITS, BERT, CNHuBERT,
and speaker-verification weights from `lj1995/GPT-SoVITS` into the current
runtime image. The model snapshot is pinned with `MODEL_REVISION`; update that
build argument and the five matching SHA-256 values in the Dockerfile
deliberately when upgrading weights. Runtime Hugging Face and Transformers
access remains offline.

The Dockerfile prunes source in a throwaway build stage, so removed files do not
remain in an earlier runtime-image layer. Local weight files remain outside the
build context because the pinned files are fetched during the image build. It
installs the CUDA 12.8 `torchaudio` wheel with `--no-deps` so pip does not
replace the PyTorch already supplied by the base image.

For a second-stage trim, run:

```bash
python patches/apply_v2proplus_lazy_imports.py
python -c 'from GPT_SoVITS.TTS_infer_pack.TTS import TTS; print("import ok")'
# Then run an actual first-chunk inference smoke test with your volume.
bash patches/prune_after_lazy_imports.sh
```

The patcher makes BigVGAN, F5/DiT, AP-BWE, and PEFT imports lazy. After the real
model test, remove `peft` from `requirements.infer.txt` and rebuild.

## RunPod settings

- Endpoint type: **Load Balancer**, not queue-based Serverless.
- Expose `8000/http`; set `PORT=8000`, `PORT_HEALTH=8000`,
  `HEALTH_CHECK_PATH=/ping`.
- Attach the network volume and one GPU per worker.
- Replace any existing RunPod model-path overrides with the values in the
  Environment section below (including `MODEL_ROOT`, `GPT_WEIGHT`, and
  `SOVITS_WEIGHT`), and remove stale `SV_MODEL_DIR` overrides. Runtime
  environment values take precedence over the image defaults.
- Uvicorn workers: exactly one (enforced by `entrypoint.sh`).
- Autoscaling: Request Count, scaler value `1`.
- Active workers: at least `1` for no cold start. Max workers is your total
  concurrent-stream cap.

The service rejects a second local request with HTTP 429 rather than queueing.
Set `ALLOWED_LANGUAGES` explicitly. When `ENABLE_CHINESE=0`, the server excludes
Mandarin and automatic language modes by default so missing G2PW assets cannot
fail halfway through a stream.
`/ping` stays HTTP 200 while busy so the worker remains healthy; the semaphore is
an overload guard in case the platform routes two requests to one worker.

## Environment

```text
VOLUME_ROOT=/runpod-volume/gpt-sovits
MODEL_ROOT=/app/models
GPT_WEIGHT=s1v3.ckpt
SOVITS_WEIGHT=v2Pro/s2Gv2ProPlus.pth
BERT_MODEL=chinese-roberta-wwm-ext-large
CNHUBERT_MODEL=chinese-hubert-base
VOICE_CONFIG=voice.json
DEVICE=cuda
IS_HALF=1
ENABLE_CHINESE=0
ALLOWED_LANGUAGES=en
MAX_TEXT_CHARS=3000
```

## Stream contract

`POST /tts` returns headerless mono signed 16-bit little-endian PCM. Read the
`X-Audio-Sample-Rate`, `X-Audio-Channels`, and `X-Audio-Format` headers.

```bash
python -m pip install httpx
python client.py \
  --url https://ENDPOINT_ID.api.runpod.ai/tts \
  --api-key "$RUNPOD_API_KEY" \
  --text "Hello from a streaming worker." \
  --lang en \
  --output output.pcm

ffplay -nodisp -autoexit -f s16le -ar SAMPLE_RATE -ac 1 output.pcm
```

The included client retries 429/502/503/504 and connection failures only before
the first PCM byte. It never resumes or retries halfway through a headerless PCM
body; a failed partial stream is left as `output.pcm.partial`.
