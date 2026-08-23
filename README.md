# GPT-SoVITS v2ProPlus RunPod load-balanced PCM streamer

Overlay these files into the root of a **pinned** GPT-SoVITS checkout.

## Network-volume layout

```text
/runpod-volume/gpt-sovits/
├── voice.json
├── voices/default/reference.wav
└── models/
    ├── gpt/v2proplus.ckpt
    ├── sovits/v2proplus.pth
    ├── chinese-roberta-wwm-ext-large/
    ├── chinese-hubert-base/
    ├── sv/pretrained_eres2netv2w24s4ep4.ckpt
    └── G2PWModel/                  # only when Chinese is enabled
```

Copy `voice.json.example` to the volume as `voice.json` and edit it. Use a WAV
reference; this slim image intentionally omits the system ffmpeg binary. Current
upstream checks that the reference WAV is 3–10 seconds long.

## Build

```bash
# Run from the GPT-SoVITS repository root after copying this bundle into it.
docker build --pull -t YOUR_REGISTRY/gpt-sovits-v2proplus:COMMIT_SHA .
docker push YOUR_REGISTRY/gpt-sovits-v2proplus:COMMIT_SHA
```

The Dockerfile prunes source in a throwaway build stage, so removed files do not remain in an earlier runtime-image layer. Keep large model files out of the build context with `.dockerignore`. It installs the CUDA 12.8 `torchaudio` wheel with `--no-deps` so pip does not replace the PyTorch already supplied by the base image.

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
MODEL_ROOT=models
GPT_WEIGHT=gpt/v2proplus.ckpt
SOVITS_WEIGHT=sovits/v2proplus.pth
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
