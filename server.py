from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("gpt-sovits-stream")

ALL_LANGUAGES = {
    "auto",
    "auto_yue",
    "en",
    "zh",
    "ja",
    "yue",
    "ko",
    "all_zh",
    "all_ja",
    "all_yue",
    "all_ko",
}


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}




def configured_languages() -> set[str]:
    raw = os.getenv("ALLOWED_LANGUAGES")
    if raw:
        allowed = {item.strip().lower() for item in raw.split(",") if item.strip()}
    elif env_bool("ENABLE_CHINESE", False):
        allowed = set(ALL_LANGUAGES)
    else:
        # Avoid auto modes when Mandarin assets are absent; auto may route text
        # into the Chinese frontend and fail after response headers are sent.
        allowed = {"en", "ja", "yue", "ko", "all_ja", "all_yue", "all_ko"}
    unknown = sorted(allowed - ALL_LANGUAGES)
    if unknown:
        raise ValueError(f"Unknown ALLOWED_LANGUAGES values: {', '.join(unknown)}")
    if not allowed:
        raise ValueError("ALLOWED_LANGUAGES must not be empty")
    return allowed

def resolve_under(
    root: Path,
    raw: str,
    *,
    require_file: bool = False,
    require_dir: bool = False,
) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if require_file and not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    if require_dir and not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")
    return path


@dataclass
class RuntimeState:
    pipeline: Any | None = None
    voice: dict[str, Any] | None = None
    sample_rate: int | None = None
    load_error: str | None = None
    allowed_languages: set[str] = field(default_factory=set)
    ready: threading.Event = field(default_factory=threading.Event)
    # Authoritative worker capacity: exactly one active stream per process/GPU.
    stream_slot: threading.Lock = field(default_factory=threading.Lock)


runtime = RuntimeState()


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=int(os.getenv("MAX_TEXT_CHARS", "3000")))
    text_lang: str = Field(default=os.getenv("DEFAULT_TEXT_LANG", "en"))

    # Upstream modes: 2 = semantic-boundary chunks; 3 = fixed-length/faster chunks.
    streaming_mode: Literal[2, 3] = 3
    overlap_length: int = Field(default=2, ge=1, le=8)
    min_chunk_length: int = Field(default=16, ge=4, le=128)

    top_k: int = Field(default=15, ge=1, le=100)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0, le=2.0)
    repetition_penalty: float = Field(default=1.35, ge=0.1, le=3.0)
    speed_factor: float = Field(default=1.0, ge=0.6, le=1.65)
    seed: int = -1
    text_split_method: Literal["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"] = "cut5"


def load_voice_manifest(volume_root: Path, allowed_languages: set[str]) -> dict[str, Any]:
    manifest_path = resolve_under(
        volume_root,
        os.getenv("VOICE_CONFIG", "voice.json"),
        require_file=True,
    )
    with manifest_path.open("r", encoding="utf-8") as f:
        voice = json.load(f)

    required = {"ref_audio_path", "prompt_lang"}
    missing = sorted(required - voice.keys())
    if missing:
        raise ValueError(f"Voice manifest missing fields: {', '.join(missing)}")

    voice["ref_audio_path"] = str(
        resolve_under(volume_root, str(voice["ref_audio_path"]), require_file=True)
    )
    voice["prompt_text"] = str(voice.get("prompt_text", ""))
    voice["prompt_lang"] = str(voice["prompt_lang"]).lower()
    if voice["prompt_lang"] not in allowed_languages:
        raise ValueError(
            f"prompt_lang {voice['prompt_lang']!r} is not in ALLOWED_LANGUAGES"
        )

    aux_paths: list[str] = []
    for raw in voice.get("aux_ref_audio_paths", []) or []:
        aux_paths.append(str(resolve_under(volume_root, str(raw), require_file=True)))
    voice["aux_ref_audio_paths"] = aux_paths
    return voice


def load_models() -> None:
    try:
        # Import in a background loader so Uvicorn binds first and /ping can
        # report HTTP 204 while CUDA/models initialize.
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS

        volume_root = Path(os.getenv("VOLUME_ROOT", "/runpod-volume/gpt-sovits")).resolve()
        model_root = resolve_under(
            volume_root,
            os.getenv("MODEL_ROOT", "models"),
            require_dir=True,
        )
        gpt_weight = resolve_under(
            model_root,
            os.getenv("GPT_WEIGHT", "gpt/v2proplus.ckpt"),
            require_file=True,
        )
        sovits_weight = resolve_under(
            model_root,
            os.getenv("SOVITS_WEIGHT", "sovits/v2proplus.pth"),
            require_file=True,
        )
        bert_dir = resolve_under(
            model_root,
            os.getenv("BERT_MODEL", "chinese-roberta-wwm-ext-large"),
            require_dir=True,
        )
        cnhubert_dir = resolve_under(
            model_root,
            os.getenv("CNHUBERT_MODEL", "chinese-hubert-base"),
            require_dir=True,
        )
        allowed_languages = configured_languages()
        voice = load_voice_manifest(volume_root, allowed_languages)

        config = {
            "custom": {
                "device": os.getenv("DEVICE", "cuda"),
                "is_half": env_bool("IS_HALF", True),
                "version": "v2ProPlus",
                "t2s_weights_path": str(gpt_weight),
                "vits_weights_path": str(sovits_weight),
                "bert_base_path": str(bert_dir),
                "cnhuhbert_base_path": str(cnhubert_dir),
            }
        }

        log.info("Loading GPT-SoVITS v2ProPlus from %s", model_root)
        pipeline = TTS(config)

        # Warm the fixed reference semantic/spec. Current upstream accepts 3-10 s.
        pipeline.set_ref_audio(voice["ref_audio_path"])

        runtime.pipeline = pipeline
        runtime.voice = voice
        runtime.allowed_languages = allowed_languages
        runtime.sample_rate = int(pipeline.configs.sampling_rate)
        runtime.ready.set()
        log.info(
            "Ready: mono s16le PCM at %d Hz; allowed languages=%s",
            runtime.sample_rate,
            ",".join(sorted(runtime.allowed_languages)),
        )
    except Exception:
        runtime.load_error = traceback.format_exc()
        log.error("Model initialization failed:\n%s", runtime.load_error)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=load_models, name="model-loader", daemon=True).start()
    yield


app = FastAPI(
    title="GPT-SoVITS v2ProPlus raw PCM streamer",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.get("/ping")
def ping() -> Response:
    if runtime.load_error is not None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": runtime.load_error[-2000:]},
        )
    if not runtime.ready.is_set():
        # RunPod load balancing treats 204 as initializing.
        return Response(status_code=204)
    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "sample_rate": runtime.sample_rate,
            "channels": 1,
            "sample_format": "s16le",
            "busy": runtime.stream_slot.locked(),
            "allowed_languages": sorted(runtime.allowed_languages),
        },
    )


@app.post("/tts")
def tts(req: TTSRequest) -> StreamingResponse:
    if runtime.load_error is not None:
        raise HTTPException(status_code=500, detail="Model failed to initialize")
    if not runtime.ready.is_set() or runtime.pipeline is None or runtime.voice is None:
        raise HTTPException(
            status_code=503,
            detail="Worker is initializing",
            headers={"Retry-After": "1"},
        )

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must contain non-whitespace characters")

    text_lang = req.text_lang.lower()
    if text_lang not in runtime.allowed_languages:
        raise HTTPException(
            status_code=400,
            detail=f"text_lang {text_lang!r} is not enabled on this endpoint",
        )

    # No in-worker queue. A second request receives 429 immediately.
    if not runtime.stream_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Worker is busy",
            headers={"Retry-After": "0"},
        )

    pipeline = runtime.pipeline
    voice = runtime.voice
    try:
        upstream_generator = pipeline.run(
            {
                "text": req.text,
                "text_lang": text_lang,
                "ref_audio_path": voice["ref_audio_path"],
                "aux_ref_audio_paths": voice.get("aux_ref_audio_paths", []),
                "prompt_text": voice.get("prompt_text", ""),
                "prompt_lang": voice["prompt_lang"],
                "top_k": req.top_k,
                "top_p": req.top_p,
                "temperature": req.temperature,
                "text_split_method": req.text_split_method,
                "batch_size": 1,
                "batch_threshold": 0.75,
                "split_bucket": False,
                "speed_factor": req.speed_factor,
                "fragment_interval": 0.0,
                "seed": req.seed,
                "parallel_infer": False,
                "repetition_penalty": req.repetition_penalty,
                "sample_steps": 32,
                "super_sampling": False,
                "streaming_mode": True,
                "return_fragment": False,
                "fixed_length_chunk": req.streaming_mode == 3,
                "overlap_length": req.overlap_length,
                "min_chunk_length": req.min_chunk_length,
            }
        )
    except Exception:
        runtime.stream_slot.release()
        raise

    def pcm_chunks() -> Generator[bytes, None, None]:
        try:
            for sr, chunk in upstream_generator:
                if int(sr) != int(runtime.sample_rate):
                    # Current upstream emits a 16 kHz silence block before
                    # reloading/raising after an internal failure. Discard it,
                    # then advance once more so upstream recovery can complete.
                    log.error(
                        "Discarding chunk with sample rate %s; expected %s",
                        sr,
                        runtime.sample_rate,
                    )
                    continue
                pcm = np.asarray(chunk)
                if pcm.dtype != np.int16:
                    pcm = pcm.astype(np.int16, copy=False)
                pcm = np.ascontiguousarray(pcm.astype("<i2", copy=False)).reshape(-1)
                if pcm.size:
                    yield pcm.tobytes(order="C")
        except GeneratorExit:
            log.info("Client disconnected from PCM stream")
            raise
        except Exception:
            # Headers may already be sent; terminate the body. Clients must never
            # retry/resume halfway through headerless PCM.
            log.exception("TTS stream failed")
            raise
        finally:
            try:
                upstream_generator.close()
            except Exception:
                log.exception("Failed to close upstream TTS generator")
            runtime.stream_slot.release()

    sample_rate = int(runtime.sample_rate)
    return StreamingResponse(
        pcm_chunks(),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store, no-transform",
            "Content-Encoding": "identity",
            "X-Accel-Buffering": "no",
            "X-Audio-Format": "s16le",
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Bits-Per-Sample": "16",
        },
    )
