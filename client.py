from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

RETRYABLE_STATUS = {429, 502, 503, 504}


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        raw = response.headers.get("retry-after")
        if raw:
            try:
                return max(0.25, min(float(raw), 10.0))
            except ValueError:
                pass
    return min(0.5 * (2**attempt), 10.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream GPT-SoVITS raw s16le PCM")
    parser.add_argument("--url", required=True, help="https://wp6fhn2lh8tqmc.api.runpod.ai/tts")
    # parser.add_argument("--api-key", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--mode", type=int, choices=(2, 3), default=3)
    parser.add_argument("--output", default="output.pcm")
    parser.add_argument("--retries", type=int, default=5, help="Retries allowed before the first PCM byte")
    args = parser.parse_args()

    output = Path(args.output)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)

    payload = {
        "text": args.text,
        "text_lang": args.lang,
        "streaming_mode": args.mode,
    }
    headers = {
        "Authorization": f"Bearer ",
        "Accept-Encoding": "identity",
    }
    timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)

    for attempt in range(args.retries + 1):
        got_pcm = False
        response: httpx.Response | None = None
        try:
            with httpx.stream(
                "POST",
                args.url,
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status_code in RETRYABLE_STATUS:
                    body = response.read().decode("utf-8", errors="replace")
                    if attempt >= args.retries:
                        print(f"HTTP {response.status_code}: {body}", file=sys.stderr)
                        return 1
                    delay = retry_delay(response, attempt)
                    print(
                        f"Worker unavailable (HTTP {response.status_code}); retrying before audio",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue

                if response.is_error:
                    body = response.read().decode("utf-8", errors="replace")
                    print(f"HTTP {response.status_code}: {body}", file=sys.stderr)
                    return 1

                sample_rate = response.headers.get("x-audio-sample-rate", "unknown")
                print(
                    f"Receiving mono s16le PCM at {sample_rate} Hz -> {output}",
                    file=sys.stderr,
                )
                with partial.open("wb") as f:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        got_pcm = True
                        f.write(chunk)

                if not got_pcm:
                    partial.unlink(missing_ok=True)
                    if attempt >= args.retries:
                        print("Stream ended before the first PCM byte", file=sys.stderr)
                        return 1
                    time.sleep(retry_delay(response, attempt))
                    continue

                os.replace(partial, output)
                return 0

        except httpx.HTTPError as exc:
            if got_pcm:
                print(
                    f"PCM stream failed after audio began; not retrying: {exc}. "
                    f"Partial data remains at {partial}",
                    file=sys.stderr,
                )
                return 2
            partial.unlink(missing_ok=True)
            if attempt >= args.retries:
                print(f"Request failed before audio: {exc}", file=sys.stderr)
                return 1
            print(f"Connection failed before audio; retrying: {exc}", file=sys.stderr)
            time.sleep(retry_delay(response, attempt))

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
