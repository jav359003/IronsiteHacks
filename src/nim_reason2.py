from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests

from .config import SETTINGS


def _auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if SETTINGS.nim_api_key:
        headers["Authorization"] = f"Bearer {SETTINGS.nim_api_key}"
    return headers


def _encode_video_b64(video_path: str) -> str:
    data = Path(video_path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def caption_video(
    video_path: str,
    prompt: str,
    fps: int | None = 2,
    model: str | None = None,
    max_tokens: int = 256,
) -> str:
    model_name = model or SETTINGS.cosmos_reason2_model

    media_io_kwargs: dict[str, Any] = {}
    if fps is not None:
        media_io_kwargs["video"] = {"fps": fps}

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes concise video captions.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": _encode_video_b64(video_path)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    if media_io_kwargs:
        payload["media_io_kwargs"] = media_io_kwargs

    url = f"{SETTINGS.cosmos_reason2_base_url}/chat/completions"
    resp = requests.post(url, headers=_auth_headers(), data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
