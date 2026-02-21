from __future__ import annotations

import base64
import json
from pathlib import Path

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


def embed_text(text: str, model: str | None = None) -> list[float]:
    model_name = model or SETTINGS.cosmos_embed1_model
    payload = {
        "model": model_name,
        "input": text,
        "request_type": "query",
        "encoding_format": "float",
    }
    url = f"{SETTINGS.cosmos_embed1_base_url}/v1/embeddings"
    resp = requests.post(url, headers=_auth_headers(), data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]


def embed_video(video_path: str, model: str | None = None) -> list[float]:
    model_name = model or SETTINGS.cosmos_embed1_model
    payload = {
        "model": model_name,
        "input": _encode_video_b64(video_path),
        "request_type": "query",
        "encoding_format": "float",
    }
    url = f"{SETTINGS.cosmos_embed1_base_url}/v1/embeddings"
    resp = requests.post(url, headers=_auth_headers(), data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]
