from __future__ import annotations

import json
from typing import Any

import requests

from .config import SETTINGS


def _auth_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if SETTINGS.qdrant_api_key:
        headers["api-key"] = SETTINGS.qdrant_api_key
    return headers


def collection_exists(name: str) -> bool:
    url = f"{SETTINGS.qdrant_url}/collections/{name}"
    resp = requests.get(url, headers=_auth_headers(), timeout=30)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return True


def create_collection(name: str, vector_size: int, distance: str = "Cosine") -> None:
    url = f"{SETTINGS.qdrant_url}/collections/{name}"
    payload = {"vectors": {"size": vector_size, "distance": distance}}
    resp = requests.put(url, headers=_auth_headers(), data=json.dumps(payload), timeout=30)
    resp.raise_for_status()


def ensure_collection(name: str, vector_size: int, distance: str = "Cosine") -> None:
    if not collection_exists(name):
        create_collection(name, vector_size, distance=distance)


def upsert_points(name: str, points: list[dict[str, Any]]) -> None:
    url = f"{SETTINGS.qdrant_url}/collections/{name}/points?wait=true"
    payload = {"points": points}
    resp = requests.put(url, headers=_auth_headers(), data=json.dumps(payload), timeout=60)
    resp.raise_for_status()


def search(name: str, vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
    url = f"{SETTINGS.qdrant_url}/collections/{name}/points/search"
    payload = {"vector": vector, "limit": limit}
    resp = requests.post(url, headers=_auth_headers(), data=json.dumps(payload), timeout=30)
    resp.raise_for_status()
    return resp.json().get("result", [])
