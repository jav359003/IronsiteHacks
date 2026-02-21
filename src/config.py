from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    cosmos_reason2_base_url: str = os.getenv("COSMOS_REASON2_BASE_URL", "http://127.0.0.1:8000/v1")
    cosmos_reason2_model: str = os.getenv("COSMOS_REASON2_MODEL", "nvidia/cosmos-reason2-2b")
    cosmos_embed1_base_url: str = os.getenv("COSMOS_EMBED1_BASE_URL", "http://127.0.0.1:8000")
    cosmos_embed1_model: str = os.getenv("COSMOS_EMBED1_MODEL", "nvidia/cosmos-embed1")
    nim_api_key: str | None = os.getenv("NIM_API_KEY") or None

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "cosmos_embed1")

    video_dir: str = os.getenv(
        "VIDEO_DIR", "/Users/damith/ironsite-rag/data/ironsite-trimmed"
    )


SETTINGS = Settings()
