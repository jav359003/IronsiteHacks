from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    cosmos_reason2_model: str = os.getenv("COSMOS_REASON2_MODEL", "nvidia/Cosmos-Reason2-8B")
    cosmos_embed1_model: str = os.getenv("COSMOS_EMBED1_MODEL", "nvidia/Cosmos-Embed1-224p")

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "cosmos_embed1")

    video_dir: str = os.getenv(
        "VIDEO_DIR", "/Users/damith/ironsite-rag/data/ironsite-trimmed"
    )


SETTINGS = Settings()
