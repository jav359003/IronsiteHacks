from __future__ import annotations

import argparse
import json

from src.config import SETTINGS
from src.nim_embed1 import embed_text
from src.vector_db_qdrant import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Qdrant with Cosmos-Embed1 text embeddings")
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    vector = embed_text(args.query)
    results = search(SETTINGS.qdrant_collection, vector, limit=args.limit)

    for rank, r in enumerate(results, start=1):
        payload = r.get("payload", {})
        print(json.dumps({
            "rank": rank,
            "score": r.get("score"),
            "video_path": payload.get("video_path"),
            "clip_path": payload.get("clip_path"),
            "caption": payload.get("caption"),
            "clip_index": payload.get("clip_index"),
        }, indent=2))


if __name__ == "__main__":
    main()
