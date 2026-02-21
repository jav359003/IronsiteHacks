from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tqdm import tqdm

from src.config import SETTINGS
from src.nim_embed1 import embed_video
from src.nim_reason2 import caption_video
from src.vector_db_qdrant import ensure_collection, upsert_points
from src.video_utils import iter_video_clips, list_videos


DEFAULT_PROMPT = (
    "Write a short, concrete caption (max 20 words) describing the key action, "
    "objects, and spatial relationships in the video."
)


def _point_id(video_path: str, clip_idx: int) -> str:
    h = hashlib.sha256(f"{video_path}:{clip_idx}".encode("utf-8")).hexdigest()[:24]
    return h


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption videos and index Cosmos-Embed1 embeddings into Qdrant")
    parser.add_argument("--video-dir", default=SETTINGS.video_dir)
    parser.add_argument("--clip-seconds", type=int, default=15)
    parser.add_argument("--stride-seconds", type=int, default=30)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos (0 = all)")
    args = parser.parse_args()

    videos = list_videos(args.video_dir)
    if args.limit:
        videos = videos[: args.limit]

    first_embedding = None
    points = []

    for video_path in tqdm(videos, desc="videos"):
        clip_idx = 0
        for clip_path in iter_video_clips(video_path, args.clip_seconds, args.stride_seconds):
            caption = caption_video(clip_path, args.prompt)
            embedding = embed_video(clip_path)

            if first_embedding is None:
                first_embedding = embedding
                ensure_collection(SETTINGS.qdrant_collection, vector_size=len(embedding))

            payload = {
                "video_path": os.path.abspath(video_path),
                "clip_path": os.path.abspath(clip_path),
                "caption": caption,
                "clip_index": clip_idx,
            }
            points.append({"id": _point_id(video_path, clip_idx), "vector": embedding, "payload": payload})
            clip_idx += 1

    if points:
        upsert_points(SETTINGS.qdrant_collection, points)
        print(json.dumps({"indexed": len(points), "collection": SETTINGS.qdrant_collection}, indent=2))
    else:
        print("No points indexed.")


if __name__ == "__main__":
    main()
