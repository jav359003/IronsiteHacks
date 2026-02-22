from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from decord import VideoReader
from transformers import AutoModel, AutoProcessor

from .config import SETTINGS


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


@lru_cache(maxsize=1)
def _load_embed_model(model_name: str) -> tuple[AutoModel, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device=_device(), dtype=_dtype())
    return model, processor


def _sample_video_frames(video_path: str, num_frames: int = 8) -> np.ndarray:
    vr = VideoReader(video_path)
    total = len(vr)
    if total == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num=num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
    frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)
    return np.expand_dims(frames, axis=0)  # (1, T, C, H, W)


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    dtype = _dtype()
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def embed_text(text: str, model: str | None = None) -> list[float]:
    model_name = model or SETTINGS.cosmos_embed1_model
    embed_model, processor = _load_embed_model(model_name)
    device = _device()

    text_inputs = processor(text=[text], return_tensors="pt")
    text_inputs = _move_to_device(text_inputs, device)
    with torch.no_grad():
        embeddings = embed_model.get_text_embeddings(**text_inputs)
    return embeddings[0].float().cpu().numpy().tolist()


def embed_video(video_path: str, model: str | None = None) -> list[float]:
    model_name = model or SETTINGS.cosmos_embed1_model
    embed_model, processor = _load_embed_model(model_name)
    device = _device()

    batch = _sample_video_frames(video_path)
    video_inputs = processor(videos=batch, return_tensors="pt")
    video_inputs = _move_to_device(video_inputs, device)
    with torch.no_grad():
        embeddings = embed_model.get_video_embeddings(**video_inputs)
    return embeddings[0].float().cpu().numpy().tolist()
