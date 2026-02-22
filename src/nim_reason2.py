from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .config import SETTINGS


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


@lru_cache(maxsize=1)
def _load_reason2(model_name: str) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    if not torch.cuda.is_available():
        model.to(_device())
    return model, processor


def caption_video(
    video_path: str,
    prompt: str,
    fps: int | None = 4,
    model: str | None = None,
    max_tokens: int = 256,
) -> str:
    model_name = model or SETTINGS.cosmos_reason2_model
    reason_model, processor = _load_reason2(model_name)

    video_url = f"file://{video_path}"
    video_content = {"type": "video", "video": video_url}
    if fps is not None:
        video_content["fps"] = fps
    content = [
        video_content,
        {"type": "text", "text": prompt},
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that writes concise video captions.",
        },
        {"role": "user", "content": content},
    ]

    apply_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if fps is not None:
        apply_kwargs["fps"] = fps
    inputs = processor.apply_chat_template(messages, **apply_kwargs)
    inputs = inputs.to(reason_model.device)

    with torch.no_grad():
        output_ids = reason_model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], output_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)
    return decoded[0].strip()
