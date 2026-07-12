"""Device / dtype auto-detection shared by every script.

This extends the cuda/mps/cpu pattern from the Colab phase. The one subtlety:
- CUDA T4 has NO bfloat16 -> fp16 there; A100/L4 get bf16.
- MPS on recent macOS/torch supports bf16, which avoids the known fp16 overflow
  issues in Qwen2 activations; we probe for it instead of assuming.
"""
from __future__ import annotations

import gc

import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        try:  # probe: bf16 on MPS needs torch>=2.3 + a recent macOS
            torch.zeros(1, dtype=torch.bfloat16, device="mps")
            return torch.bfloat16
        except Exception:
            return torch.float16
    return torch.float32


def empty_cache(device: str) -> None:
    """Free cached allocator blocks. Call between eval items / demo requests —
    on 16GB unified memory the KV cache of a 20K-token Telugu pack is ~0.8GB
    and we do not want stale blocks accumulating across requests."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
