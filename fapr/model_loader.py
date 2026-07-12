"""Model + tokenizer loading with a per-device quantization policy.

One codebase, three backends:
  cuda (Colab T4/A100) -> bitsandbytes NF4 4-bit (the T4 OOM from Attempt 1 was a
                          7B model + eager attention; 3B @ 4-bit + sdpa is ~3.5GB)
  mps  (M5 MacBook)    -> bf16/fp16 full weights (~6.2GB for 3B — fits 16GB
                          unified); optional `optimum-quanto` int8 (~3.4GB) via
                          quantize="int8-mps" if memory pressure appears.
                          bitsandbytes does NOT support MPS — do not try.
  cpu                  -> fp32 fallback (slow; only for debugging logic).

sdpa attention everywhere: memory-efficient fused kernels on both CUDA and MPS,
and flash-attention-2 is unavailable on both T4 (Turing) and Mac anyway.
"""
from __future__ import annotations

import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import pick_device, pick_dtype

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def load_model(model_id: str = DEFAULT_MODEL, quantize: str = "auto"):
    """quantize: 'auto' (4-bit on CUDA, plain half precision on MPS/CPU),
    '4bit' (force CUDA 4-bit), 'int8-mps' (quanto int8 on Apple Silicon),
    'none' (half precision everywhere)."""
    device = pick_device()
    dtype = pick_dtype(device)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        # Llama-family tokenizers ship without a pad token; leaving it unset is
        # one root cause of the old blank-generate bug (pad/eos ambiguity).
        tok.pad_token = tok.eos_token

    kwargs = dict(torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True)

    if device == "cuda" and quantize in ("auto", "4bit"):
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            kwargs["device_map"] = "auto"
        except ImportError:
            warnings.warn("bitsandbytes unavailable — loading half precision on CUDA instead.")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if "device_map" not in kwargs:
        model.to(device)

    if device == "mps" and quantize == "int8-mps":
        try:
            from optimum.quanto import freeze, qint8
            from optimum.quanto import quantize as quanto_quantize
            quanto_quantize(model, weights=qint8)
            freeze(model)
            print("[loader] quanto int8 applied (MPS)")
        except ImportError:
            warnings.warn("optimum-quanto not installed (pip install optimum-quanto) — "
                          "running unquantized on MPS.")

    model.eval()

    # Deterministic decoding for the demo/eval + silence the 'temperature set but
    # do_sample=False' warnings that Qwen's shipped generation_config triggers.
    g = model.generation_config
    g.do_sample = False
    g.temperature = None
    g.top_p = None
    g.top_k = None
    g.pad_token_id = tok.pad_token_id

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[loader] {model_id} | device={device} dtype={dtype} | ~{n_params:.1f}B params | "
          f"native ctx={getattr(model.config, 'max_position_embeddings', '?')}")
    return model, tok, device
