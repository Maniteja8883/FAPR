"""The FAPR RoPE patch — the heart of the project.

WHY THIS DESIGN SURVIVES `generate()` (the Attempt-3 trap):
HuggingFace `generate()` recomputes `position_ids` on every autoregressive step
via `prepare_inputs_for_generation` / `cache_position`, silently discarding any
custom position tensor you pass in. So we do NOT pass positions in. Instead we
wrap the model's *rotary embedding module(s)* so that every position tensor that
flows through RoPE — prefill AND each decode step — is divided by phi on its way
in. `generate()` can recompute positions all it wants; they still pass through
our wrapper. Phase continuity is preserved automatically: a prefill of length L
gets positions [0..L-1]/phi and decode step k gets (L+k)/phi.

WHY IT IS MODEL-AGNOSTIC:
We locate rotary modules by class name pattern ("*RotaryEmbedding"), which
matches LlamaRotaryEmbedding, Qwen2RotaryEmbedding, MistralRotaryEmbedding,
GemmaRotaryEmbedding, ... In transformers >=4.46 there is typically ONE shared
rotary module on `model.model.rotary_emb`; older versions kept one per
attention layer. We scan `named_modules()` and dedupe by identity, so both
layouts work. Fractional positions are mathematically fine: RoPE computes
`angle = position * inv_freq` in float, no integer lookup table exists.
"""
from __future__ import annotations

import contextlib
from typing import List

import torch
from torch import nn


def find_rotary_modules(model: nn.Module) -> List[nn.Module]:
    mods, seen = [], set()
    for _, m in model.named_modules():
        if "RotaryEmbedding" in type(m).__name__ and id(m) not in seen:
            seen.add(id(m))
            mods.append(m)
    if not mods:
        raise RuntimeError(
            "No *RotaryEmbedding module found. Either this model does not use RoPE "
            "(FAPR only applies to RoPE models: Llama/Qwen/Mistral/Gemma families), "
            "or your transformers version renamed the rotary class. "
            "Tested with transformers>=4.46,<4.58."
        )
    return mods


@contextlib.contextmanager
def fapr_rope(model: nn.Module, phi: float):
    """Context manager: while active, every position id seen by RoPE is divided
    by `phi`. phi=1.0 is an exact no-op mathematically (we still patch, so the
    English control goes through the identical code path — honesty by design).

    Usage:
        with fapr_rope(model, phi=2.6):
            out = model.generate(**inputs, max_new_tokens=64)
    """
    if phi <= 0:
        raise ValueError(f"phi must be > 0, got {phi}")
    mods = find_rotary_modules(model)
    patched: List[nn.Module] = []
    try:
        for m in mods:
            if getattr(m, "_fapr_patched", False):
                raise RuntimeError("fapr_rope() is already active on this model — do not nest.")

            def make_wrapper(orig_fwd, scale: float):
                def wrapper(x, position_ids=None, *args, **kwargs):
                    if position_ids is not None:
                        position_ids = position_ids.to(torch.float32) / scale
                    return orig_fwd(x, position_ids, *args, **kwargs)
                return wrapper

            # Instance-level override: shadows the class method for THIS module
            # only, and is fully reversible (we just delete the instance attr).
            m.forward = make_wrapper(m.forward, float(phi))
            m._fapr_patched = True
            patched.append(m)
        yield model
    finally:
        for m in patched:
            m.__dict__.pop("forward", None)  # class method resurfaces
            m._fapr_patched = False
