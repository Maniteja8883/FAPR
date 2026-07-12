"""The fixed `generate()` path — drawback #6, diagnosed and closed.

THE BLANK-OUTPUT BUG had four stacked causes in the Colab version; each guard
below names the one it kills:

1. No chat template. A chat-tuned model fed a raw completion prompt very often
   emits EOS immediately (its training says 'this conversation is over/empty').
   -> we ALWAYS build inputs with tokenizer.apply_chat_template.
2. Missing attention_mask, with pad_token == eos_token. `generate()` then
   guesses the mask by searching for the pad id and can mask out real content.
   -> we pass the tokenizer's attention_mask explicitly, always.
3. EOS legally sampled as the very first token -> empty string after
   skip_special_tokens. -> min_new_tokens=2 makes that impossible.
4. Decoding the whole sequence and stripping specials can leave "" when only
   specials were generated. -> we decode ONLY the new tokens and fall back to a
   labeled placeholder instead of silently showing nothing.

Note what is NOT here: we never pass position_ids into generate(). The Attempt-3
digest documented that generate() recomputes/overwrites them each step. FAPR
scaling lives inside the rotary module (see rope_patch.py), so every recomputed
position still flows through it — prefill and every decode step stay consistent.
"""
from __future__ import annotations

import contextlib
from typing import Tuple

import torch

from .device import empty_cache
from .rope_patch import fapr_rope


@torch.inference_mode()
def generate_answer(model, tok, device: str, user_prompt: str,
                    phi: float = 1.0, max_new_tokens: int = 160,
                    sample: bool = False, temperature: float = 0.7,
                    top_p: float = 0.9, repetition_penalty: float = 1.2) -> Tuple[str, int]:
    """Returns (answer_text, n_input_tokens). phi=1.0 means FAPR off.

    sample=False (default) is byte-identical to the original greedy path used
    by the eval harness. sample=True is for the interactive demo only —
    temperature/top_p/repetition_penalty sampling instead of pure greedy.
    """
    messages = [{"role": "user", "content": user_prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_input = int(enc["input_ids"].shape[1])

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=2,           # fix #3
        pad_token_id=tok.pad_token_id,
    )
    if sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p,
                          repetition_penalty=repetition_penalty)
    else:
        gen_kwargs.update(do_sample=False)  # deterministic: same input -> same answer

    cm = fapr_rope(model, phi) if phi != 1.0 else contextlib.nullcontext()
    with cm:
        out = model.generate(
            **enc,                      # input_ids AND attention_mask (fix #2)
            **gen_kwargs,
        )

    answer = tok.decode(out[0][n_input:], skip_special_tokens=True).strip()  # fix #4
    del out, enc
    empty_cache(device)  # a 20K-token KV cache is ~0.8GB on the 3B — release it
    if not answer:
        answer = "[no visible text generated — see README section 'The blank-output bug']"
    return answer, n_input
