#!/usr/bin/env python3
"""OPTIONAL: export a checkpoint that loads 'pre-patched' for ONE fixed language.

What can and cannot be persisted — the honest version (read this before using):

CANNOT: the monkey-patch itself. `save_pretrained()` serializes weights and
config, never Python behavior, and FAPR has no weights — nothing was trained,
so a checkpoint saved 'while patched' loads UNpatched. FAPR is inference-time
by design and cannot be trained in.

CAN:
  (a) the phi table — data/phi_*.json, handled by fapr/fertility.py;
  (b) model weights — the HuggingFace cache (~/.cache/huggingface) already
      persists them after the first download, nothing to do;
  (c) THIS script's trick: for one FIXED phi you can bake the scaling into the
      config as rope_scaling = {"rope_type": "linear", "factor": phi}. Linear
      position interpolation divides positions by the factor inside the stock
      rotary module — mathematically identical to FAPR with a constant phi. The
      exported copy loads pre-scaled with ZERO custom code (nice for serving
      exactly one language, or for the paper's 'FAPR = per-language linear
      interpolation chosen by fertility' framing).

Cost of (c): a full weight copy on disk per language (~6GB for the 3B) and the
factor is frozen — useless for the language-switching demo. The runtime patch
plus cached phi JSON remains the primary mechanism; re-applying the patch at
startup costs microseconds. The real startup costs were always the model
download (cached) and phi derivation (cached).

  python scripts/export_prepatched.py --language telugu --out models/qwen3b-fapr-te
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fapr  # noqa: F401

from fapr.fertility import load_phi_table
from fapr.model_loader import DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--language", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    table = load_phi_table(args.model)
    if not table or args.language not in table:
        sys.exit(f"No cached phi for '{args.language}' — run scripts/compute_phi.py first.")
    phi = float(table[args.language])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[export] loading {args.model} on CPU (fp16) — needs ~6GB RAM for the 3B")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    # Both keys: 'rope_type' is current transformers, 'type' the legacy alias.
    # The scaling takes effect when the EXPORTED copy is loaded (rotary inv_freq
    # is built at init from config) — that is the whole point of this script.
    model.config.rope_scaling = {"rope_type": "linear", "type": "linear", "factor": phi}

    out = Path(args.out)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(out)
    print(f"[export] wrote {out} with rope_scaling linear factor={phi} "
          f"({args.language}). Load it like any HF model — no patch needed, "
          f"but it ONLY suits {args.language}.")


if __name__ == "__main__":
    main()
