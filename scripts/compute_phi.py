#!/usr/bin/env python3
"""Build/refresh the phi lookup table. Tokenizer-only — no model download, runs
in ~1-2 minutes, and is the fastest way to answer the open question from the
Colab phase: is Telugu really 10.67x, or was that TinyLlama's tokenizer?

  python scripts/compute_phi.py                       # Qwen2.5-3B tokenizer, en/te/hi
  python scripts/compute_phi.py --langs english telugu hindi yoruba bengali
  python scripts/compute_phi.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fapr  # noqa: F401  (sets env before torch/transformers load)

from fapr.fertility import DEFAULT_LANGS, LANGS, ensure_phi, phi_cache_path
from fapr.model_loader import DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--langs", nargs="+", default=list(DEFAULT_LANGS),
                    choices=sorted(LANGS.keys()))
    ap.add_argument("--n-passages", type=int, default=300)
    ap.add_argument("--refresh", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    table = ensure_phi(tok, args.model, tuple(args.langs),
                       n_passages=args.n_passages, refresh=args.refresh)

    print(f"\nphi table for {args.model} (cached at {phi_cache_path(args.model)}):")
    for lang in args.langs:
        bar = "#" * max(1, int(round(table[lang] * 4)))
        print(f"  {lang:10s} {table[lang]:6.3f}x  {bar}")
    print("\nReminder: phi is a property of the (language, tokenizer) pair. Compare this "
          "against data/phi_TinyLlama_TinyLlama-1.1B-Chat-v1.0.json to quantify how much "
          "of the old 10.67x Telugu ratio was byte-fallback artifact.")


if __name__ == "__main__":
    main()
