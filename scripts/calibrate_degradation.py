#!/usr/bin/env python3
"""Find WHERE this model actually degrades before spending compute on the matrix.

This exists because of Attempt 2's 'robustness paradox' (see the context digest):
Qwen2.5-1.5B at 12K tokens was ~perfect, so FAPR had nothing to rescue and
position compression only distorted healthy local geometry. FAPR's operating
regime is 'evidence near the window edge'. This script sweeps English haystack
budgets x needle depths, BASELINE ONLY, and prints the accuracy matrix.

Read it like this: pick the smallest budget where deep-needle accuracy clearly
drops below shallow-needle accuracy — that is your degradation onset. Then set
run_eval.py --content-tokens at/above it. Raw Telugu length there will be about
budget x phi(telugu); that inflation is the phenomenon, so English can stay
comfortable while Telugu is stressed. If NO budget shows a drop even at the
window edge, say so honestly in the paper: this model resists lost-in-the-middle
in-window, and the out-of-window regime (run_eval --overflow) is the claim.

  python scripts/calibrate_degradation.py                 # defaults
  python scripts/calibrate_degradation.py --budgets 8000 16000 24000 --n 12
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fapr  # noqa: F401

from fapr.device import empty_cache
from fapr.harness import (LETTERS, HaystackBuilder, format_mcq, letter_token_ids,
                          load_rows, null_prompt, score_letters)
from fapr.model_loader import DEFAULT_MODEL, load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quantize", default="auto",
                    choices=["auto", "4bit", "int8-mps", "none"])
    ap.add_argument("--budgets", nargs="+", type=int,
                    default=[2000, 4000, 8000, 12000, 16000])
    ap.add_argument("--depths", nargs="+", type=int, default=[0, 50, 100])
    ap.add_argument("--n", type=int, default=16, help="questions per cell")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, tok, device = load_model(args.model, args.quantize)
    ctx = int(getattr(model.config, "max_position_embeddings", 32768))
    letter_ids = letter_token_ids(tok)
    rows = load_rows("english")
    builder = HaystackBuilder(tok, rows, rows)
    prior, _ = score_letters(model, tok, null_prompt(), letter_ids)

    qrng = random.Random(f"{args.seed}-qsample")
    q_indices = qrng.sample(range(len(rows)), min(args.n, len(rows)))

    header = "budget".rjust(8) + "".join(f"  depth{d:>3}%" for d in args.depths)
    print(f"\nEnglish baseline degradation sweep (native ctx = {ctx}):\n{header}")
    for budget in args.budgets:
        if budget > ctx:
            print(f"{budget:8d}  (skipped: exceeds native window; use run_eval --overflow)")
            continue
        cells = []
        for depth in args.depths:
            correct = 0
            for qi in q_indices:
                rng_q = random.Random(f"{args.seed}-en-{qi}-{depth}")
                docs, row = builder.build(qi, depth, budget, args.seed)
                prompt, gold = format_mcq(docs, row, rng_q)
                scores, _ = score_letters(model, tok, prompt, letter_ids)
                pred = max(LETTERS, key=lambda L: scores[L] - prior[L])
                correct += int(pred == gold)
                empty_cache(device)
            cells.append(correct / len(q_indices))
        print(f"{budget:8d}" + "".join(f"  {c:8.2f}" for c in cells), flush=True)

    print("\nPick the smallest budget whose deep-needle column clearly drops; "
          "use it (or higher) as run_eval.py --content-tokens.")


if __name__ == "__main__":
    main()
