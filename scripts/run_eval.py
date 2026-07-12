#!/usr/bin/env python3
"""FAPR evaluation matrix: {baseline, fapr, random-phi} x languages x depths.

How this script maps to the known drawbacks:
  #1 weak reader  -> `--ceiling` re-runs the zero-distractor honesty check on the
                     new model and prints a loud warning if calibrated ceiling < 50%.
                     Run it BEFORE trusting any matrix result.
  #2 letter bias  -> gold letters uniformly shuffled per question (structural fix),
                     calibrated accuracy reported alongside raw, and the predicted-
                     letter histogram is printed (constant-answer diagnostic).
  #4 small N      -> --n (default 100) + bootstrap 95% CIs in every summary line.
  #5 no ablation  -> the `random` condition applies a seeded, magnitude-matched but
                     fertility-mismatched scale. FAPR beating both baseline AND
                     random is the fertility-specificity claim.

Packing: distractors are chosen once by ENGLISH token budget and reused by link in
all languages — identical content everywhere, raw length differing only by
fertility. The per-language budget is clamped so raw tokens fit the native window
(or 1.5x it with --overflow: the out-of-window regime, where the inequity — and
FAPR's headroom — is largest).

Typical usage:
  python scripts/run_eval.py --ceiling --n 40      # honesty gate, ~minutes
  python scripts/run_eval.py --quick               # smoke matrix
  python scripts/run_eval.py --n 120 --content-tokens 8000   # paper run (long!)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fapr  # noqa: F401

from fapr.device import empty_cache
from fapr.fertility import DEFAULT_LANGS, ensure_phi
from fapr.harness import (LETTERS, HaystackBuilder, bootstrap_ci, format_mcq,
                          gini, letter_token_ids, load_rows, null_prompt,
                          random_phi, score_letters)
from fapr.model_loader import DEFAULT_MODEL, load_model
from fapr.rope_patch import fapr_rope


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quantize", default="auto",
                    choices=["auto", "4bit", "int8-mps", "none"])
    ap.add_argument("--langs", nargs="+", default=list(DEFAULT_LANGS))
    ap.add_argument("--n", type=int, default=100, help="questions per language")
    ap.add_argument("--depths", nargs="+", type=int, default=[0, 25, 50, 75, 100])
    ap.add_argument("--conditions", nargs="+", default=["baseline", "fapr", "random"])
    ap.add_argument("--content-tokens", type=int, default=8000,
                    help="haystack size in ENGLISH-equivalent tokens (raw = this x phi)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ceiling", action="store_true",
                    help="zero-distractor honesty check (drawback #1 gate)")
    ap.add_argument("--overflow", action="store_true",
                    help="allow raw tokens up to 1.5x the native window")
    ap.add_argument("--quick", action="store_true",
                    help="n=25, depths 0/50/100, content 4000 — smoke run")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.n, args.depths, args.content_tokens = 25, [0, 50, 100], 4000

    model, tok, device = load_model(args.model, args.quantize)
    ctx = int(getattr(model.config, "max_position_embeddings", 32768))
    phi_table = ensure_phi(tok, args.model, tuple(args.langs))
    letter_ids = letter_token_ids(tok)
    eng_rows = load_rows("english")
    print(f"[eval] phi = { {l: phi_table[l] for l in args.langs} }")

    records, prior_cache = [], {}
    t0 = time.time()
    for lang in args.langs:
        rows = eng_rows if lang == "english" else load_rows(lang)
        builder = HaystackBuilder(tok, rows, eng_rows)
        phi = float(phi_table[lang])

        cap = int(ctx * (1.5 if args.overflow else 0.95))
        reserve = 900  # question + options + chat-template overhead, upper bound
        budget = 0 if args.ceiling else min(
            args.content_tokens, max(0, int((cap - reserve) / max(phi, 1.0))))
        if not args.ceiling and budget < args.content_tokens:
            print(f"[eval] {lang}: content budget clamped {args.content_tokens} -> {budget} "
                  f"english-equiv tokens (raw ~{int(budget * phi)} must fit cap {cap}; "
                  f"pass --overflow for the out-of-window regime)")

        qrng = random.Random(f"{args.seed}-qsample")
        q_indices = qrng.sample(range(len(rows)), min(args.n, len(rows)))
        depths = [50] if args.ceiling else list(args.depths)

        n_done = 0
        for cond in args.conditions:
            for qi in q_indices:
                for depth in depths:
                    rng_q = random.Random(f"{args.seed}-{lang}-{qi}-{depth}")
                    docs, row = builder.build(qi, depth, budget, args.seed)
                    prompt, gold = format_mcq(docs, row, rng_q)

                    if cond == "baseline":
                        phi_used = 1.0
                    elif cond == "fapr":
                        phi_used = phi
                    else:
                        phi_used = random_phi(
                            phi, rng_q, max(float(v) for v in phi_table.values()))

                    cm = (fapr_rope(model, phi_used)
                          if cond != "baseline" else contextlib.nullcontext())
                    with cm:
                        scores, n_tok = score_letters(model, tok, prompt, letter_ids)
                        pk = round(phi_used, 3)
                        if pk not in prior_cache:  # tiny prompt, memoized by phi
                            prior_cache[pk], _ = score_letters(
                                model, tok, null_prompt(), letter_ids)
                    prior = prior_cache[pk]

                    records.append({
                        "lang": lang, "condition": cond, "depth": depth, "q": qi,
                        "gold": gold,
                        "raw_pred": max(LETTERS, key=lambda L: scores[L]),
                        "cal_pred": max(LETTERS, key=lambda L: scores[L] - prior[L]),
                        "raw_tokens": n_tok, "phi_used": round(phi_used, 4),
                    })
                    n_done += 1
                    if n_done % 10 == 0:
                        print(f"[eval] {lang}/{cond}: {n_done} forwards, "
                              f"{(time.time() - t0) / 60:.1f} min elapsed", flush=True)
                    if n_done % 8 == 0:
                        empty_cache(device)
        empty_cache(device)

    summary = summarize(records, args)
    out_dir = (Path(args.out) if args.out
               else Path(__file__).resolve().parents[1] / "data" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"eval_{args.model.replace('/', '_')}_{stamp}.json"
    out_path.write_text(json.dumps(
        {"args": vars(args), "summary": summary, "records": records}, indent=2),
        encoding="utf-8")
    print_summary(summary, args)
    print(f"\n[eval] full results -> {out_path}")


def summarize(records, args):
    by = defaultdict(list)
    for r in records:
        by[(r["lang"], r["condition"])].append(r)
    summary = {}
    for (lang, cond), rs in sorted(by.items()):
        cal = [1 if r["cal_pred"] == r["gold"] else 0 for r in rs]
        raw = [1 if r["raw_pred"] == r["gold"] else 0 for r in rs]
        depth_acc = {}
        for d in sorted({r["depth"] for r in rs}):
            flags = [1 if r["cal_pred"] == r["gold"] else 0 for r in rs if r["depth"] == d]
            depth_acc[str(d)] = round(sum(flags) / len(flags), 4)
        lo, hi = bootstrap_ci(cal, seed=args.seed)
        summary[f"{lang}/{cond}"] = {
            "n": len(rs),
            "raw_acc": round(sum(raw) / len(raw), 4),
            "cal_acc": round(sum(cal) / len(cal), 4),
            "cal_acc_ci95": [round(lo, 4), round(hi, 4)],
            "acc_by_depth": depth_acc,
            "positional_gini": round(gini(depth_acc.values()), 4),
            "pred_histogram": dict(Counter(r["cal_pred"] for r in rs)),
            "mean_raw_tokens": int(sum(r["raw_tokens"] for r in rs) / len(rs)),
        }
    return summary


def print_summary(summary, args):
    print("\n===== FAPR eval summary (accuracy is letter-bias-calibrated) =====")
    for key, s in summary.items():
        print(f"{key:22s} acc={s['cal_acc']:.3f} CI95={s['cal_acc_ci95']} "
              f"raw={s['raw_acc']:.3f} gini={s['positional_gini']:.4f} "
              f"~tokens={s['mean_raw_tokens']}")
        print(f"{'':22s} by-depth={s['acc_by_depth']} preds={s['pred_histogram']}")
    if args.ceiling:
        weak = [k for k, s in summary.items() if s["cal_acc"] < 0.5]
        if weak:
            print("\n*** CEILING WARNING (drawback #1): calibrated zero-distractor "
                  "accuracy < 50% for: " + ", ".join(weak))
            print("*** A reader this weak cannot support rescue claims — reconsider "
                  "the model before running the main matrix.")
        else:
            print("\n[eval] ceiling gate PASSED — reader is strong enough for rescue claims.")


if __name__ == "__main__":
    main()
