"""Shared evaluation harness: parallel haystack packing, letter-logit scoring,
letter-bias calibration, the random-phi ablation draw, and metrics.

Scoring stays single-forward (A/B/C/D logit comparison, no generation) exactly
as in the Colab phase — it is fast and immune to free-text parsing noise. What
is new here fixes the two measurement drawbacks:
  - option order is shuffled per question with a fixed seed, so the gold letter
    is uniform over A-D across the dataset (structural fix for letter bias);
  - we additionally report CALIBRATED accuracy: the model's letter prior,
    measured on a content-free prompt, is subtracted from the letter logits
    (contextual-calibration style), and the predicted-letter histogram is kept
    in every summary so a constant-answer collapse is visible at a glance.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import torch

from .fertility import LANGS

LETTERS = ("A", "B", "C", "D")


def load_rows(lang_name: str) -> List[dict]:
    from datasets import load_dataset  # lazy
    ds = load_dataset("facebook/belebele", LANGS[lang_name], split="test")
    # Sort by the parallel-alignment key so row i is the SAME question in every
    # language config — packing and sampling then transfer across languages.
    return sorted(ds, key=lambda r: (str(r["link"]), str(r["question_number"])))


class HaystackBuilder:
    """Packs PARALLEL content. The distractor set is chosen once, by ENGLISH
    token budget, and reused by `link` in every language. Every language then
    carries identical information; raw token length differs only by fertility.
    That inflation — Telugu drifting toward the window edge while English sits
    comfortably — is the phenomenon FAPR targets, and it resolves the Attempt-2
    'robustness paradox': you do not need a weak model, you need the regime
    where fertility pushes one language near the edge."""

    def __init__(self, tokenizer, lang_rows: List[dict], eng_rows: List[dict]):
        self.tok = tokenizer
        self.rows = lang_rows
        self.passages: Dict[str, str] = {}
        for r in lang_rows:
            self.passages.setdefault(r["link"], r["flores_passage"])
        eng_pass: Dict[str, str] = {}
        for r in eng_rows:
            eng_pass.setdefault(r["link"], r["flores_passage"])
        self.eng_tokens = {
            link: len(tokenizer.encode(p, add_special_tokens=False))
            for link, p in eng_pass.items()
        }
        self.links = [l for l in eng_pass if l in self.passages]

    def build(self, q_idx: int, depth_pct: float, content_tokens: int,
              seed: int) -> Tuple[List[str], dict]:
        """Returns (ordered passage list, question row). depth_pct is how much
        distractor content precedes the needle (0 = needle first, 100 = last)."""
        row = self.rows[q_idx]
        needle_link = row["link"]
        rng = random.Random(f"{seed}-{needle_link}-pack")
        candidates = [l for l in self.links if l != needle_link]
        rng.shuffle(candidates)

        picked, total = [], 0
        for link in candidates:
            if total >= content_tokens:
                break
            picked.append(link)
            total += self.eng_tokens[link]

        target = depth_pct / 100.0 * total
        acc, pos = 0, len(picked)
        for i, link in enumerate(picked):
            if acc >= target:
                pos = i
                break
            acc += self.eng_tokens[link]

        docs = ([self.passages[l] for l in picked[:pos]]
                + [row["flores_passage"]]
                + [self.passages[l] for l in picked[pos:]])
        return docs, row


def format_mcq(docs: List[str], row: dict, rng: random.Random) -> Tuple[str, str]:
    """Seeded shuffle of option order -> gold letter uniform over A-D
    (drawback #2, structural half of the fix). Returns (prompt, gold_letter)."""
    options = [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]]
    gold = int(row["correct_answer_num"]) - 1
    order = [0, 1, 2, 3]
    rng.shuffle(order)
    gold_letter = LETTERS[order.index(gold)]
    lines = "\n".join(f"{LETTERS[i]}) {options[order[i]]}" for i in range(4))
    prompt = ("Read the passages below, then answer the multiple-choice question.\n\n"
              + "\n\n".join(docs)
              + f"\n\nQuestion: {row['question']}\n{lines}\n\n"
              + "Answer with only the letter (A, B, C, or D).")
    return prompt, gold_letter


def null_prompt() -> str:
    """Content-free probe for the model's letter prior (calibration half of the
    drawback #2 fix): same template, all content replaced by N/A."""
    lines = "\n".join(f"{L}) N/A" for L in LETTERS)
    return ("Read the passages below, then answer the multiple-choice question.\n\n"
            "N/A\n\nQuestion: N/A\n" + lines
            + "\n\nAnswer with only the letter (A, B, C, or D).")


def letter_token_ids(tok) -> Dict[str, List[int]]:
    """First-token ids for each letter, with and without a leading space —
    covers both BPE ('A' vs 'ĠA') and SentencePiece ('▁A') conventions."""
    ids: Dict[str, List[int]] = {}
    for L in LETTERS:
        variants = set()
        for v in (L, " " + L):
            e = tok.encode(v, add_special_tokens=False)
            if e:
                variants.add(e[0])
        ids[L] = sorted(variants)
    return ids


@torch.inference_mode()
def score_letters(model, tok, prompt: str,
                  letter_ids: Dict[str, List[int]]) -> Tuple[Dict[str, float], int]:
    """One forward pass, next-token logits at the answer position. use_cache=False:
    we never decode from this pass, so skipping KV allocation saves the exact
    memory that made long-context eval fragile on 16GB."""
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**enc, use_cache=False).logits[0, -1].float()
    scores = {L: max(logits[i].item() for i in ids) for L, ids in letter_ids.items()}
    return scores, int(enc["input_ids"].shape[1])


def random_phi(true_phi: float, rng: random.Random, table_max: float) -> float:
    """Drawback #5 ablation: a scaling factor of comparable magnitude that is
    deliberately NOT the language's fertility — log-uniform on [1.3, max(3, 2x
    the largest phi in the table)], rejected if within 15% of the true phi.
    If FAPR's gain were 'any position perturbation helps', this condition would
    match it; if the gain is fertility-specific, this condition should not."""
    lo, hi = 1.3, max(3.0, 2.0 * table_max)
    while True:
        cand = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        if abs(cand - true_phi) / max(true_phi, 1e-9) > 0.15:
            return cand


def gini(values) -> float:
    """Positional Gini over per-depth accuracies (carried over from Colab):
    0 = perfectly even across depths, higher = more positional inequality."""
    vals = [max(float(v), 1e-9) for v in values]
    n = len(vals)
    mu = sum(vals) / n
    return sum(abs(a - b) for a in vals for b in vals) / (2 * n * n * mu)


def bootstrap_ci(flags: List[int], n_boot: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """95% bootstrap CI on mean accuracy — drawback #4's honesty bar: with small
    N the interval is wide, and the summary shows exactly how wide."""
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    means = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]
