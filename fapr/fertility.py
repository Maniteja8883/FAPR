"""Fertility ratio (phi) computation and on-disk caching.

phi(language) = tokens this model's tokenizer needs for a set of parallel
passages / tokens it needs for the same passages in English. Computed from
Belebele's `flores_passage` field: Belebele passages ARE FLORES-200 paragraphs,
they are parallel across all language configs, parquet-hosted and ungated — one
dataset dependency for both phi and the eval instead of two.

The cache is keyed by model id because phi is a property of the
(language, tokenizer) PAIR, not of the language: TinyLlama's Llama-2 32K vocab
byte-fallbacks Telugu script into ~3 tokens per character (the 10.67x artifact),
while Qwen2.5's 151K multilingual vocab does not. All Qwen2.5 sizes share one
tokenizer, so a table computed once is valid for 0.5B through 72B.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

LANGS: Dict[str, str] = {
    "english": "eng_Latn",
    "telugu": "tel_Telu",
    "hindi": "hin_Deva",
    # Registered for drawback #3 — the briefing promised these but they were
    # never run. Adding one to an experiment is now just:
    #   python scripts/run_eval.py --langs english telugu hindi yoruba
    "yoruba": "yor_Latn",
    "bengali": "ben_Beng",
    "swahili": "swh_Latn",
    "thai": "tha_Thai",
    "finnish": "fin_Latn",
}
DEFAULT_LANGS = ("english", "telugu", "hindi")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def phi_cache_path(model_id: str) -> Path:
    return DATA_DIR / ("phi_" + model_id.replace("/", "_") + ".json")


def load_phi_table(model_id: str) -> Optional[Dict[str, float]]:
    p = phi_cache_path(model_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["phi"]
    return None


def _unique_passages(rows) -> Dict[str, str]:
    """Belebele has ~900 questions over fewer unique passages; dedupe by the
    `link` field, which is the parallel-alignment key across language configs."""
    out: Dict[str, str] = {}
    for r in rows:
        out.setdefault(r["link"], r["flores_passage"])
    return out


def compute_phi(tokenizer, langs: Iterable[str], n_passages: int = 300) -> Dict[str, float]:
    from datasets import load_dataset  # lazy import keeps `import fapr` light

    def n_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    eng = _unique_passages(load_dataset("facebook/belebele", LANGS["english"], split="test"))
    links = list(eng.keys())[:n_passages]

    phi: Dict[str, float] = {}
    for name in langs:
        if name == "english":
            phi[name] = 1.0
            continue
        if name not in LANGS:
            raise KeyError(f"Unknown language '{name}'. Add its FLORES code to fapr/fertility.py LANGS.")
        lang_pass = _unique_passages(
            load_dataset("facebook/belebele", LANGS[name], split="test"))
        common = [l for l in links if l in lang_pass]
        if len(common) < len(links):
            warnings.warn(f"{name}: only {len(common)}/{len(links)} parallel passages matched")
        eng_total = sum(n_tokens(eng[l]) for l in common)
        lang_total = sum(n_tokens(lang_pass[l]) for l in common)
        phi[name] = round(lang_total / eng_total, 3)
    return phi


def ensure_phi(tokenizer, model_id: str, langs: Iterable[str] = DEFAULT_LANGS,
               n_passages: int = 300, refresh: bool = False) -> Dict[str, float]:
    """Load phi from disk; compute (and persist) only what is missing.

    This is the 'never re-derive at startup' path: after the first run, demo and
    eval boot cost for phi is a single JSON read."""
    langs = list(langs)
    cached = None if refresh else load_phi_table(model_id)
    if cached and all(l in cached for l in langs):
        return cached
    missing = langs if refresh else [l for l in langs if not cached or l not in cached]
    computed = compute_phi(tokenizer, missing, n_passages=n_passages)
    table = dict(cached or {})
    table.update(computed)
    table.setdefault("english", 1.0)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "dataset": "facebook/belebele (flores_passage field; parallel FLORES-200 paragraphs)",
        "n_passages": n_passages,
        "computed": datetime.now().isoformat(timespec="seconds"),
        "phi": table,
    }
    phi_cache_path(model_id).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return table
