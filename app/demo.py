#!/usr/bin/env python3
"""Interactive FAPR demo.

WHY GRADIO (firm choice, not a menu): (1) `launch(share=True)` gives a public
HTTPS link from BOTH a Mac terminal and a Colab cell with zero deployment work —
Streamlit needs a hosted server or tunneling hacks to share; (2) Gradio keeps
the model resident in process memory across interactions, while Streamlit
re-runs the whole script on every widget change and needs cache decorators to
avoid reloading 6GB of weights; (3) the half-built Colab demo was already
Gradio, so this finishes that thread instead of starting a new one.

The demo intentionally exposes the ablation as a third mode ("random phi") so a
live audience can SEE that the benefit is fertility-specific, not generic
position noise (drawback #5, demo-side).

  python app/demo.py             # local:  http://127.0.0.1:7860
  python app/demo.py --share     # + public link (works on Mac and Colab)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fapr  # noqa: F401  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch loads)

import gradio as gr
import pandas as pd

from fapr.device import empty_cache
from fapr.fertility import DEFAULT_LANGS, ensure_phi
from fapr.generation import generate_answer
from fapr.harness import random_phi
from fapr.model_loader import DEFAULT_MODEL, load_model

MODES = ("Baseline (FAPR off)", "FAPR (fertility phi)", "Ablation: random phi")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "results"


# --------------------------------------------------------------------------
# Tab 1 helpers — benchmarks & analytics
# --------------------------------------------------------------------------

def _load_latest_results():
    """Newest data/results/eval_*.json by filename timestamp, or (None, None)."""
    files = sorted(glob.glob(str(RESULTS_DIR / "eval_*.json")))
    if not files:
        return None, None
    path = files[-1]
    return Path(path).name, json.loads(Path(path).read_text(encoding="utf-8"))


def build_summary_table(payload) -> pd.DataFrame:
    """language x condition pivot, values = cal_acc, from payload['summary']."""
    rows = []
    for key, stats in payload["summary"].items():
        lang, cond = key.split("/", 1)
        rows.append({"language": lang, "condition": cond, "cal_acc": stats["cal_acc"]})
    df = pd.DataFrame(rows)
    return df.pivot_table(index="language", columns="condition", values="cal_acc").reset_index()


def build_depth_figure(payload):
    """Matplotlib figure: accuracy vs depth%, one line per language/condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, stats in sorted(payload["summary"].items()):
        lang, cond = key.split("/", 1)
        depth_acc = stats.get("acc_by_depth", {})
        if not depth_acc:
            continue
        depths = sorted(depth_acc.keys(), key=lambda d: int(d))
        xs = [int(d) for d in depths]
        ys = [depth_acc[d] for d in depths]
        style = {"baseline": "--", "fapr": "-", "random": ":"}.get(cond, "-")
        ax.plot(xs, ys, style, marker="o", label=f"{lang}/{cond}")
    ax.set_xlabel("needle depth (%)")
    ax.set_ylabel("calibrated accuracy")
    ax.set_title("Accuracy vs. needle depth, by language/condition")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def load_benchmarks_tab():
    fname, payload = _load_latest_results()
    if payload is None:
        msg = ("No results found yet. Run `scripts/run_eval.py` first to "
               "generate `data/results/eval_*.json`.")
        return msg, None, None
    df = build_summary_table(payload)
    fig = build_depth_figure(payload)
    return f"Loaded `{fname}`.", df, fig


# --------------------------------------------------------------------------
# Tab 2 helpers — adaptive phi gating + token capacity bar
# --------------------------------------------------------------------------

def gate_phi(phi_lang: float, n_input_tokens: int, native_ctx: int) -> float:
    """Cross-lingual research finding (Attempt 2 — 'the robustness paradox'):
    applying the full fertility ratio to text nowhere near the context limit
    over-compresses healthy local attention geometry. Gate phi to how full the
    window actually is."""
    return max(1.0, min(phi_lang, n_input_tokens / native_ctx))


def _bar_html(label: str, pct: float, warn: bool) -> str:
    pct_clamped = max(0.0, min(pct, 100.0))
    color = "#dc2626" if warn else ("#f59e0b" if pct >= 80 else "#22c55e")
    warn_text = (' <b style="color:#dc2626;">Memory Warning: exceeds native window</b>'
                 if warn else "")
    return (
        f'<div style="margin-bottom:6px;">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.85em;">'
        f'<span>{label}</span><span>{pct:.1f}%{warn_text}</span></div>'
        f'<div style="background:#e5e7eb;border-radius:4px;height:10px;width:100%;">'
        f'<div style="background:{color};height:10px;border-radius:4px;width:{pct_clamped}%;"></div>'
        f'</div></div>'
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

def build_app(model_id: str, quantize: str):
    model, tok, device = load_model(model_id, quantize)
    ctx = int(getattr(model.config, "max_position_embeddings", 32768))
    # First launch computes phi (downloads Belebele once); every later launch is
    # a JSON read — the "don't redo everything on every run" path.
    phi_table = ensure_phi(tok, model_id, DEFAULT_LANGS)
    max_phi = max(float(v) for v in phi_table.values())

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(tok.encode(text, add_special_tokens=False))

    def update_capacity_bar(language, document, question):
        phi_lang = float(phi_table[language])
        prompt = ("Use the document to answer the question.\n\n"
                  f"Document:\n{document}\n\nQuestion: {question}")
        n_tokens = count_tokens(prompt)
        phi_eff = gate_phi(phi_lang, n_tokens, ctx)
        effective_tokens = n_tokens / phi_eff

        raw_pct = 100.0 * n_tokens / ctx
        eff_pct = 100.0 * effective_tokens / ctx
        html = _bar_html(f"raw tokens ({n_tokens:,} / {ctx:,})", raw_pct, warn=n_tokens > ctx)
        html += _bar_html(
            f"phi-effective usage (phi_lang={phi_lang:.3f}, phi_effective={phi_eff:.3f})",
            eff_pct, warn=False)
        if phi_eff < phi_lang:
            html += ('<div style="font-size:0.8em;color:#2563eb;">gating active — '
                     f"full fertility phi ({phi_lang:.3f}) was reduced to phi_effective "
                     f"({phi_eff:.3f}) because this input is not near the context limit.</div>")
        return html

    def infer(language, document, question, mode, max_new_tokens, use_sampling):
        if not document.strip() or not question.strip():
            return "Please provide both a document and a question.", ""

        phi_lang = float(phi_table[language])
        prompt = ("Use the document to answer the question.\n\n"
                  f"Document:\n{document}\n\nQuestion: {question}")
        n_input_est = count_tokens(prompt)
        phi_eff = gate_phi(phi_lang, n_input_est, ctx)

        if mode == MODES[0]:
            phi_used = 1.0
        elif mode == MODES[1]:
            phi_used = phi_eff
        else:
            # Seeded by the document so repeated clicks are reproducible.
            seed = int(hashlib.sha1(document.encode("utf-8")).hexdigest()[:8], 16)
            phi_used = random_phi(phi_eff, random.Random(seed), max_phi)

        answer, n_tokens = generate_answer(
            model, tok, device, prompt, phi=phi_used, max_new_tokens=int(max_new_tokens),
            sample=bool(use_sampling))
        empty_cache(device)

        effective = n_tokens / phi_used
        raw_pct, eff_pct = 100.0 * n_tokens / ctx, 100.0 * effective / ctx
        over = (f"\n\n> **Note:** raw tokens exceed the native window. Baseline positions "
                f"are out-of-distribution here — this is exactly the regime FAPR re-anchors. "
                f"(Memory still scales with raw tokens.)" if n_tokens > ctx else "")
        gating_note = (f" (gated down from table value {phi_lang:.3f})"
                       if mode != MODES[0] and phi_eff < phi_lang else "")
        stats = (
            f"| metric | value |\n|---|---|\n"
            f"| mode | {mode} |\n"
            f"| phi_lang (table) | {phi_lang:.3f} |\n"
            f"| phi_effective (applied){gating_note} | {phi_used:.3f} |\n"
            f"| raw input tokens | {n_tokens:,} |\n"
            f"| effective (phi-compressed) positions | {effective:,.0f} |\n"
            f"| native context window | {ctx:,} |\n"
            f"| window used — raw | {raw_pct:.1f}% |\n"
            f"| window used — effective | {eff_pct:.1f}% |{over}"
        )
        return answer, stats

    with gr.Blocks(title="FAPR — Fertility-Normalized Positional Re-Anchoring",
                    theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# FAPR demo — `{model_id}` on `{device}`")

        with gr.Tab("Performance Benchmarks & Analytics"):
            gr.Markdown(
                "## Research finding: phi-driven positional inequity across languages\n"
                "Fertility ratio (phi) — how many more tokens a language needs to say the same "
                "thing as English — pushes higher-fertility languages toward the edge of the "
                "context window sooner. FAPR re-anchors positions by dividing by phi so that "
                "equivalent *information*, not equivalent *token count*, occupies equivalent "
                "positional space. The table/plot below compare Baseline (no patch), FAPR "
                "(fertility phi), and a magnitude-matched random-phi ablation, from the most "
                "recent `scripts/run_eval.py` run.")
            load_btn = gr.Button("Load latest results")
            status_md = gr.Markdown()
            with gr.Row():
                summary_table = gr.Dataframe(label="calibrated accuracy — language x condition")
                depth_plot = gr.Plot(label="accuracy vs. needle depth")
            load_btn.click(load_benchmarks_tab, None, [status_md, summary_table, depth_plot])
            demo.load(load_benchmarks_tab, None, [status_md, summary_table, depth_plot])

        with gr.Tab("Interactive Playground"):
            gr.Markdown(
                f"Pick a language, paste a document + question, and compare the three modes. "
                f"phi table: " + ", ".join(f"{l}={phi_table[l]:.2f}" for l in DEFAULT_LANGS))
            with gr.Row():
                language = gr.Dropdown(choices=list(DEFAULT_LANGS), value="telugu", label="Language")
                mode = gr.Radio(choices=list(MODES), value=MODES[1], label="Positional mode")
                max_new = gr.Slider(16, 512, value=160, step=16, label="Max new tokens")
            use_sampling = gr.Checkbox(
                value=False,
                label="Sampling mode (temperature=0.7, top_p=0.9, repetition_penalty=1.2) — "
                      "off = deterministic greedy")
            document = gr.Textbox(lines=12, label="Document (paste long text here)")
            question = gr.Textbox(lines=2, label="Question")
            capacity_bar = gr.HTML(label="Token capacity")
            btn = gr.Button("Answer", variant="primary")
            answer = gr.Textbox(lines=6, label="Model answer")
            stats = gr.Markdown(label="Token / window stats")

            capacity_inputs = [language, document, question]
            document.change(update_capacity_bar, capacity_inputs, capacity_bar)
            question.change(update_capacity_bar, capacity_inputs, capacity_bar)
            language.change(update_capacity_bar, capacity_inputs, capacity_bar)

            btn.click(infer, [language, document, question, mode, max_new, use_sampling],
                     [answer, stats])

    # One request at a time: a single 20K-token KV cache is ~0.8GB on the 3B —
    # two concurrent ones on 16GB unified memory is how demos die mid-talk.
    demo.queue(default_concurrency_limit=1)
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quantize", default="auto",
                    choices=["auto", "4bit", "int8-mps", "none"])
    ap.add_argument("--share", action="store_true", help="create a public link")
    args, _ = ap.parse_known_args()  # tolerate notebook/Colab extra argv
    build_app(args.model, args.quantize).launch(share=args.share)


if __name__ == "__main__":
    main()
