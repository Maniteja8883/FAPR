# FAPR: Fertility-Normalized Anchored Positional Re-Anchoring

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.4-ee4c2c.svg)
![Status](https://img.shields.io/badge/status-research%20preview-yellow.svg)

**A training-free, inference-time RoPE position-rescaling method for equitable cross-lingual long-context retrieval and question answering.**

---

## Abstract

Subword tokenizers exhibit substantial **fertility disparity** across languages: the number of tokens required to encode semantically equivalent content varies sharply depending on how well a script is represented in the tokenizer's training vocabulary. This ratio — the **fertility gap**, denoted φ — is a property of the *(language, tokenizer) pair*, not of the language alone; the same Telugu passage can cost anywhere from ~4× to ~11× the English token budget depending on which vocabulary encodes it (see [Methodology](#methodology)).

This disparity is not cosmetic. In long-context retrieval-augmented generation, tokenizer fertility directly determines how much of a model's fixed *positional* budget — its context window — a passage consumes. High-fertility languages are consequently pushed toward, and past, the edges of the window at a **lower information budget** than English, compounding the well-documented "lost-in-the-middle" attention degradation with a second, entirely positional penalty that has nothing to do with the model's linguistic competence. During early-stage development, the combination of long, high-fertility sequences with the memory scaling of full-precision attention was also implicated in out-of-memory failures on constrained hardware, which shaped several of the architectural choices documented below (§ Environment Setup).

**FAPR** corrects the positional penalty directly, at inference time, with no fine-tuning and no added parameters. FAPR computes a per-language fertility ratio φ offline from parallel FLORES-200 text, then divides every rotary position index flowing through the model's RoPE module by φ. Because RoPE angles are continuous functions of position, this rescaling is mathematically well-defined for fractional indices and requires no architectural modification: a high-fertility passage's tokens are compressed onto the same rotational phase range that an information-equivalent English passage occupies, restoring parity in positional "real estate" across languages. The intervention is applied as a reversible wrapper around the model's rotary embedding module at load time, is never serialized into model weights, and adds negligible (microsecond-scale) overhead.

## Repository Structure

```
FAPR/
├── fapr/                    # Core library
│   ├── device.py            #   backend auto-detection (CUDA / MPS / CPU) + dtype policy
│   ├── model_loader.py      #   model + tokenizer loading, per-device quantization
│   ├── rope_patch.py        #   the FAPR intervention: fapr_rope(model, phi) context manager
│   ├── fertility.py         #   phi computation from FLORES-200 + on-disk cache
│   ├── generation.py        #   generation wrapper (chat template, attention mask, decoding)
│   └── harness.py           #   haystack packing, letter-logit scoring, calibration, metrics
├── scripts/                 # CLI entry points
│   ├── compute_phi.py       #   builds/refreshes the fertility lookup table
│   ├── run_eval.py          #   baseline / FAPR / random-phi evaluation matrix
│   ├── calibrate_degradation.py  # locates the context regime where the baseline degrades
│   └── export_prepatched.py #   optional: bakes a fixed phi into a standalone checkpoint
├── app/
│   └── demo.py               # Interactive Gradio UI (benchmarks + live playground)
├── data/
│   ├── phi_*.json            # cached fertility tables (tracked)
│   └── results/               # evaluation run outputs (gitignored)
├── docs/
│   ├── briefing.md            # original project briefing / motivation
│   └── defense_guide.md       # technical blueprint + viva-voce defense guide
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Environment Setup

FAPR runs on any standard Python environment with a CUDA, Apple Silicon (MPS), or CPU backend — no platform-specific setup is required.

```bash
git clone https://github.com/<your-org>/FAPR.git
cd FAPR

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python -m pip install -U pip
pip install -r requirements.txt
```

**Backend selection is automatic** (`fapr/device.py`, `fapr/model_loader.py`): the library detects the available accelerator at runtime and applies the corresponding quantization policy —

| Backend | Precision | Quantization |
|---|---|---|
| CUDA | bf16 / fp16 | 4-bit NF4 (`bitsandbytes`, Linux only) |
| Apple Silicon (MPS) | bf16 / fp16 | optional int8 (`optimum-quanto`) |
| CPU | fp32 | none (debugging fallback) |

No manual configuration is needed beyond installing `requirements.txt`; pass `--quantize {auto,4bit,int8-mps,none}` to any script to override the automatic policy.

Model weights are cached under `~/.cache/huggingface` on first load and reused across all subsequent runs.

## Usage

### 1. Build the fertility table

```bash
python scripts/compute_phi.py --model Qwen/Qwen2.5-3B-Instruct
```
Tokenizer-only, no model download. Computes and caches φ per language to `data/phi_<model>.json`, keyed by model id (φ is a tokenizer property and must be recomputed whenever the model changes).

### 2. Evaluation harness — baseline vs. FAPR vs. random-φ ablation

```bash
# Honesty gate: confirm the model is a strong enough zero-distractor reader
# before trusting any rescue claim (fails loudly if calibrated accuracy < 50%)
python scripts/run_eval.py --ceiling --n 40

# Locate the context regime where the baseline model actually degrades
python scripts/calibrate_degradation.py

# Full matrix: {baseline, fapr, random-phi} x languages x depths
python scripts/run_eval.py --n 100 --content-tokens 8000 --langs english telugu hindi

# Push into the out-of-window regime (raw tokens up to 1.5x native context)
python scripts/run_eval.py --overflow
```
Results are written to `data/results/eval_<model>_<timestamp>.json` (records + summary), including per-depth accuracy, bootstrap 95% confidence intervals, positional Gini coefficients, and predicted-letter histograms (a diagnostic for constant-answer collapse under multiple-choice scoring).

### 3. Interactive Gradio demo

```bash
python app/demo.py             # local: http://127.0.0.1:7860
python app/demo.py --share     # + public HTTPS link
```
The demo exposes a benchmarks tab (loads the latest `data/results/*.json`) and an interactive playground with three positional modes — baseline, FAPR, and the random-φ ablation — so the fertility-specificity of any effect is directly observable rather than asserted.

## Methodology

**Fertility ratio.** For language *L*, φ is measured over parallel FLORES-200 passages (via the Belebele `flores_passage` field) as the ratio of tokens required by *L* to tokens required by English for identical content:

$$\phi_L = \frac{\text{tokens}_L}{\text{tokens}_{\text{en}}}$$

**Core intervention.** Every absolute rotary position index *t* flowing through the model's RoPE module is rescaled by the language's fertility ratio before the rotation angle is computed:

$$t' = \frac{t}{\phi}$$

This is applied as a wrapper around the rotary embedding module's forward pass — not as an externally supplied `position_ids` tensor — because autoregressive generation recomputes position indices internally at every decoding step, silently discarding any custom position tensor supplied from outside the model. Patching the rotary module itself guarantees every recomputed step, prefill and decode alike, passes through the same rescaling.

**Adaptive gating (φ_eff).** Applying the full fertility ratio uninformatively — even when a sequence is far from the context limit — over-compresses positional geometry the model was never stressed on and can *degrade* accuracy relative to no intervention at all. The corrective is to gate φ by how full the context window actually is:

$$\phi_{\text{eff}} = \max\left(1.0,\ \min\left(\phi_{\text{raw}},\ \frac{\text{len}_{\text{actual}}}{\text{native\_ctx}}\right)\right)$$

**Implementation status:** this gate is implemented and applied in the interactive demo (`app/demo.py`). The evaluation harness (`scripts/run_eval.py`) and the core patch (`fapr/rope_patch.py`) currently apply **raw, ungated φ**, deliberately restricted (via `calibrate_degradation.py`) to a context regime independently verified to produce baseline degradation. Promoting the gate into the evaluation path — with its own ablation — is ongoing work; see [Results](#results).

**Fertility-specificity ablation.** To distinguish "fertility-conditioned rescaling helps" from "any position perturbation helps," a control condition applies a magnitude-matched but deliberately fertility-*mismatched* scale (log-uniform, rejected if within 15% of the true φ). FAPR is only considered validated where it outperforms both the unmodified baseline **and** this random-φ control.

## Results

> Populate this section from `data/results/eval_<model>_<timestamp>.json` after running `scripts/run_eval.py`. Result artifacts are gitignored by default (`data/results/`) — export the summary table below manually or via the demo's benchmarks tab.

| Language | Condition | N | Calibrated Acc. | 95% CI | Positional Gini |
|---|---|---|---|---|---|
| english | baseline | — | — | — | — |
| english | fapr | — | — | — | — |
| telugu  | baseline | — | — | — | — |
| telugu  | fapr | — | — | — | — |
| telugu  | random-φ | — | — | — | — |
| hindi   | baseline | — | — | — | — |
| hindi   | fapr | — | — | — | — |

**Current coverage.** Three of eight registered languages (english, telugu, hindi) have been run end-to-end; yoruba, bengali, swahili, thai, and finnish are registered in `fapr/fertility.py` but have not yet been evaluated. Extending coverage requires no code changes — `--langs english telugu hindi yoruba` — only additional compute.

## Citation

If you use FAPR in your research, please cite:

```bibtex
@misc{fapr2026,
  title        = {FAPR: Fertility-Normalized Anchored Positional Re-Anchoring for Cross-Lingual Long-Context Grounding},
  author       = {TODO: Author list},
  year         = {2026},
  howpublished = {\url{https://github.com/<your-org>/FAPR}},
  note         = {Preprint / research preview}
}
```

## License

This repository's code is released under the [MIT License](LICENSE). Note that the default model, `Qwen/Qwen2.5-3B-Instruct`, is distributed under the separate **Qwen Research License** (Qwen2.5 sizes 0.5B/1.5B/7B are Apache-2.0); consult the model card before commercial use.
