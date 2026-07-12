# FAPR — Technical Blueprint & Project Defense Guide

**Fertility-Normalized Anchored Positional Re-Anchoring** — a training-free, inference-time RoPE position-scaling method for cross-lingual long-context QA/RAG.

*Prepared for viva-voce defense. Every claim below is grounded in the code as it stands at `~/Desktop/FAPR/`, not the earlier Colab-phase design. Where the two diverge, the divergence is called out explicitly — the divergences are the point.*

**Reading key.** Each section pairs two voices:
- 🎓 **The record** — precise, authoritative language for the panel and the paper.
- 🧒 **The picture** — a zero-jargon mental model so you can defend the *idea* even if the *syntax* slips.

---

## How to read this guide in one breath

FAPR notices that some languages need far more tokens than English to say the same thing. Those extra tokens shove the real evidence toward the far edge of a model's memory, where it reads worst. FAPR divides every token's *position number* by that language's "fertility ratio" φ, so the same *information* sits in the same *positional space* it would in English — no retraining, no extra memory, one line of math inside the model's rotary layer. The rest of this document is (1) how the data flows, (2) the mathematics, (3) what actually changes in the model's behavior, and (4) how to survive the reviewer questions — including the honest "not yet done" answers.

---

# CHAPTER 1 — Core System Architecture & Data Workflow

## 1.1 The Tokenization Disparity — why φ is a *pair* property, not a language property

🎓 **The record.** Subword tokenizers (BPE / SentencePiece) are trained to be efficient on their training mixture, which is English-dominated. A script that is rare or absent in that mixture has no dedicated merges, so the tokenizer falls back to encoding it byte-by-byte (or character-by-character). The result is **tokenizer fertility**: the ratio φ of tokens needed for a passage in language L to tokens needed for the *same content* in English. Crucially, φ is a property of the **(language, tokenizer) pair** — change the tokenizer and φ changes, because the language did not.

The project measures this rather than assuming it, and the two cached tables make the point by contrast:

| Model / tokenizer | Vocab | Telugu φ | Hindi φ | Source |
|---|---|---|---|---|
| TinyLlama-1.1B (Llama-2 tokenizer) | 32K | **10.67×** | 4.58× | `data/phi_TinyLlama_TinyLlama-1.1B-Chat-v1.0.json` |
| Qwen2.5-3B-Instruct | 151K multilingual | **6.947×** | 4.404× | `data/phi_Qwen_Qwen2.5-3B-Instruct.json` |

The 10.67× is not "Telugu is 10× harder." It is the Llama-2 32K vocabulary **byte-falling-back** Telugu script into ~3 byte-tokens per character. Swap in Qwen2.5's 151K multilingual vocabulary and the same Telugu content drops to 6.95×. **The tokenizer, not the language, produced most of the gap.**

Because φ is a tokenizer property, `fapr/fertility.py` caches the table **keyed by model id** (`phi_cache_path()` → `phi_<model>.json`) and recomputes it whenever the model changes. It is computed from Belebele's `flores_passage` field (parallel FLORES-200 paragraphs), deduped by the `link` alignment key, over the first `n_passages` (default 300), as `round(Σ tokens_lang / Σ tokens_eng, 3)`. All Qwen2.5 sizes (0.5B–72B) share one tokenizer, so a table computed once is valid across the whole family — one more reason it is cached per *tokenizer*, not per experiment.

🧒 **The picture.** Imagine a translator who has memorized whole English phrases ("nice to meet you" = one flashcard) but for Telugu only knows the alphabet, so she has to spell every word out letter by letter. She isn't slower because Telugu is harder — she's slower because *her flashcard deck* has no Telugu on it. Give her a bigger deck (Qwen's 151K vocab) and she speeds up a lot. φ measures how many flashcards *this particular translator* burns on each language. Hire a new translator, remeasure — that's why the number lives in a file named after the translator.

## 1.2 The Context Window Wall — two failure regimes, not one

🎓 **The record.** A transformer trained with `max_position_embeddings = C` has only ever seen position indices in `[0, C)`. For the current default model, Qwen2.5-3B-Instruct, **C = 32,768** — *not* the 2,048 ceiling of the earlier TinyLlama phase. Two distinct things go wrong as sequences grow:

1. **In-window "lost-in-the-middle."** Even for sequences *within* `[0, C)`, attention is not uniform over depth. Empirically, models attend most strongly to the beginning and end of the context and least to the middle — a U-shaped recall curve. Evidence pushed into that trough is read worst. This regime is probed by the ordinary `run_eval.py` matrix and mapped explicitly by `scripts/calibrate_degradation.py`, which sweeps needle depth to locate the trough.
2. **Out-of-window drift.** When the raw sequence exceeds `C`, position indices land *outside* the trained distribution. RoPE still produces angles for them, but they are rotational phases the model was never optimized on — behavior degrades sharply. `scripts/run_eval.py --overflow` deliberately enters this regime by allowing raw tokens up to **1.5× C**.

Fertility couples these: a high-φ language reaches both the middle-trough depth *and* the out-of-window wall at a *lower English-equivalent content budget* than English does. That is the inequity FAPR targets.

🧒 **The picture.** The model's memory is a hallway with a fixed number of doors (32,768 of them). It hears best near the front door and the back door, and mumbles in the middle. Two things can go wrong: your important note is stuck in the mumbly middle (lost-in-the-middle), or you brought so much paper it spills past the last door into the parking lot, where the model has never stood before (out-of-window). Telugu, needing ~7× the paper for the same story, hits both problems way earlier than English.

## 1.3 The End-to-End Pipeline — one dataset, one packing rule, one patch

🎓 **The record.** The actual data flow, module by module:

```
facebook/belebele  ──(flores_passage field == FLORES-200)──┐
        │                                                    │
        ▼                                                    ▼
  fapr/fertility.py                                   fapr/harness.py
  compute_phi()  → φ table (cached JSON)              load_rows() + HaystackBuilder
        │                                                    │
        │                                          distractors chosen ONCE by
        │                                          English token budget, reused
        │                                          by `link` across languages
        ▼                                                    ▼
   phi_<model>.json  ───────────────────────────►  format_mcq() → prompt + gold letter
                                                             │
                          ┌──────────────────────────────────┤
                          ▼                                   ▼
              scripts/run_eval.py                     app/demo.py
              fapr_rope(model, φ) +                    generate_answer()
              score_letters() (single forward)        (interactive path)
                          │                                   │
                          └────────► fapr/rope_patch.py ◄─────┘
                                     (t' = t / φ inside RoPE)
```

- **One dataset dependency, not two.** Belebele's `flores_passage` field *is* FLORES-200 — the same parallel paragraphs serve both φ calibration (`fertility.py`) and the needle/distractor source (`harness.py`). Belebele is parquet-hosted and ungated, so there is a single ingestion path.
- **Parallel packing by English budget.** `HaystackBuilder` picks the distractor set **once**, measured in **English-equivalent tokens** (`self.eng_tokens`), then reuses the same passages — matched by `link` — in every language. Every language therefore carries *identical information*; only the raw token length differs, and it differs *only by fertility*. This is what isolates the positional variable from the representational one.
- **Scoring is a single forward pass.** `score_letters()` runs one forward with `use_cache=False`, reads next-token logits at the answer position, and compares A/B/C/D letter logits. No free-text generation, no parsing noise — and no KV cache allocated, which is exactly the memory that made long-context eval fragile on 16 GB.
- **The demo path** uses `generate_answer()` instead (real text generation), because a live audience wants to *see* an answer, not a logit.

🧒 **The picture.** One box of parallel Lego instructions (Belebele/FLORES) does two jobs: it tells us how bulky each language is (φ), and it supplies both the "right page" (needle) and the "wrong pages" (distractors). We decide *how many wrong pages* to stuff in by counting them in English, then hand every language the exact same pages — so the only thing that changes between English and Telugu is how much physical space those identical pages take. That's a fair experiment: same content, different bulk.

> **⚠️ Correction from the original brief — say this plainly if asked.** The early ideation (see `FAPR Full Briefing.md` and the context digest) specified **MIRACL** (Wikipedia) as the distractor corpus. The shipped code does **not** use MIRACL — distractors are drawn from **Belebele/FLORES-200 itself**. This is a deliberate **design simplification**: it collapses two dataset dependencies into one, and — more importantly — it guarantees the distractors are *also* perfectly parallel across languages, which a separate MIRACL retrieval could not promise. Present it as a strengthening of the controlled-experiment design, not a shortcut.

---

# CHAPTER 2 — The Mathematics of the FAPR Intervention

## 2.1 RoPE and absolute position indices

🎓 **The record.** Rotary Position Embedding injects position by *rotating* each query/key vector by an angle proportional to its absolute position. For position `t` and frequency `ω_i = base^(−2i/d)`, the rotation angle on dimension pair `i` is `θ = t · ω_i`. Because attention scores depend on the *difference* of rotation angles between a query at position `m` and a key at position `n`, RoPE encodes **relative** position `(m − n)` through absolute indices. The position index `t` is an ordinary integer counter: token 0, token 1, …, token L−1.

🧒 **The picture.** Every word gets put on a clock face, and its position is *how far the hand has turned*. Word 0 at 12 o'clock, word 1 a tick later, and so on. The model judges "how far apart are these two words" by the angle between their clock hands. Position is just "how much did the hand rotate."

## 2.2 The core compression — `t' = t / φ`, inside the rotary module

🎓 **The record.** FAPR's entire intervention is one division. In `fapr/rope_patch.py::fapr_rope`, every position tensor flowing into the model's rotary embedding module is divided by φ before the angle is computed:

$$t' = \frac{t}{\phi}$$

Concretely, the context manager wraps each rotary module's `forward`:

```python
def wrapper(x, position_ids=None, *args, **kwargs):
    if position_ids is not None:
        position_ids = position_ids.to(torch.float32) / scale   # t → t/φ
    return orig_fwd(x, position_ids, *args, **kwargs)
```

**Why the patch lives *inside* the rotary module — the "Attempt-3 trap."** HuggingFace `generate()` does **not** let you supply positions from outside. On every autoregressive step it *recomputes* `position_ids` internally via `prepare_inputs_for_generation` / `cache_position`, silently discarding any custom `position_ids` tensor you pass to `generate()`. The Colab phase (Attempt 3) lost days to this: FAPR appeared to run but did nothing, because the first decode step overwrote the scaled positions. The fix is to intercept **one layer lower** — at the rotary module's `forward` — so that *whatever* positions `generate()` computes, prefill and every decode step alike, they pass through the `/φ` division on the way in. Phase continuity is automatic: a prefill of length L gets `[0…L−1]/φ`, and decode step k gets `(L+k)/φ`.

Two engineering details that make this robust and honest:
- **Model-agnostic discovery.** `find_rotary_modules()` locates modules by class-name pattern `*RotaryEmbedding` (matches Llama/Qwen/Mistral/Gemma), deduping by object identity — so it works whether transformers exposes one shared rotary module (≥4.46) or one per layer (older).
- **Fully reversible, no nesting.** The patch is an *instance-level* `forward` override; teardown just deletes the instance attribute so the class method resurfaces. A `_fapr_patched` guard refuses to nest. Nothing is mutated permanently.

🧒 **The picture.** Telugu turned the clock hand 7× too far because it used 7× the words. FAPR says: before you read the clock, divide the rotation by 7. Now Telugu's 70-tick story lands on the same 10-tick arc English uses — same shape, same spacing, just un-inflated. And the trick is we don't shout the corrected positions from outside the model (it ignores us — the "Attempt-3 trap"); we sneak inside the clock mechanism itself and divide there, so *every* time the model recomputes a position, our division is already baked in.

## 2.3 Why fractional positions are legal

🎓 **The record.** There is **no discrete position lookup table** in RoPE. The angle is computed as `position · inv_freq` in floating point. Nothing requires `position` to be an integer. Therefore `t/φ` — generally non-integer (e.g. 70 / 6.947 ≈ 10.08) — is a perfectly valid input: it simply names a rotational phase *between* two integer positions. This is the same mathematical move as linear position interpolation (see §2.5); FAPR just chooses the divisor *per language, by fertility*. A high-fertility passage's many tokens are thereby compressed onto the same continuous phase range an English passage of equivalent content occupies.

🧒 **The picture.** The clock hand can point *between* the tick marks — 10.08 o'clock is a real place on a smooth dial. So squeezing 70 words onto a 10-tick arc just means the words sit at fractional positions (0, 0.14, 0.29, …). The dial was always continuous; we're finally using that.

## 2.4 The Robustness Paradox — the honesty ledger's centerpiece

🎓 **The record — do not skip this; the panel will probe it.** FAPR's own Attempt-2 finding is that applying a language's **full** fertility φ **when the model is nowhere near its context limit actively hurts accuracy.** In that experiment, Qwen2.5-1.5B read a 12K-token Telugu baseline at ~100% — it was *not* degraded, because the model is robust in-window. Applying FAPR with φ≈3.0 then compressed *healthy, functional* local attention geometry into sub-integer spacing the model was never stressed on, and accuracy collapsed toward 0%. **FAPR only helps in a regime where a baseline trough actually exists.**

State the current implementation plainly:

> `fapr/rope_patch.py` and the evaluation path in `scripts/run_eval.py` apply the **raw** φ from the lookup table, with **no adaptive gating**. The mitigation that ships today is **procedural, not mathematical**: `scripts/calibrate_degradation.py` sweeps English-only budgets × depths on the *baseline* model first to find the context regime where the model genuinely degrades, and only then is `run_eval.py --content-tokens` set at/above that onset — so the matrix runs *only where FAPR has real headroom to rescue.*

**The proposed refinement (not shipped in the science path).** An adaptive per-request gate would make this automatic instead of a manual pre-run step:

$$\phi_{\text{eff}} = \max\!\left(1.0,\; \min\!\left(\phi_{\text{raw}},\; \frac{\text{len\_actual}}{\text{native\_ctx}}\right)\right)$$

Read it as: *use no compression until the input actually starts filling the window; never expand (floor 1.0); never exceed the true fertility.*

> **🔎 Be precise about where this gate exists (this is the one place the code is ahead of the prompt).** The formula above **is implemented** — in `app/demo.py::gate_phi` — and the interactive playground applies it live (FAPR and random-φ modes both scale by `phi_eff`, with a blue "gating active" note when it fires). What it is **not** in is (a) `fapr/rope_patch.py`, which is a pure `/φ` primitive with no gate, and (b) `scripts/run_eval.py`, the scientific matrix, which uses **raw** φ on purpose so the measurement is not confounded by an untested heuristic. So the honest one-liner for the panel is: *"The gate is a UI demonstration of the future auto-tuning idea; the paper numbers come from raw φ in a regime we first proved is degraded by hand. Promoting the gate from the demo into the eval harness — with its own ablation — is the immediate next step."* Never let the panel think the paper's accuracy figures already include adaptive gating; they do not.

🧒 **The picture.** FAPR is a corrective lens. If your eyesight is fine (model comfortable, short context), forcing strong prescription glasses on you makes you *dizzy* — that's the paradox. The glasses only help once things are genuinely blurry (context full, evidence at the edge). Right now we check "is it blurry yet?" *by hand* (the calibrate step) before putting the glasses on. The auto-focusing version of the glasses (the φ_eff formula) is built into the *demo* so you can watch it dial itself in — but the formal experiment deliberately uses the fixed prescription, in a situation we already confirmed is blurry, so nobody can argue the auto-focus was secretly doing the work.

## 2.5 Footnote: FAPR with constant φ *is* linear position interpolation

🎓 **The record.** `scripts/export_prepatched.py` bakes a single fixed φ into a checkpoint's config as `rope_scaling = {"rope_type": "linear", "factor": φ}`. Stock linear interpolation divides positions by `factor` inside the rotary module — **mathematically identical to constant-φ FAPR**. This gives the paper a clean framing: *FAPR = per-language linear interpolation, with the interpolation factor chosen by tokenizer fertility.* The novelty is not the operator; it is *making the operator heterogeneous across languages and conditioning it on φ.*

---

# CHAPTER 3 — Before vs. After: Demystifying the Generation Behavior

## 3.1 The baseline failure modes actually observed in this project

### (a) The Attempt-1 OOM

🎓 **The record.** Attempt 1 ran **Qwen2.5-7B** with **eager attention** and a 12K-token Telugu document on a 16 GB T4. Result: `OutOfMemoryError: tried to allocate 6.81 GiB`. Cause: the attention matrix and KV cache scale **quadratically / linearly with sequence length**, and a 7B model at long context blew the VRAM. **The fixes, all in `fapr/model_loader.py`:**
- smaller model (7B → **3B**);
- **`attn_implementation="sdpa"`** everywhere (memory-efficient fused kernels; flash-attn-2 is unavailable on both T4 Turing and Mac anyway);
- **4-bit NF4** quantization on CUDA (`BitsAndBytesConfig`, double-quant) → the 3B loads in ~3.5 GB. On Apple Silicon: bf16/fp16 (~6.2 GB), with optional `quanto` int8 (~3.4 GB). `bitsandbytes` does **not** support MPS — the loader never tries.

🧒 **The picture.** A 7B model reading a long document is like renting a moving truck that turns out to be too small — the memory needed grows fast as the document gets longer, and it burst. We swapped to a smaller truck (3B), a smarter loading method that stacks boxes efficiently (sdpa), and compressed the boxes (4-bit). Now it fits.

### (b) The Attempt-3 "blank output" bug — four stacked causes

🎓 **The record.** A chat-tuned model fed a raw completion prompt would emit **empty strings**. The Colab version had **four causes stacked**, each closed by a named guard in `fapr/generation.py::generate_answer`:

| # | Cause | Guard in `generate_answer` |
|---|---|---|
| 1 | Chat model on a raw prompt emits EOS immediately ("this conversation is empty/over") | **Always** `tokenizer.apply_chat_template(...)` |
| 2 | Missing `attention_mask` with `pad_token == eos_token` → `generate()` mis-guesses the mask and can hide real content | **Always** pass the tokenizer's explicit `attention_mask` (via `**enc`) |
| 3 | EOS legally sampled as the **very first** token → "" after `skip_special_tokens` | `min_new_tokens=2` makes that impossible |
| 4 | Decoding the whole sequence + stripping specials can yield "" when only specials were produced | Decode **only the new tokens** (`out[0][n_input:]`), with a **labeled fallback** string instead of silently returning "" |

Note the design also *never passes `position_ids` into `generate()`* — for the Attempt-3-trap reason in §2.2. Determinism: `sample=False` (default) is the byte-identical greedy path used by the eval; `sample=True` is demo-only.

🧒 **The picture.** The model kept handing back blank pages for four compounding reasons: it thought the conversation was already over (no chat template), it couldn't tell which words were real (bad mask), it was allowed to say "done!" before writing anything (EOS as first token), and even when it wrote, we were accidentally erasing it while cleaning up (decoding the whole thing). Four locks, four keys — now it always writes at least a couple of real words, and if it truly writes nothing we *say so* instead of showing an empty box.

## 3.2 The FAPR Rescue — what compression buys, once you're genuinely degraded

🎓 **The record.** Once the model is in a genuinely degraded regime — located by `calibrate_degradation.py`, where deep-needle accuracy has clearly dropped below shallow-needle accuracy — compressing position ids `t → t/φ` pulls the high-fertility passage's evidence *back off the window edge* and into positional territory the model reads well. It restores the ability to attend to evidence that fertility inflation had shoved toward (or past) the edge, and it does so:
- **with no fine-tuning and no extra VRAM** — FAPR has **no weights and nothing to train**;
- **at microsecond cost** — the patch is re-applied at load time and is *not* serialized by `save_pretrained()` (see README's *"what persists between runs"* table).

The persistence table is worth memorizing, because it answers "don't you have to redo everything each run?":

| Artifact | Persists? | Where |
|---|---|---|
| φ lookup table | ✅ | `data/phi_<model>.json`, auto-loaded |
| Model weights | ✅ | HF cache (`~/.cache/huggingface`) |
| Eval results | ✅ | `data/results/*.json` |
| The RoPE monkey-patch | ❌ **by design** | re-applied at load; microseconds |
| A "trained-in" FAPR | ❌ **impossible** | no weights exist to train |

🧒 **The picture.** The evidence had been pushed to the very back of the mumbly hallway (or out into the parking lot). FAPR slides it back to a spot where the model hears clearly. And because FAPR is just *arithmetic on position numbers* — not new knowledge stuffed into the model — there's nothing to save, nothing to retrain, and it costs essentially nothing to switch on. Cold-start next time = load the cached weights, read one small JSON, re-apply the one-line patch. Nothing is re-derived.

---

# CHAPTER 4 — The Viva-Voce Defense (Anticipating Reviewer Questions)

### Q1. *"How do you know your baseline model is a strong enough reader that a 'rescue' claim is even meaningful?"*

🎓 The `--ceiling` gate in `scripts/run_eval.py` is a **zero-distractor, calibrated-accuracy honesty check**: it packs no haystack (budget 0), scores the questions, and if calibrated accuracy is **< 50%** it prints a loud `*** CEILING WARNING (drawback #1)` and tells you not to trust any FAPR-vs-baseline result on a reader that weak. This is drawback #1 in the ledger — the exact failure of the TinyLlama phase (a 1.1B model too weak to read at all, so *any* "improvement" was noise). We run `--ceiling` **before** the matrix; a rescue claim on top of a non-reader is not a claim.

🧒 You can't prove a lifeguard saved a swimmer if the swimmer could already swim fine *and* you never checked they could swim at all. First we prove the model can read (clears 50% with no distractions). Only then does "FAPR rescued it" mean anything.

### Q2. *"Isn't your multiple-choice scoring vulnerable to letter-position bias?"*

🎓 Two independent fixes (drawback #2), both in `fapr/harness.py`:
- **Structural:** `format_mcq()` shuffles option order per question with a **seeded RNG**, so the gold answer is **uniformly distributed over A–D** across the dataset. A model that always guesses "C" now scores at chance, not inflated.
- **Measurement:** a content-free **`null_prompt()`** (every field replaced by "N/A") measures the model's raw letter prior; that prior is **subtracted from the letter logits** — *contextual calibration*. Both raw and calibrated accuracy are reported, and every summary prints the **`pred_histogram`** of predicted letters, so a constant-answer collapse (e.g. all-B) is visible at a glance.

🧒 If a student always circles "C," a test where the answer is *usually* C makes them look smart. So (1) we shuffle which slot the right answer sits in, and (2) we separately ask the model a totally blank question to see which letter it *likes* by habit, then subtract that bias out. And we print a tally of its guesses — if it just spammed one letter, you'd see it instantly.

### Q3. *"How do you know FAPR's benefit isn't just generic position perturbation — 'any jiggle helps'?"*

🎓 The **random-φ ablation** (`fapr/harness.py::random_phi`, condition `"random"` in the eval and a third radio button in the demo). It draws a scale that is **magnitude-matched but deliberately fertility-mismatched**: log-uniform on `[1.3, max(3, 2×table_max)]`, **rejected if within 15% of the true φ**. The logic is a clean dissociation:
- If the gain were "any position perturbation helps," random-φ would match FAPR.
- If the gain is **fertility-specific**, FAPR must beat **both** baseline **and** random-φ.

FAPR beating baseline *but not* random-φ would sink the fertility-specificity claim — and the experiment is built to expose exactly that.

🧒 Maybe the glasses help just because they're *some* correction, not because they're *your* prescription. So we also try a random wrong prescription of similar strength. If your real prescription wins and the random one doesn't, the benefit is genuinely *about your eyes* (fertility) — not about wearing any old glasses.

### Q4. *"Why default to a 3B model rather than a larger 7–8B for a conference submission?"*

🎓 Three grounded reasons:
1. **Hardware ceiling.** 7B with eager attention OOM'd on the free-tier T4 (Attempt 1). 3B + sdpa + 4-bit fits comfortably on both T4 and a 16 GB Mac — reproducible on hardware a reviewer actually has.
2. **Throughput of the matrix.** Scoring is a **single forward pass** per item (`score_letters`, `use_cache=False`), but the matrix is large: 100+ questions × multiple languages × multiple depths × 3 conditions. Long-context forwards are tens of seconds each on MPS; 3B keeps the full matrix tractable.
3. **The variable under test is context-length/fertility, not parameter scale.** Isolating a *positional* effect does not require frontier model size — it requires a reader strong enough to clear the ceiling gate and a regime where fertility stresses it. The `--model` flag keeps 7B available for a scale-up; nothing in the method is 3B-specific.

Small-N honesty is already mitigated (drawback #4): **`bootstrap_ci`** reports **95% CIs alongside every accuracy number** (default N=100), so a wide interval is *shown, not hidden*.

🧒 We picked a car that fits in the garage everyone owns (a free Colab T4 / a laptop), because the thing we're testing is *how you pack the trunk*, not *how big the engine is*. A bigger engine wouldn't teach us anything about trunk-packing, and it wouldn't fit in the garage. And when we report scores, we always show the error bars — with 100 questions they're wide, and we say so out loud.

### Q5. *"What is not yet done — and what would you say if asked directly?"*

🎓 Say it plainly; the honest "designed but not yet run" answer is a **strength** here (it's the README's *honesty ledger*), not a weakness to hide:

- **Sentinel / anchor sentences — the "A" in FAPR — are DESIGNED BUT NOT IMPLEMENTED.** The shipped codebase does **position scaling only**. The bilingual anchor-sentence mechanism (identical landmark sentences at fixed depths) exists in the ideation and the name; there is no code for it. Future work.
- **Only 3 of 8 registered languages have been run end-to-end.** `fapr/fertility.py::LANGS` registers eight (english, telugu, hindi, yoruba, bengali, swahili, thai, finnish); `DEFAULT_LANGS` and both cached φ tables cover **english / telugu / hindi** only. Adding one is *one flag* (`--langs … yoruba`) — but it **has not been run**, and I will not present yoruba/bengali/swahili/thai/finnish numbers as if they exist (drawback #3).
- **Adaptive φ-gating is a proposed refinement, not the science path.** As in §2.4: the `φ_eff` formula is live in `app/demo.py::gate_phi` as a UI demonstration, but the eval matrix uses **raw φ** in a regime pre-verified as degraded. Promoting the gate into `run_eval.py` — with its own ablation — is named future work.

🧒 If a panelist asks "did you build everything in the poster?", the winning answer is the honest inventory: the position-squeezing half is real and tested on 3 languages; the "landmark signs" half (the A in FAPR) is designed but not built yet; the self-tuning glasses live in the demo but the formal numbers use the fixed prescription; and five more languages are one command away but not yet run. Volunteering the gaps is what makes the parts that *are* done believable.

---

## Appendix A — The five-command workflow (and *why the order matters*)

```bash
python scripts/compute_phi.py                 # φ table (tokenizer only, ~2 min) — answers the 10.67× question
python scripts/run_eval.py --ceiling --n 40   # drawback-#1 gate: is this model actually a reader?
python scripts/calibrate_degradation.py       # find the regime where it degrades (robustness paradox)
python scripts/run_eval.py --quick            # smoke matrix (baseline / fapr / random-φ)
python app/demo.py                            # live UI; ablation is a deliberate third radio button
```

The order encodes the scientific argument: **measure fertility → prove the reader is real → find where it breaks → only then rescue → show it live.** Skipping the middle two steps is exactly how Attempts 1–3 failed.

## Appendix B — One-line rebuttals to keep in your pocket

| If they say… | You say… |
|---|---|
| "Telugu is just 10× harder for models." | "That 10.67× is the *Llama-2 tokenizer* byte-falling-back Telugu. Qwen's multilingual vocab measures 6.95×. φ is a (language, tokenizer) property — here's both tables." |
| "You just passed custom positions to generate()." | "That's the Attempt-3 trap — `generate()` recomputes positions every step and discards yours. We patch *inside* the rotary module, so every recomputed step still divides by φ." |
| "Fractional positions can't be valid." | "There's no position lookup table in RoPE — the angle is `position · inv_freq` in float. 70/6.947 is just a phase between two ticks." |
| "Compressing positions must always help." | "It hurt in Attempt 2 — full φ on an un-stressed model distorts healthy geometry (the robustness paradox). We only apply it in a regime we first prove is degraded." |
| "Maybe any position noise helps." | "Random-φ ablation, magnitude-matched but fertility-mismatched. FAPR must beat *both* baseline and random-φ to claim fertility-specificity." |
| "Where are the other languages / the anchors / the auto-gate?" | "Registered/designed but not yet run/built — it's in our honesty ledger. Position-scaling on 3 languages is what's tested today." |

---

*End of blueprint. Nothing above claims a feature the codebase does not implement; the "designed but not yet run" items are labeled as such by design.*
