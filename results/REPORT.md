# Decision models on fresh data: results

Status: complete for the pre-registered design (protocol v1 with changelog) plus a clearly separated post hoc block. Numbers come from `results/summary.json`; raw model outputs are in `results/raw/`. Dates: data collected 2026-09-27, runs 2026-09-27 and 2026-09-28.

## Summary

1. **On fresh data, Jev 1.13 cannot be distinguished from small general LLMs with structured output at this sample size.** On 400 arXiv papers from September 2026, Jev reaches 86.3%; GPT-6 Luna 88.7%, Gemini 3.5 Flash-Lite 88.0%, Claude Haiku 4.5 87.3%, Claude Sonnet 5 89.7%. All four LLM point estimates are above Jev (+1.0 to +3.5 points); none of the differences is significant after Holm correction (Sonnet: +3.5 [+0.8, +6.5] unadjusted, p_holm 0.12). This is not evidence of equivalence.
2. **Jev is clearly ahead of every open zero-shot or decision model in the configuration tested here**, by 5.8 points (Eikos-4B, post hoc) to 42.8 points (Laya). GLiNER2.5-Decide trails by 12.8 points. Results depend on each model's interface and prompt: the same Qwen3.5-4B scores 53.2% with our option-letter readout and 77.5% with SemIf's vendor prompt. For the configurations that fit on a 16 GB laptop, the claim that open models have caught up does not hold on this data.
3. **Supervised baselines trained on 2024 arXiv labels come close or match.** TF-IDF + logistic regression reaches 82.0% (83.0% with class weighting); logistic regression on Qwen3-Embedding-8B embeddings reaches 86.3%, equal to Jev, with better calibration (ECE 0.029 vs 0.068), the lowest AURC of all systems and about a twelfth of the cost (post hoc). The condition is labelled data.
4. **Jev's advantages are real but narrower than marketed:** cheapest per decision (USD 0.035 per 1,000 vs 0.065 for GPT-6 Luna, 0.89 for Haiku), fastest API (p50 0.47 s vs 0.77 to 2.14 s), stable to label order and wording (97 to 99% identical answers), good at "none of these" (95% detected, 1% false). It is 1.6 to 4.5 times faster than small LLMs, not 200 times, and about 2 times cheaper than the cheapest small LLM.
5. **Jev's probabilities are fragile.** 51.5% of its choice confidences are exactly 1.0, so the most confident half cannot be ranked. Asking the same decision as one yes/no question per label raises its ECE from 0.068 to 0.251; the yes-probabilities over all labels sum to a median of 1.25.
6. **The rationale-first prompt costs accuracy.** In this prompt condition (rationale field first, instruction to quote the input) accuracy drops by 4.5 to 7 points for all three LLMs and cost rises 1.5 to 1.8 times. This measures that prompt condition, not rationales in general. Luna and Gemini quote the input verbatim in 92 to 97% of their quotations; Haiku rarely quotes.
7. **Fixed-window encoders lose information in long inputs**; Jev and Luna found a single decisive sentence at every length up to 24,000 tokens.

## Setup

### Data (all text dated 2026-09-01 or later)

| Source | Task | Items | Label origin |
|---|---|---|---|
| S1 arXiv | primary category among 8 CS categories | 400 (50 per class) | author's choice, moderated before publication; verified on 60 abstract pages (60/60) |
| S2 Federal Register | issuing agency among 6 | 358 | Federal Register metadata; agency names masked (0.26% residual leakage) |
| P1 probes | probability of a stated random event | 127 | by construction |
| P2 probes | routing of minimal pairs | 74 (37 pairs) | by construction |
| P3 probes | one decisive sentence in 500 to 24,000 tokens of filler | 114 | by construction |
| P4 probes | non-CS papers asked with the S1 labels | 100 | by construction ("none") |

Descriptive statistics and quality checks: `results/data_quality.json` (no duplicates, no overlap between pilot, main, training and probes; 11.8% of S1 items cross-listed within the label set). Label noise: S1 labels are author choices among overlapping categories; all five API systems agree on a different label for 16 of 400 papers (see Bounds).

### Systems

Pre-registered: Jev 1.13 (OpenRouter), GPT-6 Luna, Gemini 3.5 Flash-Lite, Claude Haiku 4.5, Claude Sonnet 5 (S1 only), GLiNER2.5-Decide and -1B, Laya, DeBERTa-v3 NLI zero-shot, GLiClass v3, Qwen3.5-4B read through option-letter logprobs, TF-IDF + logistic regression trained on 2024 arXiv. Post hoc (after the LangWatch comparison): Eikos-4B, SemIf 4B, Kev-0.8B. Exact model IDs and revisions: `notes/local-models.md`.

All systems receive the same label names and descriptions, but each through its own interface: LLMs via structured output with a label enum and a verbalized confidence; Jev and the decision models via their typed choice APIs; DeBERTa NLI as entailment hypotheses ("This text is about {description}."); Qwen letter logprobs via lettered options. GLiNER's parenthesis rewrite did not trigger (no parentheses in any label set).

## Results

### S1 arXiv, accuracy and selective prediction

| System | Accuracy [95% CI] | Δ vs Jev [95% CI] | Holm p | Error at 80% coverage | AURC | ECE | USD / 1,000 |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 5 | 89.7% [86.4, 92.4] | +3.5 [+0.8, +6.5] | 0.12 | 6.8% | 0.051 | 0.064 | 2.480 |
| GPT-6 Luna | 88.7% [85.3, 91.5] | +2.5 [−0.5, +5.5] | 0.43 | 8.5% | 0.074 | 0.055 | 0.065 |
| Gemini 3.5 Flash-Lite | 88.0% [84.5, 90.8] | +1.8 [−1.0, +4.8] | 0.62 | 10.0% | 0.067 | 0.062 | 0.205 |
| Claude Haiku 4.5 | 87.3% [83.6, 90.2] | +1.0 [−1.5, +3.5] | 0.62 | 10.3% | 0.077 | 0.062 | 0.888 |
| **Jev 1.13** | **86.3% [82.5, 89.3]** | | | **8.6%** | 0.067 | 0.068 | **0.035** |
| TF-IDF + LR (2024 labels, 117 to 400 per class) | 82.0% [77.9, 85.5] | −4.3 [−8.3, −0.3] | 0.23 | 11.6% | 0.071 | 0.083 | local |
| GLiNER2.5-Decide | 73.5% [69.0, 77.6] | −12.8 | <0.001 | 21.6% | 0.117 | 0.088 | local |
| GLiNER2.5-Decide-1B | 69.8% [65.1, 74.1] | −16.5 | <0.001 | 25.9% | 0.182 | 0.152 | local |
| DeBERTa-v3 NLI | 66.7% [62.0, 71.2] | −19.5 | <0.001 | 26.9% | 0.159 | 0.085 | local |
| GLiClass v3 | 64.2% [59.4, 68.8] | −22.0 | <0.001 | 26.9% | 0.178 | 0.074 | local |
| Qwen3.5-4B letter logprobs | 53.2% [48.4, 58.1] | −33.0 | <0.001 | 39.1% | 0.266 | 0.108 | local |
| Laya | 43.5% [38.7, 48.4] | −42.8 | <0.001 | 50.0% | 0.343 | 0.109 | local |

Figures: `figures/s1_risk_coverage.png`, `figures/s1_reliability.png`, `figures/s1_cost_accuracy.png`.

Jev's risk–coverage curve is flat up to 51.5% coverage because 51.5% of its confidences are exactly 1.0; the expected error inside that block is constant under random tie-breaking.

**Unambiguous subset** (349 papers not cross-listed within the label set): Sonnet 91.4%, Luna 90.8%, Gemini 89.4%, Haiku 88.8%, Jev 87.7%, TF-IDF 84.2%, Eikos 82.5% (post hoc), SemIf 79.7% (post hoc), GLiNER 74.2%, GLiNER-1B 71.6%, NLI 68.8%, Kev-0.8B 67.1% (post hoc), GLiClass 65.6%, Qwen 54.7%, Laya 42.7%. The ordering is unchanged.

### Stability to wording and question form (S1, agreement with the system's own choice run)

| System | Reversed order | Paraphrase 1 | Paraphrase 2 (names only) | One yes/no per label |
|---|---|---|---|---|
| Jev | 99.0% | 97.5% | 98.0% | 97.0% |
| DeBERTa NLI | | | | 99.5% |
| Eikos-4B (post hoc) | | | | 93.0% |
| GLiClass | | | | 69.3% |
| GLiNER | | | | 66.3% |
| Qwen letter logprobs | | | | 52.5% |
| GLiNER-1B | | | | 50.0% |
| Laya | | | | 43.5% |

Accuracy under paraphrase moves by up to 16.5 points for open models (e.g. GLiNER 73.5% → 60.3% with names only, NLI 66.8% → 50.3%); Jev stays between 85.3% and 88.0%. The question form mainly changes Jev's calibration: ECE 0.068 (choice) vs 0.251 (yes/no).

### Rationale before the label (S1)

| System | Accuracy label only → with rationale | Paired Δ [95% CI] | Holm p | Cost ratio | Output tokens | Quotations: items with one / verbatim |
|---|---|---|---|---|---|---|
| GPT-6 Luna | 88.7% → 81.8% | −7.0 [−10.8, −3.5] | 0.001 | ×1.47 | 19 → 74 | 99% / 97% |
| Gemini 3.5 Flash-Lite | 88.0% → 83.5% | −4.5 [−7.8, −1.5] | 0.016 | ×1.83 | 22 → 87 | 98% / 92% |
| Claude Haiku 4.5 | 87.3% → 82.8% | −4.5 [−7.8, −1.3] | 0.016 | ×1.62 | 19 → 119 | 16% / 76% |

Paired within each system (same items, label only vs rationale first), exact McNemar with Holm correction across the three systems. The rationale run also raised the output cap from 150 to 600 tokens; no rationale run was truncated at that cap, so the cap does not explain the drop.

A rationale is a review aid, not proof of the decision path; the pilot review found fluent rationales for wrong labels. Its usefulness to a human reviewer was not measured.

### No fitting label (P4) and false "none" (S1)

| System | "none" chosen on P4 (nothing fits) | "none" chosen on S1 (a label fits) | AUROC, top confidence S1 vs P4 (post hoc) |
|---|---|---|---|
| Jev | 95% | 1.0% | 0.900 |
| Gemini 3.5 Flash-Lite | 87% | 1.3% | 0.990 |
| GPT-6 Luna | 86% | 0.5% | 0.624 |
| Claude Haiku 4.5 | 65% | 0.5% | 0.985 |
| DeBERTa NLI | 94% | 22.3% | 0.898 |
| GLiNER-1B | 92% | 36.8% | 0.818 |
| GLiNER | 25% | 0.0% | 0.915 |
| Laya | 13% | 1.0% | 0.619 |
| GLiClass | 100% | 99.8% | 0.788 |
| Qwen letter logprobs | 67% | 3.8% | 0.849 |
| Eikos-4B / SemIf / Kev-0.8B (post hoc) | 95% / 97% / 86% | 0.3% / 3.3% / 2.0% | 0.901 / 0.940 / 0.919 |

With an explicit "none" option, Jev combines the highest detection among pre-registered systems with a low false rate. Without it, GPT-6 Luna gives 53% of its forced (always wrong) P4 answers a confidence of 0.9 or more (Jev 11%, Gemini 1%, Haiku 0%), which is why its confidence barely separates fitting from non-fitting inputs. GLiClass effectively always answers "none" when offered. LLM false-none rates on S1 were added post hoc (protocol changelog).

### Long input (P3), accuracy by length; refused inputs count as errors

| System | 500 | 2k | 8k | 24k | Note |
|---|---|---|---|---|---|
| Jev | 1.00 | 1.00 | 1.00 | 1.00 | |
| GPT-6 Luna | 1.00 | 1.00 | 1.00 | 1.00 | |
| Gemini 3.5 Flash-Lite | 0.96 | 1.00 | 1.00 | 0.93 | misses 2 of 5 at 24k, 5% position |
| Claude Haiku 4.5 | 1.00 | 0.97 | 1.00 | | run up to 8k for cost |
| GLiNER chunked | 1.00 | 1.00 | 0.93 | 0.86 | vendor long-input API |
| Eikos-4B (post hoc) | 1.00 | 1.00 | 1.00 | 0.00 | 24k exceeds 16 GB memory |
| Kev-0.8B (post hoc) | 0.93 | 0.93 | 0.79 | 0.46 | |
| Qwen letter logprobs | 0.71 | 0.76 | 0.93 | 0.71 | |
| GLiClass | 0.96 | 0.76 | 0.66 | 0.50 | 1,024-token window |
| Laya | 0.75 | 0.59 | 0.55 | 0.46 | 512-token window |
| Laya long | 0.89 | 0.69 | 0.52 | 0.57 | says "yes" to most long negatives |
| DeBERTa NLI | 0.64 | 0.48 | 0.48 | 0.46 | 512-token window |
| GLiNER | 1.00 | 1.00 | 0.00 | 0.00 | refuses > 4,096 tokens (memory) |
| GLiNER-1B | 1.00 | 1.00 | 0.31 | 0.00 | refuses > 7,999 tokens |

The decisive sentence is stylistically foreign to the filler; P3 measures whether a salient sentence is found, not long-document understanding.

### Stated probabilities (P1) and minimal pairs (P2)

P1: LLMs return the exact ratio (MAE 0.000; one Luna outlier). Jev MAE 0.027 [0.024, 0.029], maximum error 0.08; the published fair-coin example (0.92) is not reproduced for explicitly stated probabilities. Encoder models and letter logprobs are unrelated to the stated probability (MAE 0.14 to 0.43), as expected for scores that are not event probabilities.

P2: ceiling. LLMs, GLiNER and Eikos 100%; Jev and Kev-0.8B 98.6%; GLiNER-1B 96%; GLiClass 88%; Laya 77% (pair consistency 54%).

### S2 Federal Register

Ceiling for API systems (99.2 to 99.7%, no significant differences). Local systems: NLI 96.7%, Eikos and SemIf 97.8%, Kev-0.8B 97.2%, GLiNER 93.0%, GLiClass and Qwen 88.0%, GLiNER-1B 84.4%, Laya 81.0%; every pre-registered local system is significantly below Jev.

### Post hoc block: additional open decision models (S1)

| System | Accuracy | Δ vs Jev [95% CI] | Holm p (own family) | Error at 80% coverage |
|---|---|---|---|---|
| Eikos-4B | 80.5% [76.3, 84.1] | −5.8 [−9.0, −2.8] | <0.001 | 12.8% |
| SemIf 4B | 77.5% [73.2, 81.3] | −8.8 [−12.5, −5.3] | <0.001 | 17.2% |
| Kev-0.8B | 64.5% [59.7, 69.0] | −21.8 [−26.8, −17.0] | <0.001 | 29.4% |

SemIf uses the same Qwen3.5-4B base as our letter-logprob run and reaches 77.5% instead of 53.2%: reading an LLM as a classifier depends strongly on the prompt. Kev-4B was dropped (about 50 s per item on the laptop); 27B models and Shisa DE-1 were out of scope.

### Post hoc: closing three gaps (stability of LLMs, batching, repeatability)

Added after the limitations review (protocol changelog, `results/gaps.json`).

**Stability, agreement with the system's own choice run (S1):** Jev 99.0% (reversed), 97.5% and 98.0% (paraphrases); GPT-6 Luna 94.0%, 95.3%, 95.3%; Gemini 92.3%, 92.3%, 93.0%. LLM accuracy stays between 85.8% and 89.0% across variants. Jev is the most stable of the three.

**Ten decisions per request (S1, items shuffled so each request mixes categories):**

| System | Accuracy single → batch of 10 | Paired Δ [95% CI] | USD / 1,000 single → batch | Latency per decision, batch (p50) |
|---|---|---|---|---|
| Jev | 86.3% → 84.8% | −1.5 [−3.8, +0.5], McNemar p 0.24 | 0.035 → 0.026 | 0.07 s (0.69 s per request) |
| GPT-6 Luna | 88.8% → 89.5% | +0.8 [−1.8, +3.3], McNemar p 0.69 | 0.065 → 0.048 | 0.23 s (2.29 s per request) |

Batching saves about a quarter of the cost for both; per decision Jev is then about 3.3 times faster and 1.8 times cheaper than Luna.

**Repeatability (98 S1 items re-run with the cache bypassed):** label agreement Jev 99.0%, Haiku 99.0%, Gemini 98.0%, Luna 94.9%. Mean absolute confidence change: Haiku 0.004, Jev 0.007 (max 0.06), Gemini 0.017, Luna 0.036 (max 0.48). Luna's accuracy on these items moved from 89.8% to 85.7% between runs: single-run differences of a few points between LLMs are within run-to-run variation.

### Post hoc: supervised baselines with modern embeddings (S1)

Same temporal split as the pre-registered TF-IDF baseline (train: 2,806 arXiv abstracts from 2024, test: the 400 S1 papers), logistic regression with C chosen by 5-fold cross-validation on training data only. Separate Holm family.

| System | Accuracy [95% CI] | Δ vs Jev [95% CI] | ECE | Error at 80% coverage | AURC | AUROC "none fits" | USD / 1,000 |
|---|---|---|---|---|---|---|---|
| Qwen3-Embedding-8B + LR, balanced | 86.3% [82.5, 89.3] | 0.0 [−3.3, +3.0] | 0.029 | 7.2% | **0.038** | 0.925 | 0.003 (embeddings) |
| Qwen3-Embedding-8B + LR | 85.5% [81.7, 88.6] | −0.8 [−4.0, +2.5] | 0.040 | 8.1% | 0.042 | | 0.003 |
| OpenAI text-embedding-3-large + LR, balanced | 85.3% [81.4, 88.4] | −1.0 [−4.3, +2.3] | 0.038 | 7.5% | 0.042 | 0.901 | 0.037 |
| TF-IDF + LR, balanced | 83.0% [79.0, 86.4] | −3.3 [−7.3, +0.8] | 0.096 | 11.6% | 0.069 | 0.942 | local |
| Jev 1.13 (reference) | 86.3% [82.5, 89.3] | | 0.068 | 8.6% | 0.067 | 0.900 | 0.035 |

With about 2,800 labelled examples, an off-the-shelf embedding model plus logistic regression matches Jev's accuracy, is better calibrated, has the lowest AURC of all 23 systems, and costs about a twelfth per decision (embedding only; the classifier runs in microseconds). None of the accuracy or error-at-80% differences to Jev is significant. Class weighting adds 0.8 to 1 point. The balanced TF-IDF model chose the largest C in its grid (100); a wider grid might improve it slightly. The condition is labelled data: here it came free from arXiv; in practice it has to be collected.

### Bounds on attainable accuracy (S1, exploratory)

At least one of the five API systems is right on 95.5% of papers; all five are wrong on 4.5%, and on 16 of those 18 they agree on the same other category. Majority vote reaches 90.0% (2 items with a 2–2 tie, broken by system order; the alternative rule gives 90.25%). Excluding the 16 consensus items: Jev 89.8%, Luna 92.5%, Gemini 91.7%, Haiku 90.9%, Sonnet 93.5%, TF-IDF 84.1%, Eikos 83.9%. This suggests, but does not measure, a practical ceiling below 100%: the label is the author's choice among overlapping categories, and the 16 consensus items were not independently adjudicated.

### Cost and latency

Isolated latency, p50 / p95 seconds (API: concurrency 1, 30 items, cache bypassed; local: Apple M1 Pro, batch 1, 50 items, nothing else loaded): TF-IDF 0.001; Laya 0.14 / 0.15; Kev-0.8B 0.21 / 0.26; GLiNER-1B 0.36 / 0.46; GLiClass 0.46 / 0.66; Jev 0.47 / 0.60; GLiNER 0.54 / 0.80; Gemini 0.77 / 1.06; Haiku 1.07 / 1.65; Luna 1.14 / 2.29; Qwen 1.66 / 2.16; SemIf 1.70 / 2.10; NLI 1.73 / 3.03; Sonnet 2.14 / 6.54; Eikos 2.69 / 4.71. Local numbers are laptop numbers, not GPU serving.

Total OpenRouter spend: USD 4.69 including pilot and all reruns; every request is in the request cache and reconciles with the account.

## Limitations

- **Freshness** is established only for disclosed cutoffs (Claude Haiku 4.5: training data to July 2025; GPT-6 Luna: knowledge cutoff 2026-05-18). Jev, Gemini and the open models do not disclose cutoffs; arXiv and Federal Register texts from September 2026 are nevertheless unlikely to be in any training set.
- **Label noise** in S1 (author choice among overlapping categories) and S2 (NAGPRA notices) caps accuracy; comparisons are paired on identical items.
- **Ceiling effects** in S2, P1 and P2 limit what these sources can separate.
- **Probe templates** were written by the orchestrating model (Claude). No model under test generated text, but Claude Haiku 4.5 and Sonnet 5 share the author's family.
- **Prompting** was fixed a priori with one wording per LLM; no tuning on a dev split. The Qwen letter-logprob result understates the model (SemIf control).
- **Verbalized LLM confidences** are coarse (8 to 28 distinct values) and not model probabilities; calibration metrics for LLMs describe these numbers.
- **Local runs** are limited by 16 GB unified memory (refusals and failures on long inputs; Kev-4B dropped). Latencies of the main local run were affected by memory pressure and are not used.
- **The post hoc block** (Eikos-4B, SemIf, Kev-0.8B; OOD AUROC; confidence informativeness; LLM false-none) was added after interim results and is exploratory.
- **Sample sizes**: 400 items give about ±3.5 points at 86% accuracy; differences below 3 points between API systems are not resolvable.

- **Single run per condition** in the main analysis. Re-running 98 items shows 1 to 5% of labels change between identical requests (most for GPT-6 Luna), so differences of a few points between LLMs are within run-to-run variation.
- **Task type:** both real sources are topic or origin classification. Decisions with rules, exceptions, arithmetic or several interacting questions were not tested. English only; 6 to 8 labels (high-cardinality label sets not tested); a snapshot of September 2026.
- **Global multiplicity:** Holm correction is applied within each source and family, not across all sources, metrics and variants.
- **Reviewers:** the design and results were reviewed twice by another language model (Codex), not by a human domain expert. The article's thesis was formulated before the study; pre-registration and the changelog mitigate but do not remove this.
- **Latency** was measured from Germany through OpenRouter in one time window (30 to 100 items); costs are OpenRouter list prices, local hardware cost is not included.

## Reproducibility

Protocol and all deviations: `notes/protocol.md`. Pilot and incident reviews: `notes/pilot-review.md`. Model details: `notes/local-models.md`. Data builders: `src/dma/data/`. Runners: `src/dma/runners/`. Analysis: `uv run python -m dma.analysis.report` and `uv run python -m dma.analysis.figures`. Scripts: `scripts/run_main_api.sh`, `scripts/run_main_local.sh`, `scripts/run_followup.sh`, `scripts/run_latency.sh`.
