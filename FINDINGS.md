# Findings log

Chronological. Each entry states what was checked, how, and the result, including what we could not find or verify.

## 2026-09-27

- **Jev is callable directly on OpenRouter** as `typesafe/jev-1.13` (context 32,000, input text only, output modality "decisions"). It does not appear in the default `/api/v1/models` listing; only `typesafe/jev-router` does. Verified via `/api/v1/models/typesafe/jev-1.13/endpoints`.
- **GLiNER2.5-Decide has no hard context cap in its config** (`max_len: null`). The encoder is DeBERTa-v3-large with relative position attention, so longer inputs run, but the base model was pretrained at 512 positions. The GLiNER2 library ships a chunking API (`classify_long`, default chunk 384 words, overlap 64) that aggregates chunk logits. Whether quality holds on long inputs is an open question for experiment 3.
- **Fastino publishes three Decide checkpoints**: `GLiNER2.5-Decide` (DeBERTa-v3-large), `GLiNER2.5-multi-Decide` (mDeBERTa-v3-base, `max_len` 4096) and `GLiNER2.5-Decide-1B` (Ettin encoder, 7,999 positions, local attention 128).
- **Dataset strategy is open.** Public classification sets are contaminated for LLMs; GitHub issues rejected because label provenance is unverifiable; no hand labeling. Research on sealed test sets, synthetic data with labels by construction and outcome-labeled data is in progress.

## 2026-09-27, after source verification

- **Jev calibration depends on question form.** Zenodo 22935043 (Meng, UCLA, 2026-09-24): the same 872 SST-2 sentences give ECE 0.091 as a yes/no question and 0.020 as a choice question. Jev beats non-task-exposed DeBERTa-v3 zero-shot in all 16 configurations, loses to a task-exposed DeBERTa-v3 checkpoint on AG News (-2.30 pp) and is indistinguishable on SST-2. The paper reports ECE, accuracy, AURC and conformal coverage, no Brier score.
- **Jev has no abstention option** in its API contract (Choice, Score, Noul only). Independent audits report degraded accuracy and calibration in Russian and Korean on entailment tasks, not on intent classification.
- **JevK5 is not Jev.** It is an independent open-weight model (Qwen3.5-4B plus LoRA, `allebee/jevk5`). Fastino's leaderboard does not measure TypeSafe's model.
- **Many Jev alternatives appeared within two weeks:** Kev (Qwen family, 0.8B to 27B), Tev1 (Together AI), SemIf (logits from frozen Qwen3.5-4B), GLiClass (Knowledgator, since 2024), plus a community list. Not all will be benchmarked; selection is documented in the article plan.
- **Training cutoffs:** Claude Haiku 4.5 training data to July 2025; gpt-5.4-nano knowledge cutoff 2025-08-31; Gemini 3.1 Flash-Lite and Jev undisclosed. Data after 2026-09-01 is past all disclosed cutoffs.
- **Data strategy candidates:** labels by construction (templated probes, minimal pairs, calibration and position probes) plus real text with institutional labels set before any model saw it (arXiv `primary_category`, Federal Register agency and document type, CourtListener `suitNature`). arXiv volume since 2026-09-01: cs.CL 1,891, cs.LG 3,351, cs.AI 3,716.
- **Not found:** contamination evidence for Banking77 or CLINC150; official calibration numbers for GLiNER2.5-Decide; a TypeSafe response to the independent audits.

## 2026-09-27, main run: API systems (interim, local systems pending)

Numbers from `results/summary.json` (protocol v1 with changelog). 95% Wilson intervals; paired tests against Jev with Holm correction. Latency measured with 8 concurrent requests (Sonnet 4), p50 per request.

**S1 arXiv primary category (400 papers, 8 classes)**

| System | Accuracy | Error at 80% coverage | ECE | Cost per 1,000 | Latency p50 |
|---|---|---|---|---|---|
| Jev 1.13 | 86.3% [82.5, 89.3] | 8.6% | 0.064 | $0.035 | 0.50 s |
| GPT-6 Luna | 88.7% [85.3, 91.5] | 8.5% | 0.055 | $0.065 | 1.15 s |
| Gemini 3.5 Flash-Lite | 88.0% [84.5, 90.8] | 10.0% | 0.062 | $0.205 | 0.69 s |
| Claude Haiku 4.5 | 87.3% [83.6, 90.2] | 10.3% | 0.063 | $0.888 | 1.16 s |
| Claude Sonnet 5 | 89.7% [86.4, 92.4] | 6.8% | 0.079 | $2.480 | 1.93 s |

- No difference to Jev is significant after Holm correction (largest: Sonnet +3.5 pp, raw bootstrap CI [0.75, 6.5], McNemar p_holm 0.15).
- **Rationale first lowers accuracy**: Luna 88.7 → 81.8%, Gemini 88.0 → 83.5%, Haiku 87.3 → 82.8%; cost ×1.5 to ×1.8. Luna vs Jev with rationale: −4.5 pp [−8.0, −1.0].
- **Quote fidelity (exploratory)**: share of rationales with a quotation / share of quotations found verbatim in the input: Luna 99% / 97%, Gemini 98% / 92%, Haiku 16% / 76%.
- **Jev calibration depends on question form**: ECE 0.064 as one choice question, 0.251 as one yes/no question per label on the same items. Per-item sum of yes probabilities: median 1.25, 82% outside [0.9, 1.1].
- **Jev is stable to wording**: agreement with the base run 99% (reversed order), 97.5% and 98% (two paraphrases). 52% of Jev choice confidences are exactly 1.
- **Attainable accuracy is below 100% (exploratory)**: the label is the author's choice among overlapping categories. All five systems wrong on 4.5% of papers, 16 of these 18 with the same alternative label; at least one system right on 95.5%; majority vote 90.0%. Cross-listed papers (12.8%) have at least one system wrong in 47% of cases, others in 20%. Excluding the 16 consensus items: Jev 89.8%, Luna 92.5%, Gemini 91.7%, Haiku 90.9%, Sonnet 93.5%.

**S2 Federal Register agency (358 documents, 6 classes)**: ceiling for all systems (99.2 to 99.7%). With a "none" option Jev falls to 89.4%, mostly NAGPRA notices written by museums and universities, where "none" is defensible.

**P1 stated probability (127)**: LLMs return the exact ratio (MAE 0.000; Luna one outlier). Jev MAE 0.027 [0.024, 0.029], max 0.08. The published fair-coin 0.92 example is not reproduced for explicitly stated probabilities.

**P2 minimal pairs (74)**: 100% for all LLMs, Jev 98.6% (one error). Ceiling.

**P3 long input (114; Haiku ≤ 8,000 tokens, 83)**: Jev and Luna find the sentence at every length and position up to 24,000 tokens. Gemini misses it in 2 of 5 cases at 24,000 tokens, 5% position. No false positives. Limitation: the sentence is stylistically foreign to the filler.

**P4 no fitting label (100 non-CS papers)**: with a "none" option: Jev 95%, Gemini 87%, Luna 86%, Haiku 65%. Forced choice (all answers wrong by construction): share with confidence ≥ 0.9: Luna 53%, Jev 11%, Gemini 1%, Haiku 0%.

**Spend**: USD 3.84 in total (pilot 0.29). Remaining balance USD 1.96.

## 2026-09-28, main run: local systems (pre-registered block)

From `results/summary.json` (commit d35e0a1). Paired against Jev (choice) with Holm correction within the pre-registered family. Local latencies of this run are not used (see protocol changelog).

**S1 arXiv, all pre-registered systems (400 papers)**

| System | Accuracy [95% CI] | Δ vs Jev [95% CI] | Holm p | Error at 80% coverage | ECE |
|---|---|---|---|---|---|
| Claude Sonnet 5 | 89.7% [86.4, 92.4] | +3.5 [+0.8, +6.5] | 0.17 | 6.8% | 0.079 |
| GPT-6 Luna | 88.7% [85.3, 91.5] | +2.5 [−0.5, +5.5] | 0.57 | 8.5% | 0.055 |
| Gemini 3.5 Flash-Lite | 88.0% [84.5, 90.8] | +1.8 [−1.0, +4.5] | 0.62 | 10.0% | 0.062 |
| Claude Haiku 4.5 | 87.3% [83.6, 90.2] | +1.0 [−1.8, +3.5] | 0.62 | 10.3% | 0.063 |
| **Jev 1.13** | **86.3% [82.5, 89.3]** | | | 8.6% | 0.064 |
| TF-IDF + logistic regression (trained on 2024 arXiv) | 82.0% [77.9, 85.5] | −4.3 [−8.3, −0.3] | 0.34 | 11.6% | 0.083 |
| GLiNER2.5-Decide | 73.5% [69.0, 77.6] | −12.8 [−16.8, −8.8] | <0.001 | 21.6% | 0.088 |
| GLiNER2.5-Decide-1B | 69.8% [65.1, 74.1] | −16.5 [−20.8, −12.3] | <0.001 | 25.9% | 0.152 |
| DeBERTa-v3 NLI zero-shot | 66.7% [62.0, 71.2] | −19.5 [−24.0, −15.0] | <0.001 | 26.9% | 0.085 |
| GLiClass large v3 | 64.2% [59.4, 68.8] | −22.0 [−27.0, −17.0] | <0.001 | 26.9% | 0.074 |
| Qwen3.5-4B, letter logprobs | 53.2% [48.4, 58.1] | −33.0 [−38.0, −28.0] | <0.001 | 39.1% | 0.108 |
| Laya | 43.5% [38.7, 48.4] | −42.8 [−48.5, −37.0] | <0.001 | 50.0% | 0.109 |

- On fresh data Jev is clearly ahead of every open zero-shot decision model tested, by 13 to 43 points. Laya's weakness is consistent with LangWatch (Laya-typed 38.2% there).
- A supervised TF-IDF baseline trained on 2024 arXiv labels trails Jev by 4.3 points; not significant after Holm correction. It needs labelled training data, which is exactly what zero-shot avoids.
- Qwen3.5-4B via letter logprobs shows a strong first-option bias; SemIf (same base, vendor prompt) is the control (post hoc block).
- **Question form (yes/no per label vs one choice) changes answers for most open models**: agreement with the choice run Laya 43.5%, GLiNER-1B 50%, Qwen 52.5%, GLiNER 66%, GLiClass 69%; Jev 97%, NLI 99.5%.

**S2 Federal Register (358)**: API systems 99.2 to 99.7%; NLI 96.7%, GLiNER 93.0%, GLiClass 88.0%, Qwen 88.0%, GLiNER-1B 84.4%, Laya 81.0%. Every local system is significantly below Jev (Holm p ≤ 0.008).

**P1 stated probability**: encoder models and Qwen letter probabilities are unrelated to the stated probability (MAE 0.28 to 0.43); LLMs exact; Jev MAE 0.027.

**P3 long input (accuracy by length 500 / 2k / 8k / 24k tokens; refused inputs count as errors)**: Jev and Luna 1.00 throughout. GLiNER chunked 1.00 / 1.00 / 0.93 / 0.86. GLiClass 0.96 / 0.76 / 0.66 / 0.50, NLI 0.64 / 0.48 / 0.48 / 0.46 and Laya 0.75 / 0.59 / 0.55 / 0.46 (truncation: they only see the start). Laya-long answers "yes" to almost everything at length (specificity 0.50 at 2k, 0.14 at 8k). GLiNER refuses > 4,096 tokens, GLiNER-1B > 7,999 (memory and position limits on the M1 Pro). Qwen 0.71 / 0.76 / 0.93 / 0.71.

**P4 no fitting label, with a "none" option**: GLiClass 100%, Jev 95%, NLI 94%, GLiNER-1B 92%, Gemini 87%, Luna 86%, Qwen 67%, Haiku 65%, GLiNER 25%, Laya 13%.

**Post hoc block (Eikos-4B, SemIf, Kev-0.8B)**: runs complete; analysis pending. Kev-4B dropped (hardware), see protocol changelog.

## 2026-09-28, post hoc block and additional analyses (exploratory)

**Additional decision models, S1 arXiv** (separate Holm family): Eikos-4B 80.5% [76.3, 84.1], −5.8 pp vs Jev [−9.0, −2.8], p_holm < 0.001; SemIf 4B 77.5% [73.2, 81.3], −8.8 pp, p_holm < 0.001; Kev-0.8B 64.5% [59.7, 69.0], −21.8 pp, p_holm < 0.001. S2: 97.2 to 97.8%. SemIf (same Qwen3.5-4B base as our letter-logprob run, vendor prompt) reaches 77.5% versus 53.2% for our prompt: the logprob approach depends strongly on the prompt. Agreement between yes/no and choice form: Eikos 93%, Kev-0.8B 88%, SemIf 84%. P3: Eikos 1.00 up to 8k, fails at 24k (memory); Kev-0.8B 0.93 / 0.93 / 0.79 / 0.46; SemIf refuses > 4,096 tokens. P4 with "none": SemIf 97%, Eikos 95%, Kev-0.8B 86%.

**Detecting "no fitting label" without a threshold** (AUROC of top confidence, S1 in-set vs P4 out-of-set): Gemini 0.990, Haiku 0.985, SemIf 0.940, TF-IDF 0.939, Kev-0.8B 0.919, GLiNER 0.915, Eikos 0.901, **Jev 0.900**, NLI 0.898, Qwen 0.849, GLiNER-1B 0.818, GLiClass 0.788, **Luna 0.624**, Laya 0.619.

**Informativeness of confidences (S1)**: Jev returns 53 distinct values and exactly 1.0 for 51.5% of items, so its top half cannot be ranked (flat segment in the risk–coverage curve up to about 60% coverage). Verbalized LLM confidences are coarse: Gemini 8 distinct values, Haiku 9, Luna 25, Sonnet 28. Encoders and logprob readouts are continuous.

**Latency, isolated** (p50 / p95 seconds; local on M1 Pro at batch 1 with nothing else loaded, API at concurrency 1 with cache bypassed, 50 or 30 S1 items): TF-IDF 0.001; Laya 0.14 / 0.15; Kev-0.8B 0.21 / 0.26; GLiNER-1B 0.36 / 0.46; GLiClass 0.46 / 0.66; **Jev 0.47 / 0.60**; GLiNER 0.54 / 0.80; Gemini 0.77 / 1.06; Haiku 1.07 / 1.65; Luna 1.14 / 2.29; Qwen letter logprobs 1.66 / 2.16; SemIf 1.70 / 2.10; DeBERTa NLI 1.73 / 3.03; Sonnet 2.14 / 6.54; Eikos 2.69 / 4.71. Local numbers are laptop numbers and not comparable to GPU serving.

**Spend reconciled**: every OpenRouter request is in the cache; cached costs plus the uncached latency run sum to USD 4.263 against USD 4.249 account usage since project start. Remaining balance USD 1.55.

**Figures**: `results/figures/` (risk–coverage, reliability, cost against accuracy, long input).

## 2026-09-28, results review by Codex and corrections

An independent review (`results/review-codex.md`) recomputed five headline numbers from raw rows (all reproduced) and found no bug in Wilson, McNemar, Holm, bootstrap or risk at coverage. Corrections applied: wording changed from "as accurate as" to "not distinguishable at this sample size" (all LLM point estimates are above Jev); open-model results qualified by tested interface; accuracy ceiling reworded as indicative; ECE made independent of row order (Jev 0.064 → 0.068); 10,000 bootstrap resamples; Holm families separated (pre-registered systems, rationale variants, post hoc); unlogged deviations recorded in the protocol changelog; unambiguous-subset results added (ordering unchanged).

**Rationale-first prompt, paired within each system:** Luna −7.0 points [−10.8, −3.5], p_holm 0.001; Gemini −4.5 [−7.8, −1.5], p_holm 0.016; Haiku −4.5 [−7.8, −1.3], p_holm 0.016. No rationale output hit the 600-token cap.
