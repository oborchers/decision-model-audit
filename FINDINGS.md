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
