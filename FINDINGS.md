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
