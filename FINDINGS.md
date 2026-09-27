# Findings log

Chronological. Each entry states what was checked, how, and the result, including what we could not find or verify.

## 2026-09-27

- **Jev is callable directly on OpenRouter** as `typesafe/jev-1.13` (context 32,000, input text only, output modality "decisions"). It does not appear in the default `/api/v1/models` listing; only `typesafe/jev-router` does. Verified via `/api/v1/models/typesafe/jev-1.13/endpoints`.
- **GLiNER2.5-Decide has no hard context cap in its config** (`max_len: null`). The encoder is DeBERTa-v3-large with relative position attention, so longer inputs run, but the base model was pretrained at 512 positions. The GLiNER2 library ships a chunking API (`classify_long`, default chunk 384 words, overlap 64) that aggregates chunk logits. Whether quality holds on long inputs is an open question for experiment 3.
- **Fastino publishes three Decide checkpoints**: `GLiNER2.5-Decide` (DeBERTa-v3-large), `GLiNER2.5-multi-Decide` (mDeBERTa-v3-base, `max_len` 4096) and `GLiNER2.5-Decide-1B` (Ettin encoder, 7,999 positions, local attention 128).
- **Dataset strategy is open.** Public classification sets are contaminated for LLMs; GitHub issues rejected because label provenance is unverifiable; no hand labeling. Research on sealed test sets, synthetic data with labels by construction and outcome-labeled data is in progress.
