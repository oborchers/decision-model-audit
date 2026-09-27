# decision-model-audit

Reproducible comparison of "decision models" (zero-shot classifiers such as TypeSafe Jev, Fastino GLiNER2.5-Decide and Laya) against small LLMs with structured outputs.

Status: work in progress. Findings are logged iteratively in [`FINDINGS.md`](FINDINGS.md), including negative and inconclusive results.

## Questions

1. Accuracy and calibration of the returned probabilities
2. Sensitivity to label wording
3. Behaviour on inputs longer than the encoder window
4. Behaviour when no label fits
5. Cost of a human-readable rationale (LLMs) versus no rationale (decision models)
6. Cost and latency per 1,000 decisions

## Layout

```
src/dma/          runners per model family, shared schema, metrics
probes/           generators for constructed probes (ground truth by construction)
data/             evaluation data or fetch scripts (no contaminated public sets as primary measure)
results/          raw model outputs (JSONL) and computed metrics, committed
notes/            method notes and decisions
```

## Setup

```bash
uv sync
```

API keys are read from the environment at runtime and never committed.
