# decision-model-audit

Reproducible comparison of "decision models" (zero-shot classifiers such as TypeSafe Jev, Fastino GLiNER2.5-Decide and Laya) against small LLMs with structured outputs.

Status: complete (September 2026 snapshot). The full results are in [`results/REPORT.md`](results/REPORT.md). The pre-registered protocol and every deviation from it are in [`notes/protocol.md`](notes/protocol.md). Findings were logged chronologically in [`FINDINGS.md`](FINDINGS.md), including negative and inconclusive results.

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

API calls go through OpenRouter. Set `OPENROUTER_API_KEY` in the environment, or set `DMA_KEY_FILE` to a dotenv file that contains it. Keys are never committed.

Every API response is cached, so the analysis can be rerun from the committed raw outputs without any API access:

```bash
uv run python -m dma.analysis.report
uv run python -m dma.analysis.figures
```

The run scripts in `scripts/` reproduce the model calls. Local models were run on an Apple M1 Pro with 16 GB of memory.

## License

Code and analysis: MIT, see [`LICENSE`](LICENSE). Third-party code in `third_party/` keeps its own license. The evaluation data keeps the terms of its sources: arXiv metadata and Federal Register documents.
