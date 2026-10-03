# decision-model-audit

Reproducible comparison of "decision models" (zero-shot classifiers such as TypeSafe Jev, Fastino GLiNER2.5-Decide and Laya) against small LLMs with structured outputs.

Status: complete (September 2026 snapshot), with a post hoc addition of Cloudflare's Clef and Clef-flash (October 2026). The full results are in [`results/REPORT.md`](results/REPORT.md). The pre-registered protocol and every deviation from it are in [`notes/protocol.md`](notes/protocol.md). Findings were logged chronologically in [`FINDINGS.md`](FINDINGS.md), including negative and inconclusive results.

Write-up: [Ist Jev wirklich gut? Ein Feldtest auf frischen Daten](https://www.drborchers.com/blog/jev-feldtest/) (German) and [Is Jev Actually Any Good? A Field Test on Fresh Data](https://www.drborchers.com/en/blog/jev-field-test/) (English).

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

The post hoc Clef and Clef-flash runs (October 2026) go through Cloudflare Workers AI instead. Set `CLOUDFLARE_API_TOKEN` (an account token with Workers AI read access) and `CLOUDFLARE_ACCOUNT_ID`, or set `DMA_CF_KEY_FILE` to a dotenv file that contains them, then run `scripts/run_clef.sh <block>`. Spend is booked in `results/raw/clef/spend.jsonl` and capped by `DMA_CF_BUDGET` and `DMA_CF_BLOCK_BUDGET`.

Every API response is cached, so the analysis can be rerun from the committed raw outputs without any API access:

```bash
uv run python -m dma.analysis.report
uv run python -m dma.analysis.figures
```

The run scripts in `scripts/` reproduce the model calls. Local models were run on an Apple M1 Pro with 16 GB of memory.

## License

Code and analysis: MIT, see [`LICENSE`](LICENSE). Third-party code in `third_party/` keeps its own license. The evaluation data keeps the terms of its sources: arXiv metadata and Federal Register documents.
