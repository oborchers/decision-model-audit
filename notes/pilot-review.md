# Pilot review (qualitative and quantitative), 2026-09-27

Done after the pilot and after the first Jev main runs, before any other main run. The main run was paused for this review.

## Quantitative
- All systems return valid, parseable answers on all pilot sets. Costs and latencies extrapolated; projected main cost about USD 3.90.
- Bugs found and fixed: cache hashed sorted JSON keys (reversed variant served from cache); OpenRouter key read concurrently from a FIFO.

## Qualitative
1. **S1 errors are mostly genuine boundary cases**, not obvious failures (fuzzing: cs.CR by the author, cs.SE by all four models; GUI grounding: cs.CV vs. cs.HC/cs.CL; LLM analysis of game reviews: cs.HC vs. cs.CL). The cross-listing ambiguity marker flags none of them, so it is a weak proxy. Absolute accuracy is capped by label noise; comparisons between systems remain valid because all see the same items. **Added (exploratory, post hoc):** sensitivity analysis excluding items where all API models agree on the same non-gold label.
2. **Rationales** read plausibly and justify wrong answers as fluently as right ones (GPT-6 Luna: wrong label, confidence 0.98, coherent rationale). **Added (exploratory):** quote fidelity, the share of quoted phrases that occur verbatim in the input. Gemini quotes verbatim; Haiku mostly paraphrases.
3. **P4 works as intended:** without a "none" option Jev assigns an astrophysics paper to cs.CL with confidence 0.96.
4. **P3 limitation:** the decisive sentence is stylistically foreign to the Federal Register filler. P3 measures whether a salient sentence is found in long input, not subtle long-document understanding.
5. **S2 label noise:** NAGPRA notices are published under the National Park Service but written by museums and universities; "none" is defensible there.
6. **Ceiling effects:** S2, P1 and P2 are near ceiling for all systems. Reported as findings; probes are not made harder after seeing results.

## Local systems pilot (S1 40, P4 10, P1 8 items)

- All local systems valid on all pilot items after one fix: `qwen-lp` supported at most 8 option letters; S1 with `none` needs 9. Extended to A–J (single-token check enforced in code).
- `qwen-lp` shows a strong first-option bias (19 of 40 S1 pilot answers are option A, `cs.CL`). Kept as is; the `reversed` variant measures it.
- On P1, `qwen-lp` letter probabilities are near 0.1 for "yes" regardless of the stated probability, and the encoder models (GLiNER, GLiClass, NLI, Laya) return values unrelated to the stated probability. Expected: token or entailment scores are not event probabilities. Reported as a finding, not treated as a bug.
- Laya is weak on S1 (17 of 40) but not degenerate (predictions spread over 7 labels).
- `gliner-1b` loads in about 95 s per process; the main run script accepts this.

## Main local run incident (2026-09-27)

- Swap reached 24.5 of 24.6 GB. Causes: a memory leak in `qwen-lp` (MLX kept freed per-item KV-cache buffers; the process grew to 12 GB RSS) and a concurrent toy test of an additional model. The toy test was stopped; the leak was fixed by releasing the cache and calling MLX `clear_cache()` after every item; the runner gained a resume step that skips items with an existing valid row. The qwen lane was restarted and resumed at 306 of 400 S1 items. Results are unaffected; latencies of this run are not used (see protocol changelog).
