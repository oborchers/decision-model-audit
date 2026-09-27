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
