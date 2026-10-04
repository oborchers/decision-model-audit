# Decision models on fresh data, part 2: equally accurate, not equally usable

Status: complete. Every analysis in this report is post hoc relative to the part 1 protocol and is logged in the changelog of `notes/protocol.md` (entries from 2026-10-02 to 2026-10-04) before the respective evaluation calls, except where marked. Part 1 results (`results/REPORT.md`, tag `v1-september-2026`, and the Clef addition merged as `c074d28`) are unchanged: rerunning all analyses with the part 2 data reproduces every published number and figure (`results/summary.json` and `results/gaps.json` compared field by field with both states outside the new systems; the four part 1 figures are pixel-identical to `c074d28`). Runs: 2026-10-02 to 2026-10-04.

Numbers come from `results/summary.json` (S1 to P4 for the new systems), `results/gaps.json` (batches, repetition), `results/probes_teil2.json` (P5 to P7), `results/teil2.json` (automation rate, FWF, interface, stability) and `results/finetune.json` (fine-tuning).

## Summary

1. **Accuracy has become a commodity; price has not.** On the 400 fresh arXiv papers of part 1 (S1), five of thirteen new decision models cannot be distinguished from Jev 1.13 (86.3%): Perplexity pplx-decider 87.3%, Fastino GLiDE 86.0%, Kev 4B 85.5%, Mercury 84.3%, APUS-OpenJev 9B 83.8% (Holm family "part 2", all p_holm ≥ 0.49). Clef from part 1 adds a sixth (85.8%). Among the hosted models in this group, the list price per 1,000 S1 decisions ranges from USD 0.020 (Kev 4B) to 0.255 (GLiDE), a factor of about 13 (Mercury ran on a free tier without list price).
2. **Usability differs far more than accuracy.** As an exploratory estimate on the same 400 papers (threshold chosen on the evaluation data, wide intervals), the share of S1 papers a system could have decided alone while keeping the error among them at or below 5% (from its own confidence) ranges from 0% to 61% among decision models of similar accuracy: GLiDE 60.5%, Decision 2.0 Nox 4B 58.3%, pplx-decider 56.5%, Clef 56.3%, Kev 4B 55.3%, but Jev 0% and Solar Decide 0%. Jev's 51.5% saturated confidences of exactly 1.0 contain 5.3% errors, so its rate jumps between 0% and about 51% with the target. An embedding classifier trained on 2024 labels reaches 67.0%. Intervals are wide (Jev 0 to 78.8%).
3. **"Jev-compatible" does not mean interchangeable.** The same request returns fields with different meanings: `confidence` differs from the top probability for 79 to 100% of answers at nine providers, matches it exactly at three (Tev1, APUS 4B and 9B), and Jev (39.8%) and Mercury (59.0%) sit in between. For `score` questions, most return an expected level, GLiDE the most likely level. Context limits range from silent truncation (Clef through Workers AI) over refusals (Kev 4B, APUS, Decision 2.0 Kai and Nox) to complete reading (pplx-decider, GLiDE, D1).
4. **Long inputs and batches separate the models.** Only pplx-decider, GLiDE and D1 found a single decisive sentence at every length up to 24,000 tokens, as Jev does. Bundling ten papers per request keeps or improves accuracy for GLiDE (+2.8 points), D1 (+3.0) and pplx-decider (+1.3), and collapses it for Kev 4B (−49.8), Decision 2.0 Kai (−45.3) and Strands Decider 2B (−36.0). For most hosted models, the billed cost per decision rises 4 to 7 times when bundled.
5. **A "none of these" option is cheap for some and costly for others.** All models detect most off-topic papers (93 to 100%), but D1, Tev1 and Solar reject 13.8%, 14.0% and 30.5% of S1 papers that do fit (Jev 1.0%).
6. **Stated chances are reproduced only by Jev and a general LLM.** Mean absolute error between the returned and the stated probability of a described random event: GPT-6 Luna 0.006, Jev 0.027, all other decision models 0.083 (Solar) to 0.362 (APUS 4B).
7. **New probes:** per-item options (P5, choose an abstract's own title among five similar titles) are solved by all decision models except CLM (97.3 to 99.5%) and equally by a training-free embedding comparison (97.3%). Source against claim (P6) 86.0 to 100%. Counting events in a JSON log (P7) is hard for all: Nox 4B 63.4%, Jev 61.3%, GLiDE 55.6%, all others 26.1 to 49.3% (chance 25%).
8. **German costs almost no accuracy.** On 600 FWF research projects with summaries in both languages by the same applicants, nine of ten systems stay within one point between German and English; only Clef-flash loses 2.2 points (p_holm 0.044). German inputs need 21 to 35% more tokens, and the two summaries of a project are not translations of each other, so this compares the languages as applicants write them.
9. **In these recipes, fine-tuning a decision model brought no advantage over fine-tuning a plain encoder.** With 400 S1 training labels (plus 400 further labelled papers used to choose learning rate and epoch), Laya and ModernBERT-large (Laya's backbone) reach 78.8% and 79.8%, below the embedding classifier of part 1, which used the 400 training labels only (85.0%). With all 2,806 labels and the same recipe, added after earlier results, ModernBERT reaches 87.0%, Laya 84.3% (descriptive, three seeds).
10. **Models with the same label serve different purposes.** CLM-8B, built for selecting agent actions, reaches 39.3% on S1; Laya, described by its vendor as a base model for fine-tuning, reached 43.5% without training in part 1.

## Setup

### New systems (part 2)

| System | Access | Exact model | Where it ran | Programme |
|---|---|---|---|---|
| Liquid D1 | OpenRouter `/alpha/decisions` | `liquid/d1` | hosted | full |
| Upstage Solar Decide | OpenRouter | `upstage/solar-decide` | hosted | full |
| Inception Mercury Decide | OpenRouter, free tier | `inception/mercury-decide:free` | hosted | S1 `choice` and `choice_none`, P4, P5 to P7 (rate limit) |
| Together Tev1-4B experimental | OpenRouter | `togethercomputer/tev1-4b-experimental` | hosted | full |
| Kev 4B | OpenRouter | `jaredpalmer/kev-4b` (8,192-token context) | hosted | full |
| Perplexity pplx-decider | `api.perplexity.ai/v1/decisions` | `pplx-decider-v1-27b` | hosted | full, P5 to P7, FWF |
| Fastino GLiDE | `api.fastino.ai/v1/systemone` | `fastino/GLiDE` | hosted | full, P5 to P7, FWF |
| Strands Decider 2B | local, `strands-decider serve` | `StrandsAgents/strands-decider-2B-hobson-v19` @ `bb282d78` | M1 Pro, Apple GPU (MPS), serial | full, P5 to P7 |
| APUS-OpenJev 4B, 9B | local, own shim with the vendor's renderer (`src/dma/local_servers/apus.py`) | `apus-ailab/APUS-OpenJev-v1-4B-MLX-4bit` @ `8ff01a46`, `-9B-MLX-4bit` @ `2615dc82` | M1 Pro, MLX | full, P5 to P7 |
| CLM-8B | local, `clm-tune-mlx-serve` | heads `Contrastive-LM/CLM-v0.1-8B` @ `e939398d`, encoder `mlx-community/Qwen3-8B-8bit` @ `48a0b75b`, `clm-tune-mlx` @ `3a81a90b` | M1 Pro, MLX | S1 choice variants, P4 with "none", P5 |
| Decision 2.0 Kai 0.6B | local, vendor `system_one` (`src/dma/local_servers/hf_system_one.py`) | `vllm-sr/Decision-2.0-Kai-0.6B` @ `cd49ea38` | M1 Pro, MPS | full, P5 to P7 |
| Decision 2.0 Nox 4B | same | `vllm-sr/Decision-2.0-Nox-4B` @ `25e8f67d` | one NVIDIA A40 (Runpod), CUDA | full, P5 to P7 |

All systems receive the identical request body as Jev (state, typed questions, label names and descriptions); APUS's own contract is reached through a translating shim, one question at a time. Jev, Clef, Clef-flash and GPT-6 Luna were additionally run on the new probes and FWF. The model list was frozen on 2026-10-03. Hosted weights expose no revision.

### New data

| Source | Task | Items | Label origin |
|---|---|---|---|
| P5 titles | choose the abstract's own title among five; the four distractors are the lexically closest papers of the same category | 184 | by construction |
| P6 claims | is a sentence supported by the abstract (own sentence; sentence of the lexically closest abstract; own sentence with one number changed) | 179 | by construction |
| P7 counting | count events of one type in a JSON log of 40 to 120 events, four ordered levels; near-miss event names must not count | 142 | by construction |
| FWF | research field among eight ÖFOS fields, same project summary in English and German | 600 × 3 conditions | dominant field of the project (FWF Open API, CC0) |

P5 to P7 use arXiv papers created on or after 2026-09-01 that are in neither S1 split (`src/dma/data/build_probes_teil2.py`, seed 20261002). They were built as P5a to P5c and renamed before any write-up; items and answers are unchanged (protocol). The first version was solved 10 of 10 by every system in the pilot and was replaced by harder items before any main call. FWF summaries are not fresh (public for months to years); the paired design keeps this from biasing the language difference, but absolute FWF accuracies are not contamination-free.

### New analyses

- **Automation rate** (`teil2.py`): on S1, the largest share of papers whose expected error stays at or below 2% or 5% when a system decides the papers it is most confident about, using the tie-aware risk-at-coverage of part 1 (tied blocks contribute their mean error), bootstrap over papers. The threshold is chosen on the evaluation data, so absolute rates are optimistic; the comparison between systems is the purpose.
- **Interface behaviour**: whether `confidence` equals the top probability, saturation, distinct values, refusals per probe, `score` semantics.
- **Stability**: agreement of reversed label order, two label paraphrases and the yes/no form with the own choice run, as defined in part 1.
- **Fine-tuning** (`finetune.py`): Laya and ModernBERT-large trained on S1 labels, descriptive.

Paired comparisons against Jev use exact McNemar tests. Holm families: "part 2, new decision models" (S1 and S2 separately, 13 systems), one family per probe for P5 to P7 (all systems on that probe), one over systems for FWF. Earlier families and their adjusted p values are unchanged.

## Results

### S1 arXiv, accuracy and selective prediction

| System | Accuracy [95% CI] | Δ vs Jev [95% CI] | Holm p | Error at 80% coverage | AURC | ECE | USD / 1,000 |
|---|---|---|---|---|---|---|---|
| pplx-decider | 87.3% [83.6, 90.2] | +1.0 [−1.0, +3.0] | 1.00 | 8.4% | 0.051 | 0.062 | 0.022 |
| *Jev 1.13 (part 1)* | *86.3% [82.5, 89.3]* | | | *8.6%* | *0.067* | *0.068* | *0.035* |
| GLiDE | 86.0% [82.3, 89.1] | −0.3 [−2.5, +2.0] | 1.00 | 9.7% | 0.055 | 0.054 | 0.255 |
| *Clef (part 1)* | *85.8% [82.0, 88.8]* | *−0.5 [−3.0, +2.0]* | *0.84* | *8.8%* | *0.073* | *0.095* | *0.155* |
| Kev 4B | 85.5% [81.7, 88.6] | −0.8 [−3.0, +1.5] | 1.00 | 8.4% | 0.052 | 0.182 | 0.020 |
| Mercury | 84.3% [80.4, 87.5] | −2.0 [−5.0, +1.0] | 1.00 | 10.3% | 0.062 | 0.085 | free tier |
| APUS 9B | 83.8% [79.8, 87.0] | −2.5 [−5.3, +0.3] | 0.49 | 7.8% | 0.058 | 0.100 | local |
| APUS 4B | 82.0% [77.9, 85.5] | −4.3 [−7.3, −1.3] | 0.046 | 11.3% | 0.066 | 0.121 | local |
| Nox 4B | 81.5% [77.4, 85.0] | −4.8 [−7.8, −2.0] | 0.015 | 11.6% | 0.068 | 0.114 | local |
| D1 | 81.3% [77.1, 84.8] | −5.0 [−7.8, −2.3] | 0.005 | 12.5% | 0.075 | 0.040 | 0.019 |
| Tev1 | 81.3% [77.1, 84.8] | −5.0 [−8.3, −2.0] | 0.020 | 10.9% | 0.069 | 0.084 | 0.025 |
| Solar Decide | 79.3% [75.0, 82.9] | −7.0 [−10.3, −4.0] | <0.001 | 17.8% | 0.172 | 0.092 | 0.039 |
| Strands 2B | 74.3% [69.8, 78.3] | −12.0 [−15.5, −8.5] | <0.001 | 18.8% | 0.108 | 0.043 | local |
| Kai 0.6B | 71.5% [66.9, 75.7] | −14.8 [−19.0, −10.8] | <0.001 | 23.1% | 0.151 | 0.356 | local |
| CLM-8B | 39.3% [34.6, 44.1] | −47.0 [−52.5, −41.3] | <0.001 | 55.6% | 0.401 | 0.242 | local |

Not significant against Jev does not mean equivalent; at n = 400 differences below about 3 points cannot be resolved. Reference points from part 1: GPT-6 Luna 88.8%, logistic regression on Qwen3-Embedding-8B embeddings with 2024 labels 86.3% (ECE 0.029, AURC 0.038).

### Automation rate (S1)

Share of papers decided alone at a maximum expected error of 5% and 2%, from the returned `confidence`, with bootstrap 95% intervals for the 5% target. Figure: `figures/teil2_risk_coverage.png`.

| System | Accuracy | ≤ 5% error [95% CI] | ≤ 2% error |
|---|---|---|---|
| Embedding + LR (part 1) | 86.3% | 67.0% [47.0, 82.8] | 46.8% |
| GLiDE | 86.0% | 60.5% [24.0, 75.5] | 25.0% |
| Nox 4B | 81.5% | 58.3% [17.8, 67.8] | 16.5% |
| pplx-decider | 87.3% | 56.5% [26.5, 73.8] | 26.0% |
| Clef | 85.8% | 56.3% [0.8, 71.5] | 0.8% |
| Claude Sonnet 5 | 89.8% | 55.8% [21.7, 84.5] | |
| Kev 4B | 85.5% | 55.3% [26.0, 78.5] | 26.5% |
| APUS 4B | 82.0% | 51.3% [15.8, 67.3] | 15.8% |
| Mercury | 84.3% | 46.5% [22.0, 71.5] | 20.8% |
| Tev1 | 81.3% | 46.5% [17.5, 72.8] | 18.3% |
| D1 | 81.3% | 41.8% [14.5, 63.5] | 15.0% |
| APUS 9B | 83.8% | 41.3% [27.3, 81.5] | 29.8% |
| Clef-flash | 82.3% | 38.3% [30.5, 77.5] | 31.3% |
| Strands 2B | 74.3% | 36.5% [9.3, 52.3] | 9.0% |
| Claude Haiku 4.5 | 87.3% | 24.3% [15.0, 45.0] | |
| Kai 0.6B | 71.5% | 15.5% [7.5, 35.3] | 9.3% |
| CLM-8B | 39.3% | 6.5% | 1.0% |
| Jev 1.13 | 86.3% | 0.0% [0.0, 78.8] | 0.0% |
| GPT-6 Luna | 88.8% | 0.0% [0.0, 72.0] | 0.0% |
| Solar Decide | 79.3% | 0.0% [0.0, 10.5] | 0.0% |

Jev returns exactly 1.0 for 206 of 400 papers, 11 of them wrong (5.3%). The most confident block therefore cannot be split, and the rate at a 5% target is 0% while a 5.5% target would allow at least 51.5%; the interval reflects this jump. Solar's confidence does not rank its errors (AURC 0.172, error 17.8% even at 80% coverage). Using the top probability instead of `confidence` changes the rates by at most 5.5 points for every decision model (`teil2.json`).

### Interface: what the same fields mean

| System | `confidence` ≠ top probability | Exactly 1.0 | Distinct values (400 papers) | Refusals |
|---|---|---|---|---|
| Jev | 39.8% | 51.5% | 53 | none |
| Clef, Clef-flash, Kev 4B, pplx-decider, Strands 2B | 100% | 0% | 373 to 396 | Kev 4B: P3 41/114 |
| Kai 0.6B, Nox 4B | 99.8% | 0% | 400 | P3 39/114 and 28/114 |
| CLM-8B | 97.3% | 0% | 400 | |
| Solar | 95.8% | 0% | 399 | |
| D1 | 85.5% | 0% | 400 | |
| GLiDE | 78.8% | 0% | 399 | P4 forced choice 1/100 |
| Mercury | 59.0% | 1.5% | 396 | |
| Tev1 | 0% | 0.8% | 400 | P4 forced choice 6/100 (answers that no option fits) |
| APUS 4B, 9B | 0% | 0% | 400, 389 | P3 47/114 each (8,192-token limit) |

**Score questions (P7).** The API's `score` field is the expected level (probability-weighted mean) for all systems except GLiDE, which returns the most likely level as an integer in all 141 valid answers. Code that thresholds `score` behaves differently by provider without any change in the request.

**Context limits.** Kev 4B (8,192 tokens), APUS (8,192 tokens) and Decision 2.0 refuse longer inputs per question (`max_length_exceeded` for Kai and Nox); these count as errors. Workers AI truncates Clef silently at about 2,200 tokens (part 1). Tev1 refuses instead of choosing when no option fits.

### Long input (P3), accuracy by length; refused inputs count as errors

| System | 500 | 2k | 8k | 24k | Refused |
|---|---|---|---|---|---|
| Jev, pplx-decider, GLiDE, D1 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| Tev1 | 1.00 | 1.00 | 0.97 | 0.96 | 0 |
| Solar | 0.93 | 1.00 | 0.93 | 0.79 | 0 |
| Nox 4B | 1.00 | 1.00 | 1.00 | 0.00 | 28 |
| Kai 0.6B | 1.00 | 1.00 | 0.59 | 0.00 | 39 |
| Kev 4B | 1.00 | 1.00 | 0.55 | 0.00 | 41 |
| APUS 4B | 0.93 | 0.97 | 0.34 | 0.00 | 47 |
| APUS 9B | 0.82 | 0.86 | 0.31 | 0.00 | 47 |
| Strands 2B | 0.54 | 0.62 | 0.48 | 0.46 | 0 |

Strands misses the decisive sentence at every length without refusing.

### Stability and repeatability (S1)

Agreement with the own choice run (part 1 definition; an invalid answer counts as disagreement):

| System | Reversed order | Paraphrase 1 | Paraphrase 2 | Yes/no form | Repeated request |
|---|---|---|---|---|---|
| Jev | 99.0% | 97.5% | 98.0% | 97.0% | 99.0% |
| GLiDE | 98.0% | 96.8% | 96.8% | 94.5% | 99.0% |
| pplx-decider | 95.5% | 97.8% | 97.0% | 94.3% | 100% |
| Nox 4B | 94.8% | 89.3% | 94.5% | 95.0% | 100% |
| D1 | 93.5% | 93.5% | 95.8% | 92.5% | 100% |
| Kev 4B | 93.3% | 93.3% | 94.8% | 95.3% | 100% |
| Strands 2B | 93.3% | 86.5% | 89.3% | 90.0% | 100% |
| Tev1 | 92.8% | 86.8% | 91.0% | 91.8% | 100% |
| Kai 0.6B | 92.5% | 83.0% | 88.0% | 85.8% | 100% |
| Solar | 86.3% | 90.0% | 92.0% | 80.3% | 98.0% |
| APUS 4B | 83.8% | 90.5% | 90.3% | 91.0% | 100% |
| APUS 9B | 83.8% | 93.3% | 90.8% | 89.0% | 100% |
| CLM-8B | 100% | 29.5% | 53.3% | not run | |

Repeated requests: 98 S1 papers with the cache bypassed. CLM embeds each label description separately, so the order of labels cannot matter, and changing the descriptions changes its answers fundamentally. The yes/no form asks one question per label; accuracy in that form ranges from 67.0% (Kai) to 85.5% (Kev 4B), with ECE from 0.089 (APUS 9B) to 0.473 (Kai).

### No fitting label (P4) and false "none" (S1)

| System | "None" chosen on off-topic papers (P4) | False "none" on fitting S1 papers | AUROC in-domain vs off-topic |
|---|---|---|---|
| Jev | 95% | 1.0% | 0.900 |
| Solar | 100% | 30.5% | 0.913 |
| D1 | 99% | 13.8% | 0.909 |
| Tev1 | 98% | 14.0% | 0.899 |
| Mercury | 98% | 1.8% | 0.935 |
| APUS 4B | 98% | 5.5% | 0.931 |
| pplx-decider | 97% | 3.8% | 0.961 |
| GLiDE | 97% | 1.3% | 0.852 |
| Kev 4B | 96% | 6.0% | 0.923 |
| APUS 9B | 96% | 1.5% | 0.930 |
| Kai 0.6B | 95% | 10.5% | 0.969 |
| CLM-8B | 95% | 15.3% | |
| Nox 4B | 94% | 0.5% | 0.965 |
| Strands 2B | 93% | 2.8% | 0.954 |

The high P4 detection rates of Solar, D1 and Tev1 are bought with many false rejections of fitting papers. The `choice_none` accuracy on S1 drops accordingly (Solar 58.8%, D1 71.3%, Tev1 72.8%).

### Stated probabilities (P1) and minimal pairs (P2)

Mean absolute error between the returned and the stated chance of a described random event (127 items): GPT-6 Luna 0.006, Jev 0.027, Solar 0.083, GLiDE 0.147, Clef-flash 0.156, Strands 0.168, Kai 0.204, Clef 0.207, pplx-decider 0.213, D1 0.220, Nox 0.241, Tev1 0.244, APUS 9B 0.246, Kev 4B 0.293, APUS 4B 0.362. P2 minimal pairs: 100% for all new systems except Kev 4B (98.6%, as Jev).

### Batches of ten papers per request (S1)

| System | Single | Batch of 10 | Δ [95% CI] | p (McNemar) | USD / 1,000 single → batch |
|---|---|---|---|---|---|
| D1 | 81.3% | 84.3% | +3.0 [+0.5, +5.5] | 0.036 | 0.019 → 0.128 |
| GLiDE | 86.0% | 88.8% | +2.8 [+0.8, +4.8] | 0.013 | 0.255 → 1.179 |
| pplx-decider | 87.3% | 88.5% | +1.3 [−0.8, +3.3] | 0.33 | 0.022 → 0.129 |
| Solar | 79.3% | 80.3% | +1.0 [−3.0, +5.0] | 0.72 | 0.039 → 0.165 |
| Jev | 86.3% | 84.8% | −1.5 [−3.8, +0.5] | 0.24 | 0.035 → 0.026 |
| Tev1 | 81.3% | 75.5% | −5.8 [−9.3, −2.3] | 0.003 | 0.025 → 0.136 |
| APUS 9B | 83.8% | 77.8% | −6.0 [−9.5, −2.8] | <0.001 | local |
| Nox 4B | 81.5% | 74.3% | −7.3 [−11.0, −3.8] | <0.001 | local |
| APUS 4B | 82.0% | 72.0% | −10.0 [−14.3, −5.8] | <0.001 | local |
| Strands 2B | 74.3% | 38.3% | −36.0 [−41.8, −30.5] | <0.001 | local |
| Kai 0.6B | 71.5% | 26.3% | −45.3 [−50.5, −39.8] | <0.001 | local |
| Kev 4B | 85.5% | 35.8% | −49.8 [−55.3, −44.3] | <0.001 | 0.020 → 0.021 |

p values here are unadjusted, as in part 1's gap analysis. The costs are the billed amounts per decision. For most hosted decision models, one bundled request is billed at several times the cost of ten single ones (the providers' billing rules were not examined); for Jev bundling saves about a quarter. Median latency per bundled request: Nox 7.5 s (A40), GLiDE 9.7 s, Strands 11.3 s, Solar 14.6 s, APUS 4B 97 s and APUS 9B 196 s (laptop).

### S2 Federal Register

All new systems reach 96.6% to 99.7% (Jev 99.4%). Only Kai (−2.8 points, p_holm 0.021) and Strands (−2.5, p_holm 0.039) are below Jev after Holm correction; S2 is near ceiling.

### Probes P5 to P7

| System | P5 titles | P6 claims (balanced) | P6 number changed | P7 counting exact | P7 within one level | P7 ECE |
|---|---|---|---|---|---|---|
| Jev | 98.4% | 97.8% (98.3%) | 98.3% | 61.3% | 76.8% | 0.165 |
| Nox 4B | 98.9% | 93.3% (95.0%) | 81.4% | 63.4% | 83.1% | 0.498 |
| GLiDE | 98.9% | 98.3% (98.7%) | 98.3% | 55.6% | 84.5% | 0.294 |
| GPT-6 Luna | 98.9% | 95.0% (96.2%) | 86.4% | 49.3% | 79.6% | 0.497 |
| Solar | 97.8% | 98.9% (99.2%) | 100% | 48.6% | 68.3% | 0.307 |
| Mercury | 98.9% | 100% (100%) | 100% | 46.5% | 82.4% | 0.447 |
| Kev 4B | 97.8% | 96.1% (97.1%) | 93.2% | 45.8% | 71.8% | 0.276 |
| Tev1 | 97.8% | 96.1% (97.1%) | 93.2% | 45.1% | 72.5% | 0.284 |
| pplx-decider | 99.5% | 97.8% (98.3%) | 98.3% | 42.3% | 74.6% | 0.244 |
| D1 | 98.9% | 98.3% (98.7%) | 100% | 39.4% | 74.6% | 0.405 |
| Clef-flash | 97.8% | 96.1% (97.1%) | 93.2% | 38.7% | 66.9% | 0.366 |
| Clef | 98.4% | 97.2% (97.9%) | 96.6% | 34.5% | 64.1% | 0.464 |
| APUS 4B | 97.8% | 97.2% (97.9%) | 96.6% | 33.8% | 64.8% | 0.379 |
| APUS 9B | 97.3% | 97.2% (97.9%) | 96.6% | 30.3% | 70.4% | 0.375 |
| Strands 2B | 98.4% | 92.7% (94.5%) | 84.7% | 28.9% | 54.2% | 0.140 |
| Kai 0.6B | 97.3% | 86.0% (89.5%) | 61.0% | 26.1% | 50.7% | 0.198 |
| CLM-8B | 54.3% | not run | | not run | | |
| Embedding cosine, no training | 97.3% | | | | | |

n = 184, 179 and 142. Against Jev after Holm (per probe): on P5 only CLM differs; on P6 only Kai (p_holm < 0.001); on P7 every system except Nox (+2.1 points, p_holm 0.77), GLiDE (−5.6, p_holm 0.56) and Solar (−12.7, p_holm 0.060) is below Jev. Chance on P7 is 25% (balanced levels).

- **P5 does not separate the models.** Choosing among options that change per item works for every decision model and for an untrained embedding comparison of abstract and titles (Qwen3-Embedding-8B, cosine). It shows that per-item options work, not that a decision model is needed for them.
- **P6** is near ceiling except for the changed-number items, where small models drop (Kai 61.0%, Nox 81.4%, Strands 84.7%). AUROC of p(yes) is 0.999 to 1.000 for every decision model (Luna 0.867). Some unsupported sentences taken from the lexically closest abstract are arguably supported, and one number change hit an enumeration marker; labels were kept as constructed (protocol).
- **P7** measures counting under distraction. Errors are mostly off by one level (within-one accuracy 50.7 to 84.5%), and confidences are poorly calibrated (ECE up to 0.498).

### German: FWF paired test

Accuracy on the same 600 projects, English text with English task (en), German text with English task (de-en), German text with German task (de):

| System | en | de-en | de | Δ de − en [95% CI] | Holm p |
|---|---|---|---|---|---|
| GLiDE | 89.7% | 90.2% | 89.7% | +0.0 [−1.3, +1.3] | 1.00 |
| D1 | 89.8% | 89.3% | 88.8% | −1.0 [−2.3, +0.2] | 1.00 |
| Jev | 89.3% | 89.0% | 89.0% | −0.3 [−1.5, +0.8] | 1.00 |
| Clef-flash | 89.2% | 87.3% | 87.0% | −2.2 [−3.7, −0.8] | 0.044 |
| Clef | 88.2% | 88.8% | 88.8% | +0.7 [−0.3, +1.7] | 1.00 |
| Solar | 88.5% | 87.3% | 87.7% | −0.8 [−2.3, +0.7] | 1.00 |
| pplx-decider | 87.8% | 88.5% | 88.8% | +1.0 [−0.5, +2.5] | 1.00 |
| Kev 4B | 87.2% | 86.3% | 86.8% | −0.3 [−1.5, +0.8] | 1.00 |
| GPT-6 Luna | 86.8% | 85.7% | 86.8% | +0.0 [−1.8, +1.8] | 1.00 |
| Tev1 | 85.0% | 85.0% | 84.3% | −0.7 [−2.3, +1.0] | 1.00 |

German summaries need 21 to 35% more input tokens for the same project (median per system; Jev 903 English, 1,139 German), which raises the cost per decision by the same factor for token-priced models. Local models and Mercury were not run on FWF.

### Fine-tuning (S1, descriptive)

Does a decision model trained on the task's labels beat a plain encoder trained on the same labels? Laya (`convaiinnovations/laya`; the Hub revision changed on 2026-10-03 by adding a root `config.json` only, weights unchanged) and ModernBERT-large (`answerdotai/ModernBERT-large`, Laya's backbone, standard classification head), trained on S1 `train_2024` labels on one A40, evaluated on S1 main. Mean over three seeds, range in brackets.

| Recipe | Laya | ModernBERT-large |
|---|---|---|
| 400 labels, first recipe (3 epochs; Laya 1e-5 constant, ModernBERT 2e-5 with decay) | 78.4% [77.8, 79.8] | 66.9% [55.8, 74.0] |
| 400 labels, rate chosen on validation (3e-5 both), up to 15 epochs, epoch chosen on validation | 78.8% [77.3, 79.8] | 79.8% [79.8, 79.8] |
| 400 labels, best rate per model on S1 main (for orientation only) | 79.3% at 1e-5 | 79.8% at 3e-5 |
| 2,406 labels, constant rate 3e-5, epoch chosen on validation | 81.8% [81.8, 82.0] | 82.7% [82.3, 83.0] |
| 2,806 labels, first recipe | 77.8% [77.3, 78.8] | 86.1% [85.5, 86.8] |
| 2,806 labels, same recipe for both (3e-5, 3 epochs, linear decay) | 84.3% [83.8, 84.8] | 87.0% [86.5, 87.8] |
| Reference: embedding + LR (part 1), 400 / 2,806 labels | 85.0% / 86.3% | |
| Reference: Jev without training | 86.3% | |

Curves: `figures/teil2_finetune_curves_400.png`, `figures/teil2_finetune_curves_2406.png`; per-epoch values in `ft_curves/`.

- **The first ModernBERT recipe was undertrained** (75 optimizer steps at 400 labels, mean confidence 0.58 for the weakest seed). With a higher rate and more epochs all three seeds reach 79.75%.
- **Both models memorise 400 papers.** At the chosen rate, training loss falls below 0.001; validation loss is lowest after 1 to 4 epochs and rises afterwards (Laya to 1.2 to 1.7), while validation accuracy stays flat. Selecting the epoch on validation handles this.
- **With many labels the learning rate must decay.** A constant rate of 3e-5 gives noisy curves and 82 to 83%; the same rate with linear decay over three epochs gives 84 to 87%.
- **Learning-rate grid at 400 labels** (mean best validation accuracy): ModernBERT 1e-5 80.5%, 3e-5 83.8%, 8e-5 78.1% (one seed failed with 66.5% on S1 main); Laya 3e-6 78.8%, 1e-5 80.6%, 3e-5 82.0%.

With 400 training labels the classical recipe of part 1 (frozen embeddings plus logistic regression, trained in seconds, no validation set) is 5 to 6 points ahead of both fine-tuned models, which additionally used 400 validation labels. With all labels, the plain encoder catches up with Jev, and Laya's decision pretraining shows no advantage in these recipes; other budgets or methods were not tested. Choosing the epoch used 400 additional labelled papers for validation, a real cost in a 400-label setting. The last row was added after the long-run results and is reported as such (protocol).

### Latency and cost

Isolated latency, p50 / p95 seconds, 30 S1 papers, concurrency 1, cache bypassed:

- Hosted: pplx-decider 0.26 / 0.29, Tev1 0.39 / 0.48, Jev 0.47 / 0.60, D1 0.52 / 1.05, Kev 4B 0.75 / 1.00, Solar 0.82 / 1.34, GLiDE 1.10 / 1.62 (all from Germany; Mercury not measured).
- Laptop (M1 Pro, 16 GB): Kai 0.67 / 11.0 (MPS), Strands 0.93 / 1.29 (MPS), APUS 4B 1.76 / 2.24, APUS 9B 5.15 / 6.63 (MLX, 4-bit).
- GPU (one A40): Nox 4B 0.15 / 0.19.
- CLM-8B returned answers in about 2 ms, below a forward pass of an 8B encoder on this laptop; its server had already embedded these papers during the main run, so the number is not reported.

Local and GPU numbers are not comparable with each other or with vendor claims measured on other hardware.

Spend for part 2 at list prices, from the ledgers: OpenRouter USD 1.75 (`results/raw/teil2/spend_openrouter.jsonl`, including Jev and Luna on the new probes), Fastino USD 3.43 and Perplexity USD 0.33 (`spend_ext.jsonl`), Cloudflare USD 0.65 (`results/raw/clef/spend.jsonl` minus the 1.12 of the Clef run), Runpod about USD 2.66 (one A40 for 5.4 hours: Nox 4B and all fine-tuning). Total about USD 8.80.

## Limitations

- **Post hoc.** Everything here was added after part 1 and decided with knowledge of its results. Each addition was logged before its evaluation calls; three steps were decided after seeing results and are marked (P5 to P7 revision after a pilot at ceiling, fine-tuning with longer training, the shared fine-tuning recipe).
- **One task family.** S1 is topic classification near its practical ceiling (part 1: majority vote of five API systems 90.0%). The new probes are constructed, single-purpose and partly near ceiling (P5, P6, S2). Decisions with rules, exceptions or several interacting questions were not tested.
- **Interfaces differ.** All decision models received the identical request, but each provider interprets fields in its own way (`confidence`, `score`, limits). Results describe the hosted or vendor-served path at the time of the run, not the models in isolation. Hosted weights have no revision and may change.
- **Local hardware.** Strands, APUS, CLM and Kai ran on a 16 GB laptop, Nox on a rented A40. Latencies are not comparable across these setups. Strands crashed under concurrent requests on the Apple GPU and was rerun serially; all local models ran one request at a time.
- **Mercury** ran a reduced programme on the free tier (no stability, batches, latency, P1 to P3, FWF).
- **Automation rate** chooses the threshold on the evaluation data and has wide intervals; it compares systems, it does not predict production coverage.
- **Probes were authored by the orchestrating model (Claude)** where templates were needed (P7 logs). No model under test generated text.
- **FWF** is not fresh, and only hosted models were run on it.
- **Fine-tuning** is descriptive with three seeds, one task and one GPU; no significance tests. A different training budget or parameter-efficient method might change the comparison.
- **Sample sizes** as in part 1: about ±3.5 points at 86% accuracy on S1; probes have 142 to 184 items.

## Reproducibility

Protocol and deviations: `notes/protocol.md` (changelog from 2026-10-02). Data builders: `src/dma/data/build_probes_teil2.py`, `src/dma/data/build_fwf.py` (FWF snapshot `data/raw/fwf/projects_2026-10-02.jsonl.gz`). Clients and runners: `src/dma/ext_client.py`, `src/dma/runners/api.py`, `src/dma/runners/p5_embed.py`, `src/dma/runners/laya_finetune.py`, `src/dma/runners/encoder_finetune.py`, `src/dma/runners/ft_curves.py`. Local servers: `src/dma/local_servers/`. Run script: `scripts/run_teil2.sh` (blocks `pilot`, `main`, `gaps`, `latency`, `probes`, `fwf`). Analyses: `uv run python -m dma.analysis.report`, `dma.analysis.gaps`, `dma.analysis.probes_teil2`, `dma.analysis.teil2`, `dma.analysis.finetune`, `dma.analysis.figures`.
