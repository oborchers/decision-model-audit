# Local models

How the local systems of protocol section 3 are run, what their probabilities mean, and where they deviate from the protocol. Runner: `src/dma/runners/local.py`.

```
uv run python -m dma.runners.local --system <id> --task <task.json> --items <items.jsonl> \
    --variant <variant> --out <results.jsonl> [--train <train.jsonl>] [--device mps|cpu] [--limit N]
```

One `result_row` is appended per item, flushed after every item. Each model is loaded once per process. Before timing, one warm-up call is made on the first item cut to 64 words; it is not written. `latency_s` is wall time per item at batch size 1 and covers every model call the variant needs for that item (k calls for `yesno` with k labels). `cost_usd` is 0. A failure gives `valid=false` with `error`. Every row carries `model_id` and `model_revision` (the Hugging Face commit of the cached snapshot). A JSON summary (median and max latency, load time, peak memory) goes to stderr.

## Models and revisions

| System | Model | Revision (HF commit) | Encoder / size | Device |
|---|---|---|---|---|
| `gliner` | `fastino/GLiNER2.5-Decide` | `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6` | DeBERTa-v3-large, 340M, fp32 | mps |
| `gliner-long` | same | same | same | mps |
| `gliner-1b` | `fastino/GLiNER2.5-Decide-1B` | `52c94d3b698bf6d2619df9d898bdc1523ea3f1ca` | ModernBERT (`jhu-clsp/ettin-enc-from-dec-1b`), fp32 | mps |
| `laya` | `convaiinnovations/laya` (repo root = English checkpoint) | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | ModernBERT-large, 421M | mps |
| `laya-long` | `convaiinnovations/laya`, subfolder `multilingual` | same | mmBERT-base, 322M | mps |
| `nli` | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` | `cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2` | DeBERTa-v3-large, 435M | mps |
| `gliclass` | `knowledgator/gliclass-large-v3.0` | `e065d1844f913a9aa611cf33623a9538b8aa8841` | DeBERTa-v3-large uni-encoder, 439M | mps |
| `qwen-lp` | `mlx-community/Qwen3.5-4B-MLX-4bit` (conversion of `Qwen/Qwen3.5-4B`) | `32f3e8ecf65426fc3306969496342d504bfa13f3` | 4-bit affine, group 64 | MLX (Metal) |
| `tfidf` | scikit-learn TF-IDF + logistic regression | trained per run on `--train` | | cpu |

The Laya package names its models under `convaiinnovations/*`; the code lives on GitHub under `NandhaKishorM/laya`. The Qwen weights come only from the MLX conversion; the base repo `Qwen/Qwen3.5-4B` was at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on 2026-09-27, but the conversion does not state which base commit it was made from. The conversion is the vision-language one (`preprocessor_config.json`, vision weights); `mlx-lm` drops the vision tower on load and runs the text model only.

Package versions (uv.lock): torch 2.14.0, transformers 5.17.0, gliner2 2.0.0, peft 0.21.0 (import-time dependency of gliner2, not declared by it), laya 0.3.20, gliclass 0.1.20, mlx 0.32.2, mlx-lm 0.31.3, scikit-learn 1.9.1, tokenizers 0.23.2, huggingface-hub 1.33.0. Python 3.12.12, macOS on Apple M1 Pro 16 GB.

## Shared prompt pieces

- Label names and descriptions come from `Task.label_map(variant)`; `yesno` uses the base label order.
- `yesno` asks, once per label: `{instructions} Is the answer "{label}": {description without final period}?` (`yesno_question` in the runner, importable by other runners). `pred` = label with the highest yes probability, `probs` = per-label yes probabilities (they do not sum to 1).
- `noul`: the question is `item["question"]` if present, else `task.instructions`. `probs = {"yes": p}`, `pred = p >= 0.5`, `confidence = max(p, 1-p)`.
- Choice variants: `pred` = argmax of `probs`, `confidence` = its probability.

## How probabilities are obtained

**gliner / gliner-1b.** `gliner2.classification.Classifier` with one `.single("label", {label: description}, instruction=task.instructions)` task, decoder `independent`. `probs` = `result.probabilities("label")`: softmax over the per-label logits (activation `auto` = softmax for an exclusive task, temperature 1.0). This is the documented API that returns scores for all labels; `classify_text` returns only the winner and its confidence. `noul`: `.single("answer", ["yes", "no"], instruction=question)`, which is the model card's "question over a passage" pattern; `p` = softmax probability of `yes`. `yesno`: one such call per label with the `yesno_question`.

**gliner-long.** Same schema through `Classifier.classify_long(chunk_size=384, chunk_overlap=64, aggregate="max")`, the documented long-document path (tutorial 12 and 14). The text is split into whitespace-word chunks of 384 with 64 overlap; per-label *logits* are max-aggregated over chunks, then softmax is applied once. For `noul`, the `yes` and `no` logits are maxed independently, so the aggregate can mix chunks. On short inputs (one chunk) it equals `gliner`.

**laya.** `laya.Agent("convaiinnovations/laya")`, i.e. the English checkpoint pinned directly instead of the recommended `Router`, so that routing heuristics cannot switch checkpoints between items (the Router would send English text to the same checkpoint unless the question schema matches one of its typed-decisions workflows). Choice: one `choice` question with `instructions` and `criteria` = label map; `probs` = returned `probabilities` (temperature-scaled softmax with the checkpoint's per-option-count temperature). `noul`: one `noul` question, `p` = returned `noul`. `yesno`: all k `noul` questions in one `predict` call; Laya encodes each question as its own sequence, so this equals k separate calls. The checkpoint's temperature for the `choice:11+` bucket (0.10) is outside the package's valid range and is clamped to 0.5 with a warning; it only affects questions with 11 or more options, not S1 (8 or 9).

**laya-long.** Multilingual checkpoint with `max_len=8192` per call, as the Laya README prescribes for long documents.

**nli.** transformers `zero-shot-classification` pipeline, hypothesis template `This text is about {}.`, class text = label description without its final period (label names are not shown). Choice variants: `multi_label=False`, i.e. softmax of the entailment logits across labels. `yesno`: `multi_label=True`, per label softmax over [not_entailment, entailment] (the v2.0 model is binary: `0 entailment, 1 not_entailment`). `noul`: the question itself is the hypothesis (template `{}`), `multi_label=True`. NLI has no question form; a question as hypothesis is outside its training distribution (see limitations).

**gliclass.** `ZeroShotClassificationPipeline` (uni-encoder), `classification_type="multi-label"`, `threshold=0`, `prompt=task.instructions`. Each label is passed as the string `"{label}: {description}"`. Raw scores are independent sigmoids. Choice variants: `probs` = raw scores divided by their sum; raw scores are kept in `raw_scores`. `yesno`: one call per label with that label alone; `probs` = raw sigmoid. `noul`: the question is the single label and no prompt is given (model card: "represent your premise as a text and hypothesis as a label").

**qwen-lp.** mlx-lm, chat template with `enable_thinking=False` (the template inserts an empty `<think></think>` block). System message: instructions, then `Options:` with `A) label: description` lines, then "The user message contains the text to decide on. Answer with the letter of the correct option only." User message: the text. The prompt ends at the assistant turn, so the next token is the answer letter. Letters A to H and " A" to " H" are all single tokens (ids 32 to 39 and 357, 417, 351, 414, 458, 426, 469, 462); a letter's log-probability is the logsumexp of its two variants, and `probs` = softmax over the offered letters. `letter_mass` (total probability of all offered letter tokens over the full vocabulary) and `top_token` are recorded per row to detect off-format answers; on the toy items `letter_mass` was at least 0.993. `noul` and `yesno`: options `A) yes`, `B) no`, `p` = probability of A. The prompt is prefilled in chunks of 1,024 tokens through the KV cache, so full-sequence logits are never materialised.

**tfidf.** `TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)` + `LogisticRegression(max_iter=5000)`, C chosen from {0.01, 0.1, 1, 10, 100} by 5-fold stratified CV (seed 20260927, scoring log loss) on the `--train` file only, then refit on all of it. `probs` = `predict_proba`. Rows record `C`, CV log loss, train size and file. Supports `choice`, `choice_none`, `reversed`, `para<k>`; all give identical predictions because the model never sees descriptions or order. `choice_none`: `none` gets probability 0 and cannot be predicted (row `note`). `yesno` and `noul` are refused.

## Input length and truncation

Measured with toy items made of repeated Federal Register filler and one decisive sentence (3,000 words = about 3,250 DeBERTa tokens with the sentence at 50%; 18,000 words = about 19,450 tokens with the sentence at 95%).

| System | Window | What happens beyond it |
|---|---|---|
| `gliner` | none enforced (gliner2 `max_len=None`, DeBERTa relative positions accept any length) | Whole text encoded, memory grows quadratically. 3.2k tokens: 30 s and 6.7 to 14 GB Metal memory on MPS, 9.4 s on CPU. 19k tokens on MPS failed with `Invalid buffer size: 22.60 GiB`; 7k tokens on CPU swapped for over 10 minutes (17 GB swap) and was killed. **Runner limit:** inputs over 4,096 tokens are refused with `valid=false` (`MemoryError`). |
| `gliner-long` | 384-word chunks, 64 overlap | Nothing is dropped; 3k words 5.5 s (MPS) or 3.9 s (CPU); 18k words 36 s (MPS). |
| `gliner-1b` | ModernBERT `max_position_embeddings` 7999 | 19k tokens ran on MPS (106 s, 12 GB) but past the trained positions and wrong on the toy item. **Runner limit:** over 7,999 tokens refused with `valid=false`. |
| `laya` | 512 tokens total (config `max_len`) | Sequence is `[CLS] question [SEP] options [SEP] state [SEP]`; question plus options are capped at 192 tokens (`head_max_len`), each option description at its first 48 tokens; the state (text) is right-truncated to the remaining room, so only the beginning of the text is seen. |
| `laya-long` | 8,192 tokens (`max_len=8192`, `head_max_len` 256) | Same layout, text right-truncated at 8,192 total. |
| `nli` | 512 tokens (tokenizer `model_max_length`) | Pair encoding with `truncation="only_first"`: the text (premise) is cut to fit 512 with the hypothesis; the hypothesis is kept whole. All three long toy items got the identical score 0.836. |
| `gliclass` | 1,024 tokens (pipeline `max_length`) | Input order is labels, `<<SEP>>`, prompt, text (`prompt_first=true`); truncation cuts the end, i.e. the text beyond about 1,024 tokens minus label and prompt tokens. |
| `qwen-lp` | 262,144 tokens | No truncation. 3.2k tokens 10 s, 19k tokens 73 s, MLX peak 5.3 GB. |
| `tfidf` | none | Bag of n-grams over the whole text. |

Recommendation for P3: run `gliner-long` (and optionally `gliner` for the cells up to 4,096 tokens), and run the gliner family with `--device cpu` for long items, which was 1.4 to 3.3 times faster than MPS there. On short items MPS is faster.

## Latency and memory (toy run)

Toy task: 3-label support-ticket choice task (with `none` label and one paraphrase set) and 6 items; `noul` task with 6 items, 2 of them with their own question (urn, raffle). Median latency per item in seconds, M1 Pro, default device, after warm-up. Peak RSS from `getrusage`; Metal memory from `torch.mps.driver_allocated_memory()` at the end of the run, MLX from `mx.get_peak_memory()`.

| System | choice | choice_none | reversed | para0 | yesno (3 labels) | noul | long 3.2k tok | long 19k tok | Peak RSS GB | Metal/MLX GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `gliner` | 0.092 | 0.096 | 0.091 | 0.090 | 0.219 | 0.071 | 30.6 (CPU 9.4) | refused | 4.0 to 4.5 | 2.2 short, 6.7 to 14 long |
| `gliner-long` | 0.092 | | | | | 0.071 | 5.5 (CPU 3.9) | 35.4 | 4.5 | 2.2 short, 3.2 long |
| `gliner-1b` | 0.087 | 0.092 | 0.088 | 0.088 | 0.187 | 0.062 | 4.1 | refused | 5.0 to 6.3 | 5.4 short, 6.5 long |
| `laya` | 0.048 | 0.048 | 0.048 | 0.047 | 0.081 | 0.039 | 0.15 (truncated) | 0.17 (truncated) | 3.1 | 2.2 |
| `laya-long` | 0.025 | | | | | 0.020 | 0.72 | 3.3 (truncated at 8,192) | 2.9 | 1.3 short, 4.0 long |
| `nli` | 0.171 | 0.222 | 0.168 | 0.166 | 0.171 | 0.059 | 0.51 (truncated) | 0.51 (truncated) | 0.8 | 1.1 |
| `gliclass` | 0.083 | 0.087 | 0.083 | 0.082 | 0.197 | 0.064 | 1.88 (truncated) | 1.88 (truncated) | 0.7 | 2.2 short, 3.3 long |
| `qwen-lp` | 0.428 | 0.431 | 0.428 | 0.426 | 0.983 | 0.321 | 10.2 | 69.9 | 3.0 to 3.3 | 2.7 to 2.8 short, 5.3 long |
| `tfidf` | 0.0004 | 0.0003 | 0.0003 | 0.0003 | refused | refused | | | 0.16 | |

Toy accuracy (correct/items; `inv` = invalid rows) is only a smoke test, not evidence: gliner 6/6 on every choice variant and yesno, 4/6 noul; gliner-1b 4 to 5/6 choice, 6/6 noul; laya 4/6 choice, 2/6 choice_none (predicts `none` for 4 of 6 in-set tickets), 6/6 reversed and para0; nli 6/6 choice variants, 5/6 noul; gliclass 6/6 choice, 4/6 choice_none and yesno; qwen-lp 5 to 6/6 choice, 6/6 yesno, 5/6 noul; tfidf 6/6 (templated train set). On the long toy items only `gliner-long` (3/3) found the decisive sentence reliably; nli's 2/3 is an artefact (constant 0.836 on truncated filler).

Load times: `gliner` about 15 s, `gliner-1b` about 97 s (transformers first random-initialises the 1B ModernBERT encoder, 77 s in a profile, before loading the checkpoint), `laya` 4 s, `nli` 5 s, `gliclass` 6 to 8 s, `qwen-lp` 5 to 6 s, `tfidf` training time on the train file.

## Deviations from the protocol and open points

1. **Parentheses are removed for GLiNER.** gliner2 rejects `(` and `)` in labels, descriptions and instructions because they are reserved prompt tokens (`SchemaError`). The gliner runners rewrite `x (y)` as `x, y,` and mark such rows `parens_removed: true`. All other systems see the original text. The task files in `data/` as of 2026-09-27 (S1, S2, P1 to P4) contain no parentheses, so this rewrite does not fire on them; it matters only if descriptions change. (`[AGENCY]` in the S2 instructions is not a reserved token.)
2. **Runner length limits** for `gliner` (4,096 tokens) and `gliner-1b` (7,999 tokens), see above. These are refusals recorded as invalid rows, not truncation. In P3 (about 400, 1,600, 6,000 and 18,000 words) this makes `gliner` invalid on the 8,000 and 24,000-token cells and `gliner-1b` invalid on the 24,000-token cells; `gliner-long` covers all cells. Expected P3 runtime at the measured speeds: `qwen-lp` about 70 s per 24,000-token item (about 35 min for the 28 items), `gliner-long` about 35 s (MPS).
3. **Laya is pinned to the English checkpoint** via `Agent`, not the `Router`. `laya-long` is an extra system (not in the protocol table) for Q4.
4. **`gliner-long` and `laya-long`** are additional system IDs for Q4; they are the vendors' documented long-input paths.
5. **NLI `noul`** uses the question verbatim as the hypothesis, which is not a statement. On the toy probes it answered 0.07 and 0.09 for p = 0.3 and p = 0.8. Treat NLI `noul` results (P1, P3) as out-of-design.
6. **NLI class text is the description only**, as specified; label names are not shown to it. All other systems see names and descriptions.
7. **GLiClass `noul`** passes no prompt, only the question as the single label; choice variants use the instructions as prompt.
8. **tfidf** returns identical predictions for all choice variants and never predicts `none`; do not report it for Q3 or Q5.
9. **yesno wording** is defined in this runner (`yesno_question`); an LLM runner for the same variant should import it so the wording is shared.
10. **Latency includes tokenisation** of the full text; for GLiClass and NLI, tokenising an 18k-word text before truncation costs about 0.4 to 1.5 s.
11. **`gliner-1b` load time** is about 97 s per process (redundant random weight init inside transformers), so run its variants in few processes.

## Additional decision models (added after interim results)

Runner: `src/dma/runners/local_extra.py`. It uses the same CLI, row format, warm-up, latency definition and stderr summary as `local.py`, because it registers its systems in `local.SYSTEMS` (in its own process only) and calls `local.main`. `local.py` is unchanged. These systems are reported in a separate block, not in the pre-registered comparison (protocol changelog).

```
uv run python -m dma.runners.local_extra --system eikos-4b|kev-0.8b|kev-4b|semif-4b \
    --task <task.json> --items <items.jsonl> --variant <variant> --out <results.jsonl> [--device mlx|mps|cpu] [--limit N]
```

Every system runs its vendor's own inference code. The Eikos and SemIf code comes from their Hugging Face snapshots. The Kev, SemIf and jevbench code is vendored unmodified at pinned commits in `third_party/` (see `third_party/COMMITS.txt`, licences alongside). It is put on `sys.path` at load time and is not installed. The reason: `kev` pins `torch<2.9` and would conflict with the project's torch 2.14, and SemIf and jevbench would have to be installed with `--no-deps`. `pydantic` 2.13.5, which `kev.api` needs, is already in the environment.

### Models and revisions

| System | Model (HF commit) | Base | Code | Device | Vendor temperature / calibration |
|---|---|---|---|---|---|
| `eikos-4b` | `caiovicentino1/Eikos-4B-MLX-8bit` `5b1a88542bcf916a375a45418bc02af76f61e267` | `caiovicentino1/Eikos-4B` (`2b0f4d13…`, merged fine-tune of Qwen3.5-4B), MLX 8-bit, group 64 | shipped `mlx_decide.py` and `decision_core.py` from the snapshot | MLX | `calib.json` mode `T1`: w = 0, b = 0, so T = exp(0) = 1 for every question. Recorded per row as `vendor_temperature` and in `vendor_calibration`. |
| `kev-0.8b` | `jaredpalmer/kev-0.8b` `9a45d25eb2ab761841196625383fa1dff0e56c1e` | `Qwen/Qwen3.5-0.8B-Base` `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` + LoRA r16 + pointer head | github.com/jaredpalmer/kev `5920c5fe` (`kev/`) | MLX bf16 (kev.serve default on Apple Silicon) | "as served": pointer-head logits divided by T = 2.3511 from `head.pt` |
| `kev-4b` | `jaredpalmer/kev-4b` `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` | `Qwen/Qwen3.5-4B-Base` `1001bb4d826a52d1f399e183466143f4da7b741b` | same | MLX bf16 | T = 2.41 from `head.pt` (card). Not yet loaded here. |
| `semif-4b` | `vinci00/semif-qwen3.5-4b-mlx-4bit` `4ef6ec3e6e41f95ce6a6cf6deb0e6184753dddde` | frozen `Qwen/Qwen3.5-4B` `851bf6e8…`, MLX 4-bit affine, group 64, no fine-tuning | shipped `decision_mlx.py`; `semif_phase1` (SemIf `1f2dea3e`) and `jevbench.adapters.semif_direct` (jevbench `75e6224e`) | MLX | none: a raw softmax over the option-letter logits (T = 1). SemIf's per-workload temperature scaling has to be fitted on labelled workload data, and none ships with the model. It is not applied. |

### How probabilities are obtained

**eikos-4b.** `MLXDecider(snapshot).dist(text, q, options_of(q))`. The prompt style `letter-v1-semif` has a fixed system message and a JSON user message `{evidence, criterion, options[{letter, description: "label: description"}]}`. The chat template runs with `enable_thinking=False`. The readout takes the final hidden state times the tied embedding, restricted to the option-letter tokens A, B, …, then a softmax with the calibrated T. Choice variants: `criteria` is the label map. `noul`: `{"type": "noul", "instructions": question}` without criteria. `options_of` turns this into `yes`/`no`, shown as `true: yes` and `false: no`, and `p` = P(yes). `yesno`: the k yes/no questions go through the shipped `dist_many_cached` (the state prefix once, then the questions as one right-padded batch; the vendor documents this as equal to separate passes). `input_tokens` is recorded per row.

**kev-0.8b / kev-4b.** `Checkpoint(repo).load(device, opts)` with `kev.serve.main`'s defaults (bf16; `attn=sdpa` on MPS; `backend=auto`, which gives MLX for the hybrid Qwen3.5 base on Apple Silicon; the LoRA is merged in fp32 and rounded once). The request is built as `/v1/systemone` builds it: `SystemOneRequest(state=text, questions=…)`, then `kev.api.to_record` and `model.encode(..., max_state=65536, max_branch=73728)`, then `model.probs`. Choice: `{"type": "choice", "instructions": task.instructions, "criteria": label map}`. Options are rendered as `name: description` and `probs` = the pointer-head softmax per label. `kev_confidence` records TypeSafe's (p_max − 1/K)/(1 − 1/K), while row `confidence` = p_max as for every other system. `noul`: `{"type": "noul", "instructions": question}`, options `[no, yes]`, `p` = P(yes). `yesno`: all k noul questions in one request. Kev runs each question as its own causal row on the shared state, so this equals k separate requests. Differences from kev.serve: no HTTP layer, and no cross-request state-prefix cache (it only saves time). Probabilities are kept unrounded, whereas the API rounds to 4 decimals. `KEV_DATE_FACTS` stays off (the serve default); the row field `date_facts` records it. `--device mps` or `cpu` selects Kev's PyTorch backend, which is slow because MPS has no Gated DeltaNet kernels.

**semif-4b.** `decision_mlx.configure()` (vendor settings: Metal, cache limit 256 MiB, memory limit 12 GiB, seed 42), `mlx_lm.load(snapshot)`, then `predict(model, tok, "semif", task)`. jevbench's `SemIfDirectAdapter.build_request` maps the question. Choice: one option per label, described as `label: description`. `noul`: options `true`/`false`, described as `true: The proposition is true.` and `false: The proposition is false.` SemIf's `encode_prompt` then builds the same system message and JSON evidence/criterion/options payload as Eikos, with thinking disabled, and checks that the answer letters stay single tokens. Last-position logits are restricted to the letters, followed by a plain softmax. `yesno` uses the default loop, one call per label. `option_logits` and `input_tokens` are recorded per row. At most 16 options.

### Context window and truncation

| System | Window | What happens beyond it |
|---|---|---|
| `eikos-4b` | Base: 262,144 positions. Trained on inputs up to 32k tokens (card). | No truncation. The shipped `dist` prefills the whole prompt in one forward pass, without chunking, so memory grows with length. Long inputs are not yet measured on this machine. |
| `kev-0.8b`, `kev-4b` | Serving limit: 65,536 state tokens (`SERVE_MAX_STATE`). Trained on states up to 7,552 tokens (`max_state` in `training_config.json`; decision-v7 used 384). | The state is right-truncated to 65,536 tokens without an error (kev.serve encodes non-strictly). The row flag `state_truncated` records this. A question branch over its row limit raises `ContextOverflow`, which gives an invalid row. |
| `semif-4b` | 4,096 prompt tokens (SemIf's encoder budget) | Refused, no truncation: `ValueError`, which gives `valid=false`. In P3 this makes every cell above about 3,900 text tokens invalid (the 8k and 24k cells, and probably most 2k-token items once the prompt is added). |

### Declared training data (from the cards)

- **Eikos-4B** (card, `NOTICE`, dataset `caiovicentino1/eikos-decisions`): about 23k rows. The data is distilled from GLM-5.3-Flash at maximum reasoning effort. Items were written by Qwen3.8-27B and GLM-5.3-Flash and kept only where the blind teacher agreed with the gold answer. Programmatic items cover probability, calendar and finance arithmetic, trading and trade-finance rules, compositional rules and rulebooks. Answer verification uses GSM8K train (with Qwen3.5-0.8B solutions) and TAT-QA. Entity sentiment comes from FinEntity. There are long-context dossiers of 6k to 32k tokens made from other training items, and EN↔PT translations. Focus: finance, trading, trade finance, English and Portuguese. **arXiv, Federal Register or generic topic-classification sets are not declared.** Eval-only sets: MMLU-Pro, RewardBench, ASSIN2, BoolQ, Banking77, CLINC150, LegalBench, XNLI-es, CUAD, FinQA and others. The card reports 8-gram decontamination against its own evaluation sets only. The P1 urn and raffle probes are close to its programmatic "probability" family.
- **Kev-0.8B / Kev-4B** (cards, `training_config.json`, suite manifests in the Kev repo): `decision-v7` has 10,000 public records, 1,000 each from Banking77, BoolQ, AG News, MultiNLI, SST-5, Yelp full, TREC, DBpedia-14, Amazon reviews (en) and IMDB. It adds 896 policy minimal pairs and 1,680 generated rule-structure records. The deltas add: 1,425 generated date and unknowable records; `documents-v1` with 5,219 CFPB consumer-complaint narratives (2015 to 2024; in Kev-4B round 8 and Kev-0.8B round 15); `hard-v1` with 6,000 programmatic skill records; and `devtools-v1` with 5,320 records from CodeReviewer, CommitPackFT, FlakeFlagger and Aegis. **These include generic topic-classification sets (AG News, DBpedia-14, TREC) and support-style intent data (Banking77, CFPB complaints).** arXiv and the Federal Register are not declared. MultiNLI's government genre contains a few Federal Register-style sentences (seen in the suites). The labels come from the datasets or from programs. The card states that no Jev outputs were used.
- **SemIf** (card): no training or fine-tuning. The model is frozen Qwen3.5-4B, so its training data is Qwen's pre- and post-training data, which is undisclosed. The card says "evaluation overlap is unknown" and notes that public JevBench items were used in upstream development.

### Latency and memory (toy run)

Toy task and items as in the section above (6 ticket items, 6 noul items). These measurements ran **while the two main local lanes (qwen-lp on MLX, laya on MPS) were running and the machine was swapping** (memory pressure level "warning", 19 to 23 GB of swap in use). Latencies are therefore upper bounds and noisy; rerun them on an idle machine.

| System | choice | choice_none | reversed | para0 | yesno (3) | noul | Peak RSS GB | MLX peak GB | Load s |
|---|---|---|---|---|---|---|---|---|---|
| `kev-0.8b` (MLX bf16) | 0.120 | 0.191 | 0.138 | 0.148 | 0.184 | 0.099 | 0.8 to 1.6 | 2.57 | 10 to 21 |
| `semif-4b` | 8.5 to 19 s on 3 items (swap-bound); the process ended before its summary line | | | | | | | | |
| `eikos-4b` | not run yet (memory) | | | | | | | | |
| `kev-4b` | not run yet (memory) | | | | | | | | |

Toy accuracy (smoke test only): kev-0.8b 6/6 on choice, reversed, para0 and yesno; 5/6 on choice_none and noul. On the probes it gave the urn (p = 0.3) P(yes) = 0.60 and the raffle (p = 0.8) P(yes) = 0.66. semif-4b got 3/3 on the choice items it completed, with probabilities of at least 0.998.

**Memory feasibility on the M1 Pro 16 GB.** Expected footprints:
- Eikos-4B-MLX-8bit: 4.5 GB of weights.
- SemIf 4-bit: 2.2 GB of weights (card: 3.37 GiB peak MLX on its benchmark).
- Kev-4B bf16: 8.4 GB of base weights plus a per-tensor merge transient. The card says "~9 GB for serving"; an unmerged fp32 path would take about 17 GB. Kev-4B is not feasible next to the running main lanes (swap 20 to 23 of about 25 GB in use). On an otherwise idle machine it probably fits, but this is untested. Kev-0.8B needs 2.6 GB MLX peak.
- Long inputs will raise the Eikos and Kev peaks (the full-attention layers' KV and the unchunked prefill in Eikos).

### Deviations and open points

1. **Code not installed but vendored** at pinned commits (see above). SemIf's card pins mlx-lm at git commit `a63e24c3…`; this project uses mlx-lm 0.31.3 (release), which provides the same APIs `decision_mlx.py` imports. Parity against the vendor's recorded probabilities has not been checked yet.
2. **Kev code version.** The released checkpoints were trained at Kev commits `45923b7a` (0.8B) and `6d02f5d0` (4B). Serving here uses the current `5920c5fe`. Between them, the bodies of `encode` and `to_record` are unchanged. The serving state limit rose from 8,192 to 65,536 tokens: at the training commits, states over 8,192 tokens were right-truncated there, and now they are kept up to 65,536. Other changes: serving memory handling (per-tensor merge, MLX cache limit, rows per pass) and TypeSafe's confidence formula, which only affects `kev_confidence`.
3. **Kev runs on MLX, not MPS**, because that is kev.serve's Apple Silicon default. `--device mps` gives Kev's PyTorch backend if an MPS comparison is needed.
4. **Kev probabilities are unrounded.** The HTTP API rounds to 4 decimals, and `kev_confidence` is taken from the rounded API answer.
5. **SemIf has no calibration**, and **Eikos ships T = 1**. Neither is re-fitted here. The protocol does not refit any system.
6. **Eikos `yesno` uses the batched cached path** (`dist_many_cached`), and Kev puts k questions in one request. Laya does the same; the other systems make k calls.
7. **SemIf refuses prompts over 4,096 tokens** (invalid rows), in the same way the runner limits `gliner` and `gliner-1b`.
8. **Toy latencies were measured under heavy swap** and must be remeasured before they are reported (Q7). Eikos and Kev-4B are untested pending memory.
