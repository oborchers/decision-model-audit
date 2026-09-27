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
