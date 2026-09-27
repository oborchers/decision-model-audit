"""Local model runners (no API calls, no cost).

    uv run python -m dma.runners.local --system <id> --task <task.json> --items <items.jsonl> \
        --variant <variant> --out <results.jsonl> [--train <train.jsonl>] [--device mps|cpu]

Appends one ``result_row`` per item. Each model is loaded once per process. One warm-up call on the
first item is made before timing and is excluded from the results. Latency is wall time per item at
batch size 1 and covers every model call the variant needs for that item (k calls for ``yesno``).

Systems (see notes/local-models.md for model revisions and exact probability definitions):
    gliner       fastino/GLiNER2.5-Decide, gliner2 ``Classifier`` (single-label softmax)
    gliner-long  same model, ``Classifier.classify_long`` (chunk 384, overlap 64 words, max-aggregated logits)
    gliner-1b    fastino/GLiNER2.5-Decide-1B, same API as ``gliner``
    laya         convaiinnovations/laya (English ModernBERT checkpoint, max_len 512)
    laya-long    convaiinnovations/laya subfolder multilingual, max_len=8192 (documented long-document path)
    nli          MoritzLaurer/deberta-v3-large-zeroshot-v2.0, transformers zero-shot pipeline
    gliclass     knowledgator/gliclass-large-v3.0, gliclass uni-encoder pipeline
    qwen-lp      Qwen3.5-4B (MLX 4-bit), softmax over the logprobs of the option letters
    tfidf        TF-IDF + logistic regression trained on --train (choice variants only)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import os
import resource
import sys
import time
from pathlib import Path

from dma.tasks import Task, load_items, result_row, write_rows

LETTERS = "ABCDEFGHIJ"
CHOICE_VARIANTS = ("choice", "choice_none", "reversed")


# --------------------------------------------------------------------------------------------
# Shared prompt pieces. Kept here so that other runners can import the same wording.
# --------------------------------------------------------------------------------------------

def _strip_period(s: str) -> str:
    s = s.strip()
    return s[:-1] if s.endswith(".") else s


def yesno_question(task: Task, label: str, description: str) -> str:
    """The yes/no question asked once per label in the ``yesno`` variant."""
    return f'{task.instructions} Is the answer "{label}": {_strip_period(description)}?'


def _no_parens(s: str) -> str:
    """GLiNER2 rejects '(' and ')' in labels, descriptions and instructions (reserved prompt
    tokens). '(x)' becomes ', x,' ; applied only by the gliner runners, recorded per row."""
    out = re.sub(r"\s*\(([^()]*)\)", r", \1,", s).replace("(", ",").replace(")", ",")
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r",\s*([.?!:;])", r"\1", out)
    return out.strip().rstrip(",").strip()


def noul_question(task: Task, item: dict) -> str:
    return item.get("question") or task.instructions


def is_choice_variant(v: str) -> bool:
    return v in CHOICE_VARIANTS or (v.startswith("para") and v[4:].isdigit())


def hf_revision(repo: str) -> str | None:
    """Commit hash of the locally cached snapshot of ``repo`` (refs/main)."""
    try:
        from huggingface_hub import constants
        ref = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        return ref.read_text().strip()
    except Exception:
        return None


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def _torch_device(pref: str | None):
    import torch
    if pref:
        return pref
    return "mps" if torch.backends.mps.is_available() else "cpu"


# --------------------------------------------------------------------------------------------
# Runners. Each implements choice(text, task, label_map) -> (probs, extra),
# yes(text, question_or_label...) -> (p_yes, extra) via yesno()/noul().
# --------------------------------------------------------------------------------------------

class Runner:
    system: str = ""
    model_id: str = ""
    supports = ("choice", "choice_none", "reversed", "para", "yesno", "noul")

    def __init__(self, device: str | None = None, **kw):
        self.device_pref = device

    def meta(self) -> dict:
        return {"model_id": self.model_id, "model_revision": hf_revision(self.model_id)}

    def choice(self, text: str, task: Task, labels: dict[str, str]) -> tuple[dict, dict]:
        raise NotImplementedError

    def yesno(self, text: str, task: Task, labels: dict[str, str]) -> tuple[dict, dict]:
        """Default: one independent yes/no call per label."""
        probs, extra = {}, {}
        for lab, desc in labels.items():
            p, _ = self.noul(text, yesno_question(task, lab, desc))
            probs[lab] = p
        return probs, extra

    def noul(self, text: str, question: str) -> tuple[float, dict]:
        raise NotImplementedError

    def memory(self) -> dict:
        return {}


# ---- GLiNER2.5-Decide ----------------------------------------------------------------------

class GlinerRunner(Runner):
    system = "gliner"
    model_id = "fastino/GLiNER2.5-Decide"
    long = False
    # gliner2 does not truncate (max_len=None) and attention memory grows quadratically.
    # Measured on the M1 Pro 16 GB: 3.2k tokens = 9 s CPU / 30 s and 14 GB on MPS;
    # 7k tokens on CPU swapped for >10 min; 19k tokens on MPS failed (22.6 GiB buffer).
    # Longer inputs are refused (valid=False) instead of risking a swap storm.
    max_tokens = 4096

    def __init__(self, device=None, **kw):
        super().__init__(device)
        import torch
        from gliner2.classification import Classifier, ClassificationConfig, ClassificationSchema
        self._Schema = ClassificationSchema
        self.device = _torch_device(device)
        self.clf = Classifier.from_pretrained(self.model_id)
        self.clf.to(device=self.device)
        self.clf.eval()
        # independent decoder: a single task has no constraints; max_len=None = library default
        self.cfg = ClassificationConfig(decoder="independent", include_confidence=True)
        self._torch = torch

    def _run(self, text, schema):
        if not self.long and self.max_tokens:
            n = len(self.clf.model.processor.tokenizer(text, add_special_tokens=False)["input_ids"])
            if n > self.max_tokens:
                raise MemoryError(f"input has {n} tokens > runner limit {self.max_tokens} "
                                  f"(gliner2 does not truncate; gliner-long chunks long inputs)")
        if self.long:
            return self.clf.classify_long(text, schema, config=self.cfg,
                                          chunk_size=384, chunk_overlap=64, aggregate="max")
        return self.clf.classify(text, schema, config=self.cfg)

    def choice(self, text, task, labels):
        clean = {k: _no_parens(v) for k, v in labels.items()}
        ins = _no_parens(task.instructions)
        schema = self._Schema().single("label", clean, instruction=ins)
        res = self._run(text, schema)
        pr = dict(res.probabilities("label"))
        probs = {k: float(pr[k]) for k in labels}
        extra = {}
        if clean != dict(labels) or ins != task.instructions:
            extra["parens_removed"] = True
        return probs, extra

    def noul(self, text, question):
        q = _no_parens(question)
        schema = self._Schema().single("answer", ["yes", "no"], instruction=q)
        res = self._run(text, schema)
        pr = dict(res.probabilities("answer"))
        return float(pr["yes"]), ({"parens_removed": True} if q != question else {})

    def memory(self):
        return _torch_mem(self.device)


class GlinerLongRunner(GlinerRunner):
    system = "gliner-long"
    long = True
    max_tokens = None


class Gliner1BRunner(GlinerRunner):
    system = "gliner-1b"
    model_id = "fastino/GLiNER2.5-Decide-1B"
    # ModernBERT (Ettin) encoder, max_position_embeddings 7999. 19k tokens ran (106 s, 12 GB
    # MPS) but past the trained positions; refused above 7999 tokens.
    max_tokens = 7999


# ---- Laya -----------------------------------------------------------------------------------

class LayaRunner(Runner):
    system = "laya"
    model_id = "convaiinnovations/laya"
    subfolder = None
    max_len = None  # checkpoint default (rl_agent_config.json)

    def __init__(self, device=None, **kw):
        super().__init__(device)
        from laya import Agent
        self.device = _torch_device(device)
        self.agent = Agent(self.model_id, device=self.device, subfolder=self.subfolder)

    def meta(self):
        m = super().meta()
        if self.subfolder:
            m["model_subfolder"] = self.subfolder
        m["max_len"] = self.max_len or self.agent.cfg.get("max_len")
        return m

    def _predict(self, text, questions):
        return self.agent.predict(text, questions, max_len=self.max_len)["answers"]

    def choice(self, text, task, labels):
        ans = self._predict(text, {"q": {"type": "choice", "instructions": task.instructions,
                                          "criteria": dict(labels)}})["q"]
        probs = {k: float(ans["probabilities"][k]) for k in labels}
        return probs, {"laya_confidence": ans.get("confidence"),
                       "laya_answer_confidence": ans.get("answer_confidence")}

    def yesno(self, text, task, labels):
        # One noul question per label in one call. Laya encodes every question as its own row
        # ([CLS] question [SEP] options [SEP] state), so this equals k separate calls.
        qs = {f"q{i}": {"type": "noul", "instructions": yesno_question(task, lab, desc)}
              for i, (lab, desc) in enumerate(labels.items())}
        ans = self._predict(text, qs)
        return {lab: float(ans[f"q{i}"]["noul"]) for i, lab in enumerate(labels)}, {}

    def noul(self, text, question):
        ans = self._predict(text, {"q": {"type": "noul", "instructions": question}})["q"]
        return float(ans["noul"]), {}

    def memory(self):
        return _torch_mem(self.device)


class LayaLongRunner(LayaRunner):
    system = "laya-long"
    subfolder = "multilingual"
    max_len = 8192


# ---- NLI (Laurer DeBERTa-v3-large zero-shot v2.0) ------------------------------------------

class NLIRunner(Runner):
    system = "nli"
    model_id = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    template = "This text is about {}."

    def __init__(self, device=None, **kw):
        super().__init__(device)
        from transformers import pipeline
        self.device = _torch_device(device)
        self.pipe = pipeline("zero-shot-classification", model=self.model_id, device=self.device)

    def _classes(self, labels):
        return [_strip_period(d) for d in labels.values()]

    def choice(self, text, task, labels):
        classes = self._classes(labels)
        if len(set(classes)) != len(classes):
            raise ValueError("duplicate label descriptions")
        out = self.pipe(text, classes, hypothesis_template=self.template, multi_label=False)
        sc = dict(zip(out["labels"], out["scores"]))
        return {lab: float(sc[c]) for lab, c in zip(labels, classes)}, {}

    def yesno(self, text, task, labels):
        classes = self._classes(labels)
        out = self.pipe(text, classes, hypothesis_template=self.template, multi_label=True)
        sc = dict(zip(out["labels"], out["scores"]))
        return {lab: float(sc[c]) for lab, c in zip(labels, classes)}, {}

    def noul(self, text, question):
        # The question itself is the hypothesis (template "{}"): NLI has no native question form.
        out = self.pipe(text, [question], hypothesis_template="{}", multi_label=True)
        return float(out["scores"][0]), {"hypothesis": question}

    def memory(self):
        return _torch_mem(self.device)


# ---- GLiClass large v3.0 --------------------------------------------------------------------

class GLiClassRunner(Runner):
    system = "gliclass"
    model_id = "knowledgator/gliclass-large-v3.0"

    def __init__(self, device=None, **kw):
        super().__init__(device)
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
        self.device = _torch_device(device)
        model = GLiClassModel.from_pretrained(self.model_id)
        tok = AutoTokenizer.from_pretrained(self.model_id, add_prefix_space=True)
        # multi-label = independent sigmoid per label; normalised below for choice variants
        import torch
        # gliclass maps any device *string* other than cuda to cpu, so pass a torch.device
        self.pipe = ZeroShotClassificationPipeline(model, tok, classification_type="multi-label",
                                                   device=torch.device(self.device),
                                                   progress_bar=False)

    @staticmethod
    def _label_text(lab, desc):
        return f"{lab}: {_strip_period(desc)}"

    def _scores(self, text, label_texts, prompt):
        out = self.pipe(text, label_texts, threshold=0.0, prompt=prompt)[0]
        return {r["label"]: float(r["score"]) for r in out}

    def choice(self, text, task, labels):
        lt = {lab: self._label_text(lab, d) for lab, d in labels.items()}
        raw = self._scores(text, list(lt.values()), task.instructions)
        raw = {lab: raw[t] for lab, t in lt.items()}
        s = sum(raw.values())
        probs = {k: (v / s if s > 0 else 1 / len(raw)) for k, v in raw.items()}
        return probs, {"raw_scores": raw}

    def yesno(self, text, task, labels):
        probs = {}
        for lab, d in labels.items():
            t = self._label_text(lab, d)
            probs[lab] = self._scores(text, [t], task.instructions)[t]
        return probs, {}

    def noul(self, text, question):
        # Model card: for NLI-type use put the hypothesis/question as the single label.
        return self._scores(text, [question], None)[question], {}

    def memory(self):
        return _torch_mem(self.device)


# ---- Qwen3.5-4B via MLX, letter logprobs ---------------------------------------------------

class QwenLPRunner(Runner):
    system = "qwen-lp"
    model_id = "mlx-community/Qwen3.5-4B-MLX-4bit"
    base_model = "Qwen/Qwen3.5-4B"
    prefill_step = 1024

    def __init__(self, device=None, **kw):
        super().__init__(device)
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.model, self.tok = load(self.model_id)
        self.letter_ids = {}
        vocab = self.tok.get_vocab() if hasattr(self.tok, "get_vocab") else None
        for L in LETTERS:
            ids = set()
            for form in (L, " " + L):
                enc = self.tok.encode(form, add_special_tokens=False)
                if len(enc) != 1:
                    raise RuntimeError(f"letter {form!r} is not a single token: {enc}")
                ids.add(enc[0])
            self.letter_ids[L] = sorted(ids)

    def meta(self):
        m = super().meta()
        m["base_model"] = self.base_model
        m["base_model_revision"] = None  # weights come from the MLX conversion only
        return m

    def _prompt(self, system: str, text: str) -> list[int]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        return self.tok.encode(s, add_special_tokens=False)

    def _last_logprobs(self, ids: list[int]):
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        arr = mx.array(ids)
        n = len(ids)
        pos = 0
        # chunked prefill so that full-sequence logits (L x vocab) are never materialised
        while n - pos > self.prefill_step:
            self.model(arr[pos:pos + self.prefill_step][None], cache=cache)
            mx.eval([c.state for c in cache])
            pos += self.prefill_step
        logits = self.model(arr[pos:][None], cache=cache)[0, -1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        mx.eval(lp)
        return lp

    def _letters(self, system, text, k):
        lp = self._last_logprobs(self._prompt(system, text))
        scores, mass = [], 0.0
        for L in LETTERS[:k]:
            vals = [float(lp[i].item()) for i in self.letter_ids[L]]
            m = max(vals)
            s = m + math.log(sum(math.exp(v - m) for v in vals))
            scores.append(s)
            mass += math.exp(s)
        top_id = int(self.mx.argmax(lp).item())
        return _softmax(scores), {"letter_mass": round(mass, 6),
                                  "top_token": self.tok.decode([top_id])}

    @staticmethod
    def _system(instructions: str, options: list[str]) -> str:
        opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
        return (f"{instructions}\n\nOptions:\n{opts}\n\n"
                f"The user message contains the text to decide on. "
                f"Answer with the letter of the correct option only.")

    def choice(self, text, task, labels):
        if len(labels) > len(LETTERS):
            raise ValueError(f"at most {len(LETTERS)} options supported")
        opts = [f"{lab}: {_strip_period(d)}" for lab, d in labels.items()]
        p, extra = self._letters(self._system(task.instructions, opts), text, len(opts))
        return dict(zip(labels, p)), extra

    def noul(self, text, question):
        p, extra = self._letters(self._system(question, ["yes", "no"]), text, 2)
        return p[0], extra

    def memory(self):
        mx = self.mx
        try:
            return {"mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 3)}
        except AttributeError:
            return {"mlx_peak_gb": round(mx.metal.get_peak_memory() / 1e9, 3)}


# ---- TF-IDF + logistic regression ----------------------------------------------------------

class TfidfRunner(Runner):
    system = "tfidf"
    model_id = "sklearn TfidfVectorizer+LogisticRegression"
    supports = ("choice", "choice_none", "reversed", "para")

    def __init__(self, device=None, train=None, task: Task | None = None, **kw):
        super().__init__(device)
        if not train:
            raise SystemExit("tfidf needs --train <train.jsonl>")
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.pipeline import make_pipeline
        rows = load_items(train)
        X = [r["text"] for r in rows]
        y = [r["label"] for r in rows]
        if task is not None:
            unknown = sorted(set(y) - set(task.labels))
            if unknown:
                raise SystemExit(f"train labels not in task label set: {unknown}")
        pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
                             LogisticRegression(max_iter=5000))
        grid = {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260927)
        self.gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_log_loss", n_jobs=1)
        self.gs.fit(X, y)
        self.classes = list(self.gs.best_estimator_.classes_)
        self.train_info = {"train_file": str(train), "n_train": len(X),
                           "C": self.gs.best_params_["logisticregression__C"],
                           "cv_neg_log_loss": round(float(self.gs.best_score_), 4),
                           "sklearn": sklearn.__version__}

    def meta(self):
        return {"model_id": self.model_id, **self.train_info}

    def choice(self, text, task, labels):
        p = self.gs.predict_proba([text])[0]
        pr = dict(zip(self.classes, map(float, p)))
        probs = {lab: pr.get(lab, 0.0) for lab in labels}
        extra = {}
        missing = [lab for lab in labels if lab not in pr]
        if missing:
            extra["note"] = f"labels not learnable from training data, probability 0: {missing}"
        return probs, extra


# --------------------------------------------------------------------------------------------

SYSTEMS = {c.system: c for c in (GlinerRunner, GlinerLongRunner, Gliner1BRunner, LayaRunner,
                                  LayaLongRunner, NLIRunner, GLiClassRunner, QwenLPRunner,
                                  TfidfRunner)}


def _torch_mem(device) -> dict:
    import torch
    out = {}
    if str(device).startswith("mps"):
        try:
            out["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 3)
        except Exception:
            pass
    return out


def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 1e9 if sys.platform == "darwin" else r / 1e6, 3)


def check_variant(runner: type[Runner], task: Task, variant: str) -> None:
    kind = "para" if variant.startswith("para") else variant
    if kind == "rationale":
        raise SystemExit("rationale is an LLM-only variant")
    if kind not in runner.supports:
        raise SystemExit(f"{runner.system} does not support variant {variant}")
    if task.type == "noul" and variant != "noul":
        raise SystemExit(f"noul task only supports variant noul, got {variant}")
    if task.type == "choice" and variant == "noul":
        raise SystemExit("variant noul needs a noul task")
    if variant == "choice_none" and not task.none_label:
        raise SystemExit("task has no none_label")
    if kind == "para" and int(variant[4:]) >= len(task.paraphrases):
        raise SystemExit(f"task has no paraphrase {variant[4:]}")


def predict(runner: Runner, task: Task, item: dict, variant: str) -> dict:
    """Return kwargs for result_row (pred, probs, confidence, extra)."""
    text = item["text"]
    if variant == "noul":
        p, extra = runner.noul(text, noul_question(task, item))
        return {"pred": p >= 0.5, "probs": {"yes": p}, "confidence": max(p, 1 - p), "extra": extra}
    labels = task.label_map(variant if variant != "yesno" else "choice")
    if variant == "yesno":
        probs, extra = runner.yesno(text, task, labels)
    else:
        probs, extra = runner.choice(text, task, labels)
    pred = max(probs, key=probs.get)
    if variant == "choice_none" and runner.system == "tfidf":
        extra = {**extra, "note": "tfidf cannot output none; probs cover trained labels only"}
    return {"pred": pred, "probs": probs, "confidence": probs[pred], "extra": extra}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--task", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", help="training JSONL for tfidf")
    ap.add_argument("--device", help="torch device override (mps or cpu)")
    ap.add_argument("--limit", type=int, help="only the first N items")
    args = ap.parse_args(argv)

    task = Task.load(args.task)
    items = load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    n_all = len(items)
    items = [it for it in items if it.get("task", task.name) == task.name]
    if len(items) < n_all:
        print(f"skipped {n_all - len(items)} items of other tasks", file=sys.stderr)
    if not items:
        raise SystemExit(f"no items for task {task.name}")

    check_variant(SYSTEMS[args.system], task, args.variant)
    t0 = time.perf_counter()
    runner = SYSTEMS[args.system](device=args.device, train=args.train, task=task)
    load_s = time.perf_counter() - t0
    meta = runner.meta()

    # warm-up on the first item (cut to 64 words so a long item does not run twice), excluded
    warm = {**items[0], "text": " ".join(items[0]["text"].split()[:64])}
    try:
        predict(runner, task, warm, args.variant)
    except Exception:  # the timed calls below record the error per row
        import traceback
        print("warm-up failed:\n" + traceback.format_exc(), file=sys.stderr)

    lat = []
    n_err = 0
    for it in items:
        t = time.perf_counter()
        try:
            r = predict(runner, task, it, args.variant)
            dt = time.perf_counter() - t
            row = result_row(it, args.system, args.variant, pred=r["pred"], probs=r["probs"],
                             confidence=r["confidence"], latency_s=round(dt, 4), cost_usd=0.0,
                             extra={**meta, **r["extra"]})
            lat.append(dt)
        except Exception as e:
            dt = time.perf_counter() - t
            n_err += 1
            row = result_row(it, args.system, args.variant, latency_s=round(dt, 4), cost_usd=0.0,
                             valid=False, extra={**meta, "error": f"{type(e).__name__}: {e}"[:2000]})
        write_rows(args.out, [row])

    lat.sort()
    summary = {"system": args.system, "variant": args.variant, "n": len(items), "errors": n_err,
               "load_s": round(load_s, 2),
               "latency_median_s": round(lat[len(lat) // 2], 4) if lat else None,
               "latency_max_s": round(lat[-1], 4) if lat else None,
               "peak_rss_gb": peak_rss_gb(), **runner.memory()}
    print(json.dumps(summary), file=sys.stderr)


if __name__ == "__main__":
    main()
