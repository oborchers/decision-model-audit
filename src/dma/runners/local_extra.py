"""Runners for the open decision models added after the interim results (no API calls, no cost).

    uv run python -m dma.runners.local_extra --system <id> --task <task.json> --items <items.jsonl> \
        --variant <variant> --out <results.jsonl> [--device mlx|mps|cpu] [--limit N]

Same CLI, row format, warm-up, latency definition and summary as ``dma.runners.local`` (this module registers
its systems there and calls its ``main``). Every system runs the vendor's own inference code, not a
re-implementation (see notes/local-models.md, "Additional decision models"):

    eikos-4b   caiovicentino1/Eikos-4B-MLX-8bit, shipped ``mlx_decide.MLXDecider`` + ``decision_core.options_of``
               from the HF snapshot; letter-logit readout, ``calib.json`` temperature (T = 1).
    kev-0.8b   jaredpalmer/kev-0.8b (LoRA + pointer head on Qwen/Qwen3.5-0.8B-Base), vendored ``kev`` package
               (third_party/kev): ``Checkpoint.load`` with kev.serve's defaults (MLX on Apple Silicon, bf16),
               TypeSafe ``/v1/systemone`` request mapping (``kev.api``), temperature from ``head.pt``.
    kev-4b     jaredpalmer/kev-4b, same code on Qwen/Qwen3.5-4B-Base.
    semif-4b   vinci00/semif-qwen3.5-4b-mlx-4bit, shipped ``decision_mlx.predict`` (SemIf direct option-logit
               readout via vendored ``semif_phase1`` and jevbench's ``SemIfDirectAdapter``), 4,096-token limit,
               raw softmax (no temperature).

``--device``: default and ``mlx`` = the vendor's Apple Silicon path (MLX). For Kev, ``mps`` or ``cpu`` select
Kev's PyTorch backend instead (slow: no Gated DeltaNet kernels on MPS). Eikos and SemIf are MLX only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from dma.runners.local import Runner, hf_revision, yesno_question

THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party"
THIRD_PARTY_COMMITS = {"kev": "5920c5fe4ca8e0970ed4209ac2c9b8e18bea5109",
                       "semif": "1f2dea3e25379f9dfc98cb83c324f00ab5deda37",
                       "jevbench": "75e6224ed8103bbc3485ca74820a2eaf7ce8abe0"}


def _add_path(p) -> None:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def _snapshot(repo: str, **kw) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, **kw)


def _mlx_mem() -> dict:
    import mlx.core as mx
    try:
        return {"mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 3)}
    except AttributeError:
        return {"mlx_peak_gb": round(mx.metal.get_peak_memory() / 1e9, 3)}


# ---- Eikos-4B (MLX 8-bit) -------------------------------------------------------------------

class EikosRunner(Runner):
    """Eikos's shipped MLX decider. One ``dist`` call per question; ``yesno`` uses the shipped multi-question
    path ``dist_many_cached`` (state prefix once, the k yes/no questions as one batch)."""
    system = "eikos-4b"
    model_id = "caiovicentino1/Eikos-4B-MLX-8bit"
    base_model = "caiovicentino1/Eikos-4B (fine-tune of Qwen/Qwen3.5-4B)"

    def __init__(self, device=None, **kw):
        super().__init__(device)
        if device not in (None, "mlx"):
            raise SystemExit("eikos-4b runs on MLX only (vendor MLX build)")
        self.path = _snapshot(self.model_id)
        _add_path(self.path)
        from mlx_decide import MLXDecider  # shipped code, sets PROMPT_STYLE before importing decision_core
        self.d = MLXDecider(self.path)
        import decision_core as dc
        self.dc = dc
        self.cfg = json.loads((Path(self.path) / "decision_config.json").read_text())

    def meta(self):
        return {"model_id": self.model_id, "model_revision": Path(self.path).name,
                "base_model": self.base_model, "prompt_version": self.cfg.get("prompt_version"),
                "readout": self.cfg.get("readout"), "vendor_calibration": self.d.calib,
                "max_one_pass": self.cfg.get("max_one_pass")}

    def _dist(self, text, q):
        opts = self.dc.options_of(q)
        p, n = self.d.dist(text, q, opts)
        T = self.dc.temp_for(self.d.calib, 1.0, n, len(opts), q["type"])
        return p, {"input_tokens": n, "vendor_temperature": round(T, 6)}

    def choice(self, text, task, labels):
        q = {"type": "choice", "instructions": task.instructions, "criteria": dict(labels)}
        p, extra = self._dist(text, q)
        return {k: float(p[k]) for k in labels}, extra

    def noul(self, text, question):
        p, extra = self._dist(text, {"type": "noul", "instructions": question})
        return float(p["yes"]), extra

    def yesno(self, text, task, labels):
        qs = [{"type": "noul", "instructions": yesno_question(task, lab, desc)} for lab, desc in labels.items()]
        res = self.d.dist_many_cached(text, [(q, self.dc.options_of(q)) for q in qs])
        return ({lab: float(p["yes"]) for lab, (p, _) in zip(labels, res)},
                {"input_tokens": sum(n for _, n in res), "yesno_mode": "dist_many_cached"})

    def memory(self):
        return _mlx_mem()


# ---- Kev (LoRA + pointer head on Qwen3.5 base) -----------------------------------------------

class KevRunner(Runner):
    """Kev through its own loader and TypeSafe request mapping, as ``kev.serve`` answers a /v1/systemone request
    (without the HTTP layer and without the cross-request state-prefix cache, which only saves time)."""
    system = "kev-0.8b"
    model_id = "jaredpalmer/kev-0.8b"

    def __init__(self, device=None, **kw):
        super().__init__(device)
        _add_path(THIRD_PARTY / "kev")
        from dataclasses import replace

        import torch
        from kev.api import SystemOneRequest, to_answers, to_record, with_date_facts
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.device import default_device
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
        self._req, self._to_record, self._to_answers = SystemOneRequest, to_record, to_answers
        self._date_facts = with_date_facts if os.environ.get("KEV_DATE_FACTS", "0") == "1" else None
        self._limits = (SERVE_MAX_STATE, SERVE_MAX_BRANCH)
        # kev.serve.main's defaults, in the same order
        dev = default_device() if device in (None, "mlx") else device
        opts = LoadOptions.from_env()
        if device in ("mps", "cpu") and opts.backend is None:
            opts = replace(opts, backend="torch")        # explicit torch device: Kev's PyTorch backend
        if dev == "mps" and opts.attn is None:
            opts = replace(opts, attn="sdpa")
        if dev != "cpu" and opts.dtype is None:
            opts = replace(opts, dtype=torch.bfloat16)
        if opts.backend is None:
            opts = replace(opts, backend="auto")         # MLX for the hybrid Qwen3.5 bases on Apple Silicon
        self.device = dev
        self.ck = Checkpoint(self.model_id)
        self.tok, self.model = self.ck.load(dev, opts)
        self.opts = opts

    def meta(self):
        m = self.ck.meta
        return {"model_id": self.model_id, "model_revision": Path(self.ck.path).name,
                "base_model": m.base, "base_revision": m.base_revision,
                "backend": self.model.backend, "dtype": str(self.model.dtype),
                "device": self.device if self.model.backend == "torch" else "mlx",
                "vendor_temperature": float(self.model.head.temperature),
                "date_facts": self._date_facts is not None,
                "kev_code_commit": THIRD_PARTY_COMMITS["kev"]}

    def _ask(self, text, questions: dict):
        state = self._date_facts(text) if self._date_facts else text
        rec, qmeta = self._to_record(self._req(state=state, questions=questions))
        max_state, max_branch = self._limits
        enc = self.model.encode(self.tok, rec, max_state=max_state, max_branch=max_branch)
        ps = [p.float().tolist() for p in self.model.probs(enc)]
        answers = self._to_answers(ps, qmeta)
        info = {"input_tokens": len(enc["ids"]), "state_tokens": enc["seg"].count(0),
                "state_truncated": bool(enc["state_truncated"])}
        return ps, answers, info

    def choice(self, text, task, labels):
        ps, ans, info = self._ask(text, {"q": {"type": "choice", "instructions": task.instructions,
                                               "criteria": dict(labels)}})
        return dict(zip(labels, ps[0])), {**info, "kev_confidence": ans["q"]["confidence"]}

    def noul(self, text, question):
        ps, _, info = self._ask(text, {"q": {"type": "noul", "instructions": question}})
        return ps[0][1], info            # options are [no, yes]; TypeSafe's noul = p(true)

    def yesno(self, text, task, labels):
        # k noul questions in one request: Kev runs every question as its own causal row on the shared state
        qs = {f"q{i}": {"type": "noul", "instructions": yesno_question(task, lab, desc)}
              for i, (lab, desc) in enumerate(labels.items())}
        ps, _, info = self._ask(text, qs)
        return {lab: p[1] for lab, p in zip(labels, ps)}, info

    def memory(self):
        if self.model.backend == "mlx":
            return _mlx_mem()
        from dma.runners.local import _torch_mem
        return _torch_mem(self.device)


class Kev4BRunner(KevRunner):
    system = "kev-4b"
    model_id = "jaredpalmer/kev-4b"


# ---- SemIf Qwen3.5-4B (MLX 4-bit) -----------------------------------------------------------

class SemIfRunner(Runner):
    """The shipped ``decision_mlx.predict(model, tokenizer, "semif", task)`` (card, "Minimal inference")."""
    system = "semif-4b"
    model_id = "vinci00/semif-qwen3.5-4b-mlx-4bit"
    max_tokens = 4096   # SemIf's encoder budget; longer prompts are refused (no truncation)

    def __init__(self, device=None, **kw):
        super().__init__(device)
        if device not in (None, "mlx"):
            raise SystemExit("semif-4b runs on MLX only (vendor MLX build)")
        self.path = _snapshot(self.model_id)
        _add_path(THIRD_PARTY / "jevbench")
        _add_path(THIRD_PARTY / "semif" / "src")
        _add_path(self.path)
        from decision_mlx import configure, predict
        from mlx_lm import load
        configure()   # vendor settings: Metal, cache limit 256 MiB, memory limit 12 GiB, seed 42
        self.model, self.tok = load(self.path, tokenizer_config={"trust_remote_code": False})
        self._predict = predict
        self.manifest = json.loads((Path(self.path) / "decision_manifest.json").read_text())
        self._n = 0

    def meta(self):
        return {"model_id": self.model_id, "model_revision": Path(self.path).name,
                "base_model": "Qwen/Qwen3.5-4B", "base_revision": self.manifest.get("base_revision"),
                "vendor_temperature": None, "max_tokens": self.max_tokens,
                "semif_code_commit": THIRD_PARTY_COMMITS["semif"],
                "jevbench_code_commit": THIRD_PARTY_COMMITS["jevbench"]}

    def _run(self, text, question):
        self._n += 1
        task = SimpleNamespace(id=f"dma-{self._n}", state=text, question=question)
        probs, md = self._predict(self.model, self.tok, "semif", task)
        return probs, {"input_tokens": md["input_tokens"], "option_logits": md["option_logits"]}

    def choice(self, text, task, labels):
        p, extra = self._run(text, {"type": "choice", "instructions": task.instructions,
                                    "criteria": dict(labels)})
        return {k: float(p[k]) for k in labels}, extra

    def noul(self, text, question):
        p, extra = self._run(text, {"type": "noul", "instructions": question})
        return float(p["yes"]), extra

    def memory(self):
        return _mlx_mem()


# --------------------------------------------------------------------------------------------

EXTRA_SYSTEMS = {c.system: c for c in (EikosRunner, KevRunner, Kev4BRunner, SemIfRunner)}


def main(argv=None):
    from dma.runners import local
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--system" in argv:
        i = argv.index("--system")
        if i + 1 < len(argv) and argv[i + 1] not in EXTRA_SYSTEMS:
            raise SystemExit(f"--system must be one of {sorted(EXTRA_SYSTEMS)} "
                             f"(the other systems run through dma.runners.local)")
    local.SYSTEMS.update(EXTRA_SYSTEMS)   # this process only
    local.main(argv)


if __name__ == "__main__":
    main()
