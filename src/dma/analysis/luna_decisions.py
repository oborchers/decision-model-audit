"""GPT-6 Luna through OpenAI's Decisions API (post hoc, 2026-10-07). Writes results/luna_decisions.json.

Runs the unchanged part 1 and part 2 analyses on a temporary copy of the raw rows in which the
`luna-decisions` rows are appended to the part 2 files, then keeps only the entries for `luna-decisions`
and its two references, Jev and GPT-6 Luna through OpenRouter. Published result files are not touched.
Holm-adjusted p values computed inside these analyses include the extra system and are not reported;
the paired comparisons against Jev are given unadjusted.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SYS = "luna-decisions"
KEEP = (SYS, "jev", "luna")


def merged_root() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="luna_decisions_"))
    (tmp / "data").symlink_to(ROOT / "data")
    shutil.copytree(ROOT / "results/raw", tmp / "results/raw")
    for f in (tmp / "results/raw").glob(f"*/*.{SYS}.jsonl"):
        target = f.with_name(f.name.replace(f".{SYS}.jsonl", ".teil2.jsonl"))
        with target.open("a") as out:
            out.write(f.read_text())
        f.unlink()
    return tmp


def pick(d, keys=KEEP):
    if isinstance(d, dict):
        hit = {k: v for k, v in d.items() if any(k == s or k.startswith(s + "/") for s in keys)}
        if hit:
            return hit
        return {k: pick(v, keys) for k, v in d.items() if isinstance(v, dict) and pick(v, keys)}
    return None


def main(tmp: Path | None = None):
    tmp = tmp or merged_root()
    from dma.analysis import gaps, probes_teil2, report, teil2
    for m in (report, teil2, gaps, probes_teil2):
        m.ROOT = tmp
    teil2.RAW = probes_teil2.RAW = tmp / "results/raw/main"
    gaps.G = tmp / "results/raw/gaps"
    teil2.DECISION.add(SYS)
    (tmp / "results/figures").mkdir(parents=True, exist_ok=True)
    teil2.figure_risk_coverage = lambda *a, **k: None  # its default path is bound to the real results folder
    report.main()
    teil2.main()
    gaps.main()
    probes_teil2.main()
    out = {}
    for f in sorted((tmp / "results").glob("*.json")):
        sel = pick(json.loads(f.read_text()))
        if sel:
            out[f.stem] = sel
    (ROOT / "results/luna_decisions.json").write_text(json.dumps(out, indent=1) + "\n")
    shutil.rmtree(tmp)
    print("written results/luna_decisions.json")


if __name__ == "__main__":
    main()
