"""Neutral reranker-to-JevBench mapping and public-only calibration.

Rerankers score query/document relevance rather than typed decisions.  This
module deliberately contains no model-specific label wording: every model sees
the same query, answer-option documents and calibration grids.  Provider-native
instruction plumbing belongs to the transport, but the instruction text comes
from :func:`task_instruction` here.
"""

from __future__ import annotations

import json
import math
from typing import Iterable

TEMPERATURE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
NOUL_THRESHOLD_GRID = (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)

INSTRUCTIONS = {
    "choice": "Classify: which option correctly answers the question for the given situation? Rate how well each option fits.",
    "noul": "Classify: does yes or no correctly answer the question for the given situation? Rate how well each option fits.",
    "score": "Classify: which rubric level correctly rates the situation? Rate how well each level fits.",
}


def _text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _criteria(task) -> dict[str, str]:
    q = task.question
    criteria = q.get("criteria")
    if q["type"] == "score":
        return {str(i): _text(v) for i, v in enumerate(criteria or [])}
    if isinstance(criteria, dict):
        if q["type"] == "noul":
            return {
                "no": _text(criteria.get("false", criteria.get("no", "No"))),
                "yes": _text(criteria.get("true", criteria.get("yes", "Yes"))),
            }
        return {str(k): _text(v) for k, v in criteria.items()}
    return {str(label): str(label) for label in task.labels}


def task_instruction(task) -> str:
    """The one frozen instruction wording for each primitive type."""
    return INSTRUCTIONS[task.question["type"]]


def query_text(task) -> str:
    """Render state + question + full rubric without the optional instruction."""
    rubric = _criteria(task)
    lines = [
        "Situation:",
        _text(task.state),
        "",
        "Question:",
        _text(task.question["instructions"]),
        "",
        "Answer rubric:",
    ]
    lines.extend(f"- {label}: {rubric.get(str(label), str(label))}" for label in task.labels)
    return "\n".join(lines)


def option_documents(task) -> list[str]:
    """One relevance document per exact answer label, in canonical order."""
    rubric = _criteria(task)
    kind = "Rubric level" if task.question["type"] == "score" else "Answer option"
    return [f"{kind} {label}: {rubric.get(str(label), str(label))}" for label in task.labels]


def softmax(scores: Iterable[float], temperature: float) -> list[float]:
    values = [float(x) for x in scores]
    if not values or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("scores must be non-empty and temperature must be positive")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("reranker scores must be finite")
    scaled = [x / temperature for x in values]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def apply_noul_threshold(probs: dict[str, float], threshold: float) -> dict[str, float]:
    """Apply a frozen yes/no threshold as a calibrated log-odds intercept.

    This keeps a genuine two-class distribution while making its 0.5 boundary
    correspond to ``P(yes) >= threshold`` in the temperature-scaled distribution.
    """
    if set(probs) != {"no", "yes"} or not 0 < threshold < 1:
        raise ValueError("noul calibration requires no/yes probabilities and a threshold in (0,1)")
    eps = 1e-12
    p = min(1 - eps, max(eps, float(probs["yes"])))
    shifted = math.log(p / (1 - p)) - math.log(threshold / (1 - threshold))
    yes = 1 / (1 + math.exp(-shifted))
    return {"no": 1 - yes, "yes": yes}


def calibrated_probs(task, scores: Iterable[float], temperature: float, noul_threshold: float) -> dict[str, float]:
    values = softmax(scores, temperature)
    probs = {str(label): values[i] for i, label in enumerate(task.labels)}
    return apply_noul_threshold(probs, noul_threshold) if task.question["type"] == "noul" else probs


def multiclass_log_loss(records: Iterable[tuple[object, list[float]]], temperature: float) -> float:
    losses = []
    for task, scores in records:
        if task.expected is None:
            continue
        probs = softmax(scores, temperature)
        expected = str(task.expected)
        index = [str(x) for x in task.labels].index(expected)
        losses.append(-math.log(max(1e-12, probs[index])))
    if not losses:
        raise ValueError("no labelled public records")
    return sum(losses) / len(losses)


def fit_temperature(records: Iterable[tuple[object, list[float]]]) -> tuple[float, list[dict]]:
    rows = []
    frozen = list(records)
    for temperature in TEMPERATURE_GRID:
        rows.append({"temperature": temperature, "log_loss": multiclass_log_loss(frozen, temperature)})
    chosen = min(rows, key=lambda row: (row["log_loss"], abs(math.log(row["temperature"]))))
    return chosen["temperature"], rows


def fit_noul_threshold(records: Iterable[tuple[object, list[float]]], temperature: float) -> tuple[float, list[dict]]:
    frozen = [(task, scores) for task, scores in records if task.question["type"] == "noul" and task.expected is not None]
    if not frozen:
        raise ValueError("no labelled public noul records")
    rows = []
    for threshold in NOUL_THRESHOLD_GRID:
        correct = 0
        conf_outcome = []
        for task, scores in frozen:
            raw = {str(label): value for label, value in zip(task.labels, softmax(scores, temperature))}
            probs = apply_noul_threshold(raw, threshold)
            predicted = "yes" if probs["yes"] >= 0.5 else "no"
            correct += predicted == str(task.expected)
            confidence = max(probs.values())
            conf_outcome.append((confidence, predicted == str(task.expected)))
        # Equal-width 10-bin ECE, matching JevBench's published calibration convention.
        ece = 0.0
        for b in range(10):
            bucket = [(c, ok) for c, ok in conf_outcome if min(9, int(c * 10)) == b]
            if bucket:
                ece += len(bucket) / len(conf_outcome) * abs(
                    sum(c for c, _ in bucket) / len(bucket) - sum(ok for _, ok in bucket) / len(bucket)
                )
        rows.append({"threshold": threshold, "accuracy": correct / len(frozen), "ece": ece})
    chosen = min(rows, key=lambda row: (-row["accuracy"], row["ece"], abs(row["threshold"] - 0.5)))
    return chosen["threshold"], rows
