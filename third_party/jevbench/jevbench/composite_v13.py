"""JevBench v1.3 scoring.

The task set and all measurements are unchanged from v1.2.  Intelligence is
now accuracy above an item-specific uniform-guessing baseline.  The published
tier baselines below are computed from every frozen item as ``1 / options``
(``1 / levels`` for score items); the option-count histograms make the
calculation reproducible without publishing held-out item text.
"""
import math

TIER_WEIGHTS = {"easy": 0.14, "standard": 0.28, "judge": 0.28, "hard": 0.30}
TIER_OPTION_COUNTS = {
    "easy": {2: 18, 4: 13, 5: 41},
    "standard": {2: 32, 4: 40, 5: 12, 6: 12},
    "judge": {2: 68, 9: 78},
    "hard": {2: 77, 3: 26, 4: 73, 5: 38, 6: 6},
}


def chance_from_option_counts(counts):
    n = sum(counts.values())
    if not n:
        raise ValueError("a tier needs at least one item")
    return sum(count / options for options, count in counts.items()) / n


TIER_CHANCES = {tier: chance_from_option_counts(counts) for tier, counts in TIER_OPTION_COUNTS.items()}
AXES = ("intelligence", "calibration", "speed", "cost")
WEIGHTS = {axis: 0.25 for axis in AXES}
SPEED_BEST_S, SPEED_PER_DECADE = 0.1, 20
COST_BEST_USD, COST_PER_DECADE = 0.001, 30
PRODUCTION = {"api"}
LOAD_FACTOR = 2.0
OWN_SERVER_ADD_S = 0.15
OWN_SERVERS = {"gpu", "cpu"}
SPEED_NOTE = ("Latency of self-hosted and demo endpoints is adjusted ×2 (+0.15 s on our own servers) to approximate "
              "production load — an assumption, not a measurement; raw measurements are in the table and the repo.")

PRESETS = {
    "JevBench Score (25:25:25:25)": (0.25, 0.25, 0.25, 0.25),
    "Balanced 33:33:33 (no calibration)": (1 / 3, 0.0, 1 / 3, 1 / 3),
    "Emphasis on Accuracy 60:20:20": (0.6, 0.0, 0.2, 0.2),
    "Emphasis on Speed 20:60:20": (0.2, 0.0, 0.6, 0.2),
    "Emphasis on Cost 20:20:60": (0.2, 0.0, 0.2, 0.6),
    "Intelligence only": (1.0, 0.0, 0.0, 0.0),
}
MAIN = "JevBench Score (25:25:25:25)"


def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def chance_corrected_accuracy(accuracy, chance):
    return clamp(100 * (accuracy - chance) / (1 - chance))


def intelligence(tiers, tier_chances=TIER_CHANCES):
    """Weighted chance-corrected tier accuracy; missing tiers are renormalised."""
    score = weight = 0.0
    for tier, tier_weight in TIER_WEIGHTS.items():
        if tiers.get(tier) is not None:
            score += tier_weight * chance_corrected_accuracy(tiers[tier], tier_chances[tier])
            weight += tier_weight
    return score / weight if weight else None


def adjusted_latency(seconds, endpoint_kind):
    if seconds is None:
        return None
    if endpoint_kind in PRODUCTION:
        return seconds
    return seconds * LOAD_FACTOR + (OWN_SERVER_ADD_S if endpoint_kind in OWN_SERVERS else 0.0)


def speed_point(seconds):
    return clamp(100 - SPEED_PER_DECADE * math.log10(seconds / SPEED_BEST_S))


def speed(p50_s, p95_s, endpoint_kind):
    a, b = adjusted_latency(p50_s, endpoint_kind), adjusted_latency(p95_s, endpoint_kind)
    return None if a is None or b is None else (speed_point(a) + speed_point(b)) / 2


def cost(usd_per_1000):
    if usd_per_1000 is None or not usd_per_1000 > 0:
        raise ValueError("every system needs a positive price; a missing price is never scored as 100")
    return clamp(100 - COST_PER_DECADE * math.log10(usd_per_1000 / COST_BEST_USD))


def tvd(p, q, labels):
    return 0.5 * sum(abs(p.get(label, 0.0) - q.get(label, 0.0)) for label in labels)


def calibration(ece, mean_tvd=None):
    if ece is None:
        return None
    ece_score = max(0.0, 100 * (1 - ece / 0.5))
    return ece_score if mean_tvd is None else (ece_score + 100 * (1 - mean_tvd)) / 2


def near_chance_multiplier(intelligence_score):
    if intelligence_score is None or intelligence_score >= 50:
        return 1.0
    return (max(intelligence_score, 0.0) / 50) ** 2


def geometric(axes, weights=WEIGHTS):
    total = sum(weights[axis] for axis in AXES)
    base = math.exp(sum(weights[axis] / total * math.log(max(axes.get(axis) or 0.0, 1.0))
                        for axis in AXES if weights[axis] > 0))
    return base * near_chance_multiplier(axes.get("intelligence"))


def jevbench_score(axes):
    return geometric(axes, WEIGHTS)


def preset_score(axes, weights):
    return geometric(axes, dict(zip(AXES, weights)))
