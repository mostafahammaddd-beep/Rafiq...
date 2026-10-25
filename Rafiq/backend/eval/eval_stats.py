"""
Evaluation statistics for Rafiq's Tajweed assessment (numpy only).

Why this exists: the release targets in the brief (>= 95 % precision of high-confidence corrections, < 1 % false-positive
rate on clean recitations, per-category recall, RTF p95 < 1) are claims about a POPULATION. A point estimate from a few
dozen clips cannot support them. Every verdict below is therefore decided on a confidence bound, never on the point
estimate, and a target without enough samples is reported as UNVERIFIED with the number of samples that would be needed.

Definitions (fixed here so that every report uses the same ones)
  unit            one (clip, category) opportunity: e.g. the madd muttasil in word 39 of clip c17.
  truth           True when a qualified human reviewer marked an error of that category in that unit.
  predicted       True when the system reported a correction of that category for that unit
                  (an abstention is NOT a prediction; abstentions are counted separately).
  precision       TP / (TP + FP)            over predicted units, per category (and overall, micro-averaged)
  recall          TP / (TP + FN)            over truth-positive units (an abstention on an error counts as a miss)
  FPR (clean)     FP / (FP + TN)            over truth-negative units, i.e. clean recitation
  coverage        1 - abstained / total     the share of units on which the system committed to an answer
  RTF             processing seconds / audio seconds of the same span (p95 by nearest rank)
Confidence intervals: Wilson score (two-sided 95 %) for precision / recall; exact one-sided 95 % upper bound
(Clopper-Pearson, closed form for 0 events) for FPR.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Z95 = 1.959963984540054          # two-sided 95 %
TARGET_PRECISION = 0.95
TARGET_FPR = 0.01
TARGET_RTF_P95 = 1.0
MIN_RTF_SAMPLES = 20


# ------------------------------------------------------------------------------ intervals
def wilson(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) when n == 0 (no information)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def upper_bound_zero_events(n: int, conf: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper bound on a rate after 0 events in n trials: 1 - (1-conf)^(1/n)."""
    if n <= 0:
        return 1.0
    return 1.0 - (1.0 - conf) ** (1.0 / n)


def upper_bound_binomial(k: int, n: int, conf: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper bound for k events in n trials (bisection on the binomial CDF)."""
    if n <= 0:
        return 1.0
    if k <= 0:
        return upper_bound_zero_events(n, conf)
    if k >= n:
        return 1.0
    alpha = 1.0 - conf

    def cdf(p: float) -> float:            # P[X <= k | p], computed in log space to stay stable for large n
        s = 0.0
        for i in range(k + 1):
            lg = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            if p <= 0.0:
                term = 1.0 if i == 0 else 0.0
            elif p >= 1.0:
                term = 1.0 if i == n else 0.0
            else:
                term = math.exp(lg + i * math.log(p) + (n - i) * math.log1p(-p))
            s += term
        return s

    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def min_clean_units_for_fpr(target: float = TARGET_FPR, conf: float = 0.95) -> int:
    """Smallest number of clean units that can demonstrate FPR < target with ZERO false positives (299 for 1 % / 95 %)."""
    return math.ceil(math.log(1.0 - conf) / math.log(1.0 - target))


def min_predicted_units_for_precision(target: float = TARGET_PRECISION, z: float = Z95) -> int:
    """Smallest n such that n/n correct predictions give a Wilson lower bound >= target (the best case)."""
    n = 1
    while wilson(n, n, z)[0] < target:
        n += 1
    return n


# ------------------------------------------------------------------------------ unit-level confusion
CATEGORIES_NOTE = ("Keep categories separate (e.g. madd_muttasil, madd_munfasil, ghunna, qalqala, ikhfaa, iqlab, idgham, "
                   "tafkhim, letter_substitution, vowel, skipped_word, ...): pooling hides the categories that do not work.")


def _rate(k: int, n: int) -> Optional[float]:
    return None if n <= 0 else k / n


def confusion(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """records: {clip_id, speaker_id, category, truth: bool, predicted: bool, abstained: bool (optional)}.
    Returns per-category + 'ALL' dictionaries with counts, rates and intervals."""
    buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "abstain": 0, "n": 0})
    for r in records:
        for key in (r["category"], "ALL"):
            b = buckets[key]
            b["n"] += 1
            if r.get("abstained"):
                b["abstain"] += 1
                if r["truth"]:
                    b["fn"] += 1                      # an abstention on a real error is a miss, never a hit
                else:
                    b["tn"] += 1                      # an abstention on clean audio costs nothing
                continue
            t, p = bool(r["truth"]), bool(r["predicted"])
            b["tp" if (t and p) else "fp" if (p and not t) else "fn" if (t and not p) else "tn"] += 1
    out: Dict[str, Dict[str, Any]] = {}
    for key, b in buckets.items():
        tp, fp, fn, tn = b["tp"], b["fp"], b["fn"], b["tn"]
        pred, pos, neg = tp + fp, tp + fn, fp + tn
        out[key] = dict(b, predicted=pred, truth_pos=pos, truth_neg=neg,
                        precision=_rate(tp, pred), precision_ci95=wilson(tp, pred),
                        recall=_rate(tp, pos), recall_ci95=wilson(tp, pos),
                        fpr=_rate(fp, neg), fpr_upper95=upper_bound_binomial(fp, neg),
                        coverage=_rate(b["n"] - b["abstain"], b["n"]))
    return out


def verdict(c: Dict[str, Any]) -> Dict[str, str]:
    """VERIFIED only when the CONFIDENCE BOUND meets the target; otherwise UNVERIFIED (with the reason) or FAIL
    (the bound proves the target is missed)."""
    v: Dict[str, str] = {}
    lo, hi = c["precision_ci95"]
    if c["predicted"] == 0:
        v["precision>=95%"] = "UNVERIFIED (no predictions in this category)"
    elif lo >= TARGET_PRECISION:
        v["precision>=95%"] = "VERIFIED (95%% Wilson lower bound %.3f, n=%d)" % (lo, c["predicted"])
    elif hi < TARGET_PRECISION:
        v["precision>=95%"] = "FAIL (95%% Wilson upper bound %.3f < 0.95, n=%d)" % (hi, c["predicted"])
    else:
        need = min_predicted_units_for_precision()
        v["precision>=95%"] = ("UNVERIFIED (point %.3f, 95%% CI %.3f-%.3f, n=%d; even a perfect record needs >= %d predictions)"
                               % (c["precision"], lo, hi, c["predicted"], need))
    if c["truth_neg"] == 0:
        v["FPR<1%"] = "UNVERIFIED (no clean units)"
    elif c["fpr_upper95"] < TARGET_FPR:
        v["FPR<1%"] = "VERIFIED (95%% one-sided upper bound %.4f, %d clean units, %d FP)" % (c["fpr_upper95"], c["truth_neg"], c["fp"])
    elif _lower_fpr(c) >= TARGET_FPR:
        v["FPR<1%"] = "FAIL (observed %.4f; the 95%% Wilson lower bound %.4f is already >= 1%%)" % (c["fp"] / c["truth_neg"], _lower_fpr(c))
    else:
        v["FPR<1%"] = ("UNVERIFIED (observed %.4f, 95%% upper bound %.4f, %d clean units, %d FP; zero FP needs >= %d clean units)"
                       % (c["fp"] / c["truth_neg"], c["fpr_upper95"], c["truth_neg"], c["fp"], min_clean_units_for_fpr()))
    return v


def _lower_fpr(c: Dict[str, Any]) -> float:
    return wilson(c["fp"], c["truth_neg"])[0] if c["truth_neg"] else 0.0


# ------------------------------------------------------------------------------ latency
def percentile_nearest_rank(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = max(1, math.ceil(q / 100.0 * len(s)))
    return s[k - 1]


def rtf_report(process_s: Sequence[float], audio_s: Sequence[float], hardware: Optional[str]) -> Dict[str, Any]:
    """RTF per measured span (e.g. per session or per model call). p95 by nearest rank. Verified only with >= 20 spans AND a
    hardware description (an RTF without the hardware it was measured on is meaningless)."""
    if len(process_s) != len(audio_s):
        raise ValueError("process_s and audio_s must pair up")
    rtfs = [p / a for p, a in zip(process_s, audio_s) if a > 0]
    p95 = percentile_nearest_rank(rtfs, 95)
    rep: Dict[str, Any] = {"n": len(rtfs), "p50": percentile_nearest_rank(rtfs, 50), "p95": p95, "max": max(rtfs) if rtfs else None,
                           "hardware": hardware}
    if p95 is None:
        rep["verdict"] = "UNVERIFIED (no measurements)"
    elif not hardware:
        rep["verdict"] = "UNVERIFIED (hardware not specified)"
    elif len(rtfs) < MIN_RTF_SAMPLES:
        rep["verdict"] = "UNVERIFIED (p95 %.3f from only %d spans; need >= %d)" % (p95, len(rtfs), MIN_RTF_SAMPLES)
    elif p95 < TARGET_RTF_P95:
        rep["verdict"] = "VERIFIED on this hardware only (p95 %.3f < 1, n=%d)" % (p95, len(rtfs))
    else:
        rep["verdict"] = "FAIL (p95 %.3f >= 1, n=%d)" % (p95, len(rtfs))
    return rep


# ------------------------------------------------------------------------------ data hygiene
def leakage_report(splits: Dict[str, Sequence[Dict[str, Any]]]) -> List[str]:
    """splits: name -> [{clip_id, speaker_id, audio_sha256 (optional)}]. Returns human-readable violations:
    a speaker present in more than one partition, a clip id or an identical audio hash in more than one partition."""
    problems: List[str] = []
    for field, label in (("speaker_id", "speaker"), ("clip_id", "clip id"), ("audio_sha256", "identical audio")):
        seen: Dict[str, str] = {}
        flagged = set()
        for name, items in splits.items():
            for it in items:
                key = it.get(field)
                if key is None:
                    continue
                if key in seen and seen[key] != name and (key, name, seen[key]) not in flagged:
                    flagged.add((key, name, seen[key]))
                    problems.append("%s %r appears in both %r and %r" % (label, key, seen[key], name))
                seen.setdefault(key, name)
    if any(it.get("speaker_id") is None for items in splits.values() for it in items):
        problems.append("some clips have no speaker_id: speaker-disjointness cannot be verified")
    return problems


# ------------------------------------------------------------------------------ calibration
def reliability(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> Dict[str, Any]:
    """Expected calibration error with equal-width bins. Fine only as a diagnostic: ECE is biased for small n."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must pair up")
    n = len(confidences)
    rows = []
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        rows.append({"bin": "%.1f-%.1f" % (lo, hi), "n": len(idx), "mean_confidence": conf, "accuracy": acc})
        ece += len(idx) / n * abs(acc - conf)
    return {"n": n, "ece": ece if n else None, "bins": rows}


def threshold_for_precision(confidences: Sequence[float], correct: Sequence[bool], target: float = TARGET_PRECISION,
                            z: float = Z95) -> Optional[float]:
    """Lowest confidence threshold t such that, among predictions with confidence >= t, the Wilson LOWER bound of the
    precision is >= target. None = no threshold achieves it on this data: the category must abstain (or collect more
    data). Select the threshold on a validation partition and report it on a separate test partition."""
    pairs = sorted(zip(confidences, correct), key=lambda x: -x[0])
    best: Optional[float] = None
    tp = 0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:       # ties are taken together
            tp += 1 if pairs[j][1] else 0
            j += 1
        if wilson(tp, j, z)[0] >= target:
            best = pairs[i][0]
        i = j
    return best
