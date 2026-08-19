"""
Rule-based fuzzy driver alertness classifier.

Inputs used by the fuzzy inference system:
  perclos   : PERCLOS percentage, 0-100
  ear       : Eye Aspect Ratio, usually 0.0-0.4
  hrv_score : Derived HRV health/alertness score, 0.0-1.0

The public classify_drowsiness() function still accepts the older HRV metric
arguments so the rest of the project does not need to change. Internally, the
HRV metrics are condensed into one interpretable HRV score before rule firing.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


_OUT_ALERT_CENTER = 0.85
_OUT_DROWSY_CENTER = 0.45
_OUT_CRITICAL_CENTER = 0.10

_ALERT_THRESHOLD = 0.65
_CRITICAL_THRESHOLD = 0.30


def _is_valid_number(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and not math.isnan(float(value))


def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-10)
    return (d - x) / (d - c + 1e-10)


def _triangle(x: float, a: float, b: float, c: float) -> float:
    return _trapezoid(x, a, b, b, c)


def _shoulder_left(x: float, a: float, b: float) -> float:
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a + 1e-10)


def _shoulder_right(x: float, a: float, b: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a + 1e-10)


def _fuzzify_perclos(perclos: Optional[float]) -> Optional[Dict[str, float]]:
    if not _is_valid_number(perclos):
        return None

    value = float(perclos)
    return {
        "low": _shoulder_left(value, 12.0, 20.0),
        "medium": _trapezoid(value, 12.0, 20.0, 38.0, 48.0),
        "high": _shoulder_right(value, 40.0, 55.0),
    }


def _fuzzify_ear(ear: Optional[float]) -> Optional[Dict[str, float]]:
    if not _is_valid_number(ear):
        return None

    value = float(ear)
    return {
        "low": _shoulder_left(value, 0.18, 0.24),
        "medium": _triangle(value, 0.20, 0.26, 0.32),
        "high": _shoulder_right(value, 0.28, 0.34),
    }


def _fuzzify_hrv_score(hrv_score: Optional[float]) -> Optional[Dict[str, float]]:
    if not _is_valid_number(hrv_score):
        return None

    value = float(np.clip(hrv_score, 0.0, 1.0))
    return {
        "low": _shoulder_left(value, 0.30, 0.45),
        "medium": _triangle(value, 0.35, 0.55, 0.75),
        "high": _shoulder_right(value, 0.65, 0.80),
    }


def _score_range(value: Optional[float], good_low: float, good_high: float, outer_low: float, outer_high: float) -> Optional[float]:
    if not _is_valid_number(value):
        return None

    x = float(value)
    if good_low <= x <= good_high:
        return 1.0
    if outer_low < x < good_low:
        return (x - outer_low) / (good_low - outer_low + 1e-10)
    if good_high < x < outer_high:
        return (outer_high - x) / (outer_high - good_high + 1e-10)
    return 0.0


def _score_minimum(value: Optional[float], weak: float, good: float) -> Optional[float]:
    if not _is_valid_number(value):
        return None

    x = float(value)
    if x <= weak:
        return 0.0
    if x >= good:
        return 1.0
    return (x - weak) / (good - weak + 1e-10)


def _derive_hrv_score(
    hr_bpm: Optional[float] = None,
    sdnn_ms: Optional[float] = None,
    rmssd_ms: Optional[float] = None,
    pnn50_pct: Optional[float] = None,
    lf_hf: Optional[float] = None,
) -> Optional[float]:
    """Condense HRV metrics to one 0-1 score for fuzzy rules."""
    scores = []

    hr_score = _score_range(hr_bpm, good_low=60.0, good_high=90.0, outer_low=45.0, outer_high=110.0)
    sdnn_score = _score_minimum(sdnn_ms, weak=15.0, good=55.0)
    rmssd_score = _score_range(rmssd_ms, good_low=25.0, good_high=80.0, outer_low=8.0, outer_high=120.0)
    pnn50_score = _score_range(pnn50_pct, good_low=8.0, good_high=45.0, outer_low=0.0, outer_high=80.0)
    lf_hf_score = _score_range(lf_hf, good_low=0.8, good_high=2.5, outer_low=0.2, outer_high=5.0)

    for score in (hr_score, sdnn_score, rmssd_score, pnn50_score, lf_hf_score):
        if score is not None:
            scores.append(score)

    if not scores:
        return None

    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _and(*degrees: Optional[float]) -> float:
    valid = [float(degree) for degree in degrees if degree is not None]
    return min(valid) if valid else 0.0


def _or(*degrees: Optional[float]) -> float:
    valid = [float(degree) for degree in degrees if degree is not None]
    return max(valid) if valid else 0.0


def _rule_strength(inputs: Dict[str, Optional[Dict[str, float]]], terms: Iterable[Tuple[str, str]]) -> float:
    degrees = []
    for variable, term in terms:
        membership = inputs.get(variable)
        if membership is None:
            return 0.0
        degrees.append(membership.get(term, 0.0))
    return _and(*degrees)


_RULES = [
    ("critical", (("perclos", "high"),), "PERCLOS is High -> Critical"),
    ("critical", (("perclos", "high"), ("ear", "low")), "PERCLOS is High AND EAR is Low -> Critical"),
    ("critical", (("perclos", "high"), ("hrv", "low")), "PERCLOS is High AND HRV is Low -> Critical"),
    ("critical", (("ear", "low"), ("hrv", "low")), "EAR is Low AND HRV is Low -> Critical"),
    ("critical", (("perclos", "medium"), ("ear", "low"), ("hrv", "low")), "PERCLOS is Medium AND EAR is Low AND HRV is Low -> Critical"),
    ("drowsy", (("perclos", "medium"),), "PERCLOS is Medium -> Drowsy"),
    ("drowsy", (("perclos", "medium"), ("hrv", "low")), "PERCLOS is Medium AND HRV is Low -> Drowsy"),
    ("drowsy", (("perclos", "medium"), ("ear", "medium")), "PERCLOS is Medium AND EAR is Medium -> Drowsy"),
    ("drowsy", (("ear", "low"), ("perclos", "low")), "EAR is Low AND PERCLOS is Low -> Drowsy"),
    ("drowsy", (("ear", "medium"), ("hrv", "low")), "EAR is Medium AND HRV is Low -> Drowsy"),
    ("drowsy", (("perclos", "low"), ("ear", "medium"), ("hrv", "low")), "PERCLOS is Low AND EAR is Medium AND HRV is Low -> Drowsy"),
    ("alert", (("ear", "high"), ("perclos", "low")), "EAR is High AND PERCLOS is Low -> Alert"),
    ("alert", (("ear", "high"), ("perclos", "low"), ("hrv", "high")), "EAR is High AND PERCLOS is Low AND HRV is High -> Alert"),
    ("alert", (("perclos", "low"), ("ear", "medium"), ("hrv", "high")), "PERCLOS is Low AND EAR is Medium AND HRV is High -> Alert"),
    ("alert", (("perclos", "low"), ("ear", "high"), ("hrv", "medium")), "PERCLOS is Low AND EAR is High AND HRV is Medium -> Alert"),
]


def _evaluate_rules(inputs: Dict[str, Optional[Dict[str, float]]], debug: bool) -> Tuple[Dict[str, float], Dict[str, float]]:
    strengths = {"alert": 0.0, "drowsy": 0.0, "critical": 0.0}
    fired_rules = {}

    for output, terms, label in _RULES:
        strength = _rule_strength(inputs, terms)
        if strength <= 0.0:
            continue

        strengths[output] = max(strengths[output], strength)
        fired_rules[label] = round(float(strength), 3)

        if debug:
            print(f"[fuzzy] rule={label} strength={strength:.3f}")

    return strengths, fired_rules


def _fallback_strengths(inputs: Dict[str, Optional[Dict[str, float]]]) -> Dict[str, float]:
    """Handle cases where only one or two fuzzy inputs are available."""
    strengths = {"alert": 0.0, "drowsy": 0.0, "critical": 0.0}

    perclos = inputs.get("perclos")
    ear = inputs.get("ear")
    hrv = inputs.get("hrv")

    if perclos is not None:
        strengths["alert"] = max(strengths["alert"], perclos["low"])
        strengths["drowsy"] = max(strengths["drowsy"], perclos["medium"])
        strengths["critical"] = max(strengths["critical"], perclos["high"])

    if ear is not None:
        strengths["alert"] = max(strengths["alert"], 0.8 * ear["high"])
        strengths["drowsy"] = max(strengths["drowsy"], 0.7 * ear["medium"])
        strengths["critical"] = max(strengths["critical"], 0.8 * ear["low"])

    if hrv is not None and perclos is None and ear is None:
        strengths["alert"] = max(strengths["alert"], 0.6 * hrv["high"])
        strengths["drowsy"] = max(strengths["drowsy"], 0.5 * hrv["medium"])
        strengths["critical"] = max(strengths["critical"], 0.4 * hrv["low"])

    return strengths


def _defuzzify(strengths: Dict[str, float]) -> Tuple[float, float]:
    alert = strengths["alert"]
    drowsy = strengths["drowsy"]
    critical = strengths["critical"]
    denom = alert + drowsy + critical

    if denom <= 0.0:
        return 0.5, 0.0

    score = (
        alert * _OUT_ALERT_CENTER
        + drowsy * _OUT_DROWSY_CENTER
        + critical * _OUT_CRITICAL_CENTER
    ) / denom
    score = float(np.clip(score, 0.0, 1.0))

    ordered = sorted(strengths.values(), reverse=True)
    confidence = float(np.clip((ordered[0] - ordered[1]) / (ordered[0] + 1e-10), 0.0, 1.0))
    return score, confidence


def _classify_state(score: float) -> str:
    if score >= _ALERT_THRESHOLD:
        return "ALERT"
    if score <= _CRITICAL_THRESHOLD:
        return "CRITICAL"
    return "DROWSY"


def classify_drowsiness(
    ear=None,
    perclos=None,
    blink_rate=None,
    hr_bpm=None,
    sdnn_ms=None,
    rmssd_ms=None,
    pnn50_pct=None,
    lf_hf=None,
    hrv_score=None,
    debug=False,
):
    """
    Fuse PERCLOS, EAR, and a derived HRV score into driver alertness.

    blink_rate is accepted for backward compatibility but is no longer used by
    the fuzzy rule base.
    """
    if hrv_score is None:
        hrv_score = _derive_hrv_score(
            hr_bpm=hr_bpm,
            sdnn_ms=sdnn_ms,
            rmssd_ms=rmssd_ms,
            pnn50_pct=pnn50_pct,
            lf_hf=lf_hf,
        )

    inputs = {
        "perclos": _fuzzify_perclos(perclos),
        "ear": _fuzzify_ear(ear),
        "hrv": _fuzzify_hrv_score(hrv_score),
    }

    if all(value is None for value in inputs.values()):
        print("[fuzzy] WARNING: No valid fuzzy inputs provided; returning neutral.", file=sys.stderr)
        return {
            "alertness_score": 0.5,
            "state": "UNKNOWN",
            "confidence": 0.0,
            "contributions": {},
            "hrv_score": None,
            "rule_strengths": {"alert": 0.0, "drowsy": 0.0, "critical": 0.0},
        }

    rule_strengths, fired_rules = _evaluate_rules(inputs, debug)

    if max(rule_strengths.values()) <= 0.0:
        rule_strengths = _fallback_strengths(inputs)

    score, confidence = _defuzzify(rule_strengths)
    state = _classify_state(score)

    contributions = {
        "perclos": round(inputs["perclos"]["low"] - inputs["perclos"]["high"], 3) if inputs["perclos"] else 0.0,
        "ear": round(inputs["ear"]["high"] - inputs["ear"]["low"], 3) if inputs["ear"] else 0.0,
        "hrv_score": round((inputs["hrv"]["high"] - inputs["hrv"]["low"]), 3) if inputs["hrv"] else 0.0,
    }

    if debug:
        print(
            "[fuzzy] strengths "
            f"alert={rule_strengths['alert']:.3f} "
            f"drowsy={rule_strengths['drowsy']:.3f} "
            f"critical={rule_strengths['critical']:.3f} "
            f"score={score:.3f} confidence={confidence:.2%}"
        )

    return {
        "alertness_score": round(score, 3),
        "state": state,
        "confidence": round(confidence, 3),
        "contributions": contributions,
        "hrv_score": round(float(hrv_score), 3) if _is_valid_number(hrv_score) else None,
        "rule_strengths": {key: round(float(value), 3) for key, value in rule_strengths.items()},
        "fired_rules": fired_rules,
    }


def print_result(result):
    print("Drowsiness Assessment")
    print(f"  State           : {result['state']}")
    print(f"  Alertness Score : {result['alertness_score']:.3f} (1=alert, 0=critical)")
    print(f"  Confidence      : {result['confidence']:.1%}")
    print(f"  HRV Score       : {result.get('hrv_score')}")
    print(f"  Rule strengths  : {result.get('rule_strengths')}")
    if result.get("fired_rules"):
        print("  Fired rules:")
        for label, strength in result["fired_rules"].items():
            print(f"    {strength:.3f} - {label}")
    print()


if __name__ == "__main__":
    examples = [
        ("Moderately drowsy", dict(ear=0.196, perclos=34.61, hr_bpm=84.0, sdnn_ms=60.4, rmssd_ms=99.1, pnn50_pct=85.7)),
        ("Alert driver", dict(ear=0.31, perclos=8.0, hr_bpm=72.0, sdnn_ms=55.0, rmssd_ms=42.0, pnn50_pct=18.0, lf_hf=1.4)),
        ("Critical eyes", dict(ear=0.13, perclos=62.0, hr_bpm=56.0, sdnn_ms=18.0, rmssd_ms=110.0, pnn50_pct=92.0, lf_hf=0.3)),
        ("PERCLOS only", dict(ear=0.21, perclos=28.0)),
    ]

    for title, kwargs in examples:
        print(f"=== {title} ===")
        print_result(classify_drowsiness(**kwargs, debug=True))
