"""Unified driver alertness state engine.

This module combines PERCLOS, EAR, blink count, and optional HRV metrics
into a single driver state label using the fuzzy logic classifier already
present in fuzzy_drowsiness_proc.py.

It also detects a sustained 10% HRV drop by comparing current SDNN/RMSSD
values against a rolling baseline.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import perclos_engine as pe
import fuzzy_drowsiness_proc as fuzzy


class DriverStateMonitor:
    """Track PERCLOS and optional HRV metrics to emit a fused driver state."""

    def __init__(self, perclos_window: int = 1800, hrv_window: int = 8, debug: bool = False):
        self.perclos_state = pe.create_perclos_state(window_size=perclos_window)
        self.hrv_history: deque[Dict[str, float]] = deque(maxlen=hrv_window)
        self.debug = debug
        self._baseline: Optional[Dict[str, float]] = None

    def update(
        self,
        avg_ear: float,
        hrv_metrics: Optional[Dict[str, Any]] = None,
        blink_rate: Optional[float] = None,
        use_perclos_calibration: bool = True,
    ) -> Dict[str, Any]:
        """Update the rolling state and return the latest driver assessment."""
        if use_perclos_calibration and self.perclos_state["calibration_frames_collected"] < 150:
            pe.calibrate_perclos_baseline(self.perclos_state, avg_ear)

        pe.update_perclos_metrics(self.perclos_state, avg_ear)
        perclos_pct = float(self.perclos_state["metrics"]["perclos_percentage"])

        hrv_drop = self._update_hrv_drop(hrv_metrics)

        if blink_rate is None:
            blink_rate = float(self.perclos_state["metrics"]["total_blinks"])

        hrv_values = self._extract_hrv_features(hrv_metrics)
        result = fuzzy.classify_drowsiness(
            ear=avg_ear,
            perclos=perclos_pct,
            blink_rate=blink_rate,
            hr_bpm=hrv_values.get("hr_bpm"),
            sdnn_ms=hrv_values.get("sdnn_ms"),
            rmssd_ms=hrv_values.get("rmssd_ms"),
            pnn50_pct=hrv_values.get("pnn50_pct"),
            lf_hf=hrv_values.get("lf_hf"),
            debug=self.debug,
        )

        result.update(
            {
                "perclos_percentage": round(perclos_pct, 2),
                "blink_count": int(self.perclos_state["metrics"]["total_blinks"]),
                "hrv_drop_detected": hrv_drop["detected"],
                "hrv_drop_percent": round(hrv_drop["drop_percent"], 2),
                "hrv_baseline": self._baseline,
                "ear_closed_threshold": round(self.perclos_state["ear_closed_threshold"], 4),
            }
        )
        return result

    def _update_hrv_drop(self, hrv_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not hrv_metrics:
            return {"detected": False, "drop_percent": 0.0}

        self.hrv_history.append({k: float(v) for k, v in hrv_metrics.items() if isinstance(v, (int, float))})
        if len(self.hrv_history) < 2:
            self._baseline = self._current_hrv_baseline()
            return {"detected": False, "drop_percent": 0.0}

        baseline = self._current_hrv_baseline()
        self._baseline = baseline
        if not baseline:
            return {"detected": False, "drop_percent": 0.0}

        latest = self._latest_hrv_reference(hrv_metrics)
        if not latest:
            return {"detected": False, "drop_percent": 0.0}

        drop_percent = 0.0
        for name in ("sdnn_ms", "rmssd_ms"):
            baseline_val = baseline.get(name)
            latest_val = latest.get(name)
            if baseline_val and latest_val and baseline_val > 0:
                drop = (baseline_val - latest_val) / baseline_val * 100.0
                if drop > drop_percent:
                    drop_percent = drop

        detected = drop_percent >= 10.0
        return {"detected": detected, "drop_percent": drop_percent}

    def _current_hrv_baseline(self) -> Optional[Dict[str, float]]:
        if not self.hrv_history:
            return None
        baseline = {
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50_pct": 0.0,
            "lf_hf": 0.0,
            "hr_bpm": 0.0,
        }
        for item in self.hrv_history:
            for key in baseline:
                baseline[key] += float(item.get(key, 0.0))
        count = len(self.hrv_history)
        return {k: v / count for k, v in baseline.items()}

    def _latest_hrv_reference(self, hrv_metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if not hrv_metrics:
            return None
        return {k: float(v) for k, v in hrv_metrics.items() if isinstance(v, (int, float)) and k in {"sdnn_ms", "rmssd_ms", "pnn50_pct", "lf_hf", "hr_bpm"}}

    def _extract_hrv_features(self, hrv_metrics: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        if not hrv_metrics:
            return {"hr_bpm": None, "sdnn_ms": None, "rmssd_ms": None, "pnn50_pct": None, "lf_hf": None}
        return {
            "hr_bpm": hrv_metrics.get("hr_bpm"),
            "sdnn_ms": hrv_metrics.get("sdnn_ms"),
            "rmssd_ms": hrv_metrics.get("rmssd_ms"),
            "pnn50_pct": hrv_metrics.get("pnn50_pct"),
            "lf_hf": hrv_metrics.get("lf_hf"),
        }


def evaluate_driver_state(
    avg_ear: float,
    perclos_percentage: Optional[float] = None,
    monitor: Optional[DriverStateMonitor] = None,
    hrv_metrics: Optional[Dict[str, Any]] = None,
    blink_rate: Optional[float] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper for a single-shot evaluation."""
    if monitor is None:
        monitor = DriverStateMonitor(debug=debug)
    if perclos_percentage is not None:
        monitor.perclos_state["metrics"]["perclos_percentage"] = float(perclos_percentage)
    return monitor.update(avg_ear=avg_ear, hrv_metrics=hrv_metrics, blink_rate=blink_rate, use_perclos_calibration=False)
