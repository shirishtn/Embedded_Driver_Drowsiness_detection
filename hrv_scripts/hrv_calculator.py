"""
hrv_calculator.py — Compute HRV metrics from an rPPG waveform.

Time-domain:
  HR_mean, SDNN, RMSSD, pNN50

Frequency-domain (Welch PSD):
  LF (0.04–0.15 Hz), HF (0.15–0.4 Hz), LF/HF ratio

Peak detection: scipy find_peaks with physiologically constrained
minimum distance to avoid spurious peaks.
"""

from __future__ import annotations

import sys
import numpy as np

try:
    from scipy.signal import find_peaks, welch
    from scipy.interpolate import interp1d
except ImportError as exc:
    raise ImportError("[hrv_calculator] scipy is required.") from exc


# Frequency-domain bands (Hz)
_LF_BAND = (0.04, 0.15)
_HF_BAND = (0.15, 0.40)


class HRVCalculator:
    """Compute HRV metrics from a bandpass-filtered rPPG signal."""

    def __init__(self, config: dict, debug: bool = False):
        self.debug   = debug
        self.min_hr  = config.get("min_hr_bpm", 45)
        self.max_hr  = config.get("max_hr_bpm", 180)

    # ── Public ────────────────────────────────────────────────────────────────

    def compute(self, signal: np.ndarray, fps: float) -> dict | None:
        """
        Parameters
        ----------
        signal : 1-D filtered rPPG waveform (uniform sample rate = fps)
        fps    : samples per second

        Returns dict of HRV metrics, or None on failure.
        """
        if signal is None or len(signal) < int(fps * 5):
            print("[hrv_calculator] Signal too short for HRV computation.")
            return None

        # ── Peak detection ────────────────────────────────────────────────
        peaks = self._detect_peaks(signal, fps)
        if peaks is None or len(peaks) < 4:
            print("[hrv_calculator] Insufficient peaks detected.")
            return None

        if self.debug:
            print(f"[hrv_calculator] {len(peaks)} peaks found.")

        # ── RR intervals (seconds) ────────────────────────────────────────
        rr = np.diff(peaks) / fps   # seconds between consecutive peaks
        rr = self._filter_rr(rr)

        if len(rr) < 3:
            print("[hrv_calculator] Insufficient valid RR intervals.")
            return None

        # ── Time-domain metrics ───────────────────────────────────────────
        rr_ms   = rr * 1000.0
        hr_bpm  = 60.0 / rr.mean()
        sdnn    = rr_ms.std(ddof=1)
        rmssd   = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
        nn50    = int(np.sum(np.abs(np.diff(rr_ms)) > 50.0))
        pnn50   = 100.0 * nn50 / len(rr_ms) if len(rr_ms) > 1 else 0.0

        # ── Frequency-domain metrics ──────────────────────────────────────
        lf_power, hf_power, lf_hf = self._frequency_domain(rr, fps)

        # ── SNR of rPPG signal ────────────────────────────────────────────
        snr_db = self._estimate_snr(signal, fps, hr_bpm)

        metrics = {
            "hr_bpm":    round(hr_bpm, 2),
            "sdnn_ms":   round(sdnn, 2),
            "rmssd_ms":  round(rmssd, 2),
            "pnn50_pct": round(pnn50, 2),
            "lf_power":  round(lf_power, 4),
            "hf_power":  round(hf_power, 4),
            "lf_hf":     round(lf_hf, 4),
            "snr_db":    round(snr_db, 2),
            "n_peaks":   len(peaks),
            "n_rr":      len(rr),
        }

        if self.debug:
            print(f"[hrv_calculator] Metrics: {metrics}")

        return metrics

    # ── Peak detection ────────────────────────────────────────────────────────

    def _detect_peaks(self, signal: np.ndarray, fps: float):
        # Minimum inter-peak distance in samples derived from max HR
        min_dist = int(fps * 60.0 / self.max_hr)
        min_dist = max(min_dist, 1)

        # Invert if the waveform has downward peaks (PPG polarity varies)
        peaks_pos, _ = find_peaks(signal,  distance=min_dist, prominence=0.01)
        peaks_neg, _ = find_peaks(-signal, distance=min_dist, prominence=0.01)

        if self.debug:
            print(f"[hrv_calculator] Positive peaks: {len(peaks_pos)}  "
                  f"Negative peaks: {len(peaks_neg)}")

        # Use whichever polarity gives more peaks
        peaks = peaks_pos if len(peaks_pos) >= len(peaks_neg) else peaks_neg
        return peaks if len(peaks) >= 4 else None

    # ── RR filtering ──────────────────────────────────────────────────────────

    def _filter_rr(self, rr: np.ndarray) -> np.ndarray:
        """Remove physiologically impossible RR intervals."""
        rr_min = 60.0 / self.max_hr
        rr_max = 60.0 / self.min_hr

        mask = (rr >= rr_min) & (rr <= rr_max)
        n_removed = (~mask).sum()

        if self.debug and n_removed > 0:
            print(f"[hrv_calculator] Removed {n_removed} outlier RR intervals.")

        rr_clean = rr[mask]

        # Ectopic beat removal: > 20% deviation from median
        if len(rr_clean) >= 3:
            med = np.median(rr_clean)
            mask2 = np.abs(rr_clean - med) / (med + 1e-8) < 0.20
            n_ec = (~mask2).sum()
            if self.debug and n_ec > 0:
                print(f"[hrv_calculator] Removed {n_ec} suspected ectopic intervals.")
            rr_clean = rr_clean[mask2]

        return rr_clean

    # ── Frequency domain ─────────────────────────────────────────────────────

    def _frequency_domain(self, rr: np.ndarray, fps: float):
        """
        Interpolate RR tachogram to uniform grid, then use Welch PSD.
        Returns (lf_power, hf_power, lf_hf).
        """
        try:
            # Cumulative time axis of RR intervals
            t = np.cumsum(rr)
            t = np.insert(t, 0, 0.0)[:-1]

            # Interpolate to 4 Hz uniform grid (standard for HRV)
            fs_interp = 4.0
            t_interp  = np.arange(t[0], t[-1], 1.0 / fs_interp)

            if len(t_interp) < 8:
                return 0.0, 0.0, float("nan")

            interp_fn  = interp1d(t, rr, kind="cubic", bounds_error=False,
                                  fill_value="extrapolate")
            rr_interp  = interp_fn(t_interp)

            # Welch PSD
            nperseg = min(len(rr_interp), 256)
            freqs, psd = welch(rr_interp, fs=fs_interp, nperseg=nperseg)

            def band_power(f_low, f_high):
                idx = (freqs >= f_low) & (freqs < f_high)
                if not idx.any():
                    return 0.0
                return float(np.trapz(psd[idx], freqs[idx]))

            lf = band_power(*_LF_BAND)
            hf = band_power(*_HF_BAND)
            lf_hf = lf / (hf + 1e-12)

            return lf, hf, lf_hf

        except Exception as exc:
            print(f"[hrv_calculator] Frequency-domain analysis failed: {exc}",
                  file=sys.stderr)
            return 0.0, 0.0, float("nan")

    # ── SNR estimation ────────────────────────────────────────────────────────

    def _estimate_snr(self, signal: np.ndarray, fps: float, hr_bpm: float) -> float:
        """
        Simple SNR: power at HR fundamental vs. out-of-band power.
        """
        try:
            nperseg = min(len(signal), 512)
            freqs, psd = welch(signal, fs=fps, nperseg=nperseg)

            hr_hz = hr_bpm / 60.0
            bw    = 0.05  # ±0.05 Hz around fundamental

            signal_idx = np.abs(freqs - hr_hz) < bw
            noise_idx  = (freqs > 0.1) & ~signal_idx

            if not signal_idx.any() or not noise_idx.any():
                return float("nan")

            sig_power   = psd[signal_idx].mean()
            noise_power = psd[noise_idx].mean()
            snr_db = 10.0 * np.log10(sig_power / (noise_power + 1e-12))
            return float(snr_db)

        except Exception as exc:
            print(f"[hrv_calculator] SNR estimation failed: {exc}", file=sys.stderr)
            return float("nan")
