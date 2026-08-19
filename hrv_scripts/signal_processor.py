"""
signal_processor.py — rPPG signal extraction.

Implements two motion-robust methods:
  • CHROM  — de Haan & Jeanne (2013), recommended default
  • POS    — Wang et al. (2017), good under varying skin tones

Buffer management keeps memory bounded.
All filtering done with scipy (low footprint).
"""

import sys
from collections import deque

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
except ImportError as exc:
    raise ImportError("[signal_processor] scipy is required.") from exc


class SignalProcessor:
    """
    Maintains a rolling RGB buffer and extracts the rPPG signal on demand.
    """

    def __init__(self, config: dict, debug: bool = False):
        self.fps      = config["fps"]
        self.method   = config.get("rppg_method", "chrom")
        self.bp_low   = config["bandpass_low"]
        self.bp_high  = config["bandpass_high"]
        self.bp_order = config["bandpass_order"]
        self.debug    = debug

        max_len = config["buffer_seconds"] * self.fps
        self._rgb_buffer = deque(maxlen=int(max_len))
        self._ts_buffer  = deque(maxlen=int(max_len))

        self._sos = self._design_bandpass()
        print(f"[signal_processor] Method={self.method}  "
              f"BPF={self.bp_low}–{self.bp_high} Hz  buffer={int(max_len)} samples")

    # ── Public ────────────────────────────────────────────────────────────────

    def push(self, rgb: np.ndarray):
        """Append a [R, G, B] sample."""
        import time
        self._rgb_buffer.append(rgb.copy())
        self._ts_buffer.append(time.time())

    def extract_rppg(self):
        """
        Returns (rppg_signal, timestamps) as numpy arrays, or (None, None).
        """
        if len(self._rgb_buffer) < 2:
            return None, None

        rgb = np.array(self._rgb_buffer)          # (N, 3)
        ts  = np.array(self._ts_buffer)           # (N,)

        try:
            if self.method == "chrom":
                raw = self._chrom(rgb)
            elif self.method == "pos":
                raw = self._pos(rgb)
            else:
                print(f"[signal_processor] Unknown method '{self.method}', using chrom.")
                raw = self._chrom(rgb)
        except Exception as exc:
            print(f"[signal_processor] rPPG extraction failed: {exc}", file=sys.stderr)
            return None, None

        # Bandpass filter
        filtered = self._bandpass(raw)

        if self.debug:
            print(f"[signal_processor] rPPG extracted: N={len(filtered)}  "
                  f"range=[{filtered.min():.4f}, {filtered.max():.4f}]")

        return filtered, ts

    def sample_count(self) -> int:
        return len(self._rgb_buffer)

    # ── rPPG methods ──────────────────────────────────────────────────────────

    def _chrom(self, rgb: np.ndarray) -> np.ndarray:
        """
        CHROM method (de Haan & Jeanne, 2013).
        Decomposes RGB into two chrominance channels orthogonal to the
        specular-reflection direction, then combines to cancel motion.
        """
        # Normalise each channel by its mean (temporal normalisation)
        mean = rgb.mean(axis=0) + 1e-8
        rgb_n = rgb / mean                          # (N, 3)

        # Chrominance signals
        Xs = 3 * rgb_n[:, 0] - 2 * rgb_n[:, 1]     # 3R - 2G
        Ys = 1.5 * rgb_n[:, 0] + rgb_n[:, 1] - 1.5 * rgb_n[:, 2]  # 1.5R+G-1.5B

        # Normalise to unit variance
        std_x = Xs.std() + 1e-8
        std_y = Ys.std() + 1e-8
        alpha = std_x / std_y

        signal = Xs - alpha * Ys
        return signal

    def _pos(self, rgb: np.ndarray) -> np.ndarray:
        """
        POS method (Wang et al., 2017).
        Projects normalised RGB onto a plane orthogonal to the skin-tone direction.
        """
        mean = rgb.mean(axis=0) + 1e-8
        C = rgb / mean                              # (N, 3)

        # Projection matrix P (skin-normalised)
        # P maps [R/Rm, G/Gm, B/Bm] → two orthogonal projections
        S1 = C[:, 0] - C[:, 1]
        S2 = C[:, 0] + C[:, 1] - 2 * C[:, 2]

        # Tune alpha to minimise specular/motion component
        std1 = S1.std() + 1e-8
        std2 = S2.std() + 1e-8
        alpha = std1 / std2

        signal = S1 + alpha * S2
        return signal

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _design_bandpass(self):
        """Second-order sections Butterworth bandpass (zero-phase via sosfiltfilt)."""
        nyq = self.fps / 2.0
        low  = self.bp_low  / nyq
        high = self.bp_high / nyq

        # Clamp to avoid degenerate filter
        low  = max(low,  1e-3)
        high = min(high, 0.999)

        if low >= high:
            print(f"[signal_processor] WARNING: bandpass range invalid "
                  f"({self.bp_low}–{self.bp_high} Hz @ {self.fps} fps). "
                  f"Check fps/config.", file=sys.stderr)
            # Return identity (no filtering)
            return None

        try:
            from scipy.signal import butter
            sos = butter(self.bp_order, [low, high], btype="band", output="sos")
            return sos
        except Exception as exc:
            print(f"[signal_processor] Filter design failed: {exc}", file=sys.stderr)
            return None

    def _bandpass(self, signal: np.ndarray) -> np.ndarray:
        if self._sos is None or len(signal) < 2 * self.bp_order + 1:
            return signal
        try:
            return sosfiltfilt(self._sos, signal)
        except Exception as exc:
            print(f"[signal_processor] Bandpass filter failed: {exc}", file=sys.stderr)
            return signal
