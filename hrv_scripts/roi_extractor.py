"""
roi_extractor.py — Extract mean RGB from skin ROIs with
motion and illumination artefact detection.

Motion detection: frame-differencing on forehead patch (fast, no optical-flow library needed).
Illumination detection: relative change in mean luminance.
"""

from __future__ import annotations

import sys
import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("[roi_extractor] opencv-python is required.") from exc


class ROIExtractor:
    """
    Given a BGR frame, face bbox and landmark dict, returns:
        roi_rgb     : np.array([R_mean, G_mean, B_mean]) float64
        motion_flag : bool — True if motion artefact detected
        illum_flag  : bool — True if illumination artefact detected
    """

    # Fraction of face bounding box used for each ROI patch
    _FOREHEAD_REGION   = dict(x_frac=(0.30, 0.70), y_frac=(0.07, 0.22))
    _LEFT_CHEEK_REGION  = dict(x_frac=(0.05, 0.35), y_frac=(0.45, 0.65))
    _RIGHT_CHEEK_REGION = dict(x_frac=(0.65, 0.95), y_frac=(0.45, 0.65))

    def __init__(self, config: dict | None = None, debug: bool = False):
        self.debug = debug
        self._motion_threshold = (config or {}).get("motion_threshold", 0.15)
        self._illum_threshold  = (config or {}).get("illumination_threshold", 0.20)

        self._prev_forehead_gray: np.ndarray | None = None
        self._prev_lum: float | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(self, frame_bgr: np.ndarray, face_bbox, landmarks):
        """
        Returns (roi_rgb, motion_flag, illum_flag).
        roi_rgb is None if the patch is invalid.
        """
        x, y, w, h = face_bbox

        # ── Illumination check (whole frame luminance) ─────────────────────
        illum_flag = self._check_illumination(frame_bgr)

        # ── Extract patches ────────────────────────────────────────────────
        patches = []
        for region in (self._FOREHEAD_REGION,
                       self._LEFT_CHEEK_REGION,
                       self._RIGHT_CHEEK_REGION):
            patch = self._crop_patch(frame_bgr, x, y, w, h, region)
            if patch is not None and patch.size > 0:
                patches.append(patch)

        if not patches:
            print("[roi_extractor] No valid ROI patches extracted.", file=sys.stderr)
            return None, False, illum_flag

        # ── Motion check on forehead patch ────────────────────────────────
        forehead_patch = self._crop_patch(frame_bgr, x, y, w, h, self._FOREHEAD_REGION)
        motion_flag = self._check_motion(forehead_patch)

        # ── Mean RGB across all patches ───────────────────────────────────
        all_pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
        # patches are BGR; convert to RGB
        roi_rgb = np.array([
            all_pixels[:, 2].mean(),  # R
            all_pixels[:, 1].mean(),  # G
            all_pixels[:, 0].mean(),  # B
        ], dtype=np.float64)

        if self.debug:
            print(f"[roi_extractor] RGB={roi_rgb.round(1)}  "
                  f"motion={motion_flag}  illum={illum_flag}")

        return roi_rgb, motion_flag, illum_flag

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _crop_patch(self, frame, fx, fy, fw, fh, region) -> np.ndarray | None:
        """Crop a sub-region defined as fractions of the face bounding box."""
        x0 = fx + int(fw * region["x_frac"][0])
        x1 = fx + int(fw * region["x_frac"][1])
        y0 = fy + int(fh * region["y_frac"][0])
        y1 = fy + int(fh * region["y_frac"][1])

        H, W = frame.shape[:2]
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)

        if x1 <= x0 or y1 <= y0:
            return None

        return frame[y0:y1, x0:x1]

    def _check_motion(self, patch: np.ndarray | None) -> bool:
        """Frame-differencing on greyscale forehead patch."""
        if patch is None or patch.size == 0:
            self._prev_forehead_gray = None
            return False

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._prev_forehead_gray is None or \
                self._prev_forehead_gray.shape != gray.shape:
            self._prev_forehead_gray = gray
            return False

        diff = np.abs(gray - self._prev_forehead_gray)
        # Normalise by max possible value (255)
        motion_score = diff.mean() / 255.0
        self._prev_forehead_gray = gray

        if self.debug and motion_score > self._motion_threshold * 0.5:
            print(f"[roi_extractor] Motion score: {motion_score:.4f}")

        return float(motion_score) > self._motion_threshold

    def _check_illumination(self, frame_bgr: np.ndarray) -> bool:
        """Detect sudden global illumination change via mean luminance."""
        # Fast: subsample 1-in-4 pixels
        sub = frame_bgr[::4, ::4]
        lum = 0.114 * sub[:, :, 0].mean() + \
              0.587 * sub[:, :, 1].mean() + \
              0.299 * sub[:, :, 2].mean()

        if self._prev_lum is None:
            self._prev_lum = lum
            return False

        rel_change = abs(lum - self._prev_lum) / (self._prev_lum + 1e-6)
        self._prev_lum = lum

        if self.debug and rel_change > self._illum_threshold * 0.5:
            print(f"[roi_extractor] Illumination change: {rel_change:.4f}")

        return float(rel_change) > self._illum_threshold
