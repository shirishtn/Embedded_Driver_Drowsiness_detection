"""
face_detector.py — Lightweight face + landmark detector for RPi.

Uses OpenCV Haar cascade (CPU-only, < 2 MB).
For landmarks we use a simple 5-point dlib predictor or fall back to
estimating forehead/cheek positions from the bounding box alone.
"""

import sys
import os
import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("[face_detector] opencv-python is required.") from exc

try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False


# Paths to cascade / predictor files
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# 68-point shape predictor (optional, ~100 MB) — use 5-point if absent
_DLIB_PREDICTOR_PATH = os.path.join(
    os.path.dirname(__file__), "models", "shape_predictor_68_face_landmarks.dat"
)
_DLIB_5PT_PATH = os.path.join(
    os.path.dirname(__file__), "models", "shape_predictor_5_face_landmarks.dat"
)


class FaceDetector:
    """
    Detects the largest frontal face and returns its bounding box plus
    approximate landmark coordinates (dict with 'forehead', 'left_cheek',
    'right_cheek' keys as (x, y) tuples in pixel space).
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._haar  = None
        self._dlib_detector   = None
        self._dlib_predictor  = None
        self._init_detectors()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_detectors(self):
        # Haar cascade — always available
        if not os.path.exists(_HAAR_PATH):
            raise FileNotFoundError(f"[face_detector] Haar XML not found: {_HAAR_PATH}")
        self._haar = cv2.CascadeClassifier(_HAAR_PATH)
        print("[face_detector] Haar cascade loaded.")

        # Optional dlib landmarks
        if _DLIB_AVAILABLE:
            self._dlib_detector = dlib.get_frontal_face_detector()
            for predictor_path in (_DLIB_5PT_PATH, _DLIB_PREDICTOR_PATH):
                if os.path.exists(predictor_path):
                    self._dlib_predictor = dlib.shape_predictor(predictor_path)
                    print(f"[face_detector] dlib predictor loaded: {predictor_path}")
                    break
            if self._dlib_predictor is None:
                print("[face_detector] dlib predictor .dat not found — using bbox landmarks.")
        else:
            print("[face_detector] dlib not available — using bbox-estimated landmarks.")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns:
            face_bbox  : (x, y, w, h) or None
            landmarks  : dict with 'forehead', 'left_cheek', 'right_cheek'
                         each as (cx, cy) pixel coords, or None
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # improve detection under variable light

        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return None, None

        # Largest face by area
        face_bbox = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = face_bbox

        landmarks = self._estimate_landmarks(frame_bgr, face_bbox)

        if self.debug:
            print(f"[face_detector] Face @ ({x},{y}) size={w}×{h}  "
                  f"forehead={landmarks['forehead']} cheeks=("
                  f"{landmarks['left_cheek']}, {landmarks['right_cheek']})")

        return face_bbox, landmarks

    # ── Landmark helpers ──────────────────────────────────────────────────────

    def _estimate_landmarks(self, frame_bgr, face_bbox) -> dict:
        x, y, w, h = face_bbox

        if _DLIB_AVAILABLE and self._dlib_predictor is not None:
            return self._dlib_landmarks(frame_bgr, face_bbox)

        # Geometric estimates relative to bounding box
        forehead     = (x + w // 2, y + int(h * 0.12))
        left_cheek   = (x + int(w * 0.20), y + int(h * 0.55))
        right_cheek  = (x + int(w * 0.80), y + int(h * 0.55))

        return {
            "forehead":    forehead,
            "left_cheek":  left_cheek,
            "right_cheek": right_cheek,
        }

    def _dlib_landmarks(self, frame_bgr, face_bbox) -> dict:
        x, y, w, h = face_bbox
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = self._dlib_predictor(frame_bgr, rect)

        pts = np.array([[shape.part(i).x, shape.part(i).y]
                        for i in range(shape.num_parts)], dtype=np.int32)

        # For 68-pt model: forehead ~above nose bridge (pt 27)
        # For 5-pt model: less granular — fall back to bbox estimate
        if shape.num_parts >= 68:
            nose_bridge = pts[27]
            forehead    = (nose_bridge[0], nose_bridge[1] - int(h * 0.20))
            left_cheek  = tuple(pts[1])   # jaw-line approx
            right_cheek = tuple(pts[15])
        else:
            forehead    = (x + w // 2, y + int(h * 0.12))
            left_cheek  = (x + int(w * 0.20), y + int(h * 0.55))
            right_cheek = (x + int(w * 0.80), y + int(h * 0.55))

        return {
            "forehead":    tuple(forehead),
            "left_cheek":  tuple(left_cheek),
            "right_cheek": tuple(right_cheek),
        }
