"""
Lightweight landmark engine for Raspberry Pi.
Detects face and eyes using OpenCV Haar cascades and synthesizes a MediaPipe-like
`points` list with normalized coordinates so the rest of the pipeline (PERCLOS,
HRV fusion) can remain unchanged.

This is a best-effort, low-cost replacement for MediaPipe FaceMesh on low-RAM
RPi devices. It returns a `points` list of length 478 (filled with zeros) and
sets a few key indices used by the rest of the code.
"""

import cv2
import numpy as np
import os


def _find_haar_cascade(filename):
    """Locate an OpenCV Haar cascade on both pip and apt-installed OpenCV."""
    directories = []
    cv2_data = getattr(cv2, "data", None)
    if cv2_data is not None:
        directories.append(getattr(cv2_data, "haarcascades", ""))

    # Raspberry Pi OS's apt package commonly installs cascades in one of these.
    directories.extend((
        "/usr/share/opencv4/haarcascades",
        "/usr/share/opencv/haarcascades",
        "/usr/local/share/opencv4/haarcascades",
    ))

    for directory in directories:
        path = os.path.join(directory, filename)
        if directory and os.path.isfile(path):
            return path

    raise RuntimeError(
        f"Cannot find {filename}. Install OpenCV Haar cascade data, for example: "
        "sudo apt install opencv-data"
    )


# Supports both pip OpenCV (`cv2.data`) and Raspberry Pi OS's apt OpenCV.
_HAAR_FACE = _find_haar_cascade("haarcascade_frontalface_default.xml")
_HAAR_EYE = _find_haar_cascade("haarcascade_eye.xml")

# Indices we will populate (MediaPipe-like indices used elsewhere)
_KEY_INDICES = {
    "nose": 1,
    "chin": 199,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
    # Eye lid / detail indices for EAR computations (approx locations)
    "left_upper_lid": 159,
    "left_lower_lid": 153,
    "left_inner_lid": 145,
    "left_inner2": 154,
    "right_upper_lid": 386,
    "right_lower_lid": 374,
    "right_inner_lid": 380,
    "right_inner2": 381,
}


def create_landmark_tracker_state():
    face_cascade = cv2.CascadeClassifier(_HAAR_FACE)
    eye_cascade = cv2.CascadeClassifier(_HAAR_EYE)

    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Haar cascade files not found in OpenCV data folder.")

    return {"face_cascade": face_cascade, "eye_cascade": eye_cascade}


def _make_empty_points(n=478):
    return [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(n)]


def _normalize(pt_x, pt_y, w, h):
    # Return normalized coordinates similar to MediaPipe (0..1)
    return float(pt_x) / float(w) if w > 0 else 0.0, float(pt_y) / float(h) if h > 0 else 0.0


def process_face_landmarks(tracker_state, frame_packet):
    output = {"points": [], "timestamp_us": frame_packet.get("timestamp_us", 0),
              "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "detection_valid": 0}

    if frame_packet is None or "data" not in frame_packet or frame_packet["data"] is None:
        return output

    frame = frame_packet["data"]
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = tracker_state["face_cascade"].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        # No face found
        output["points"] = _make_empty_points()
        return output

    # Use largest face
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])

    # Search for eyes inside face ROI
    face_roi_gray = gray[y:y+fh, x:x+fw]
    eyes = tracker_state["eye_cascade"].detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 10))

    # Prepare empty points and populate a few key indices
    pts = _make_empty_points()

    # Approximate nose as center-top of face box
    nose_x = x + fw * 0.5
    nose_y = y + fh * 0.35
    nx, ny = _normalize(nose_x, nose_y, w, h)
    pts[_KEY_INDICES["nose"]] = {"x": nx, "y": ny, "z": 0.0}

    # Chin approximated as bottom-center
    chin_x = x + fw * 0.5
    chin_y = y + fh * 0.95
    cx, cy = _normalize(chin_x, chin_y, w, h)
    pts[_KEY_INDICES["chin"]] = {"x": cx, "y": cy, "z": 0.0}

    # Mouth corners approximate
    left_mouth_x = x + fw * 0.32
    left_mouth_y = y + fh * 0.75
    right_mouth_x = x + fw * 0.68
    right_mouth_y = y + fh * 0.75
    lm_x, lm_y = _normalize(left_mouth_x, left_mouth_y, w, h)
    rm_x, rm_y = _normalize(right_mouth_x, right_mouth_y, w, h)
    pts[_KEY_INDICES["left_mouth"]] = {"x": lm_x, "y": lm_y, "z": 0.0}
    pts[_KEY_INDICES["right_mouth"]] = {"x": rm_x, "y": rm_y, "z": 0.0}

    # Eye positions: if Haar found eyes, use them; otherwise estimate from face box
    left_eye_bbox = None
    right_eye_bbox = None
    for (ex, ey, ew, eh) in eyes:
        # convert to coordinates in whole frame
        ex_abs = x + ex
        ey_abs = y + ey
        cx_eye = ex_abs + ew * 0.5
        cy_eye = ey_abs + eh * 0.5
        # Heuristic: decide left/right based on center relative to face center
        if cx_eye < x + fw * 0.5 and left_eye_bbox is None:
            left_eye_bbox = (ex_abs, ey_abs, ew, eh)
        elif cx_eye >= x + fw * 0.5 and right_eye_bbox is None:
            right_eye_bbox = (ex_abs, ey_abs, ew, eh)

    if left_eye_bbox is None:
        # estimate
        left_eye_bbox = (x + int(fw * 0.18), y + int(fh * 0.28), int(fw * 0.18), int(fh * 0.12))
    if right_eye_bbox is None:
        right_eye_bbox = (x + int(fw * 0.64), y + int(fh * 0.28), int(fw * 0.18), int(fh * 0.12))

    # Left eye key points
    lex, ley, lew, leh = left_eye_bbox
    l_cx = lex + lew * 0.5
    l_cy = ley + leh * 0.5
    l_upper = ley + leh * 0.25
    l_lower = ley + leh * 0.75

    lx_c, ly_c = _normalize(l_cx, l_cy, w, h)
    l_up_x, l_up_y = _normalize(l_cx, l_upper, w, h)
    l_lo_x, l_lo_y = _normalize(l_cx, l_lower, w, h)

    pts[_KEY_INDICES["left_eye_outer"]] = {"x": lx_c, "y": ly_c, "z": 0.0}
    pts[_KEY_INDICES["left_upper_lid"]] = {"x": l_up_x, "y": l_up_y, "z": 0.0}
    pts[_KEY_INDICES["left_lower_lid"]] = {"x": l_lo_x, "y": l_lo_y, "z": 0.0}
    pts[_KEY_INDICES["left_inner_lid"]] = {"x": _normalize(lex + lew * 0.3, ley + leh * 0.5, w, h)[0], "y": _normalize(lex + lew * 0.3, ley + leh * 0.5, w, h)[1], "z": 0.0}
    pts[_KEY_INDICES["left_inner2"]] = {"x": _normalize(lex + lew * 0.7, ley + leh * 0.5, w, h)[0], "y": _normalize(lex + lew * 0.7, ley + leh * 0.5, w, h)[1], "z": 0.0}

    # Right eye key points
    rex, rey, rew, reh = right_eye_bbox
    r_cx = rex + rew * 0.5
    r_cy = rey + reh * 0.5
    r_upper = rey + reh * 0.25
    r_lower = rey + reh * 0.75

    rx_c, ry_c = _normalize(r_cx, r_cy, w, h)
    r_up_x, r_up_y = _normalize(r_cx, r_upper, w, h)
    r_lo_x, r_lo_y = _normalize(r_cx, r_lower, w, h)

    pts[_KEY_INDICES["right_eye_outer"]] = {"x": rx_c, "y": ry_c, "z": 0.0}
    pts[_KEY_INDICES["right_upper_lid"]] = {"x": r_up_x, "y": r_up_y, "z": 0.0}
    pts[_KEY_INDICES["right_lower_lid"]] = {"x": r_lo_x, "y": r_lo_y, "z": 0.0}
    pts[_KEY_INDICES["right_inner_lid"]] = {"x": _normalize(rex + rew * 0.3, rey + reh * 0.5, w, h)[0], "y": _normalize(rex + rew * 0.3, rey + reh * 0.5, w, h)[1], "z": 0.0}
    pts[_KEY_INDICES["right_inner2"]] = {"x": _normalize(rex + rew * 0.7, rey + reh * 0.5, w, h)[0], "y": _normalize(rex + rew * 0.7, rey + reh * 0.5, w, h)[1], "z": 0.0}

    output["points"] = pts
    output["detection_valid"] = 1
    # yaw/pitch/roll remain 0 (not computed) — sufficient for PERCLOS and HRV
    return output
