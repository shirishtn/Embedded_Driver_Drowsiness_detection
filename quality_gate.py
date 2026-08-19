import cv2
import numpy as np

def create_quality_gate_state(blur_threshold=100.0, min_brightness=30, max_brightness=225, max_motion=5.0):
    """Instantiates config parameters and tracking caches for the Quality filter."""
    return {
        "blur_threshold": blur_threshold,
        "min_brightness": min_brightness,
        "max_brightness": max_brightness,
        "max_motion": max_motion,
        "prev_gray_small": None
    }

def process_quality_checks(gate_state, frame_packet):
    """Vector-SIMD style mathematical evaluation over raw pixel buffers."""
    gray = cv2.cvtColor(frame_packet["data"], cv2.COLOR_BGR2GRAY)
    
    # 1. Illumination check
    mean_intensity = int(np.mean(gray))
    illumination_ok = 1 if gate_state["min_brightness"] <= mean_intensity <= gate_state["max_brightness"] else 0
        
    # 2. Blur detection (Laplacian Variance Method)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_ok = 1 if blur_score >= gate_state["blur_threshold"] else 0
        
    # 3. Motion Magnitude computation via downsampled frame diffs
    gray_small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    motion_ok = 0
    motion_magnitude = 0.0
    
    if gate_state["prev_gray_small"] is not None:
        diff = cv2.absdiff(gate_state["prev_gray_small"], gray_small)
        motion_magnitude = float(np.mean(diff))
        if motion_magnitude <= gate_state["max_motion"]:
            motion_ok = 1
    else:
        motion_ok = 1  # First frame default
        
    gate_state["prev_gray_small"] = gray_small
    overall_pass = 1 if (illumination_ok and blur_ok and motion_ok) else 0
    
    return {
        "illumination_ok": illumination_ok,
        "blur_ok": blur_ok,
        "motion_ok": motion_ok,
        "blur_score": blur_score,
        "motion_magnitude": motion_magnitude,
        "mean_intensity": mean_intensity,
        "overall_pass": overall_pass
    }