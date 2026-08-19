import math
import numpy as np

def create_perclos_state(window_size=1800):
    """Maintains sequential tracking queues and configuration scales."""
    return {
        "window_size": window_size,
        "ear_history": np.zeros(window_size, dtype=np.float32),
        "buffer_idx": 0,
        "buffer_count": 0,
        "ear_open_threshold": 0.28,
        "ear_closed_threshold": 0.18,
        "calibration_frames_collected": 0,
        "closed_frame_count": 0,
        "blink_counter": 0,
        "blink_durations": [],
        "metrics": {
            "perclos_percentage": 0.0,
            "total_blinks": 0,
            "avg_blink_duration_ms": 0.0
        }
    }

def _euclidean_distance(p1, p2):
    return math.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2 + (p1["z"] - p2["z"])**2)

def calculate_ear_metrics(landmarks_packet):
    """Pure mathematical transformation extraction across raw landmark points arrays."""
    pts = landmarks_packet["points"]
    
    # Left and Right vertical/horizontal configurations definitions
    l_ear = (_euclidean_distance(pts[159], pts[153]) + _euclidean_distance(pts[145], pts[154])) / (2.0 * _euclidean_distance(pts[33], pts[133]))
    r_ear = (_euclidean_distance(pts[386], pts[374]) + _euclidean_distance(pts[380], pts[381])) / (2.0 * _euclidean_distance(pts[362], pts[263]))
    
    return (l_ear + r_ear) / 2.0

def calibrate_perclos_baseline(perclos_state, avg_ear):
    """Modifies calibration configuration thresholds dynamically inside execution loops."""
    if perclos_state["calibration_frames_collected"] < 900:
        if avg_ear > 0.20:
            perclos_state["ear_open_threshold"] = (0.95 * perclos_state["ear_open_threshold"]) + (0.05 * avg_ear)
            perclos_state["calibration_frames_collected"] += 1
        perclos_state["ear_closed_threshold"] = perclos_state["ear_open_threshold"] * 0.65

def update_perclos_metrics(perclos_state, avg_ear):
    """Processes historical streaming structures to refresh state metrics outputs."""
    is_closed = 1 if avg_ear < perclos_state["ear_closed_threshold"] else 0
    idx = perclos_state["buffer_idx"]
    w_size = perclos_state["window_size"]
    
    # Step data directly into tracking buffer
    perclos_state["ear_history"][idx] = avg_ear
    perclos_state["buffer_idx"] = (idx + 1) % w_size
    perclos_state["buffer_count"] = min(perclos_state["buffer_count"] + 1, w_size)
    
    if is_closed == 1:
        perclos_state["closed_frame_count"] += 1
    else:
        if perclos_state["closed_frame_count"] > 0:
            duration_ms = perclos_state["closed_frame_count"] * 33.3
            perclos_state["blink_durations"].append(duration_ms)
            perclos_state["blink_counter"] += 1
            perclos_state["closed_frame_count"] = 0
            
    if len(perclos_state["blink_durations"]) > 50:
        perclos_state["blink_durations"].pop(0)
        
    b_count = perclos_state["buffer_count"]
    closed_samples = np.sum(perclos_state["ear_history"][:b_count] < perclos_state["ear_closed_threshold"])
    
    # Refresh metrics field states directly 
    perclos_state["metrics"]["perclos_percentage"] = (closed_samples / b_count) * 100.0 if b_count > 0 else 0.0
    perclos_state["metrics"]["total_blinks"] = perclos_state["blink_counter"]
    perclos_state["metrics"]["avg_blink_duration_ms"] = float(np.mean(perclos_state["blink_durations"])) if perclos_state["blink_durations"] else 0.0