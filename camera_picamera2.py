"""
Picamera2 camera backend exposing the same minimal API as `camera_interface.py`.
This module provides `create_camera_state`, `initialize_camera`, `start_camera`,
`get_latest_frame`, and `stop_camera` using Picamera2 for Raspberry Pi.
"""

import threading
import time

try:
    from picamera2 import Picamera2
except Exception as exc:
    raise ImportError("picamera2 is required for this backend") from exc


def create_camera_state(video_source=0, width=320, height=240, target_fps=15):
    return {
        "width": width,
        "height": height,
        "target_fps": target_fps,
        "buffers": [None, None],
        "write_idx": 0,
        "read_idx": 1,
        "lock": threading.Lock(),
        "frame_id_counter": 0,
        "is_running": False,
        "picam": None,
        "thread": None,
    }


def initialize_camera(camera_state):
    picam = Picamera2()
    # Use a small preview configuration to reduce CPU work
    config = picam.create_preview_configuration({"main": {"size": (camera_state["width"], camera_state["height"])}})
    picam.configure(config)
    picam.start()
    camera_state["picam"] = picam
    return True


def _capture_thread_loop(camera_state):
    picam = camera_state["picam"]
    frame_duration = 1.0 / camera_state["target_fps"]
    while camera_state["is_running"]:
        start_time = time.perf_counter()
        try:
            frame = picam.capture_array()
        except Exception:
            # if capture fails, skip this iteration
            continue
        timestamp_us = int(time.time() * 1_000_000)
        camera_state["frame_id_counter"] += 1
        frame_packet = {
            "data": frame,
            "width": camera_state["width"],
            "height": camera_state["height"],
            "stride": frame.strides[0],
            "timestamp_us": timestamp_us,
            "frame_id": camera_state["frame_id_counter"],
        }
        with camera_state["lock"]:
            w_idx = camera_state["write_idx"]
            camera_state["buffers"][w_idx] = frame_packet
            camera_state["write_idx"], camera_state["read_idx"] = camera_state["read_idx"], camera_state["write_idx"]
        elapsed = time.perf_counter() - start_time
        sleep_time = frame_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def start_camera(camera_state):
    """Spawns the capture thread."""
    if camera_state.get("picam") is None:
        raise RuntimeError("Picamera2 must be initialized before starting.")
    camera_state["is_running"] = True
    camera_state["thread"] = threading.Thread(target=_capture_thread_loop, args=(camera_state,), daemon=True)
    camera_state["thread"].start()


def get_latest_frame(camera_state):
    """Thread-safe access to the newest available read buffer frame."""
    with camera_state["lock"]:
        return camera_state["buffers"][camera_state["read_idx"]]


def stop_camera(camera_state):
    camera_state["is_running"] = False
    if camera_state.get("thread"):
        camera_state["thread"].join(timeout=1.0)
    if camera_state.get("picam"):
        try:
            camera_state["picam"].stop()
        except Exception:
            pass