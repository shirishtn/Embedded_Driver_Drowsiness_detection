import argparse
import os
import sys
import time

import cv2

# Allow both `python -m rpi_project.main_pipeline` and direct script execution.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if __package__ in (None, ""):
        from rpi_project import landmark_engine as le
    else:
        from . import landmark_engine as le
except ImportError:
    import landmark_engine as le

try:
    if __package__ in (None, ""):
        from rpi_project import camera_picamera2 as cam
    else:
        from . import camera_picamera2 as cam
    print("[INFO] Using Picamera2 camera backend")
except ImportError:
    import camera_interface as cam
    print("[INFO] Using OpenCV camera backend")

import quality_gate as qg
from driver_state_engine import DriverStateMonitor
from HRV_scripts.face_detector import FaceDetector
from HRV_scripts.hrv_calculator import HRVCalculator
from HRV_scripts.roi_extractor import ROIExtractor
from HRV_scripts.signal_processor import SignalProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Run fused PERCLOS + HRV driver-state monitoring (RPi-optimized)")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument("--show-video", action="store_true", default=False, help="Display the video stream")
    return parser.parse_args()


def main():
    args = parse_args()
    print("[INIT] Starting RPi-optimized driver-state monitoring pipeline")

    # Lower resolution and FPS for Raspberry Pi 1GB
    width = 320
    height = 240
    fps = 15

    camera_state = cam.create_camera_state(video_source=args.source, width=width, height=height, target_fps=fps)
    gate_state = qg.create_quality_gate_state(blur_threshold=10.0, min_brightness=30, max_brightness=230, max_motion=8.0)
    tracker_state = le.create_landmark_tracker_state()
    monitor = DriverStateMonitor(perclos_window=int(fps * 60), hrv_window=6, debug=False)

    hrv_config = {
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "buffer_seconds": 20,
        "rppg_method": "chrom",
        "bandpass_low": 0.7,
        "bandpass_high": 3.5,
        "bandpass_order": 4,
        "motion_threshold": 0.18,
        "illumination_threshold": 0.22,
    }
    face_detector = FaceDetector(debug=False)
    roi_extractor = ROIExtractor(hrv_config, debug=False)
    signal_processor = SignalProcessor(hrv_config, debug=False)
    hrv_calculator = HRVCalculator(hrv_config, debug=False)

    print("[DEBUG] Attempting to open camera/video source")
    if not cam.initialize_camera(camera_state):
        print("[FATAL] Unable to open the requested camera/video source")
        return

    cam.start_camera(camera_state)

    warmup_start = time.time()
    first_frame = None
    while first_frame is None:
        first_frame = cam.get_latest_frame(camera_state)
        time.sleep(0.05)
        if time.time() - warmup_start > 5.0:
            print("[FATAL] Camera source did not deliver a frame in time")
            cam.stop_camera(camera_state)
            return

    frame_process_count = 0
    last_hrv_update = 0.0

    try:
        while True:
            start_loop_time = time.perf_counter()
            frame_packet = cam.get_latest_frame(camera_state)
            if frame_packet is None:
                time.sleep(0.005)
                continue

            frame_process_count += 1
            frame_id = frame_packet["frame_id"]
            frame = frame_packet["data"]

            q_metrics = qg.process_quality_checks(gate_state, frame_packet)
            if args.show_video:
                cv2.imshow("Driver State Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if q_metrics["overall_pass"] == 0:
                continue

            # Use lightweight landmark engine
            landmarks_packet = le.process_face_landmarks(tracker_state, frame_packet)
            if landmarks_packet["detection_valid"] == 0:
                continue

            avg_ear = 0.0
            if len(landmarks_packet["points"]) >= 160:
                import perclos_engine as pe
                avg_ear = pe.calculate_ear_metrics(landmarks_packet)

            hrv_metrics = None
            # Less frequent HRV updates on low-power device
            if time.time() - last_hrv_update >= 8.0:
                last_hrv_update = time.time()
                face_bbox, _ = face_detector.detect(frame)
                if face_bbox is not None:
                    roi_rgb, motion_flag, illum_flag = roi_extractor.extract(frame, face_bbox, None)
                    if roi_rgb is not None and not motion_flag and not illum_flag:
                        signal_processor.push(roi_rgb)
                        rppg_signal, _ = signal_processor.extract_rppg()
                        if rppg_signal is not None and len(rppg_signal) >= int(hrv_config["fps"] * 5):
                            hrv_metrics = hrv_calculator.compute(rppg_signal, hrv_config["fps"])

            state = monitor.update(avg_ear=avg_ear, hrv_metrics=hrv_metrics, blink_rate=None, use_perclos_calibration=True)

            if frame_id % (int(fps * 1.0)) == 0:
                print(
                    f"[STATE] frame={frame_id} state={state['state']} score={state['alertness_score']:.3f} "
                    f"perclos={state['perclos_percentage']:.2f}% hrv_drop={state['hrv_drop_detected']}"
                )

            loop_duration = time.perf_counter() - start_loop_time
            sleep_slack = (1.0 / fps) - loop_duration
            if sleep_slack > 0:
                time.sleep(sleep_slack)

    except KeyboardInterrupt:
        print("[TEARDOWN] Interrupted by user")
    finally:
        cam.stop_camera(camera_state)
        cv2.destroyAllWindows()
        print(f"[SUCCESS] Pipeline stopped after {frame_process_count} frames")


if __name__ == "__main__":
    main()
