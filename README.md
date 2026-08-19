RPi-optimized driver drowsiness pipeline

This is a self-contained, light-weight runtime folder and a simplified
`landmark_engine` that avoids MediaPipe, targeting Raspberry Pi with 1GB RAM.

What changed:
- `main_pipeline.py`: lower resolution (320x240), lower FPS (15), reduced HRV
  update frequency, and uses the lightweight `landmark_engine` in this folder.
- `landmark_engine.py`: Haar cascade based face+eye detector that synthesizes a
  MediaPipe-like `points` list so the rest of the pipeline (PERCLOS, HRV,
  fuzzy decision) can remain unchanged.
- `requirements.txt`: RPi-friendly package suggestions.

How to run on the Pi:

1. Copy the complete `rpi_project` folder to the Pi. Do not copy only
   `main_pipeline.py`; its sibling modules and `hrv_scripts` folder are needed.

2. Create a virtualenv and install dependencies (prefer `opencv-python-headless`):

```bash
python3 -m venv venv
source venv/bin/activate
cd rpi_project
pip install -r requirements.txt
```

3. From inside that folder run:

```bash
python3 main_pipeline.py --source 0
```

The pipeline deliberately uses only same-folder imports, so it does not need
the parent project directory or `PYTHONPATH` configuration.

The default is headless, suitable for deployment without a desktop session.
For development with an OpenCV preview window, add `--display` and press `q`
in that window to stop:

```bash
python3 main_pipeline.py --source 0 --display
```

Pi camera usage (Picamera2)
---------------------------
If you have the official Raspberry Pi camera and `python3-picamera2` is installed,
the pipeline will prefer the Picamera2 backend automatically. To install and
run the helper install script (requires sudo):

```bash
sudo ./rpi_project/install_pi.sh
source ~/rpi_project_venv/bin/activate
cd rpi_project
python3 main_pipeline.py --source 0
```

If Picamera2 is not available, the code will fall back to the OpenCV camera
backend.

Notes and limitations:
- This lightweight engine trades landmark precision for CPU/memory efficiency.
  HRV and PERCLOS logic are preserved, but absolute accuracy will be lower
  than MediaPipe-based runs.
- For best performance on RPi, install optimized OpenCV from system packages
  or use a prebuilt wheel compatible with your Pi's architecture.
