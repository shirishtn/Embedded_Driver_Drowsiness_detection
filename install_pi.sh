#!/usr/bin/env bash
set -euo pipefail

# Minimal installer script for Raspberry Pi (Debian/Ubuntu based)
# Run with: sudo ./install_pi.sh

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root: sudo $0" >&2
  exit 1
fi

apt update
apt install -y python3 python3-venv python3-pip libopenblas-dev libopenblas0-pthread libjpeg-dev libopenjp2-7-dev libtiff-dev

# Picamera2 and libcamera (Bookworm or newer). On some OS versions picamera2
# is available via apt. If not, install via pip after enabling backports.
apt install -y python3-picamera2 libcamera-apps || true

# Create a venv and install Python requirements
sudo -u $SUDO_USER bash -lc '
python3 -m venv ~/rpi_project_venv
source ~/rpi_project_venv/bin/activate
pip install --upgrade pip
pip install -r rpi_project/requirements.txt
'

echo "Installation complete. Activate the venv with: source ~/rpi_project_venv/bin/activate"
