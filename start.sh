#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
source /opt/ros/humble/install/setup.bash
source /ros_ws/install/setup.bash
source /etc/dla-fallback.env

exec python3 /work/main.py
