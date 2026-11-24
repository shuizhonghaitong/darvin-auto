#!/usr/bin/env bash
set -euo pipefail

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
# ROS setup scripts may reference optional variables while the platform
# contract launches with a strict shell; keep that setup phase nounset-safe.
set +u
source /opt/ros/humble/install/setup.bash
source /ros_ws/install/setup.bash
source /etc/dla-fallback.env
set -u

exec python3 /work/main.py
