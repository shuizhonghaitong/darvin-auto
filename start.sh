#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
# ROS setup scripts may reference optional variables while the platform
# contract launches with a strict shell; keep that setup phase nounset-safe.
set +u
[ -f /opt/ros/humble/install/setup.bash ] && source /opt/ros/humble/install/setup.bash
[ -f /ros_ws/install/setup.bash ] && source /ros_ws/install/setup.bash
[ -f /etc/dla-fallback.env ] && source /etc/dla-fallback.env
set -u

exec python3 perception/main.py
