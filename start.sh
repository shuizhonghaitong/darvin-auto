#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# The SUT chart's readiness probe curls http://localhost:80/ready and the
# platform wrapper may launch without the build-time ENV, so the competition
# ports get defaults here as well as in main.py (env wins over config).
export MCP_PORT="${MCP_PORT:-80}"
export WS_PORT="${WS_PORT:-15721}"

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
# ROS setup scripts may reference optional variables while the platform
# contract launches with a strict shell; keep that setup phase nounset-safe.
set +u
[ -f /opt/ros/humble/install/setup.bash ] && source /opt/ros/humble/install/setup.bash
[ -f /ros_ws/install/setup.bash ] && source /ros_ws/install/setup.bash
[ -f /etc/dla-fallback.env ] && source /etc/dla-fallback.env
set -u

exec python3 perception/main.py
