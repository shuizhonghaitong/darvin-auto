#!/usr/bin/env bash
# Darwin embodied-perception OCR (subnet 276) startup wrapper.
# The platform sut shell launches this from the checked-out repo. We make the
# ROS setup phase nounset-safe (setup scripts reference optional variables),
# then hand off to perception/main.py, which reads MCP_PORT / WS_PORT from env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer the repo layout (perception/main.py); fall back to a flat /work layout
# if the sut shell copies main.py to /work.
if [ -f "$SCRIPT_DIR/perception/main.py" ]; then
  MAIN_PY="$SCRIPT_DIR/perception/main.py"
elif [ -f "$SCRIPT_DIR/main.py" ]; then
  MAIN_PY="$SCRIPT_DIR/main.py"
else
  MAIN_PY="/work/main.py"
fi

# Jetson/aarch64 OpenMP preload (present on the JP6 runtime; skip gracefully
# when the library path differs so a mismatch cannot kill startup).
if [ -f /usr/lib/aarch64-linux-gnu/libgomp.so.1 ]; then
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
fi

# ROS 2 Humble + perception workspace setup; keep this phase nounset-safe.
set +u
if [ -f /opt/ros/humble/install/setup.bash ]; then
  source /opt/ros/humble/install/setup.bash
fi
if [ -f /ros_ws/install/setup.bash ]; then
  source /ros_ws/install/setup.bash
fi
if [ -f /etc/dla-fallback.env ]; then
  source /etc/dla-fallback.env
fi
set -u

exec python3 "$MAIN_PY"
