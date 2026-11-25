#!/usr/bin/env bash
# Darwin subnet 273 — 具身智能-TTS基础能力-不限音色榜v5 (JP_VERSION=6) startup wrapper.
# Launched by the platform sut shell (harbor-contest.4pd.io/wanglimin/sut/robot_tts:0.0.4)
# from the checked-out repo at PHANTHYMOTUS_COMMIT_ID. The sut chart readiness probe
# is `curl -s -f http://localhost:80/ready`; perception/main.py implements /ready
# (200 once the TTS engine is loaded, 503 while still starting).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer the repo layout (perception/main.py); fall back to a flat /work layout
# if the sut shell copies main.py to /work.
if [ -f "$SCRIPT_DIR/perception/main.py" ]; then
  MAIN_PY="$SCRIPT_DIR/perception/main.py"
  PERCEPTION_DIR="$SCRIPT_DIR/perception"
elif [ -f "$SCRIPT_DIR/main.py" ]; then
  MAIN_PY="$SCRIPT_DIR/main.py"
  PERCEPTION_DIR="$SCRIPT_DIR"
else
  MAIN_PY="/work/main.py"
  PERCEPTION_DIR="/work"
fi

cd "$PERCEPTION_DIR"

# Jetson/aarch64 OpenMP preload (present on the JP6 runtime; skip gracefully
# when the library path differs so a mismatch cannot kill startup).
if [ -f /usr/lib/aarch64-linux-gnu/libgomp.so.1 ]; then
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
fi

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

# Port contract: the sut chart probes http://localhost:80/ready. main.py reads
# MCP_PORT/WS_PORT from the environment (official template requirement); the
# :-defaults only fill in when the shell did not export them, so a shell that
# pins different ports is never overridden.
export MCP_PORT="${MCP_PORT:-80}"
export WS_PORT="${WS_PORT:-15721}"

# Bounded per-instance footprint: the platform evaluates up to 3 service
# instances concurrently on the shared-memory Orin, so each process caps its
# CPU thread pools (matches the config's own num_threads: 2 guidance) and uses
# lazy CUDA module loading to keep the context's init-time memory low. Only
# applied when the shell has not set them.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

exec python3 "$MAIN_PY"
