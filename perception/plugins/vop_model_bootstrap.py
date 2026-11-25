#!/usr/bin/env python3
"""
plugins/vop_model_bootstrap.py — fetch the light-path ONNX + vocab from the
run's pinned public release into a writable local dir, before the runtime
TensorRT build (see plugins/vop_light_runtime.py).

The official sut shell image is fixed; it clones this repo at the pinned
commit and runs perception, so the model cannot be baked into an image. It is
hosted as a GitHub release asset of this very repository (github.com is
reachable from the Jetson eval nodes — same conclusion the upstream export
script records for its own hosts) and downloaded here at service start:

  <VOP_LIGHT_MODEL_BASE>/vop220.onnx
  <VOP_LIGHT_MODEL_BASE>/vocab220.json

Both are pinned by SHA-256 below; a mismatch is treated as a failed fetch
(the reference COS bundle fallback in vop.py then applies, so the service
never silently runs different weights than the ones it was reviewed with).

Env:
  VOP_LIGHT_MODEL_BASE   release asset base URL (default pinned below)
  VOP_LIGHT_MODEL_DIR    writable destination (default /models/vop-light,
                         private /tmp fallback when /models is read-only)
  VOP_LIGHT_FETCH_TIMEOUT  per-file download timeout seconds (default 600)
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import tempfile
import threading
import time
import urllib.request

log = logging.getLogger(__name__)

DEFAULT_BASE = (
    "https://github.com/shuizhonghaitong/darvin-auto/releases/download/"
    "agent-cb294026112500000000000000000294-vop-models-v1"
)

# Pinned assets (sha256 of the *uploaded* release file, verified on re-download
# exactly like the reference VOP_MODEL_BUNDLES pins in utils/model_downloader.py).
ASSETS: dict[str, dict] = {
    "vop220.onnx": {
        "sha256": "65fa109b0ccc8f1748458a7134d1c4b6d71ee60a3cf9112c4d9333085e870f70",
        "size": 42132435,
    },
    "vocab220.json": {
        "sha256": "7dfc0fae36eb34f6205af1994fb3a24a8a853ba478fd8dad55fe64fec747b707",
        "size": 3052,
    },
}

_local = threading.Lock()
_done: dict[str, str | None] = {}


def _base_url() -> str:
    return (os.environ.get("VOP_LIGHT_MODEL_BASE", "").strip() or DEFAULT_BASE).rstrip("/")


def _dest_dir() -> str:
    configured = os.environ.get("VOP_LIGHT_MODEL_DIR", "").strip()
    candidates = [configured, "/models/vop-light"] if configured else ["/models/vop-light"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".vop-bootstrap-probe")
            with open(probe, "w") as handle:
                handle.write("ok")
            os.unlink(probe)
            return candidate
        except OSError:
            continue
    fallback = os.path.join(tempfile.gettempdir(), f"vop-light-models-{os.getuid()}")
    os.makedirs(fallback, exist_ok=True)
    log.warning("[vop-bootstrap] /models not writable; using %s", fallback)
    return fallback


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(base: str, name: str, dest_dir: str, timeout: float) -> str | None:
    url = f"{base}/{name}"
    target = os.path.join(dest_dir, name)
    part = f"{target}.part-{os.getpid()}"
    log.info("[vop-bootstrap] downloading %s (%s) ...", name, url)
    started = time.time()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "darwin-vop-sut/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response, open(part, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        if os.path.getsize(part) > (1 << 30):
            raise RuntimeError(f"download size {os.path.getsize(part)} exceeds 1 GiB sanity cap")
        digest = _sha256(part)
        pinned = ASSETS[name]["sha256"]
        if digest != pinned:
            raise RuntimeError(f"sha256 mismatch for {name}: got {digest}, pinned {pinned}")
        os.replace(part, target)
        log.info("[vop-bootstrap] %s verified in %.1f s (sha256 %s…)", name,
                 time.time() - started, digest[:12])
        return target
    except Exception as exc:  # noqa: BLE001 — every failure is a fallback trigger
        log.error("[vop-bootstrap] %s failed: %s", name, exc)
        try:
            if os.path.exists(part):
                os.unlink(part)
        except OSError:
            pass
        return None


def ensure_assets() -> dict[str, str] | None:
    """Return {name: path} for all pinned assets, downloading what is missing.

    Returns None when anything is unavailable; the caller then uses the
    reference bundle. Process-single-flight via a module lock, and
    box-single-flight via a file lock, so concurrent instances share one
    download.
    """
    with _local:
        if all(name in _done for name in ASSETS):
            paths = {n: _done[n] for n in ASSETS}
            return paths if all(paths.values()) else None

    dest_dir = _dest_dir()
    lock_path = os.path.join(dest_dir, ".bootstrap.lock")
    timeout = float(os.environ.get("VOP_LIGHT_FETCH_TIMEOUT", "600"))
    base = _base_url()
    # One flat critical section: record into _done before releasing the lock.
    # (Nested acquisition of a non-reentrant Lock here deadlocked when the
    # vocab prefetch and the model load bootstrapped concurrently.)
    started = time.time()
    log.info("[vop-bootstrap] waiting for box download lock ...")
    with open(lock_path, "a+") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        log.info("[vop-bootstrap] lock acquired after %.1f s", time.time() - started)
        paths: dict[str, str] = {}
        for name, meta in ASSETS.items():
            candidate = os.path.join(dest_dir, name)
            with _local:
                cached = _done.get(name)
            if cached is not None or (os.path.isfile(candidate)
                                      and _sha256(candidate) == meta["sha256"]):
                paths[name] = cached or candidate
                with _local:
                    _done[name] = paths[name]
                continue
            path = _download(base, name, dest_dir, timeout)
            if path is None:
                return None
            paths[name] = path
            with _local:
                _done[name] = path
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return paths
