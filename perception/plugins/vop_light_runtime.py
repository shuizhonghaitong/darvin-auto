#!/usr/bin/env python3
"""
plugins/vop_light_runtime.py — lightweight 220-class vop engine for the Darwin
VOP subnet (github-commit delivery).

The reference vop bundle (yoloe-26s-seg, 144 frozen classes, prebuilt TensorRT
engine) is a heavy accelerator context on the shared Jetson Orin box: several
concurrent instances OOM the 8 GB unified memory (exit 137). This module loads
the light path instead:

  pinned ONNX (YOLOE-26s-seg, 220 competition classes, nms=False, 640x640;
  image-baked at /opt/vop-models when present, otherwise fetched at start
  from the run's pinned GitHub release by plugins/vop_model_bootstrap.py)
    -> runtime TensorRT build, static shapes, FP16, capped workspace
    -> cached engine (file-locked) so only the first instance pays the build
    -> run through utils.tensorrt_runtime.TensorRTEngine (unchanged reference
       decode path in plugins/vision_runtime.py)

If the baked ONNX is absent or unusable the caller falls back to the reference
COS bundle, so a broken light path never takes the service down.

Env:
  VOP_ONNX_PATH        baked ONNX location (default /opt/vop-models/vop220.onnx)
  VOP_VOCAB_PATH       vocabulary json beside the ONNX (default /opt/vop-models/vocab220.json)
  VOP_ENGINE_CACHE_DIR engine cache dir (default /models/vop-cache; private /tmp fallback)
  VOP_TRT_WORKSPACE_MB TensorRT build workspace cap in MB (default 192)
  VOP_LIGHT=0          disable the light path entirely (force reference bundle)
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import struct
import tempfile
import time

log = logging.getLogger(__name__)

DEFAULT_ONNX_PATH = "/opt/vop-models/vop220.onnx"
DEFAULT_VOCAB_PATH = "/opt/vop-models/vocab220.json"
DEFAULT_CACHE_DIR = "/models/vop-cache"
DEFAULT_WORKSPACE_MB = 192


def light_path_enabled() -> bool:
    return os.environ.get("VOP_LIGHT", "1").strip() not in ("0", "false", "off", "")


def _bootstrap_assets() -> dict[str, str] | None:
    """Fetch the pinned release assets (ONNX + vocab) into a local dir.

    Only attempted when neither the env override nor the image-baked default
    is present, i.e. under the official sut shell image, which is a fixed
    upstream artifact and therefore carries no model of its own.
    """
    from plugins.vop_model_bootstrap import ensure_assets
    return ensure_assets()


def find_onnx() -> str | None:
    path = os.environ.get("VOP_ONNX_PATH", "").strip() or DEFAULT_ONNX_PATH
    if os.path.isfile(path):
        return path
    assets = _bootstrap_assets()
    if assets and os.path.isfile(assets.get("vop220.onnx", "")):
        return assets["vop220.onnx"]
    return None


def _vocab_path() -> str:
    path = os.environ.get("VOCAB_PATH", "").strip() or DEFAULT_VOCAB_PATH
    if os.path.isfile(path):
        return path
    if not os.path.isfile(DEFAULT_ONNX_PATH) and not os.environ.get("VOP_ONNX_PATH", "").strip():
        assets = _bootstrap_assets()
        candidate = (assets or {}).get("vocab220.json", "")
        if candidate and os.path.isfile(candidate):
            return candidate
    return path


def find_vocab() -> list[str]:
    path = _vocab_path()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        names = data.get("classes") if isinstance(data, dict) else data
        if isinstance(names, list):
            return [str(n) for n in names]
    except (OSError, ValueError) as exc:
        log.warning("[vop-light] vocab file unreadable (%s): %s", path, exc)
    return []


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_dir() -> str:
    """A writable, box-local cache dir; /models when it is the shared mount."""
    configured = os.environ.get("VOP_ENGINE_CACHE_DIR", "").strip()
    candidates = [configured, DEFAULT_CACHE_DIR] if configured else [DEFAULT_CACHE_DIR]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".vop-light-probe")
            with open(probe, "w") as handle:
                handle.write("ok")
            os.unlink(probe)
            return candidate
        except OSError:
            continue
    fallback = os.path.join(tempfile.gettempdir(), f"vop-engine-cache-{os.getuid()}")
    os.makedirs(fallback, exist_ok=True)
    log.warning("[vop-light] /models not writable; using private cache %s", fallback)
    return fallback


class _FileLock:
    """Advisory lock so concurrent instances build the engine once."""

    def __init__(self, path: str):
        self._path = path
        self._handle = None

    def __enter__(self):
        self._handle = open(self._path, "a+")
        log.info("[vop-light] waiting for build lock %s", self._path)
        started = time.time()
        fcntl.flock(self._handle, fcntl.LOCK_EX)
        log.info("[vop-light] lock acquired after %.1f s", time.time() - started)
        return self

    def __exit__(self, *exc_info):
        try:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
            self._handle.close()
        except OSError:
            pass
        return False


def _trt_version(trt) -> str:
    return str(getattr(trt, "__version__", "unknown"))


def _set_workspace_limit(config, trt, workspace_bytes: int):
    # TRT 10 moved the pool limit to set_memory_pool_limit(); 8.5 also exposes
    # it, with max_workspace_size as the legacy spelling.
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def _write_engine_with_metadata(serialized: bytes, names: list[str], imgsz: int,
                                onnx_sha: str, dest: str,
                                model_name: str = "vop220"):
    """Serialize the ultralytics-style JSON header the reference reader expects.

    read_engine_file() accepts a plain engine, but with the header
    VisionEngineSession.class_names() can read the 220 names out of the engine
    itself — the same trust rule the reference bundle follows (the header is
    written here from the same list that was baked into the ONNX at export).
    """
    metadata = {
        "task": "detect",
        "name": model_name,
        "imgsz": [imgsz, imgsz],
        "end2end": True,
        "nms": False,
        "names": {str(i): name for i, name in enumerate(names)},
        "light": {"onnx_sha256": onnx_sha, "workspace_mb":
                  int(os.environ.get("VOP_TRT_WORKSPACE_MB", DEFAULT_WORKSPACE_MB))},
    }
    payload = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    with open(dest, "wb") as handle:
        handle.write(struct.pack("<i", len(payload)))
        handle.write(payload)
        handle.write(serialized)


def ensure_light_engine(onnx_path: str, names: list[str], imgsz: int = 640) -> str:
    """Return a ready-to-load .engine for the baked ONNX, building if needed."""
    if not names:
        raise ValueError("ensure_light_engine needs the class list (vocab)")
    onnx_sha = _sha256(onnx_path)
    cache_dir = _cache_dir()
    lock_path = os.path.join(cache_dir, f"build-{onnx_sha[:16]}.lock")
    with _FileLock(lock_path):
        import tensorrt as trt
        version = _trt_version(trt)
        engine_name = f"vop220-{onnx_sha[:16]}-imgsz{imgsz}-trt{version.replace('.', '_')}.engine"
        engine_path = os.path.join(cache_dir, engine_name)
        done_path = engine_path + ".done"
        if os.path.isfile(engine_path) and os.path.isfile(done_path):
            log.info("[vop-light] cached engine ready: %s", engine_path)
            return engine_path

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        if getattr(builder, "platform_has_fast_fp16", lambda: True)():
            config.set_flag(trt.BuilderFlag.FP16)
        try:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        except Exception:
            pass
        workspace_mb = int(os.environ.get("VOP_TRT_WORKSPACE_MB", DEFAULT_WORKSPACE_MB))
        _set_workspace_limit(config, trt, workspace_mb * (1 << 20))

        try:
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except (AttributeError, TypeError):
            flags = 0
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)

        log.info("[vop-light] parsing ONNX %s (%d classes, imgsz %d)",
                 onnx_path, len(names), imgsz)
        with open(onnx_path, "rb") as handle:
            onnx_bytes = handle.read()
        if not parser.parse(onnx_bytes):
            errors = []
            for index in range(parser.num_errors):
                errors.append(str(parser.get_error(index)))
            raise RuntimeError("ONNX parse failed: " + "; ".join(errors[:5]))

        # Static 1x3ximgszximgsz input: the smallest possible context, and the
        # only shape the judge ever sends (single photos, letterboxed).
        for index in range(network.num_bindings):
            tensor = network.get_binding(index)
            if tensor.is_input or (hasattr(tensor, "mode") and
                                   str(tensor.mode).endswith("INPUT")):
                tensor.shape = (1, 3, imgsz, imgsz)
        # Mark output dims static where the parser left them dynamic.
        for index in range(network.num_bindings):
            tensor = network.get_binding(index)
            shape = list(tensor.shape)
            if shape and shape[0] in (-1, None):
                shape[0] = 1
                tensor.shape = tuple(shape)

        started = time.time()
        log.info("[vop-light] building engine (workspace capped at %d MB) ...",
                 workspace_mb)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build returned no serialized engine")
        build_seconds = time.time() - started
        data = bytes(serialized)
        log.info("[vop-light] engine built in %.1f s, %.1f MB",
                 build_seconds, len(data) / 1e6)

        tmp_path = engine_path + f".tmp-{os.getpid()}"
        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        _write_engine_with_metadata(data, names, imgsz, onnx_sha, tmp_path, model_name)
        os.replace(tmp_path, engine_path)
        with open(done_path, "w") as handle:
            json.dump({"onnx_sha256": onnx_sha, "trt": version, "imgsz": imgsz,
                       "build_seconds": round(build_seconds, 1),
                       "engine_bytes": len(data)}, handle)
        return engine_path


def resolve_light_model(onnx_path: str | None = None,
                        names: list[str] | None = None) -> tuple[str, list[str]]:
    """(engine_path, vocab) for the light path, or raises."""
    onnx_path = onnx_path or find_onnx()
    if not onnx_path:
        raise FileNotFoundError(f"no baked ONNX at {onnx_path}")
    names = names or find_vocab()
    if not names:
        raise ValueError("no vocabulary for the light model (vocab220.json missing)")
    imgsz = 640
    return ensure_light_engine(onnx_path, names, imgsz), names
