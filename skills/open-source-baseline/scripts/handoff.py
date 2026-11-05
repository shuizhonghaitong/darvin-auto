#!/usr/bin/env python3
"""Validate a candidate handoff and optionally smoke-test the reviewed adapter."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
ENTRY = "image_auth_baseline/main.py"


def read_handoff(text, root=ROOT):
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if not match:
            raise ValueError("expected exactly one JSON fence")
        text = match.group(1)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    data = json.loads(text, object_pairs_hook=unique)
    if not isinstance(data, dict):
        raise ValueError("handoff must be an object")
    for key in ("candidate", "code_path", "weights_path", "license_status"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for key, kind in (("interface", dict), ("evidence", dict), ("adaptation_gaps", list)):
        if not isinstance(data.get(key), kind) or not data[key]:
            raise ValueError(f"{key} must be a nonempty {kind.__name__}")
    root = root.resolve()
    for key in ("code_path", "weights_path"):
        raw = Path(data[key])
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError(f"invalid {key}")
        path = (root / raw).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"{key} is missing or outside workspace")
        data[key] = path.relative_to(root).as_posix()
    return data


def smoke(data, root=ROOT):
    if data["code_path"] != ENTRY:
        raise ValueError("only the reviewed image-auth adapter is supported")
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 80, 40)).save(buf, format="PNG")
    payload = json.dumps({"image_base64": base64.b64encode(buf.getvalue()).decode(), "image_format": "png"}).encode()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(MODEL_PATH=str(root / data["weights_path"]), MOCK_SERVER_PORT=str(port),
               MOCK_SERVER_HOST="127.0.0.1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    proc = subprocess.Popen([sys.executable, str(root / ENTRY)], cwd=root / "image_auth_baseline",
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic() + 60
        while True:
            if proc.poll() is not None:
                raise RuntimeError("service exited: " + proc.stderr.read().decode(errors="replace")[-3000:])
            try:
                with opener.open(base + "/ready", timeout=2) as response:
                    if response.read() == b"True":
                        break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("service readiness timed out")
            time.sleep(0.2)
        req = urllib.request.Request(base + "/api/v1/image-authenticity", payload, {"Content-Type": "application/json"})
        with opener.open(req, timeout=60) as response:
            result = json.load(response)
        if result.get("success") is not True or result.get("result") not in ("1", "0_1", "0_2"):
            raise RuntimeError("unexpected inference response: " + repr(result))
        return {"ready": True, "weights_path": data["weights_path"], "response": result}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        data = read_handoff(args.handoff.read_text(encoding="utf-8"))
        print(json.dumps(smoke(data) if args.smoke else data, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"handoff failed: {exc}\n")


if __name__ == "__main__":
    main()
