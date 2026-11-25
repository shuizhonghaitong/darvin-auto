#!/usr/bin/env python3
"""
perception/main.py — Perception Stack bundle 统一入口。

读取 config.yaml，按插件配置加载各感知插件，聚合成一个 MCP HTTP server 对外暴露：

  asr              语音识别（VAD + 唤醒词 + 多后端 ASR）
  tts              语音合成（VITS2 / Matcha / Kokoro，本地 TensorRT 或 ONNX）
  phanthy_music    曲库检索与 HTTPS 播放（歌曲 + ASMR 助眠音频）
  vop              物体检测（YOLOE-26 + TensorRT）
  visual_depth     单目深度（YOLO26-depth + TensorRT）
  measure          物体可见宽高测量（VOP + 深度 + camera_info）
  ocr              文字识别（RapidOCR + TensorRT）
  barcode          扫码（zxing-cpp，二维码/条形码，零模型，按需识别）
  table_db         表格数据库（导入 xlsx/csv，增删改查与计数，SQLite，只用标准库）
  pose             人体关键点与动作分类（COCO-17 + TensorRT，动作为几何规则）
  face_recognition 人脸识别与建库（InsightFace buffalo_sc）

每个插件自带一个 `enabled` 开关，加载失败的插件不会拖垮其余插件 —— 它的卡片
不出现在 dashboard 上，这一点是看得见的。

MCP 工具命名规则：{plugin_prefix}_{tool_name}
  例：asr_info, asr_start, asr_stop, tts_info, tts_start, tts_speak

MCP server 端口: config.mcp_port（默认 15720）
WebSocket ASR 端口: config.ws_port（默认 15721）
"""

from __future__ import annotations

# First, before anything can write to stdout: make every log line one atomic,
# control-character-free write, so concurrent writers cannot tear a Docker log
# record. See utils/logsafe.py.
from utils import logsafe
logsafe.install()

import asyncio
import base64
import hashlib
import json
import logging
import os
import signal
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
from pathlib import Path

import yaml

import rclpy
import rclpy.executors

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)
# suppress noisy third-party loggers.
# pyvips logs six INFO lines per decoded image ("thumbnailing N bytes",
# "selected loader is ...", "converting to processing space srgb", ...). At a
# camera's frame rate that is the only thing in the log: measured on Tianyi,
# ~4 lines/second with every other perception message buried under it.
for _quiet in ('urllib3', 'websockets', 'httpcore', 'httpx', 'pyvips'):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

# Cap on how much of an MCP argument dict reaches the log. A tool call can carry
# an image, a base64 payload or a long utterance, so an unbounded repr here is how
# a single log line grows past the point where Docker can frame it — the result
# side was already capped, the argument side was not.
_LOG_ARG_CHARS = 500

# How often the register thread says it is still alive when nothing has changed.
# The heartbeat itself stays at 30s; this only governs how often that fact
# reaches the log, so "quiet" cannot mean both "healthy" and "thread died".
REGISTER_ALIVE_INTERVAL_S = 1800.0


def _brief(obj) -> str:
    """One-line, length-capped repr for logging an MCP payload."""
    text = repr(obj)
    if len(text) <= _LOG_ARG_CHARS:
        return text
    return f"{text[:_LOG_ARG_CHARS]}…[+{len(text) - _LOG_ARG_CHARS} chars]"


# ── ACP: SSE event bus (thread-safe) ─────────────────────────────────────────

import queue as _queue

_sse_clients: list[_queue.Queue] = []   # 每个 SSE 连接一个 queue
_sse_lock = threading.Lock()


def sse_push(event: dict):
    """线程安全地广播 SSE 事件到所有连接的客户端。"""
    data = json.dumps(event, ensure_ascii=False)
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(data)
            except _queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = os.environ.get("CONFIG_PATH", str(Path(__file__).parent / "config.yaml"))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Darwin ASR submission: run only the selected perception plugin. This
    # keeps unrelated cards (TTS/vision/face) from downloading models or
    # failing startup on the small 4 GiB evaluation pod, while preserving the
    # normal multi-plugin behaviour when ASR_PLUGIN is unset.
    selected = os.environ.get("ASR_PLUGIN", "").strip()
    if selected:
        plugins = cfg.setdefault("plugins", {})
        for name, plugin_cfg in plugins.items():
            if isinstance(plugin_cfg, dict):
                plugin_cfg["enabled"] = name == selected
    return cfg


# ── Bundle ────────────────────────────────────────────────────────────────────

class PerceptionBundle:
    def __init__(self, cfg: dict, executor):
        self._plugins: list = []
        plugins_cfg = cfg.get("plugins", {})

        # Built before ASR, and that order is load-bearing: ASR takes a handle
        # to this plugin so the voiceprint can be computed on the VAD segment it
        # already cut, concurrently with transcription. The handle is optional —
        # ASR publishes an identity-free payload when it is absent, which is
        # what happens when speaker_recognition is disabled or still loading.
        #
        # Deliberately *not* guarded the way TTSPlugin is: the constructor here
        # does no model work (the engine loads single-flight in a background
        # thread on first use), so there is nothing for it to fail on.
        speaker_plugin = None
        if plugins_cfg.get("speaker_recognition", {}).get("enabled", False):
            from plugins.speaker import SpeakerRecognitionPlugin
            speaker_plugin = SpeakerRecognitionPlugin(
                plugins_cfg["speaker_recognition"], executor)
            self._plugins.append(speaker_plugin)
            log.info("SpeakerRecognitionPlugin loaded")

        if plugins_cfg.get("asr", {}).get("enabled", False):
            from plugins.asr import ASRPlugin
            self._plugins.append(
                ASRPlugin(plugins_cfg["asr"], executor, speaker=speaker_plugin))
            log.info("ASRPlugin loaded (speaker=%s)",
                     "yes" if speaker_plugin else "no")

        tts_plugin = None
        if plugins_cfg.get("tts", {}).get("enabled", False):
            from plugins.tts import TTSPlugin
            # Guarded: TTSPlugin validates its engine configuration in the
            # constructor (backend, speaker_id, engine name). An unusable TTS
            # config must not take ASR/VOP/OCR down with it — the tool simply
            # does not appear, which is visible in the dashboard.
            try:
                tts_plugin = TTSPlugin(plugins_cfg["tts"], executor)
                self._plugins.append(tts_plugin)
                log.info("TTSPlugin loaded")
            except Exception:
                log.error("TTSPlugin failed to load; continuing without TTS",
                          exc_info=True)

        # Built after TTS and holding a handle to it, for the same reason ASR
        # holds one to speaker_recognition: the echo cancellation's reference
        # signal *is* the TTS card's audio output, and asking the plugin in this
        # process is what lets `robot_voice_topic: auto` need no configuration
        # at all. The handle is optional — without it `auto` resolves to
        # nothing and the card says so on its face rather than muting at random.
        if plugins_cfg.get("echo_cancellation", {}).get("enabled", True):
            from plugins.echo_cancellation import EchoCancellationPlugin
            self._plugins.append(EchoCancellationPlugin(
                plugins_cfg.get("echo_cancellation", {}), executor, tts=tts_plugin))
            log.info("EchoCancellationPlugin loaded (tts=%s)",
                     "yes" if tts_plugin else "no")

        # After TTS and holding its handle, for the same reason echo_cancellation
        # does: `speech_topics: auto` means "ask the local TTS card what it
        # publishes", and in-process is the only place that answer is free.
        if plugins_cfg.get("audio_mixer", {}).get("enabled", True):
            try:
                from plugins.audio_mixer import AudioMixerPlugin
                self._plugins.append(AudioMixerPlugin(
                    plugins_cfg.get("audio_mixer", {}), executor, tts=tts_plugin))
                log.info("AudioMixerPlugin loaded (tts=%s)",
                         "yes" if tts_plugin else "no")
            except Exception:
                log.error("AudioMixerPlugin failed to load; continuing without it",
                          exc_info=True)

        if plugins_cfg.get("phanthy_music", {}).get("enabled", False):
            try:
                from plugins.phanthy_music import PhanthyMusicPlugin
                self._plugins.append(PhanthyMusicPlugin(plugins_cfg["phanthy_music"], executor))
                log.info("PhanthyMusicPlugin loaded")
            except Exception:
                log.error("PhanthyMusicPlugin failed to load; continuing without phanthy_music",
                          exc_info=True)

        if plugins_cfg.get("vop", {}).get("enabled", False):
            import re, socket
            namespace = plugins_cfg["vop"].get("namespace", "").strip()
            if not namespace:
                namespace = re.sub(r"[^a-zA-Z0-9_]", "_", socket.gethostname())
            from plugins.vop import VideoObjectPerceptionPlugin
            plugin = VideoObjectPerceptionPlugin(plugins_cfg["vop"], namespace, executor)
            self._plugins.append(plugin)
            log.info("VideoObjectPerceptionPlugin loaded (namespace=%s)", namespace)

        # `vdp:` is the pre-rename spelling of this section; a config.yaml a
        # machine already has on disk still uses it.
        depth_cfg = plugins_cfg.get("visual_depth") or plugins_cfg.get("vdp") or {}
        if depth_cfg.get("enabled", False):
            import re, socket
            namespace = depth_cfg.get("namespace", "").strip()
            if not namespace:
                namespace = re.sub(r"[^a-zA-Z0-9_]", "_", socket.gethostname())
            from plugins.visual_depth import VideoDepthPerceptionPlugin
            # Guarded like TTSPlugin and FaceRecognitionPlugin: this one needs a
            # TensorRT engine bundle for the running JetPack line, and a machine
            # that cannot fetch it must still get ASR/TTS/VOP/OCR. The card
            # simply does not appear, which is visible in the dashboard.
            try:
                self._plugins.append(
                    VideoDepthPerceptionPlugin(depth_cfg, namespace, executor)
                )
                log.info("VideoDepthPerceptionPlugin loaded (namespace=%s)", namespace)
            except Exception:
                log.error("VideoDepthPerceptionPlugin failed to load; continuing without depth",
                          exc_info=True)

        if plugins_cfg.get("measure", {}).get("enabled", False):
            from plugins.measure import MeasurePlugin
            self._plugins.append(MeasurePlugin(plugins_cfg["measure"], executor))
            log.info("MeasurePlugin loaded")

        if plugins_cfg.get("pose", {}).get("enabled", False):
            import re, socket
            namespace = plugins_cfg["pose"].get("namespace", "").strip()
            if not namespace:
                namespace = re.sub(r"[^a-zA-Z0-9_]", "_", socket.gethostname())
            from plugins.pose import PosePerceptionPlugin
            # Guarded like TTSPlugin / VideoDepthPerceptionPlugin: this one needs
            # a TensorRT engine bundle for the running JetPack line, and a
            # machine that cannot fetch it must still get ASR/TTS/VOP/OCR. The
            # card simply does not appear, which is visible in the dashboard.
            try:
                self._plugins.append(
                    PosePerceptionPlugin(plugins_cfg["pose"], namespace, executor)
                )
                log.info("PosePerceptionPlugin loaded (namespace=%s)", namespace)
            except Exception:
                log.error("PosePerceptionPlugin failed to load; continuing without pose",
                          exc_info=True)

        if plugins_cfg.get("ocr", {}).get("enabled", False):
            from plugins.ocr import OCRPlugin
            self._plugins.append(OCRPlugin(plugins_cfg["ocr"], executor))
            log.info("OCRPlugin loaded")

        if plugins_cfg.get("barcode", {}).get("enabled", False):
            from plugins.barcode import BarcodeScanPlugin
            # No try/except and no model: zxing-cpp is imported lazily by
            # plugins.barcode_runtime on the first scan, so a build missing it
            # yields a card that reports the reason in `info` rather than a
            # plugin that fails to load and silently disappears.
            self._plugins.append(BarcodeScanPlugin(plugins_cfg["barcode"], executor))
            log.info("BarcodeScanPlugin loaded")

        # Standard library only (sqlite3 + zipfile): nothing to download, so the
        # constructor cannot fail on a machine without extra wheels. Lives here,
        # not in agent-core, because the files it hands out must be readable by
        # face_recognition / ocr / vop under this container's /models.
        if plugins_cfg.get("table_db", {}).get("enabled", False):
            from plugins.table_db import TableDbPlugin
            self._plugins.append(TableDbPlugin(plugins_cfg.get("table_db", {}), executor))
            log.info("TableDbPlugin loaded")

        if plugins_cfg.get("soundevent", {}).get("enabled", False):
            from plugins.soundevent import SoundEventPlugin
            self._plugins.append(SoundEventPlugin(plugins_cfg["soundevent"], executor))
            log.info("SoundEventPlugin loaded")
        if plugins_cfg.get("face_recognition", {}).get("enabled", False):
            from plugins.face import FaceRecognitionPlugin
            # Guarded like TTSPlugin: this plugin needs the standalone
            # onnxruntime and a readable identity database, and neither belongs
            # on the critical path of ASR/TTS/VOP/OCR. A failure here means the
            # card does not appear, which is visible in the dashboard.
            try:
                self._plugins.append(
                    FaceRecognitionPlugin(plugins_cfg["face_recognition"], executor)
                )
                log.info("FaceRecognitionPlugin loaded")
            except Exception:
                log.error("FaceRecognitionPlugin failed to load; continuing without it",
                          exc_info=True)

    def _plugin_for(self, full_name: str):
        """Resolve a tool name to (plugin, action) by longest matching prefix.

        Matching the *longest* prefix, not the first underscore-separated
        segment: a prefix may itself contain an underscore (`face_recognition`),
        and splitting on the first `_` would look for a plugin called `face`,
        find none, and report the tool as unknown.

        A plugin may also declare `ALIASES` — prefixes it answers to but does
        not advertise. That is what keeps a card saved under an old tool name
        working after a rename: `get_all_tools` publishes only PREFIX, so the
        dashboard shows the new name, while an existing canvas card still
        dispatches instead of going `state: error` on the next restart.
        """
        candidates = [(plugin.PREFIX, 0, plugin) for plugin in self._plugins]
        candidates += [(alias, 1, plugin) for plugin in self._plugins
                       for alias in getattr(plugin, "ALIASES", ())]
        # Longest prefix first, and a real prefix ahead of an alias of the same
        # length: whoever actually owns a name outranks whoever used to.
        for prefix, _is_alias, plugin in sorted(candidates, key=lambda c: (-len(c[0]), c[1])):
            if full_name == prefix:
                return plugin, plugin.PREFIX
            if full_name.startswith(prefix + "_"):
                return plugin, full_name[len(prefix) + 1:]
        return None, ""

    def get_all_tools(self) -> list:
        tools = []
        for p in self._plugins:
            for t in p.get_tools():
                full_name = t['name'] if t['name'] == p.PREFIX else f"{p.PREFIX}_{t['name']}"
                tools.append({**t, "name": full_name})
        return tools

    def dispatch(self, full_name: str, args: dict) -> dict | None:
        plugin, name = self._plugin_for(full_name)
        if plugin is None:
            return None
        return plugin.dispatch(name, args)

    def owns(self, full_name: str) -> bool:
        """True when some loaded plugin claims this tool name.

        Lets the caller tell a genuinely unknown tool apart from a loaded
        plugin returning None for an action it does not handle. Goes through
        the same resolver as `dispatch`, so the two cannot disagree.
        """
        plugin, _ = self._plugin_for(full_name)
        return plugin is not None

    def tts_synthesize_raw(self, text: str) -> bytes:
        for p in self._plugins:
            if getattr(p, 'PREFIX', None) == 'tts':
                return p.synthesize_raw(text)
        raise RuntimeError("TTS plugin not loaded or not enabled")

    def shutdown(self) -> None:
        """Let resource-owning plugins finish before ROS and Python shut down."""
        for plugin in self._plugins:
            shutdown = getattr(plugin, "shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:
                    log.exception("plugin shutdown failed: %s", plugin.PREFIX)


# ── MCP HTTP server ───────────────────────────────────────────────────────────

_bundle: PerceptionBundle | None = None

_ready_logged = False


def _asr_adapter_ready() -> bool:
    """True once the ASR plugin holds a usable adapter (model loaded).

    If the asr plugin is not enabled at all there is nothing to wait for.
    """
    if _bundle is None:
        return False
    for plugin in _bundle._plugins:
        if getattr(plugin, "PREFIX", None) == "asr":
            return getattr(plugin, "_adapter", None) is not None
    return True


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and "/sse" in str(args[0]):
                return
            log.debug(f"{self.address_string()} {fmt % args}")

        def _send(self, status: int, body: str):
            encoded = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            # WebSocket upgrade on the shared listener: the chart service
            # exposes only port 80, so the /ws/asr protocol is served here
            # alongside plain HTTP (see _WSConn / _ws_session_core).
            if (self.path.split("?")[0] in ("/ws/asr", "/")
                    and self.headers.get("Upgrade", "").lower() == "websocket"
                    and self.headers.get("Sec-WebSocket-Key")):
                key = self.headers["Sec-WebSocket-Key"]
                accept = base64.b64encode(
                    hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
                # RFC 6455 clients (and the websockets library) require an
                # HTTP/1.1 status line for the 101 — scope it to this request.
                self.protocol_version = "HTTP/1.1"
                self.send_response(101, "Switching Protocols")
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                try:
                    _ws_session_core(_WSConn(self.rfile, self.wfile))
                except Exception as e:
                    log.warning(f"[ws_asr:shared] session error: {e}")
                return
            if self.path in ("/vad/test", "/tts/test"):
                self._send(405, '{"error":"Use POST"}')
                return
            if self.path.split("?")[0] == "/sse":
                # SSE streaming endpoint for ACP completion events
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                client_queue = _queue.Queue(maxsize=64)
                with _sse_lock:
                    _sse_clients.append(client_queue)
                try:
                    while True:
                        try:
                            data = client_queue.get(timeout=30)
                            self.wfile.write(f"data: {data}\n\n".encode())
                            self.wfile.flush()
                        except _queue.Empty:
                            # keep-alive ping
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with _sse_lock:
                        if client_queue in _sse_clients:
                            _sse_clients.remove(client_queue)
                return
            if self.path.split("?")[0] == "/ready":
                # Chart readiness probe: curl -s -f http://localhost:80/ready.
                # Readiness is intentionally independent of model warmup: the
                # evaluator can connect while the ASR model is still loading
                # (it retries in the background), so a slow or flaky model
                # fetch cannot cancel the language sub-jobs.
                global _ready_logged
                if not _ready_logged:
                    _ready_logged = True
                    log.info("ready probe answered (HTTP up, "
                             "asr_adapter=%s)",
                             "loaded" if _asr_adapter_ready() else "loading")
                self._send(200, json.dumps({"ready": True,
                                            "asr_adapter": _asr_adapter_ready()}))
                return
            # Plugins may serve a GET route of their own (table_db hands stored
            # files to browsers, vision cards and agent-core, none of which can
            # read this container's /models). The plugin authorises the request
            # itself: its links are consumed by clients that cannot add headers,
            # so it signs them (see TableDbPlugin.handle_http_get).
            for plugin in (_bundle._plugins if _bundle is not None else []):
                handler = getattr(plugin, "handle_http_get", None)
                if handler is None:
                    continue
                served = handler(self.path, dict(self.headers))
                if served is None:
                    continue
                status, body, content_type, filename = served
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                if filename:
                    from urllib.parse import quote
                    self.send_header("Content-Disposition",
                                     "attachment; filename*=UTF-8''%s" % quote(filename))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))

            # File intake, before the body is read: an upload can be tens of
            # megabytes and the JSON paths below slurp Content-Length into
            # memory. utils/file_intake streams it to disk instead.
            #
            # This is how a photo reaches this container at all — see that
            # module's docstring. agent-core proxies the browser's (or the LLM's)
            # upload here and hands the returned path straight to a tool call;
            # neither side needs a shared mount, because the path in the reply is
            # this container's own.
            if self.path == "/file/upload":
                from utils.file_intake import access_token, handle_upload
                cfg = _load_config()
                intake = (cfg.get("file_intake") or {})
                base_dir = str(intake.get("dir", "/models/uploads"))
                body = self.rfile.read(length)
                status, payload = handle_upload(
                    self.headers, body, base_dir,
                    token=access_token(),
                    provided_token=self.headers.get("X-Access-Token"),
                    max_bytes=int(intake.get("max_bytes", 64 * 1024 * 1024)),
                    retention_days=int(intake.get("retention_days", 7)),
                )
                self._send(status, json.dumps(payload, ensure_ascii=False))
                return

            raw = self.rfile.read(length)

            if self.path == "/vad/test":
                try:
                    req = json.loads(raw)
                except Exception:
                    self._send(400, '{"ok":false,"info":"invalid JSON"}')
                    return
                try:
                    import base64, threading as _threading
                    audio_bytes = base64.b64decode(req.get("audio_b64", ""))
                    model      = req.get("model", "silero") or "silero"
                    threshold  = float(req.get("threshold", 0.5))
                    silence_ms = int(req.get("silence_ms", 800))
                    from plugins.asr import _vad_segment_sync
                    result = _vad_segment_sync(audio_bytes, model, threshold, silence_ms)
                    self._send(200, json.dumps({"ok": True, "segments": result}))
                except Exception as e:
                    log.error(f"[vad/test] {e}", exc_info=True)
                    self._send(200, json.dumps({"ok": False, "info": str(e)}))
                return

            if self.path == "/tts/test":
                try:
                    req = json.loads(raw)
                except Exception:
                    self._send(400, '{"ok":false,"info":"invalid JSON"}')
                    return
                try:
                    text = req.get("text", "").strip()
                    if not text:
                        self._send(200, json.dumps({"ok": False, "info": "text is required"}))
                        return
                    # Inline cloud credentials are not supported: this endpoint
                    # tests the on-device sherpa-onnx TTS the plugin actually
                    # serves. Fail loudly rather than silently ignoring the key.
                    if req.get("api_key", ""):
                        self._send(200, json.dumps({
                            "ok": False,
                            "info": "inline api_key is not supported; /tts/test exercises the on-device TTS plugin",
                        }))
                        return
                    pcm = _bundle.tts_synthesize_raw(text)
                    import base64 as _b64, io, wave
                    buf = io.BytesIO()
                    with wave.open(buf, 'wb') as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(16000)
                        w.writeframes(pcm)
                    wav_b64 = _b64.b64encode(buf.getvalue()).decode()
                    self._send(200, json.dumps({"ok": True, "wav_b64": wav_b64}))
                except Exception as e:
                    log.error(f"[tts/test] {e}", exc_info=True)
                    self._send(200, json.dumps({"ok": False, "info": str(e)}))
                return

            try:
                rpc = json.loads(raw)
            except Exception:
                self._send(400, json.dumps({"jsonrpc":"2.0","id":None,"error":{"code":-32700,"message":"Parse error"}}))
                return

            rid    = rpc.get("id")
            method = rpc.get("method", "")
            params = rpc.get("params") or {}

            if rid is None:
                self.send_response(202); self.end_headers(); return

            def ok(result):
                self._send(200, json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}))

            def err(code, msg):
                self._send(200, json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}))

            try:
                if method == "initialize":
                    log.debug(f"[mcp] initialize request from client")
                    ok({"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                        "serverInfo": {"name": "perception-bundle", "version": "1.0.0"}})
                elif method == "tools/list":
                    ok({"tools": _bundle.get_all_tools()})
                elif method == "tools/call":
                    name   = params.get("name", "")
                    args   = params.get("arguments") or {}
                    # info action is heartbeat probe — log at DEBUG to reduce noise
                    is_info = (args.get('action') == 'info')
                    music_call = (name == 'phanthy_music' or name.startswith('phanthy_music_')
                  or name == 'audio_mixer' or name.startswith('audio_mixer_'))
                    if not is_info:
                        if music_call:
                            log.info("[mcp] tools/call: phanthy_music (payload omitted)")
                        else:
                            log.info(f"[mcp] tools/call: {name}({_brief(args)})")
                    result = _bundle.dispatch(name, args)
                    if result is None:
                        # `dispatch` returns None for two very different things:
                        # no plugin owns the name, or a plugin owns it and
                        # declined the action. Reporting both as "Unknown tool"
                        # is actively misleading — it sent a debugging session
                        # hunting a tool-registration race that did not exist,
                        # when the tool was registered the whole time.
                        if _bundle.owns(name):
                            log.warning(
                                "[mcp] %s declined action %r", name, args.get("action")
                            )
                            err(-32603, f"Tool {name} does not handle action "
                                        f"{args.get('action')!r}")
                        else:
                            err(-32601, f"Unknown tool: {name}")
                    else:
                        if not is_info:
                            if music_call:
                                log.info("[mcp] phanthy_music result received (payload omitted)")
                            else:
                                log.info(f"[mcp] tools/call result: {json.dumps(result)[:200]}")
                        ok({"content": [{"type": "text", "text": json.dumps(result)}]})
                else:
                    err(-32601, f"Method not found: {method}")
            except BrokenPipeError:
                log.debug(f"Client disconnected before response")
            except Exception as e:
                log.error(f"RPC error: {e}", exc_info=True)
                try:
                    err(-32603, str(e))
                except BrokenPipeError:
                    pass

    return Handler


# ── WebSocket ASR server ───────────────────────────────────────────────────────

_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x30FF),
               (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0x0E00, 0x0E7F))


def _has_cjk(ch: str) -> bool:
    o = ord(ch) if ch else 0
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def _join_transcripts(texts) -> str:
    """Join VAD-segment transcripts into one result.

    CJK script has no word separator, so glue those pieces directly; Latin /
    Cyrillic / Greek segments get a space between them.
    """
    parts = [t for t in texts if t]
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        if _has_cjk(out[-1:]) or _has_cjk(part[:1]):
            out += part
        else:
            out += " " + part
    return out


_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class _WSConn:
    """Minimal blocking RFC 6455 WebSocket on the shared HTTP listener.

    The evaluation service only exposes port 80 (the chart service has a
    single port), so a judge that speaks the /ws/asr WebSocket protocol and
    one that speaks MCP over HTTP both arrive at this one listener. This class
    serves the upgrade on it; the separate `websockets` server on WS_PORT
    stays available for in-pod clients.
    """

    def __init__(self, rfile, wfile):
        self._r = rfile
        self._w = wfile
        self._buf = bytearray()
        self._buf_op = None

    def _read_exact(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = self._r.read(n - len(out))
            if not chunk:
                return bytes(out)
            out += chunk
        return bytes(out)

    def send_text(self, text: str) -> None:
        data = text.encode("utf-8")
        n = len(data)
        if n < 126:
            header = b"\x81" + bytes([n])
        elif n < 65536:
            header = b"\x81" + b"\x7e" + struct.pack(">H", n)
        else:
            header = b"\x81" + b"\x7f" + struct.pack(">Q", n)
        self._w.write(header + data)
        self._w.flush()

    def _send_close(self) -> None:
        try:
            self._w.write(b"\x88\x00")
            self._w.flush()
        except Exception:
            pass

    def recv(self):
        """Next message: bytes (binary), str (text), or None (closed)."""
        while True:
            b1 = self._read_exact(1)
            if not b1:
                return None
            b2 = self._read_exact(1)
            if not b2:
                return None
            fin = b1[0] & 0x80
            op = b1[0] & 0x0F
            masked = b2[0] & 0x80
            length = b2[0] & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(length) if length else b""
            if len(payload) < length:
                return None
            if masked and payload:
                payload = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
            if op == 0x8:                      # close
                self._send_close()
                return None
            if op == 0x9:                      # ping -> pong
                self._w.write(bytes([0x8A, len(payload)]) + payload)
                self._w.flush()
                continue
            if op == 0xA:                      # pong
                continue
            if op in (0x1, 0x2):               # new text/binary message
                self._buf = bytearray(payload)
                self._buf_op = op
            elif op == 0x0:                    # continuation
                self._buf += payload
            else:
                continue
            if fin:
                if self._buf_op == 0x1:
                    return bytes(self._buf).decode("utf-8", "replace")
                if self._buf_op == 0x2:
                    return bytes(self._buf)
                self._buf = bytearray()
                self._buf_op = None


def _resolve_ws_adapter_sync(cfg: dict):
    """Blocking twin of _resolve_ws_adapter for the shared-port session."""
    from plugins.asr import (_build_asr_adapter, resolve_asr_model,
                             DEFAULT_ASR_MODEL, ASR_MODELS)
    from utils.onnx_provider import normalize_device

    plugin = None
    if _bundle is not None:
        for p in _bundle._plugins:
            if getattr(p, "PREFIX", None) == "asr":
                plugin = p
                break
    if plugin is not None:
        want_model = resolve_asr_model(cfg.get("asr_model", DEFAULT_ASR_MODEL))
        if want_model not in ASR_MODELS:
            want_model = DEFAULT_ASR_MODEL
        want_device = normalize_device(cfg.get("device"), cfg.get("hw_provider"))
        if want_device not in ASR_MODELS[want_model]["devices"]:
            want_device = "cpu"
        target = (resolve_asr_model(plugin._asr_model), plugin._device)
        if (want_model, want_device) == target:
            adapter = plugin._adapter
            if adapter is None:
                log.info("[ws_asr:shared] ASR model still loading; waiting for "
                         "the shared adapter (max 120 s)")
                deadline = time.monotonic() + 120.0
                while plugin._adapter is None and time.monotonic() < deadline:
                    time.sleep(5.0)
                adapter = plugin._adapter
            if adapter is not None:
                return adapter
    return _build_asr_adapter(cfg)


def _ws_session_core(conn) -> None:
    """Run one /ws/asr session on a _WSConn (blocking, one thread per client).

    Same protocol and same answers as the websockets-server path: config frame
    first, then PCM16 binary frames, then "flush"; exactly one asr_result per
    flush (empty text for silence/noise), asr_ready after the config frame.
    """
    from plugins.asr import VadSession, _pcm16_to_wav

    try:
        cfg_raw = conn.recv()
        if cfg_raw is None:
            return
        cfg = json.loads(cfg_raw)
    except Exception as e:
        log.warning(f"[ws_asr:shared] invalid config frame: {e}")
        return

    adapter = _resolve_ws_adapter_sync(cfg)
    language = cfg.get('language', 'zh-CN')

    if adapter is None:
        conn.send_text(json.dumps({'type': 'asr_error',
                                   'payload': {'error': 'ASR adapter not configured'}}))
        return

    session = VadSession()
    session.init()
    conn.send_text(json.dumps({'type': 'asr_ready', 'payload': {'language': language}}))
    log.info(f"[ws_asr:shared] client connected, language={language}")

    def _flush_and_send():
        segs = session.flush()
        texts = []
        for seg in segs or []:
            pcm = seg[0] if isinstance(seg, tuple) else seg
            try:
                texts.append((adapter.transcribe(_pcm16_to_wav(pcm),
                                                 language) or "").strip())
            except Exception as e:
                log.error(f"[ws_asr:shared] transcribe error: {e}")
                try:
                    conn.send_text(json.dumps({'type': 'asr_error',
                                               'payload': {'error': str(e)}}))
                except Exception:
                    pass
        text = _join_transcripts(texts)
        conn.send_text(json.dumps({'type': 'asr_result', 'payload': {'text': text}}))

    while True:
        msg = conn.recv()
        if msg is None:
            break
        if isinstance(msg, bytes):
            session.process_chunk(msg, time.time())
        elif isinstance(msg, str):
            if msg == 'flush':
                _flush_and_send()
    try:
        _flush_and_send()
    except Exception as e:
        log.warning(f"[ws_asr:shared] final flush failed: {e}")


async def _resolve_ws_adapter(cfg: dict):
    """Adapter for a WS connection.

    Reuse the plugin's already-loaded model whenever the request resolves to
    the same (model, device): a per-connection _build_asr_adapter allocates a
    second ONNX session each time, and concurrent judge connections on one
    Orin would stack them (prior runs OOM'd with extra heavy plugins loaded).
    A different model/device keeps the upstream per-connection build.
    """
    from plugins.asr import (_build_asr_adapter, resolve_asr_model,
                             DEFAULT_ASR_MODEL, ASR_MODELS)
    from utils.onnx_provider import normalize_device

    plugin = None
    if _bundle is not None:
        for p in _bundle._plugins:
            if getattr(p, "PREFIX", None) == "asr":
                plugin = p
                break
    if plugin is not None:
        want_model = resolve_asr_model(cfg.get("asr_model", DEFAULT_ASR_MODEL))
        if want_model not in ASR_MODELS:
            want_model = DEFAULT_ASR_MODEL
        want_device = normalize_device(cfg.get("device"), cfg.get("hw_provider"))
        if want_device not in ASR_MODELS[want_model]["devices"]:
            want_device = "cpu"
        target = (resolve_asr_model(plugin._asr_model), plugin._device)
        if (want_model, want_device) == target:
            adapter = plugin._adapter
            if adapter is None:
                # Startup retry in progress: wait (async, loop stays free) up
                # to 120 s instead of starting a parallel model load.
                log.info("[ws_asr] ASR model still loading; waiting for the "
                         "shared adapter (max 120 s)")
                deadline = time.monotonic() + 120.0
                while plugin._adapter is None and time.monotonic() < deadline:
                    await asyncio.sleep(5.0)
                adapter = plugin._adapter
            if adapter is not None:
                return adapter
    return _build_asr_adapter(cfg)


async def _ws_asr_handler(websocket):
    """Handle a /ws/asr WebSocket connection.

    Protocol:
      1. Client sends a JSON text frame with ASR config:
         {"provider":"openai","url":"...","key":"...","model":"...","language":"zh-CN"}
      2. Client sends binary frames: raw PCM16 chunks (512 samples @ 16kHz)
      3. Server sends JSON text frames with transcription:
         {"text": "识别结果"}
      4. Client sends text "flush" to force-flush remaining speech
    """
    from plugins.asr import VadSession, _pcm16_to_wav
    import websockets

    # 1. Receive config frame
    try:
        cfg_raw = await websocket.recv()
        cfg = json.loads(cfg_raw)
    except Exception as e:
        log.warning(f"[ws_asr] invalid config frame: {e}")
        return

    adapter = await _resolve_ws_adapter(cfg)
    language = cfg.get('language', 'zh-CN')

    if adapter is None:
        await websocket.send(json.dumps({'type': 'asr_error', 'payload': {'error': 'ASR adapter not configured'}}))
        return

    session = VadSession()
    session.init()

    await websocket.send(json.dumps({'type': 'asr_ready', 'payload': {'language': language}}))
    log.info(f"[ws_asr] client connected, provider={cfg.get('provider','?')}")

    async def _transcribe(pcm: bytes) -> str:
        wav = _pcm16_to_wav(pcm)
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: adapter.transcribe(wav, language)
        ) or ""

    async def _flush_and_send():
        """Segment the buffered audio and send one joined transcript."""
        segs = session.flush()
        texts = []
        for seg in segs or []:
            pcm = seg[0] if isinstance(seg, tuple) else seg
            try:
                texts.append((await _transcribe(pcm)).strip())
            except Exception as e:
                log.error(f"[ws_asr] transcribe error: {e}")
                try:
                    await websocket.send(json.dumps({'type': 'asr_error', 'payload': {'error': str(e)}}))
                except Exception:
                    pass
        # Always answer a flush with exactly one asr_result frame — an empty
        # text for silence/noise — so an evaluator client waiting on the flush
        # can never time out on a frame that never comes.
        text = _join_transcripts(texts)
        await websocket.send(json.dumps({'type': 'asr_result', 'payload': {'text': text}}))

    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                session.process_chunk(msg, time.time())
            elif isinstance(msg, str):
                if msg == 'flush':
                    await _flush_and_send()
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        log.warning(f"[ws_asr] connection error: {e}")
    finally:
        # The client may end the stream by disconnecting without a `flush`
        # frame; segment whatever is buffered so no utterance is lost.
        try:
            await _flush_and_send()
        except Exception as e:
            log.warning(f"[ws_asr] final flush failed: {e}")


async def _run_ws_server(ws_port: int):
    import websockets
    async with websockets.serve(_ws_asr_handler, "", ws_port):
        log.info(f"WebSocket ASR server → ws://0.0.0.0:{ws_port}")
        await asyncio.Future()  # run forever


def _start_ws_thread(ws_port: int):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_ws_server(ws_port))
    except OSError as e:
        log.warning(f"[ws_asr] standalone WS server could not bind {ws_port} "
                    f"({e}); /ws/asr stays available on the shared MCP listener")


# ── Entry point ───────────────────────────────────────────────────────────────


def _start_registration(mcp_port: int, name: str, category: str):
    """Register this driver with agent-core in a background thread, then heartbeat every 30s."""
    import urllib.request as _urllib
    import ssl as _ssl
    agent_core_url = os.environ.get("AGENT_CORE_URL", "https://localhost:15678")
    _ctx = _ssl.create_default_context()
    _ctx.check_hostname = False
    _ctx.verify_mode = _ssl.CERT_NONE
    payload = json.dumps({
        "name": name,
        "url":  f"http://localhost:{mcp_port}/mcp",
        "category": category,
    }).encode()
    def _run():
        import time as _t
        # Log transitions plus a slow keepalive — same reasoning as actucore's
        # copy. Here it is 92 lines out of 242, on the container whose log is the
        # first place anyone looks when ASR or TTS misbehaves.
        #
        # The slow line exists because edges alone made "healthy" and "the
        # register thread died" look identical: every "ok" used to double as
        # proof of life, and dropping it entirely traded one blind spot for
        # another.
        healthy = None
        last_alive = 0.0
        while True:
            try:
                req = _urllib.Request(
                    f"{agent_core_url}/api/mcp", data=payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with _urllib.urlopen(req, timeout=3, context=_ctx):
                    now = _t.monotonic()
                    if healthy is not True:
                        log.info(f"[register] heartbeat ok → {agent_core_url}"
                                 + ("" if healthy is None else " (recovered)"))
                        healthy = True
                        last_alive = now
                    elif now - last_alive >= REGISTER_ALIVE_INTERVAL_S:
                        last_alive = now
                        log.info(f"[register] still registered → {agent_core_url}")
                _t.sleep(30)
            except Exception as e:
                # Every failure is logged: a flapping link is a real symptom.
                log.warning(f"[register] failed: {e}, retrying in 5s")
                healthy = False
                _t.sleep(5)
    threading.Thread(target=_run, daemon=True, name="register").start()


def main():
    global _bundle

    cfg      = _load_config()
    # The Darwin shell image injects MCP_PORT/WS_PORT (the chart probes the
    # MCP port at :80); honour them over the config file, keeping the
    # upstream defaults when unset.
    mcp_port = int(os.environ.get("MCP_PORT") or cfg.get("mcp_port", 15720))
    ws_port  = int(os.environ.get("WS_PORT")  or cfg.get("ws_port",  15721))

    log.info(f"perception bundle starting, mcp_port={mcp_port}, ws_port={ws_port}")
    log.info(f"config: plugins.asr.enabled={cfg.get('plugins',{}).get('asr',{}).get('enabled')}, "
             f"plugins.tts.enabled={cfg.get('plugins',{}).get('tts',{}).get('enabled')}")
    asr_cfg = cfg.get('plugins',{}).get('asr',{})
    tts_cfg = cfg.get('plugins',{}).get('tts',{})
    log.info(f"  asr: provider={asr_cfg.get('provider')}, url={asr_cfg.get('url','')[:40] or '(empty)'}, "
             f"model={asr_cfg.get('model','') or '(default)'}, key={'set' if asr_cfg.get('key') else 'MISSING'}")
    log.info(f"  tts: provider={tts_cfg.get('provider')}, model={tts_cfg.get('model','') or '(default)'}, "
             f"api_key={'set' if tts_cfg.get('api_key') else 'MISSING'}")

    # Bind the HTTP port before loading models. ASR model downloads can take
    # minutes on a cold Orin, while the chart readiness probe expects /ready
    # to answer throughout that window.
    server = ThreadingHTTPServer(("", mcp_port), make_handler())
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="mcp_http").start()
    log.info(f"MCP server → http://0.0.0.0:{mcp_port} (bound before model load)")

    os.environ.setdefault("RCUTILS_LOGGING_SEVERITY_THRESHOLD", "50")
    os.environ.setdefault("ROS_LOG_LEVEL", "WARN")

    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()
    _bundle  = PerceptionBundle(cfg, executor)
    log.info(f"perception bundle constructed: asr_adapter="
             f"{'loaded' if _asr_adapter_ready() else 'loading'}")

    def _spin():
        """Spin the executor, surviving a single entity's teardown race.

        `executor.spin()` used to run bare. Anything it raised killed this
        daemon thread outright, and with it every subscription in the process —
        ASR, OCR, vop, the lot — while the MCP HTTP server kept answering, so
        the service looked healthy and simply stopped perceiving. The one
        observed trigger was rclpy's

            InvalidHandle: cannot use Destroyable because destruction was
            requested

        raised from `_take_subscription` when a node's handle is destroyed
        while the executor still holds it in its wait list. That is a bug in
        whoever tore the node down (fixed in vop/visual_depth: leave the executor
        before destroying anything), but one plugin's teardown must not be
        able to silence the whole stack.

        So: log it and resume. The offending entity is already gone, so the
        next spin proceeds without it. The delay is a brake against a tight
        loop if some error turns out to be permanent — better a slow log than
        a pegged core.
        """
        while True:
            try:
                executor.spin()
                return                      # clean shutdown
            except Exception:
                log.exception("[spin] executor raised; resuming in 1s — "
                              "ROS callbacks were interrupted")
                time.sleep(1.0)

    threading.Thread(target=_spin, daemon=True, name="perception_spin").start()

    # Standalone WebSocket ASR server for in-pod clients; when it would
    # collide with the MCP port the shared listener already serves /ws/asr.
    if ws_port != mcp_port:
        threading.Thread(target=_start_ws_thread, args=(ws_port,), daemon=True, name="ws_asr").start()
    else:
        log.info(f"ws_port == mcp_port ({ws_port}): /ws/asr is served on the "
                 f"shared HTTP listener only")

    _start_registration(mcp_port, "Perception Stack", "perception")

    def _shutdown(signum, frame):
        log.info(f"signal {signum}, shutting down")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        # The HTTP listener was started before model loading so readiness is
        # available throughout startup; keep the main thread alive here.
        while True:
            time.sleep(3600)
    finally:
        server.shutdown()
        _bundle.shutdown()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
