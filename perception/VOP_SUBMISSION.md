# VOP 物体识别榜提交说明 (Darwin subnet 1000000294)

本分支用于 Darwin「具身智能-感知能力-VOP物体识别榜」的 github-commit 交付
(run: agent-e2942026112500000000000000000294)。官方壳镜像按
`PHANTHYMOTUS_REPO` + `PHANTHYMOTUS_COMMIT_ID` 检出本分支后运行 perception 服务。

## 本次相对上游 (agent-d 基线 b854902) 的改动

1. `perception/config.yaml`：仅启用 `vop` 插件；其余 13 个插件
   (phanthy_music / audio_mixer / asr / tts / visual_depth / pose / ocr /
   barcode / soundevent / echo_cancellation / speaker_recognition /
   face_recognition / measure) 全部显式 `enabled: false`。
   原因：本榜只评测物体识别；其余插件在启动期各自下载/加载重型模型，
   多插件并发加载会拖慢甚至阻塞服务就绪，且在多实例并发时放大内存压力。
   `echo_cancellation` 与 `audio_mixer` 在配置键缺失时默认启用，必须显式关闭。
2. `perception/main.py`：
   - 保留榜单 readme 要求的端口 env 化（`MCP_PORT` / `WS_PORT`，fallback 15720/15721）；
   - 新增 `_prewarm_vop()`：bundle 构建后经公开工具路径
     (`vop` action=start，无 topic) 在后台触发引擎加载并轮询到 settled。
     失败不致命——`recognize_by_photo` 走同一单飞加载路径会自动重试。
   目的：让评测首个请求不必承担 ~24MB TensorRT 引擎下载 + 解析的冷启动。
3. 其余上游代码零改动（最小 diff 原则）。

## 运行入口与端口契约

- 入口：`perception/main.py`（MCP-over-HTTP JSON-RPC 2.0：initialize / tools/list / tools/call）
- 端口：`MCP_PORT`（默认 15720）、`WS_PORT`（默认 15721），必须来自环境变量（榜单 readme 强制）
- 目标工具：`vop`，动作 `recognize_by_photo` / `recognize_by_url`（一次成图识别，
  返回 `{ok, source, image_size, confidence_threshold, count, latency_ms, objects:[{name, position, confidence, bbox, color}]}`）

## 模型与资源

- 模型：`yoloe-26s-seg` 预构建 TensorRT 引擎（开放词汇，词汇在导出时冻结），
  由 `utils/model_downloader.py` 按运行机 TensorRT 家族 (jp511/jp61) 从固定
  COS 地址下载，size + sha256 双校验；本榜 JP_VERSION=5 → jp511 引擎 (~23.7MB)。
- 参数规模 <30M、TRT 推理满足「必须用 GPU 且 GPU 占用 <10%」门禁。
- 仓库内不含任何模型文件（榜单规则：Git 仓库禁止上传模型文件）。
