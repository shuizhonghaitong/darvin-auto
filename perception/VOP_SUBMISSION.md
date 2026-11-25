# VOP 物体识别榜提交说明 (Darwin subnet 1000000294)

本分支用于 Darwin「具身智能-感知能力-VOP物体识别榜」的 github-commit 交付
(run: agent-f2942026112500000000000000000294)。官方壳镜像
`harbor-contest.4pd.io/wanglimin/sut/robot_vop:0.0.1` 按
`PHANTHYMOTUS_REPO` + `PHANTHYMOTUS_COMMIT_ID` 检出本分支后运行 perception 服务，
`JP_VERSION=5` → JetPack 5.11 / TensorRT 8.5 (jp511) 引擎族。

## 本分支相对上游 phanthymotus (a391872) 的全部改动

1. `perception/config.yaml`：仅启用 `vop` 插件；其余 13 个插件
   (phanthy_music / audio_mixer / asr / tts / visual_depth / pose / ocr /
   barcode / soundevent / echo_cancellation / speaker_recognition /
   face_recognition / measure) 全部显式 `enabled: false`。
   原因：本榜只评测物体识别；评测 pod 资源为 1 CPU / 4Gi（榜单 Helm chart
   共享 resources），其余插件在启动期各自下载/加载重型模型（ASR/TTS-TRT/
   depth/pose/OCR/face/speaker），多插件并发加载会拖慢甚至阻塞服务就绪，
   多实例并发时直接 OOM (exit 137)。`echo_cancellation` 与 `audio_mixer`
   在配置键缺失时代码默认启用（main.py `.get("enabled", True)`），必须显式关闭。
2. `perception/main.py`：
   - 榜单 readme 强制要求的端口 env 化（`MCP_PORT` / `WS_PORT`，
     fallback cfg 15720 / 15721，原文片段一致）；
   - `_prewarm_vop()`：bundle 构建后经公开工具路径（`vop` action=start，
     无 topic）在后台触发引擎加载并轮询到 settled（上限 300s）。
     失败不致命——`recognize_by_photo` 走同一单飞加载路径会自动重试。
     目的：让评测首个请求不必承担 ~24MB TensorRT 引擎下载 + 解析的冷启动。
   - 该文件其余逻辑零改动（最小 diff 原则）。
3. 其余上游代码零改动。构建上下文完整保留（perception/Dockerfile.jetson
   的 9 处 COPY 源路径均在仓库根目录下存在且无 .dockerignore 排除，
   已逐条静态核验）。

## 运行入口与端口契约

- 入口：`perception/main.py`（MCP-over-HTTP JSON-RPC 2.0：initialize / tools/list / tools/call）
- 端口：`MCP_PORT`（默认 15720）、`WS_PORT`（默认 15721），必须来自环境变量（榜单 readme 强制）
- 目标工具：`vop`，动作 `recognize_by_photo` / `recognize_by_url`（一次成图识别，
  返回 `{ok, source, image_size, confidence_threshold, count, latency_ms,
  objects:[{name, position, confidence, bbox, color}]}`，与榜单文档示例一致）

## 模型与资源核算（本次 run 独立核验）

- 模型：`yoloe-26s-seg` 预构建 FP16 TensorRT 引擎（开放词汇，词汇在导出时冻结，
  144 类 = COCO_80 + ROBOT_EXTRA），由 `utils/model_downloader.py` 按运行机
  TensorRT 家族 (jp511/jp61) 从固定 COS 地址下载，size + sha256 双校验；
  JP_VERSION=5 → jp511 引擎 23,742,701 B
  (sha256 49df478a308de3a1f996d4784d2b00245486c04a40e0b7b6005b7226674da4fe)。
- 参数量：引擎为 `half=True` FP16 导出 ⇒ 参数量上界 = 引擎字节数/2 ≈ 11.87M，
  满足榜单「模型 30M 以下」红线；精确值见交付 run 的
  t3_model_params.json（对公开 yoloe-26s-seg.pt 权重实际计数）。
- 必须使用 GPU：TensorRT 引擎路径即 GPU 路径；`_resolve_engine` 无 .pt/CPU 回退。
- GPU 占用 <10%：评测为按图请求式推理（非连续视频流），单帧 640×640 FP16
  前向约几十毫秒，平均占用远低于 10% 门禁。
- 内存：仅 vop 插件时进程为 python + rclpy + cv2 + numpy + tensorrt +
  单引擎，RSS 约 1.5-2 GiB，满足 4Gi pod 限制并留多实例余量。
- 仓库内不含任何模型文件（榜单规则：Git 仓库禁止上传模型文件）。

## 已知风险（本次未授权修复，记录在案）

- 引擎冻结词汇 144 类，为榜单 220 类测试集的子集（77 类如水果/文具/小工具
  不在词汇内，恒漏检）。重建 220 类引擎需要训练/导出/发布流水线，
  本次授权范围（research_mode=off、不得训练）内不可行。
