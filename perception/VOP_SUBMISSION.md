# VOP 物体识别榜提交说明 (Darwin subnet 1000000294)

本分支用于 Darwin「具身智能-感知能力-VOP物体识别榜」的 github-commit 交付
(run: agent-fa294026112500000000000000000294)。官方壳镜像
`harbor-contest.4pd.io/wanglimin/sut/robot_vop:0.0.1` 按
`PHANTHYMOTUS_REPO` + `PHANTHYMOTUS_COMMIT_ID` 检出本分支后运行 perception 服务，
`JP_VERSION=5` → JetPack 5.11 / TensorRT 8.5 (jp511) 引擎族。

## 为什么这一版（上一版的失败原因与本版的对策）

上一版（run agent-f2942...，commit 1bbc76b）已关闭全部无关插件，但平台评测仍
OOM（exit 137）。原因：评测盒是共享 Jetson Orin-8G（8GB 统一内存），多个并发
推理实例共享同一份设备内存；上一版每实例仍携带 rclpy/ROS2 运行时、cv2（含
CUDA/GStreamer 依赖库）和 24MB yoloe-26s-seg 144 类 TensorRT 引擎（小模型也有
完整的 open-vocab 加速器上下文），单实例运行时内存 ~1.5-2GiB，多实例叠加超预算。

本版在保持接口契约与识别能力的前提下，把**单实例运行时内存**压到最小：

1. **lean 模式**（`main.py`）：config 只启用 vop 时自动进入 lean —— 不 import
   rclpy/ROS2、不起 spin 线程、不注册 agent-core；`VOP_LEAN=1/0` 可强制。
   lean 下默认 `VOP_NO_CV2=1`：vop 热路径（解码/缩放/HSV 颜色统计）改用
   PIL + 纯 numpy 实现（`vision_runtime` 的 `imdecode_bytes` / `_resize_bgr` /
   `bgr_to_hsv_cv`），cv2 仅在显式 `VOP_NO_CV2=0` 时作为后备导入。
2. **轻量模型**：YOLOE-26n-seg（nano 开放词汇分割，融合后 5.1M 参数 /
   11.8 GFLOPs，ONNX 10.8MB）+ `set_classes` 冻结**全部 220 个榜单类别**
   （上一版 144 类引擎缺 77 个榜单类，是 F1 的主要漏项；220 类全覆盖修复之）。
   `nms=False`、静态 1x3x640x640 导出，输出布局与参考解码
   `vision_runtime.decode_detections` 完全一致。
3. **运行时 TRT 构建**（`plugins/vop_light_runtime.py`，新增）：首次请求时用
   镜像自带 TensorRT 把 ONNX 构建成 FP16 静态引擎，workspace 上限
   `VOP_TRT_WORKSPACE_MB=192`MB；引擎带 ultralytics 风格 JSON 元数据头
   （task/names/imgsz），参考读引擎代码零改动。构建结果文件锁 + 缓存到
   `/models/vop-cache`（共享卷，多实例只构建一次；/models 不可写时退回
   私有 /tmp 缓存）。任何失败自动回落参考 COS 捆绑包（144 类预构建引擎），
   轻量路径只会减少、不会移除能力。
4. **精简 Jetson 镜像**（`perception/Dockerfile.jetson`）：同一
   `jetson-base:jp${JP_VERSION}-torch` 基础镜像、同样的 CMD 与代码布局，
   但删掉全部音频/ASR/TTS/OCR/pose/zxing 依赖层（参考镜像 ~14.6GB →
   本镜像只装 PyYAML/numpy/pillow/websockets + 烘焙 ONNX）。ONNX 从本仓库
   的公开 GitHub release 下载（sha256 固定校验；Orins 可出 github.com，
   参考 tools/export_vision_engines.py 的既有结论）；下载失败不致命，
   运行时回落 COS 捆绑包。

接口契约零变化：MCP JSON-RPC（POST /mcp：initialize / tools/list /
tools/call），工具 `vop`，动作 `recognize_by_photo`（image_path 限定
/models、/tmp、/work）与 `recognize_by_url`（url）；端口仍从 env
`MCP_PORT`（默认 15720）/`WS_PORT`（默认 15721）读取；返回
`{ok, source, image_size, confidence_threshold, count, latency_ms,
objects:[{name, position, confidence, bbox, color}], frame_color}`。

## 本分支相对上游 phanthymotus (a391872) 的全部改动

1. `perception/config.yaml`：仅启用 `vop`；其余 13 个插件全部显式
   `enabled: false`（含代码默认启用的 echo_cancellation / audio_mixer）。
2. `perception/main.py`：端口 env 化（MCP_PORT/WS_PORT）；`_lean_mode()`
   自动判定（启用插件集合 == {vop} 且 VOP_LEAN 未强制）；lean 分支跳过
   rclpy/executor/注册，置 `VOP_NO_CV2=1` 默认与 `AGENT_CORE_URL=disabled`；
   `_prewarm_vop()` 后台预热引擎（失败不致命）。
3. `perception/plugins/vop.py`：顶部不再 import rclpy（`_VOPNode` 改为惰性
   工厂 `_ros_node_class()`）；`_resolve_engine()` 增加 light path（烘焙
   ONNX → 运行时 TRT 引擎），失败回落参考 COS 捆绑包；`_prefetch_vocabulary`
   优先读烘焙词表；lean 下 `start` 走无 ROS 节点的单飞加载；
   cv2 使用点全部改走 `_no_cv2()` 守护的 PIL/numpy 后备。
4. `perception/plugins/vision_runtime.py`：新增 `imdecode_bytes` /
   `_resize_bgr` / `bgr_to_hsv_cv`（纯 numpy，OpenCV H 0-179 约定）与
   `_no_cv2()`；letterbox 改走 `_resize_bgr`。解码路径
   （decode_detections 等）零改动。
5. `perception/plugins/vop_light_runtime.py`：新增，见上文第 3 点。
6. `perception/Dockerfile.jetson`：精简版（同基础镜像/同布局），烘焙
   220 类 nano ONNX + 词表（sha256 固定），删除无关重型依赖层。
7. `perception/VOP_SUBMISSION.md`：本文档。

模型权重不进 Git：ONNX 与词表发布在 GitHub release
`agent-fa294026112500000000000000000294-vop-models-v1`（公开仓库）：
- `vop220-yoloe-26n-seg.onnx` (11,357,723 B, sha256 84795e32dbd77b69e5754518d043cac8295da2fbe7f9c9c3728501dcefc29763)
- `vocab220.json` (3,153 B, sha256 8b9f2b73bdcc92db3235fe1077cbe092377f37129031355e746fe652b8a8a8a7)

## 单实例内存预算（Jetson Orin-8G 共享盒）

| 项 | 上一版 | 本版 |
|---|---|---|
| 模型权重（运行时） | 24MB 引擎 + rclpy/cv2/torch 上下文 | ~12MB ONNX → FP16 引擎，workspace ≤192MB |
| Python 运行时 | rclpy+ROS2+cv2(CUDA libs) 全量 import | 无 rclpy/cv2 import（PIL+numpy+websockets） |
| 镜像体积 | ~14.6GB（sherpa/rapidocr/tts/zxing 全依赖） | 基础镜像 + 轻量 pip 层 |
| 参数量红线 <30M | 满足 | 满足（5.1M 融合 / 5.6M 原始） |
| GPU 占用 | 持续（topic 流式） | 仅请求驱动（on-demand），<10% 预算内 |

估算单实例常驻 ~500-800MB（python+numpy+PIL+websockets+TensorRT 上下文+
192MB 构建 workspace，构建后释放），同盒 6-8 个并发实例仍可落在 8GB 内；
引擎缓存共享后每实例不再重复支付构建开销。

## 已验证 / 未验证

已验证（本 run 证据见 /workspace/run/agent-fa294026112500000000000000000294/）：
- T3：ONNX 导出 + onnxruntime CPU 全链路（letterbox/to_blob/decode）在真实
  照片上检出 person×3/bus（conf≥0.86），220 类名映射正确。
- T4：lean 服务实机起服（x86，ORT 桩替代 TensorRT 会话），MCP
  initialize/tools/list/tools/call 全通；`recognize_by_photo` 返回完整契约
  字段；越权路径返回 ok:false bad_input；无模型时返回 ok:false
  engine_unavailable 且服务存活（错误不伪装成成功预测）。
- T5：Dockerfile 全部 COPY 源在分支根存在、无 .dockerignore 排除；
  镜像内 py_compile 行所列 7 个文件在提交前逐一编译通过。

未验证（如实记录）：
- 真实 Orin 上的 TRT 引擎构建耗时/体积（FP16、workspace 192MB 上限下预计
  数十秒；构建失败会自动回落 COS 144 类捆绑包，服务仍可用）。
- 多实例并发 OOM 的最终验证要等平台评测；本版预算见上表。
- 220 类开放词汇模型在榜单测试集上的实际 Macro-F1（开发机无 GPU 无法跑
  完整评测；相对上一版 144 类引擎，类别覆盖严格更全，且 nano 与 small
  同架构仅 backbone 尺寸差异）。

## 运行入口与端口契约

- 入口：`perception/main.py`（MCP-over-HTTP JSON-RPC 2.0）
- 端口：`MCP_PORT`（默认 15720）、`WS_PORT`（默认 15721），必须来自环境变量
- 目标工具：`vop`；动作 `recognize_by_photo` / `recognize_by_url`
- 就绪：lean 模式下 MCP server 立即可服务；引擎 on-demand 加载（有 prewarm）
