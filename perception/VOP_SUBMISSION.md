# VOP 物体识别榜提交说明 (Darwin subnet 1000000294)

本分支用于 Darwin「具身智能-感知能力-VOP物体识别榜」的 github-commit 交付
(run: `agent-cb294026112500000000000000000294`)。官方壳镜像
`harbor-contest.4pd.io/wanglimin/sut/robot_vop:0.0.1` 按
`PHANTHYMOTUS_REPO` + `PHANTHYMOTUS_COMMIT_ID` 检出本分支后运行 perception 服务，
`JP_VERSION=6` → JetPack 6 机器（镜像内 TensorRT 10.4，jp61 引擎族）。

## 本版相对上一版（agent-fa2940…-vop-lean）的变化

1. **模型升级：nano → small（220 类全覆盖，精度优先）**
   - 参考基线引擎 `yoloe-26s-seg` 冻结 144 类，榜单 220 类中缺 77 类
     （水果/工具/文具/球类/家居小物等）；Macro-F1 按类平均，缺失类恒为 0，
     参考引擎理论上限 143/220≈65%。
   - 本版用同一模型家族 `yoloe-26s-seg`（开放词汇，zero-shot 文本提示）
     `set_classes()` 冻结**全部 220 个榜单类别**后导出 ONNX
     （`nms=False` 端对端头，静态 1x3x640x640，输出 (1,300,38)，与参考
     `vision_runtime.decode_detections` 完全兼容）。
   - 参数量：PyTorch 融合前 15.27M / 融合后 ~14M（ultralytics 摘要
     13,991,922）/ ONNX 初始器 10,492,535 —— 均 <30M 红线。
   - 本地 coco128 面板（80 个榜单共享类，IoU0.7 贪心一对一匹配，官方同款
     指标实现）：s220 与 s144 在共享类上**逐类 TP/FP 完全一致**
     （macro-F1 0.4181 @conf0.30），证明扩词表不回归既有 143 类；
     s220 相对 n220 高 +7.9 macro-F1（0.4181 vs 0.3393）。
2. **运行时模型引导（官方壳镜像适配）**
   - 壳镜像是平台固定件、无法烘焙模型。新增
     `plugins/vop_model_bootstrap.py`：服务启动/首请求时从本仓库
     **GitHub release**（`agent-cb294026112500000000000000000294-vop-models-v1`）
     下载 `vop220.onnx` + `vocab220.json`，**sha256 固定校验**、文件锁
     单飞、600s 超时（`VOP_LIGHT_FETCH_TIMEOUT`）、落到可写的
     `/models/vop-light`（不可写退回 `/tmp/vop-light-models-<uid>`）。
   - `vop_light_runtime.find_onnx/find_vocab` 接入引导：镜像内烘焙路径
     （`/opt/vop-models`，自建 Dockerfile 路径保留）优先，缺失时走 release 下载。
   - 任何失败（网络/哈希/TRT 构建）自动回落**参考 COS 144 类预构建引擎**
     捆绑包：轻量路径只会减少、不会移除能力；服务永不因模型缺失而不可用。
3. **修复两处 lean 路径缺陷（本 run 实测发现）**
   - `vop_model_bootstrap.ensure_assets` 嵌套获取非重入锁 → 并发启动
     （prewarm + 首请求）死锁；重构为扁平临界区（本地 4 并发实测通过）。
   - `vision_runtime.imdecode_bytes` PIL 路径未捕获损坏/非图片字节，
     异常穿透到 MCP 层；现返回 None → `ok:false bad_input`（与参考语义一致）。
4. 其余 lean 架构沿用上一版（已验证设计）：vop-only 配置、去 rclpy/ROS2、
   cv2 出热路径（PIL+numpy）、启动 prewarm、`MCP_PORT/WS_PORT` env 化。

## 模型资产（不进 Git，发布在 GitHub release）

release: `agent-cb294026112500000000000000000294-vop-models-v1`
- `vop220.onnx` — 42,132,435 B — sha256
  `65fa109b0ccc8f1748458a7134d1c4b6d71ee60a3cf9112c4d9333085e870f70`
  （yoloe-26s-seg.pt sha256 `48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654`，
  ultralytics 8.4.218 导出，opset 17，imgsz 640，nms=False，dynamic=False，
  词表 220 类按榜单 id 1..220 顺序）
- `vocab220.json` — 3,052 B — sha256
  `7dfc0fae36eb34f6205af1994fb3a24a8a853ba478fd8dad55fe64fec747b707`

## 本分支相对上游 phanthymotus (a391872) 的全部改动

1. `perception/config.yaml`：仅启用 `vop`；其余插件全部显式 `enabled: false`
   （含代码默认启用的 echo_cancellation / audio_mixer）→ lean 模式自动生效。
2. `perception/main.py`：端口 env 化（`MCP_PORT`/`WS_PORT`，竞赛 readme 要求）；
   `_lean_mode()` 自动判定（启用插件集合 == {vop}）；lean 分支跳过
   rclpy/executor/注册；`_prewarm_vop()` 后台预热（失败不致命）。
3. `perception/plugins/vop.py`：rclpy 惰性导入（`_ros_node_class()`）；
   `_resolve_engine()` 轻量路径（ONNX → 运行时 TRT 引擎），失败回落参考 COS
   捆绑包；`_prefetch_vocabulary` 优先引导/烘焙词表。
4. `perception/plugins/vision_runtime.py`：`imdecode_bytes` / `_resize_bgr` /
   `bgr_to_hsv_cv`（PIL/numpy，VOP_NO_CV2=1 热路径无 cv2）；PIL 解码失败
   返回 None（bad_input 语义）；`decode_detections` 等解码逻辑零改动。
5. `perception/plugins/vop_light_runtime.py`：运行时 TRT 构建
   （FP16、静态 640、workspace 上限 `VOP_TRT_WORKSPACE_MB`=192MB、文件锁
   缓存、ultralytics 风格 JSON 元数据头 task/names/imgsz）；接入
   `vop_model_bootstrap` 引导下载。
6. `perception/plugins/vop_model_bootstrap.py`：**新增**，release 模型引导
   （sha256 固定、单飞、可写目录探测、600s 超时、失败非致命）。
7. `perception/Dockerfile.jetson`：精简版（自建镜像路径保留）；烘焙资产
   更新为本 release 的 s220。
8. `perception/VOP_SUBMISSION.md`：本文档。

## 单实例内存预算（Jetson Orin-8G 共享盒，4Gi/pod 限制）

| 项 | 参考基线（144 类引擎+全插件） | 上一版 lean+nano | 本版 lean+small |
|---|---|---|---|
| 插件数 | 13+（ASR/TTS/OCR/pose/face…） | 1（vop） | 1（vop） |
| rclpy/ROS2、cv2 热路径 | 有 | 无 | 无 |
| 模型权重 | 24.8MB 引擎 | ~12MB ONNX→引擎 | ~25MB 引擎（FP16 估算） |
| 模型获取 | COS 预构建 | 镜像烘焙 | 启动引导（release，sha 固定） |
| 实测进程 RSS（x86 同栈，ORT 会话） | —（未测全插件栈） | ~500-800MB（估算） | **447MB 实测** |
| 构建 workspace | 无 | ≤192MB（构建后释放） | ≤192MB（构建后释放） |
| 参数量红线 <30M | 满足（~14M） | 满足（5.6M） | 满足（~14M 融合） |

Orin 上估算单实例常驻 ≤ ~1.2GiB（Python+PIL+numpy+websockets+TensorRT
上下文+FP16 引擎+激活），4Gi pod 限制内可放 3 实例、8GB 共享盒 6-8 实例
仍有余量；引擎缓存（/models/vop-cache，文件锁）使同盒多实例只构建一次。

## 已验证（本 run 证据在 /workspace 运行目录）

- T3：220 类 ONNX 导出 + onnxruntime CPU 全链路（参考 letterbox/to_blob/
  decode_detections）：bus.jpg 检出 bus(0.93)+person×4+glasses；zidane.jpg
  检出 person×2+tie(0.79) —— 类别名/框位置正确。
- T3：coco128 面板（128 张、80 个榜单共享类，官方 IoU0.7 贪心匹配指标）：
  s220 = s144（共享类零回归）> n220（+7.9 macro-F1）；conf 0.30 最优。
- T4：lean 服务 x86 实机起服（ORT 桩替代 TRT 会话，其余全真实路径）：
  MCP initialize/tools/list/tools/call 全通；bootstrap 下载+sha 校验+锁通过；
  `recognize_by_photo` 返回完整契约字段（name/position/confidence/bbox/color，
  confidence_threshold=0.3）；损坏图片 → ok:false bad_input；越权路径 →
  ok:false；4 并发 recognize 全 ok；首请求（含 42MB 引导下载）781ms。
- T5 前置：`perception` 全部改动文件 py_compile 通过。

## 未验证（如实记录）

- 真实 Orin 上的 TRT 引擎构建耗时/体积（FP16、workspace 192MB 上限，预计
  数十秒至 2 分钟；构建失败自动回落 COS 144 类捆绑包，服务仍可用）。
- Orin 评测节点到 GitHub release 的下载可达性（间接证据：参考
  `tools/export_vision_engines.py` 注释「Orins reach github.com」；
  失败同样回落参考捆绑包）。
- 壳镜像 robot_vop:0.0.1 的 /ready 实现与启动入口（registry 从开发容器
  TLS 不可达）；本分支保持参考 `perception/main.py` 入口与 MCP 协议不变。
- 榜单测试集上的实际 Macro-F1（测试集与官方训练集均不在本地；
  本地面板仅覆盖 80 个共享类）。相对参考基线的确定性增益来自 77 个
  原本恒 0 的类别覆盖；共享 143 类质量与参考同模型同词表（面板零回归）。

## 运行入口与端口契约

- 入口：`perception/main.py`（MCP-over-HTTP JSON-RPC 2.0，POST /mcp）
- 端口：`MCP_PORT`（默认 15720）、`WS_PORT`（默认 15721），必须来自环境变量
- 目标工具：`vop`；动作 `recognize_by_photo`（image_path ∈ /models,/tmp,/work）
  / `recognize_by_url`（url）
- 就绪：lean 模式下 MCP server 立即可服务；引擎 prewarm + on-demand 加载
- 交付 env：`PHANTHYMOTUS_REPO` / `PHANTHYMOTUS_COMMIT_ID` / `JP_VERSION=6`
