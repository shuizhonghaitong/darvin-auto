# darvin-auto — run agent-cb312026112500000000000000000312

Darvin subnet 312 (具身智能-asr基础能力-不限音色多语种榜v2, jp5/6):
8-language (zh/ja/ru/en/es/fr/th/de) robot ASR, ranked by 1-CER
(case_num weighted), evaluated in the official shell image on a 1 CPU / 4 GiB
pod.

This branch ships the perception stack (upstream
[4paradigm/phanthymotus](https://github.com/4paradigm/phanthymotus) @
`2b639989313d0e8a1f4c86ced8fcfcfe87863a00`) adapted for that contract. Only
the ASR plugin is enabled.

## Submission environment

| Var | Value |
|---|---|
| `PHANTHYMOTUS_REPO` | `https://github.com/shuizhonghaitong/darvin-auto.git` |
| `PHANTHYMOTUS_COMMIT_ID` | the 40-char commit SHA of this branch |
| `ASR_PLUGIN` | `asr` |
| `JP_VERSION` | `6` |
| `trainImage` | `harbor-contest.4pd.io/wanglimin/sut/robot_asr_zh:0.0.5` (shell) |

The shell image clones `PHANTHYMOTUS_REPO` at `PHANTHYMOTUS_COMMIT_ID` and
starts the service; `start.sh` (or `python3 perception/main.py`) is the entry
point. `MCP_PORT`/`WS_PORT` are honoured over `config.yaml` when set
(readiness probe answers at `GET /ready` on the MCP port, bound before any
model download starts, so a slow fetch cannot cancel the evaluation).

## ASR design (see t2 research, branch docs)

- `plugins.asr.asr_model: auto-route`: one process, two models, shared by all
  connections.
  - **SenseVoice small int8** for zh/ja/ko/yue (zh CER 0.00, ja 0.086 on the
    local Orin TTS panel; RTF ~0.13 single core).
  - **Whisper base int8** (74M, 160 MB, commit-pinned HF/hf-mirror files)
    for en/ru/es/fr/th/de (ru 0.036, en 0.00, es 0.00, fr 0.027, th 0.55,
    de 0.04 on the same panel; RTF ~0.45 single core).
  - The whisper session is a single recognizer switched per utterance via
    `set_config()` (verified on sherpa-onnx 1.13.6 on Orin: 382 MB RSS vs
    3.1 GB for eight per-language sessions). Unknown language tags fall back
    to auto-detect; the code only ever pushes a whisper language it knows,
    because the decoder aborts the process on an unknown code.
- VAD (silero) gates transcription: silence/noise never reaches the models
  (whisper answers noise with "[Music]"), multi-utterance buffers are
  segmented and joined, and >24 s chunks are cut for whisper (30 s truncation
  + repetition hallucination, verified).
- Both models download at startup with background retry every 30 s until
  usable; `/ready` stays 200 throughout.

## Layout

- `perception/` — the perception stack (only `asr` enabled in `config.yaml`).
- `start.sh` — entry point (LD_PRELOAD gomp + ROS setup + `python3 perception/main.py`).
- `skills/`, `workflows/` — darvin-auto research/automation skills (inherited
  from `main`; not part of the evaluated service).

No file in this branch exceeds 1 MB; all model weights are fetched at
runtime from the pinned URLs in `perception/utils/model_downloader.py`.
