# 真机数据跑 robogate：卡住的地方

日期：2026-09-13  
分支：`feat/v0.1.0-robogate`  
数据：[lerobot/aloha_static_coffee](https://huggingface.co/datasets/lerobot/aloha_static_coffee)

这是 Mobile ALOHA 真机往咖啡机放胶囊的示教，不是仓库里的 `runs/examples`，也不是 `aloha_sim_*`。

## 数据本身

- `meta/info.json` 是 **LeRobot v3.0**。README 卡片还写着 v2.0 / `episode_000000.parquet`。
- 50 集、每集 1100 帧、50 FPS、动作/状态 **14 维**。
- 表格在 `data/chunk-000/file-000.parquet`（3.3MB，50 集一个分片）。
- 相机键是 `cam_high` / `cam_low` / `cam_left_wrist` / `cam_right_wrist`，没有示例里的 `observation.images.top`。
- 时间戳是 episode 内相对秒（`float32`），没有 `t_ns`。
- 列名是 `action` / `observation.state`，不是 run 格式的 `value`。
- 录制里没有 `latency_ms`、`confidence`、闭环事件。
- checkpoint `gozdebaydogmus/act-aloha-static-coffee-test` 吃三路相机 + state + effort；模型卡 tag 写的训练集是 `aloha_mobile_wash_pan`，必须钉 revision。

## 当时的卡点与修复

| 步骤 | 当时 | 修复 |
| --- | --- | --- |
| `extract` 退出 2 | 空壳 | `robogate extract lerobot` 读 v3 分片，写 `slices/<id>/` + scenario |
| 手写 scenario 不读数据 | 相机名可写错 | extract 从 `info.json` features 自动填 `inputs`，钉 `source.revision` |
| `replay` 退出 2 | 无适配器 | `python-policy`：mock（单测）+ lerobot ACT（远端 GPU） |
| `assert` 对着数据集目录 | 要 `meta.json` | extract/replay 产出 slice 与 run，assert 只吃 run |
| 原始 parquet 当 run | traceback（要 `value`） | `Status.ERROR` 包住 `evaluate()`；extract 写成 `t_ns`/`value` |
| 无 policy parquet | traceback | 缺 latency/confidence → `skipped` |
| 真机 scenario + 示例 run | **假绿** | `scenario_id` 不一致直接 exit 1 |
| `eval` / `diff` / `gate` | 占位 | 仍占位（本轮不做） |
| `action_bounds` / `smoothness` / `dtw` | schema 有、实现无 | 已实现 |

公开真机 MCAP 单文件从约 968MB 起，本轮仍不做 MCAP extract。

## 首跑结果（2026-09-13）

远端 `microbo-gpu` 上跑通了 ep0 真实 ACT：

- run：`aloha-static-coffee-ep0__act-aloha-static-coffee-test__20260913T155353Z`
- checkpoint：`gozdebaydogmus/act-aloha-static-coffee-test@646846823c8473712f689f59bdd03f798a688f6a`
- 环境：lerobot 0.4.4、torch 2.11.0+cu128、cuda
- mean L2 = 2.583984 → scenario `max_l2` = 3.875976（×1.5）
- latency p95 = 78.94ms（只报告）；confidence = skipped（ACT 无该输出）

落地时多出来的坑：huggingface.co IPv6 不通，wrap 默认 `HF_ENDPOINT=https://hf-mirror.com`；`pip install -e ".[lerobot]"` 会把 torch 降到 2.10 并撑爆 30G overlay，必须 `--no-deps`；torchvision 0.26 没有 `VideoReader`，回放用 PyAV 顺序解码。
