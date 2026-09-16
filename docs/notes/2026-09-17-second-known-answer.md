# 第二个已知答案：官方 sim transfer cube ACT（2026-09-17）

跨数据集、非自训、会跟踪的正对照。用来打掉「工具是调给 coffee 的」。

这是**训练集跟踪**，不是留出泛化：官方模型在全部 50 集上训练，我们切 ep0–9。

## 数据与模型

| 项 | 值 |
| --- | --- |
| 数据 | `lerobot/aloha_sim_transfer_cube_human` v3.0（50×400 @ 50 fps，只有 `images.top` + state，视频 av1） |
| 数据 revision | `6a43d500f101255823a9d2b9dc244eeb01a2cd31` |
| 模型 | `lerobot/act_aloha_sim_transfer_cube_human@ba73b2766f1371cdc133ca4efb97eb090d744625` |
| 旧格式 | 无 processor json；weights 带 `normalize_inputs.*` 键（加载时 warning） |
| `processor_source` | `dataset_stats` |
| 远端坑 | Xet CAS 对 huggingface.co 401；`HF_HUB_DISABLE_XET=1` 后走镜像下载 |

远端 ffmpeg 有 `libdav1d`。`ACTPolicy.from_pretrained` 在 CPU 上能加载。

## 跟踪判定

示教间 L2 = 0.814（45 对）。门槛 = 1.629。数字见 [2026-09-17-tracking-cube.json](./2026-09-17-tracking-cube.json)。

| 条件 | 值 | 门槛 | 结果 |
| --- | --- | --- | --- |
| 跨集 frame0 std | 0.588 | > 0.01 | 过 |
| per-dim corr 中位数 | **0.992** | > 0.5 | 过 |
| 10 集 mean L2 | **0.118** | < 1.629 | 过 |

比 coffee B'（corr 0.902 / L2 0.584）更贴示教，符合「官方模型 + 训练集」预期。

## 标定后门禁

`write_thresholds.py --bounds-pad 0.15`：`max_l2 = 0.213`，`max_delta = 0.608`，`max_lag_frames = 7`。

B-calib：**10/10 绿**。覆盖率 1.0。

## 突变体

| 候选 | gate |
| --- | --- |
| bias / noise / drop-recorded / dim / wrong-id / wrong-hash | 红 |
| lag-10 断言 | 红（部分集） |
| lag-10 `pred_shift` | **全是 10**，`--max-shift 2` 红 |

## 与 coffee 对照

| | coffee B'（自训 30k，留出 ep0–9） | sim-cube 官方 ACT（训练集 ep0–9） |
| --- | --- | --- |
| 谁训的 | 我们 | Hugging Face / LeRobot |
| 跟踪 | corr 0.902 / L2 0.584 | corr 0.992 / L2 0.118 |
| 标定后 | 11/11 绿（含事故窗） | 10/10 绿 |
| 发布负例 | 发布 coffee ACT 10/10 红 | 本套不另放负例 |

CI：`fixtures/gate/sim-cube/{baseline,candidate}.parquet`，`gate.yml` 多一步绿门禁。
