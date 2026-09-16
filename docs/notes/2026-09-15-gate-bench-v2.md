# Robogate 门禁 benchmark v2（2026-09-15）

> 标定已被 [2026-09-16-first-gate.md](./2026-09-16-first-gate.md) 取代：bounds +15%、max_l2 2.232、B' 11/11。本文数字是 +10% 时的历史值。2026-09-16 GPU 复验见 [2026-09-16-gpu-verify.json](./2026-09-16-gpu-verify.json)。

HTML 版：[2026-09-15-gate-bench-v2.html](./2026-09-15-gate-bench-v2.html)

第二轮：修 lag 盲区、改 bounds 来源、诊断发布 ACT 的归一化错位，并重训带 shuffle 的 ACT。
基线改为会跟踪的自训 T30k（B'）。不测能不能泡咖啡，只测工具会不会漏、会不会误报。

留出集：`lerobot/aloha_static_coffee` ep0–9。训练集 ep10–49。
B'：`/root/autodl-tmp/ckpts/v2/act-coffee-30000`。
发布模型（对照）：`gozdebaydogmus/act-aloha-static-coffee-test@646846823c8473712f689f59bdd03f798a688f6a`。

## 诊断

发布 B 的 postprocessor action 统计**不是**当前数据集的 `action`，而是几乎等于 `observation.state`：

| 来源 | dim1 mean | 全维 mean \|Δ\| vs unnorm |
| --- | --- | --- |
| B unnormalizer action | 0.480 | 0 |
| 数据集 `action` | −0.313 | **0.327** |
| 数据集 `observation.state` | 0.507 | **0.015** |

所以 B 的开环偏移是归一化空间错了，不是「观测没进模型」。
交叉回放 ep0 vs ep1：mean abs diff = 0.0236，max = 0.177；chunk 边界跳变约 0.22 vs 帧间 0.03。

示教包络 + 10% 下，发布 B **10/10 红**（dim1 全出界，dim5 94%，dim4 78%）。这是对发布模型的真实发现。

可复现：`python scripts/bench/diagnose_norm.py`。数字见 `docs/notes/2026-09-15-diagnose-norm.json`。

## 重训与跟踪判定

`train_act.py`：episode 乱序 + 集内顺序 + 4 worker × buffer 64，`batch=8`，lr=1e-5。
加载 B 的 processor 后，把 action mean/std 改成数据集 `meta.stats["action"]`，否则新模型会继续解到 state 空间。
spawn worker 里补打 PyAV 解码补丁（主进程的 patch 不会继承）。

| 步数 | loss | 墙钟 |
| --- | --- | --- |
| 1 | 104.74 | 6s |
| 2k | 1.68 | 7 min |
| 10k | 0.175 | 44 min |
| 30k | 0.061 | 143 min |

跟踪判定（T30k，ep0–9）：

| 条件 | 值 | 门槛 | 结果 |
| --- | --- | --- | --- |
| 跨集 frame0 std | 0.609 | > 0.01 | 过 |
| per-dim corr 中位数 | 0.902 | > 0.5 | 过 |
| 10 集 mean L2 | 0.584 | < 1.836（示教间 ×2） | 过 |

T30k 成为 B'。T2k / T10k 是已知更差。发布 B 单列对照。

## 四个指标

| 指标 | 结果 | 目标 |
| --- | --- | --- |
| 误报（B-rerun vs B-calib） | 回归 **0**：fold 完全一致（7 pass / 3 bounds fail）。gate 因 `--checkpoint` 改写 target 使 hash 不同而红 | 0 回归 |
| 检出（硬答案） | bias/noise **10/10**；缺文件/错维/错 id/错 hash **10/10**；**lag-10：`pred_shift` 10/10 恰为 10** | error 类 100%；lag 不再漏 |
| 排序（T2k L2 > T10k L2 > T30k L2） | **7 / 10** | 与步数同序 |
| 阈值覆盖（标定后 B'） | **7 / 10** | ≥ 90% |

覆盖未达标的原因：`action_bounds` 改为示教包络 + 10% 后，B' 在 ep3/6/7 仍略出界（出界份额 3.4% / 2.9% / 0.6%）。L2 与 smoothness、`action_lag` 均过。这是包络偏紧，不是模型塌缩。

## 检出

| 候选 | gate | 说明 |
| --- | --- | --- |
| action_bias / action_noise | 红 10/10 | 与第一轮相同 |
| action_lag=10（断言） | 6/10 红 | `max_lag_frames=14`（基线 \|k\| max+2），k=10 本身不够红 |
| action_lag=10（`pred_shift`） | **红 10/10，shift 全是 10** | 不依赖是否跟示教 |
| 缺 recorded / 错维 / 错 id / 错 hash | 红 10/10 error | fail-closed |
| 发布 B vs 示教包络 | 红 10/10 | 归一化空间错位 |
| B' + `drop_camera=cam_high` | 红 10/10 | L2 0.58 → 0.76 |
| B' + `state_noise=0.05` | 与 B-calib 同折 | L2 不变；只报告 |

第一轮漏掉的 lag-10：对 B' 派生的 mutant，`diff --runs` / `gate --max-shift 2` 恰好得到 10。

## 排序与 L2

| 模型 | 10 集 mean L2 |
| --- | --- |
| 示教对示教 | 0.918 |
| T2k | 0.759 |
| T10k | 0.600 |
| T30k = B' | 0.584 |
| 发布 B | 2.287 |

步数序 7/10。未同序的 3 集 L2 差很小，不推翻「更长训练更好」。

## 标定

- `max_l2 = mean+3σ = 1.442`
- `max_delta = 1.448`
- `action_bounds` = 10 集示教包络 + 10%
- `max_lag_frames = 14`（B' 在 10 集上 \|k\| 的 max + 2）
- 写回 `scenarios/real/aloha-static-coffee-ep{0-9}.yaml`，来源 eval_id=`B`（此时 B 已是 T30k）

## 产物

- `results/{B,B-calib,B-rerun,B-drop-cam,B-state-noise,B-published,T2k,T10k,T30k}.parquet`
- `results/tracking.json`、`results/bench-v2.json`
- `results/mutant-*.parquet`（本机从 B' run 合成）
- 训练日志：远端 `/tmp/wait-train.log`（finished 8586s）
