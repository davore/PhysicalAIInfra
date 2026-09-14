# Robogate 门禁 benchmark（2026-09-14）

用已知答案测 `eval` / `diff` / `gate`：同一 checkpoint 重跑必须绿；扰动 / 少训练 / 坏 run 必须红。
不测模型是否能泡咖啡，只测工具会不会漏、会不会误报。

留出集：`lerobot/aloha_static_coffee` ep0–9（1100 帧）。训练集：ep10–49。
基线 B：`gozdebaydogmus/act-aloha-static-coffee-test@646846823c8473712f689f59bdd03f798a688f6a`。
自训 ACT：从同结构随机初始化，顺序读取视频，保存 2k / 10k / 15k / 30k（`/root/autodl-tmp/ckpts/act-coffee-*`）。
`lerobot-train` 导入失败（`lerobot.scripts.train` 不存在），走 `scripts/remote/train_act.py`。
第一个 epoch 结束时 Sequential PyAV 解码器遇到 `EOFError`，已修复并入并并从 10k 续训到 30k。

训练 loss（batch=2，顺序采样）：step1=67.3 → 2k=2.01 → 10k=0.68 → 15k=0.24 → 30k=0.57。

标定：`max_l2 = mean+3σ = 2.6687`，`max_delta = 0.3526`，`action_bounds` = 10 集 B 预测包络外扩 10%。
写回 `scenarios/real/aloha-static-coffee-ep{0-9}.yaml`，来源 eval_id=`B`。

## 四个指标

| 指标 | 结果 | 目标 |
| --- | --- | --- |
| 误报率（B-rerun vs B-calib） | **0 / 10**，gate 绿 | 0 |
| 检出率（硬答案） | bias/noise **10/10**；缺文件/错维/错 id/错 hash **10/10**；**lag-10 漏报 0/10** | error 类 100% |
| 排序一致性（T2k L2 > T10k L2 > T30k L2） | **不可用**：自训输出常数 | 与步数同序 |
| 阈值覆盖（标定后 B） | **10 / 10** | ≥ 90% |

## 误报

`results/B-rerun.parquet` 对 `B-calib`：`gate` 绿，10 条全 pass。只换 run、同一 checkpoint，没有误报。

## 检出（gate 是否红）

| 候选 | gate | 红的 scenario | 主要触发 |
| --- | --- | --- | --- |
| action_bias=0.1 / 0.5 | 红 | 10/10 | `action_bounds` |
| action_noise=0.05 / 0.2 | 红 | 10/10 | `action_bounds` |
| action_lag=10 | **绿（漏）** | 0/10 | mean L2 几乎不变（2.286 vs B 的 2.287） |
| 删 `action.recorded.parquet` | 红 | 10/10 error | fail-closed |
| 动作改 13 维 | 红 | 10/10 error | fail-closed |
| 改 `scenario_id` | 红 | 10/10 error | id 硬校验 |
| 改 parquet `scenario_hash` | 红 | 10/10 | hash ≠ 当前 suite |
| T2k / T10k / T30k | 红 | 10/10 | `action_bounds`（自训输出是常数，L2 不可比） |
| B + `drop_camera=cam_high` | 红 | 10/10 | `action_bounds`（方向成立，仅报告） |
| B + `state_noise=0.05` | **绿** | 0/10 | L2 与 B 相同；不算硬漏报 |

`lag-10` 是本轮硬答案里唯一的漏报：50 fps 下 0.2 s 延迟相对示教轨迹的 mean L2 仍低于 `mean+3σ`，也没出界。
`state_noise` 与 `drop_camera` 按计划只报告；后者这次是红。

## 排序与 L2

ep0–9 上 `action_deviation` mean L2：

| 模型 | 10 集 mean L2 | 说明 |
| --- | --- | --- |
| 示教对示教（人对人下限） | 0.918 | 人对人对齐 |
| B（发布 ACT） | 2.287（约为示教间的 2.49×） | 会随观测微弱变化，但几乎不跟示教 |
| T2k / T10k / T30k | 1.445 / 1.456 / 1.376 | **输出是常数**（每帧相同、跨集差 ~0），L2 数字不可比 |

自训模型没有「优于」发布 checkpoint：顺序采样 + 小 batch 把它塌缩到数据集均值，常数动作碰巧比弱跟踪的 B 更接近示教均值，所以 L2 更低。`diff` 排出的 4/10「步数序」没有意义，排序指标作废。

## 阈值与 chunk 漂移

- 标定后 B 通过率 100%（`results/B-calib.parquet`）。
- ep0 基线 run 按 `frame % 100` 的 mean L2：offset0=2.565，offset50=2.580，offset99=2.604，**不是单调上升**，后段略高。

## 产物

- `results/{B,B-calib,B-rerun,B-drop-cam,B-state-noise,T2k,T10k,T30k}.parquet`
- `results/mutant-*.parquet`（本机从 B run 合成）
- `scenarios/real/aloha-static-coffee-ep0.yaml` … `ep9.yaml`（只读，头部注释写了 eval_id=B）
- 训练日志：`jobs/remote-train/train.log`、`train_resume.log`
