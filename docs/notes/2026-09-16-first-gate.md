# 第一次门禁闭环（2026-09-16）

把公开 coffee 数据做成事故 scenario，并用 GitHub Action 卡住本仓库的合并。

## 标定

- B' = `/root/autodl-tmp/ckpts/v2/act-coffee-30000`（冻结，不重训）。Hub 目标：`davore/act-aloha-static-coffee-bprime`（私有；`scripts/remote/upload_bprime.py`）。本机备份 `.cache/bprime/act-coffee-30000`（gitignored）。
- `action_bounds`：示教包络 pad **0.15**（0.10 时 ep3/6/7 仍出界 0.034 / 0.029 / 0.006 L2）。
- 整套 `scenarios/real`（10 条整集 + 1 条事故窗）重写后：`max_l2 = 2.232`（mean+3σ，事故窗把 B' 的 L2 拉高了）、`max_delta = 1.446`、`max_lag_frames = 14`。
- B'：11/11 绿。同一 parquet 对自身 gate 绿（误报 0）。
- 发布 B `gozdebaydogmus/act-aloha-static-coffee-test@6468468…`：11/11 红。
- `mutant-lag-10` + `gate --max-shift 2`：`pred_shift` 全是 10，红。

删掉本地恒等回放 `…act-aloha-static-coffee-test__20260915T042008Z`（L2=0，曾让 ep0 假绿）。

## 事故窗

| 项 | 值 |
| --- | --- |
| id | `coffee-published-action-space-mismatch` |
| 源 | `lerobot/aloha_static_coffee` ep0 frames **413–663**（5.0s @ 50fps） |
| 选取 | 发布 B 对示教逐帧 L2 最高的 250 帧（窗内 mean L2 ≈ 3.00；全集 2.58） |
| tags | `grasp`, `incident:norm-space`, `model:published-b`, `open-loop` |
| B' 窗内 L2 | 2.003（过 2.232） |
| 发布 B 窗内 L2 | 3.026（红）；bounds overflow 0.170 |

这是发布事故（权重把 action 解到 `observation.state` 空间），不是掉胶囊的任务事故。

## 引擎修正

- `--checkpoint` 不再改 `scenario_hash`（hash 取磁盘定义）。
- `eval --version-contains` 挑选 run；无扰动时优先未扰动的最新 run。

## CI

- `fixtures/gate/{baseline,candidate}.parquet` = B'。
- `fixtures/gate/known-bad/published-b.parquet` = 发布 B。
- `.github/workflows/gate.yml`：pytest + 绿门禁必须过 + 发布 B 必须红。

闭环 PR（已合并）：https://github.com/davore/PhysicalAIInfra/pull/1
被卡住的 promote PR（gate 红，勿合）：https://github.com/davore/PhysicalAIInfra/pull/2
CI 失败记录：https://github.com/davore/PhysicalAIInfra/actions/runs/34940283876

## GPU 复验（2026-09-16）

数字摘要：[2026-09-16-gpu-verify.json](./2026-09-16-gpu-verify.json)。5090 与 MicroBo FDTD 共用，等卡空后 11:36–12:16 CST 重放。

| 项 | 结果 |
| --- | --- |
| B'（T30k）10 集 | 10/10 绿；tracking corr 0.902 / L2 0.584 |
| 发布 B 10 集 | 10/10 红；corr ≈ 0 / L2 2.287 |
| gate B' → 发布 B | 红 10/10；pred_shift 全 0 |
| 合成 lag-10 | pred_shift 全 10 |
| 旧 T2k / T10k 用现行阈值 | T2k 红、T10k 绿 |

ep0 最难（B' L2 1.348），仍低于 `max_l2` 2.232。这是对已提交 fixtures 的复验，不改 scenario hash。

仓库已公开：https://github.com/davore/PhysicalAIInfra  
main 分支保护：必须走 PR、必需检查 `gate`、`enforce_admins`。  
PR #2 仍开着且上次 `gate` 红（勿合）：https://github.com/davore/PhysicalAIInfra/pull/2
