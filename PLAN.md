# PhysicalAIInfra 开发计划：机器人回归测试与发布门禁（Robot CI / Eval）

> 版本 v0.2 · 2026-09-11 · 单人启动 · 纯软件、不碰实时控制、不做硬件

## 0. 一句话定位

**把一次现场失败变成一条永久的回归测试，让测试卡住发布。**

- 产品名（暂定）：`robogate`（开源 CLI + GitHub Action）；托管版 `Gate Cloud`
- 不做：日志可视化（Foxglove/Rerun 已做）、概率化根因分析（Alloy/Foxglove Agent 在做）、fleet 实时调度（InOrbit/Meili）、任何硬件
- 只做：Incident → Scenario → Replay → Assert → Report → Gate → （后期）Canary / Rollback / Attestation

### 0.1 v0.1 → v0.2 变更

1. **断言按回放模式分流。** 开环（`python-policy`）只能比较动作，看不到夹爪 / 碰撞 / 到达。每个断言声明 `requires_mode`；与 run 的 `mode` 不匹配则为 `skipped`（不算失败）。开环集：`action_deviation`、`action_bounds`、`action_smoothness`、`latency`（只报告不阻断）、`confidence_floor`。闭环集：`goal_reached`、`grasp_success`、`no_collision`、`sim_success_rate`、`recovery_within`。
2. **不再声称复用 `ACT-adapter`。** 该仓库只有 LICENSE。M1 的 `python-policy` 依赖可选 extra `lerobot`，用 Hugging Face Hub 上的 ACT checkpoint 做两版本对比。
3. **scenario 只读。** 去掉写回 `history`；结果落 `runs/<run_id>/`，历史由 `eval` 汇总到 parquet。scenario 增加 `schema_version` 与内容哈希。
4. **远端任务目录。** 真实目录是 `jobs/<id>/`；`wrap.sh` 维护 `jobs/current -> jobs/<id>` 软链，与 `.gpu/observe.yaml` 对齐。
5. **时间引用必须可解析。** MCAP 用 `from_ns`/`to_ns`（log_time 纳秒）或带时区的 ISO 8601；LeRobot 用 `episode_index + frame_from/frame_to`。
6. **M3 出口收窄。** 不再承诺「真机失败在 Isaac Lab 复现 ≥ 30%」。改为以 LeRobot 仿真数据集为闭环基线；真机 scenario 只注入关节状态 + 可得物体位姿；第 11 周设决策点。

其他：本机用 `conda create -n robogate python=3.11`；M1 先打通 LeRobot，MCAP `extract` 挪到第 7–8 周与 `ros2-node` 一起；`gate` = 基线通过的 scenario 不得失败 + `blocking: true` 必须通过。

## 1. 目标客户与核心用例

### ICP（前 12 个月只服务这一类）

- 用学习型策略（ACT / Diffusion Policy / VLA / 端到端导航）的机器人公司
- 真实部署 10–500 台，或者每周至少发一次新模型到真机
- 数据栈：ROS 2 + MCAP，或 LeRobot 数据集格式（parquet + mp4）
- 团队 5–50 名工程师，没有专职 eval/release 团队

优先细分：抓取/操作类、人形、仓储 AMR（学习型导航）。

### 三个 JTBD

1. **"这个 bug 再也不能出现"**：工程师看到一次失败（Foxglove 里、Alloy 报告里、或者客户投诉），30 分钟内把它变成一个版本化的 scenario 并进 CI。
2. **"新模型比旧模型好还是差"**：v38 对 v37 在 200 个失败场景上的通过率、每场景 diff、退化清单。
3. **"这个版本能不能发"**：CI 里一个红/绿决定，附证据；后期扩到"先发 5% 机器人、指标退化自动回滚"。

## 2. 产品范围

### 开源 CLI（MIT/Apache-2.0）

| 命令 | 作用 |
| --- | --- |
| `robogate extract <source>` | 从 LeRobot（M1）或 MCAP（M2）切出一段，生成 `scenario.yaml` + 数据切片 |
| `robogate replay <scenario> --target <adapter>` | 把输入喂给被测对象，采集输出到 `runs/<run_id>/` |
| `robogate assert <scenario> <run>` | 按 scenario 断言判定通过 / 失败 / 跳过 |
| `robogate eval <suite> --target` | 批量跑一组 scenario，输出 parquet + HTML 报告 |
| `robogate diff <run-a> <run-b>` | 两个版本的逐场景对比，列出退化 |
| `robogate gate <suite> --baseline` | CI 用：基线通过的不得失败，且 `blocking` 必须通过；退出码 0/1 + JSON 摘要 |
| `robogate validate <scenario>` | 校验 scenario.yaml（M0） |
| `robogate schema` | 导出 scenario JSON Schema（M0） |

### 托管版（M4 起）

- Scenario 库：跨机器人、跨站点、按失败类型/模块打标签
- Fleet 级回归看板：每个模型版本在全部场景上的曲线
- Release Gate API：给客户的发布流水线调用，记录每次放行/拒绝的证据
- （M5）Canary 控制：按机器人 cohort 分批发布，任务级指标退化触发回滚；签名 artifact + 启动证明才允许 rollout

### 明确的非目标（12 个月内）

- 不做自己的可视化 UI（scenario 直接给出 Foxglove 深链接）
- 不做数据采集 agent（读客户已有的 MCAP / S3 / Foxglove 导出）
- 不做多厂商 fleet 调度
- 不做 LLM 自动根因（可以调用 Alloy/Foxglove 的输出作为 scenario 来源）

## 3. 架构

```text
          客户已有                    robogate                          消费方
┌─────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────┐
│ MCAP / rosbag2      │──▶│ extract  → scenario.yaml     │   │ GitHub Actions   │
│ LeRobot dataset     │   │           + slice            │   │ GitLab CI        │
│ Foxglove event 链接 │   │ replay   → adapters:         │──▶│ 发布流水线       │
│ Alloy / 人工标记    │   │   open_loop:  python-policy  │   │ Gate Cloud 看板  │
└─────────────────────┘   │   closed_loop: ros2 / isaac  │   └──────────────────┘
                          │ assert   → pass/fail/skipped │
                          │ eval/diff/gate → runs/ + HTML│
                          └──────────────────────────────┘
```

开环 replay 只产出动作与时延；闭环 replay / 仿真才产出状态与事件。断言按 `requires_mode` 与 run 的 `mode` 对齐，对不上则 `skipped`。

### 3.1 Scenario 格式（产品的护城河，只读定义）

scenario 是不可变定义：不含运行历史。内容哈希由规范 JSON（排序键、去掉 None）计算，写入 run 的 `meta.json`。

```yaml
# scenarios/grasp/2026-09-10-robot1847-drop.yaml
schema_version: 0
id: grasp-drop-1847-20260910
blocking: true                    # gate：此条必须通过
source:
  kind: mcap
  mcap: s3://fleet-logs/robot1847/2026-09-10T14-32.mcap
  from_ns: 1757514722000000000    # log_time 纳秒；或用 from_iso / to_iso（必须带时区）
  to_ns:   1757514732000000000
  foxglove_url: https://app.foxglove.dev/...
tags: [grasp, perception, model:v37, site:shenzhen-2]
inputs:
  - /camera/wrist/image_raw
  - /joint_states
  - /tf
target:
  adapter: python-policy          # 开环：只能跑开环断言
  entry: lerobot                  # 或 module:callable；旧路径 lerobot.common.policies.act 已失效
  checkpoint: lerobot/act_aloha_sim_transfer_cube_human
expected:
  # 开环：比较策略输出的动作（requires_mode 由 type 隐含）
  - type: action_deviation
    topic: action
    reference: recorded
    max_l2: 0.05
  - type: latency                 # 非目标硬件上只报告，不阻断
    topic: /policy/action
    p95_ms: 80
  - type: confidence_floor
    topic: /policy/action
    min_confidence: 0.3
```

LeRobot 源：

```yaml
source:
  kind: lerobot
  dataset: lerobot/aloha_sim_transfer_cube_human
  episode_index: 0
  frame_from: 0
  frame_to: 400
```

### 3.2 适配器（replay 的被测对象）

| 适配器 | 模式 | M 阶段 | 说明 |
| --- | --- | --- | --- |
| `python-policy` | `open_loop` | M1 | 可选 extra `lerobot`；`entry: lerobot` 走 `ACTPolicy.from_pretrained` + processors。旧路径 `lerobot.common.policies.act` 已失效 |
| `ros2-node` | `closed_loop`（或开环订阅） | M2 | `rosbags` 回放输入 topic，docker / 本地启动被测节点，录输出。不依赖本机安装 ROS |
| `isaac-lab` | `closed_loop` | M3 | 注入可得的初始状态，N 次扰动，统计通过率（本机已有 `IsaacLab` 仓库） |

### 3.3 断言库

每个断言类声明 `requires_mode`。`run.mode` 不匹配 → `skipped`，报告里单独列出。任一 `fail` 则整体失败；`skipped` 不算失败。

**开环（M1，第 1 周先做前 3 个）**

| type | 第几周 | 含义 |
| --- | --- | --- |
| `action_deviation` | 1 | 预测动作对录制动作的逐帧 L2（或 DTW） |
| `latency` | 1 | p95 推理时延；**只报告，不把 suite 打红** |
| `confidence_floor` | 1 | 最低置信度 |
| `action_bounds` | 2 | 动作是否落在关节 / 夹爪限位内 |
| `action_smoothness` | 2 | 相邻帧动作差分 / jerk 上限 |

**闭环（仿真或真机节点）**

| type | 阶段 | 含义 |
| --- | --- | --- |
| `goal_reached` | M1 schema / M3 实现 | 末端或基座到达目标 |
| `grasp_success` | M1 schema / M3 实现 | 时限内夹爪成功 |
| `no_collision` | M1 schema / M3 实现 | 无碰撞事件 |
| `sim_success_rate` | M3 | N 次扰动的通过率 |
| `recovery_within` | M3 | 扰动后恢复时限 |

支持 `custom`：用户提供 Python 函数（M2）。`latency` 在 CI 机器上不是目标硬件，测量值写入报告，不参与红绿。

### 3.4 技术栈

- Python 3.11（`conda create -n robogate python=3.11`），`typer` CLI，`pydantic` v2 校验 scenario，`polars`/`pyarrow` 存 run parquet
- M2：`mcap` + `mcap-ros2-support` 读写，`rosbags` 反序列化（免装 ROS）
- M1 报告：`jinja2` HTML；结果表用 parquet，需要时再加 `duckdb`
- GitHub Action：composite action，缓存 slice，PR 上贴 diff 评论
- 托管版：FastAPI + Postgres + S3，前端 Next.js（M4 再决定）
- 仿真：Isaac Lab（远端 GPU）
- 远端 GPU：AutoDL 经 gpu-bridge，见第 7 节

### 3.5 Run 格式

每次 replay / 手工构造的评测结果是一个目录，**不写回** scenario：

```text
runs/<run_id>/
  meta.json
  outputs/<topic>.parquet      # 列：t_ns + 数据列（value / latency_ms / confidence …）
  outputs/<topic>.recorded.parquet   # reference: recorded 时的对照
  events.jsonl                 # 闭环事件：collision / grasp_success / goal_reached
```

`meta.json`：

```json
{
  "scenario_id": "grasp-drop-1847-20260910",
  "scenario_hash": "sha256:…",
  "adapter": "python-policy",
  "mode": "open_loop",
  "target_version": "act-v38",
  "git_sha": "abc123",
  "host": "local"
}
```

topic 路径 `/policy/action` 落盘为 `outputs/policy_action.parquet`（去掉首 `/`，其余 `/` 变 `_`）。`eval` 把多次 run 汇总成一张 parquet，作为历史，而不是改 scenario.yaml。

## 4. 仓库结构

```text
PhysicalAIInfra/
├── PLAN.md
├── pyproject.toml
├── README.md
├── .gitignore
├── .gpu/observe.yaml           # GPU 管家只读观察配方
├── robogate/                   # 开源 CLI 包
│   ├── cli.py
│   ├── scenario.py             # schema + 内容哈希
│   ├── run.py                  # Run 加载
│   ├── extract.py              # M1+
│   ├── replay/                 # M1+
│   ├── asserts/
│   ├── eval.py  diff.py  gate.py
│   └── report/templates/
├── action/                     # GitHub Action（M2）
├── scenarios/examples/         # 手写 / 公开数据切出的 scenario
├── runs/examples/              # 手工 run fixture（M0）与 replay 输出
├── jobs/                       # 远端：jobs/<id>/ + jobs/current 软链
├── scripts/remote/             # M3：wrap.sh 等
├── tests/
└── docs/adr/
```

## 5. 里程碑（12 个月）

### M0 · 第 1–2 周：定义

- scenario JSON Schema v0、Run 格式、3 个手工 scenario
- 开环断言先做 3 个，第 2 周补齐 5 个；ADR：不做可视化、开环/闭环分离
- 出口：`robogate assert` 对手写 scenario + 手工 run 给出正确红 / 绿 / skipped

### M1 · 第 3–6 周：LeRobot extract + replay(python-policy) + assert

- `extract` **先做 LeRobot**（与策略同栈），切 20 个 scenario
- `python-policy`：`lerobot` extra + HF Hub ACT checkpoint，两个 checkpoint 对比出 diff
- 开环 5 断言、parquet 结果、最简 HTML 报告
- 出口：一条命令从公开 LeRobot 数据集切 20 个 scenario，跑两个 checkpoint，输出退化清单。录 3 分钟 demo

### M2 · 第 7–10 周：CI + MCAP + 第一批 design partner

- GitHub Action：PR 上跑 suite，贴 diff 评论，`gate` 红绿
- MCAP `extract` + `ros2-node` 适配器（docker 内启动被测节点）
- 开源发布；ROS Discourse / LeRobot Discord / Physical AI 社群
- 接触 15 家 ICP，目标 3 家 design partner 各接入 ≥ 20 个真实 scenario
- 出口：至少 1 家 partner 把 `robogate gate` 放进发布流水线

### M3 · 第 11–13 周：仿真评测（远端 GPU）

- `isaac-lab` 适配器：注入关节状态 + 可得物体位姿，N 次扰动，统计通过率
- 远端任务化：`wrap.sh` 写 `jobs/<id>/status.json`，并更新 `jobs/current` 软链
- **第 11 周决策点**：若真机 MCAP 无法可靠得到物体位姿 / 场景资产，则收窄到纯仿真 scenario（LeRobot aloha sim transfer cube 等），真机只保留开环动作回归
- 出口：仿真数据集上形成闭环基线；报告同时展示开环 replay 与仿真统计。不承诺「现场失败在 sim 里复现 ≥ 30%」

### M4 · 第 4–6 月：托管版 MVP

- Scenario 库、fleet 级回归看板、Release Gate API、团队/权限
- 计费：按接入机器人数（$/robot/月）或 scenario 运行量，先不做按结果收费
- 出口：3 家付费 design partner，MRR > 0；至少 1 家把 gate 设为发布必要条件

### M5 · 第 7–12 月：从门禁到部署控制

- Canary：按 cohort 分批发布 + 任务级指标退化自动回滚（对接客户已有 OTA）
- 签名 artifact、启动证明才允许获取 rollout 凭证；输出 EU CRA / 机械法规需要的发布证据
- 出口：5 家付费客户；决策点见第 9 节

## 6. 前 90 天逐周计划（单人）

| 周 | 交付 |
| --- | --- |
| 1 | 仓库骨架、schema v0、Run 格式、3 个手写 scenario、开环 3 断言、`assert` 原型 |
| 2 | 开环补齐 `action_bounds` / `action_smoothness`；闭环 3 个进 schema + 单测；ADR；选定公开数据集（LeRobot aloha/so100 + 一个 MCAP 样例） |
| 3–4 | `extract`（LeRobot）；切出 20 个 scenario |
| 5–6 | `python-policy`（lerobot extra + HF Hub checkpoint）；两个 ACT checkpoint 的 `eval` + `diff` + HTML；demo 视频 |
| 7 | GitHub Action + `gate`；MCAP `extract`；文档站 |
| 8 | 开源发布；社群发帖；开始联系 15 家 ICP |
| 9–10 | `ros2-node` 适配器；按 partner 反馈修 schema；接入首个真实数据 |
| 11 | 远端 GPU 任务化（wrap.sh、status.json、`jobs/current` 软链）；Isaac Lab 环境；**决策是否收窄到纯仿真 scenario** |
| 12–13 | `isaac-lab` 适配器；仿真数据集闭环基线（+ 第 11 周通过则做有限真机注入） |

每周固定：周一定本周唯一目标，周五写公开 changelog（build in public，也是获客渠道）。

## 7. 远端 GPU 使用（AutoDL 经 gpu-bridge，只读观察）

Python 环境：远端同样使用 Python 3.11；本机开发用 conda 环境 `robogate`。

### 需要 GPU 的任务

| 任务 | 模块名 | 触发方式 | 日志 |
| --- | --- | --- | --- |
| 学习型策略回放推理（VLA/ACT 批量 replay） | `replay` | `scripts/remote/wrap.sh replay` | `jobs/<id>/replay.log`（经 `jobs/current`） |
| 批量 eval（几十到几百 scenario） | `eval` | `wrap.sh eval` | `jobs/<id>/eval.log` |
| Isaac Lab 仿真评测 | `sim` | `wrap.sh sim` | `jobs/<id>/sim.log` |
| 失败片段 embedding 聚类（M4 后） | `cluster` | `wrap.sh cluster` | `jobs/<id>/cluster.log` |

本地 CPU 就够的：`extract`、`assert`、`diff`、`gate`、报告生成、`ros2-node`（除非被测节点本身要 GPU）。

### 约定

- 远端仓库根：`/root/work/PhysicalAIInfra`
- 每个任务一个目录 `jobs/<id>/`，内含 `status.json`（`state/started/finished/exit_code/argv`）与日志
- `wrap.sh` 在启动时把 `jobs/current` 指到本次 `jobs/<id>`（符号链接，已存在则替换）。`.gpu/observe.yaml` 继续读 `jobs/current/*.log`，不必改
- 所有远端任务由 `scripts/remote/wrap.sh <module> [args]` 启动，进程命令行包含 `robogate` 与模块名
- 进度行：`[<tag>] scenario <i>/<n> pass=<x>%`；结束行 `ROBOGATE_REPLAY_DONE` / `ROBOGATE_EVAL_DONE` / `ROBOGATE_SIM_DONE` / `ROBOGATE_CLUSTER_DONE`
- GPU 管家只读远端，不接收推送；不改 gpu-bridge 源码，不向 8765 端口 POST
- 本计划不依赖 `gpu init` / `gpu submit`；以后要一键提交再用

## 8. GTM

- 渠道：开源 + build in public；ROS Discourse、LeRobot Discord、Hugging Face 机器人社区、Physical AI 相关 meetup；每个 design partner 争取一篇联名案例
- 定位语：**"Robot CI: turn every field failure into a permanent test, and let the test block the release."**
- 与生态的关系：Foxglove/Rerun 是上游（可视化、事件），Alloy 是上游（根因 → scenario 来源），Mender/balena 是下游（OTA 执行）。全部做集成，不做替代
- 定价（M4）：免费开源 CLI；托管 $50–150/robot/月 或 按 scenario 运行量；企业版加 SSO、审计、私有部署
- 首批 15 家目标：从 LeRobot 社区活跃的商业团队、人形公司、有真实部署的 AMR 团队里找；优先已经用 Foxglove 且每周发模型的

## 9. 指标与决策点

### 领先指标（每周看）

- 被创建的 scenario 数（尤其来自真实现场失败的）
- 在 CI 里运行的 suite 数 / 周
- 被 gate 拦下的发布次数（最核心：证明"测试卡住发布"真的发生）

### 12 个月决策点

- 成立：≥ 5 家付费 design partner，且 ≥ 1 家把 gate 设为发布必要条件 → 进入 M5，考虑融资/招第一名工程师
- 不成立：< 3 家付费或没有任何一家把它设成必要条件 → 说明"eval 变强制"在机器人行业还没到时候；切换到合规版本（EU CRA 发布证据、签名与证明），复用 gate 和 artifact 那部分代码

## 10. 风险与对策

| 风险 | 对策 |
| --- | --- |
| Foxglove 顺手把 Comparison Mode 做成 CI gate | 不做可视化、直接给 Foxglove 深链接；力气放在 scenario 格式、适配器和 CI 路径上 |
| 每个客户消息类型 / 模型输出 schema 都不同 | 适配器只依赖标准 ROS 2 消息与用户 `load()`；自定义断言用 Python 函数 |
| 真机 replay 的因果性（开环 ≠ 真实运行） | 定位为开环回归 + 仿真闭环统计；报告区分两者，不宣称等价 |
| 真机失败无法在 Isaac Lab 复现 | 第 11 周决策：收窄到仿真数据集闭环 + 真机开环动作回归 |
| ICP 太少，头部公司自建 | 从 LeRobot 生态的中小商业团队起步；提供比自建更快的 CI 集成 |
| 单人节奏 | 每周一个目标；90 天内没有 partner 就停下来重选 ICP |

## 11. 立即要做的三件事（M0 第 1 周）

1. 修订本文件为 v0.2，落地 `robogate/` 骨架：schema v0、Run 格式、`validate` / `schema` / `assert`
2. 3 个手写 scenario + pass/fail run fixture；开环断言 `action_deviation` / `latency` / `confidence_floor`；红 / 绿 / skipped 单测
3. ADR：不做可视化、开环与闭环分离。LeRobot 数据集下载与 15 家 ICP 名单放到第 2 周 / M2 预热，不堵第 1 周出口
