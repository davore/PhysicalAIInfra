# robogate

**把一次现场失败变成一条永久的回归测试，让测试卡住发布。**

[![gate](https://github.com/davore/PhysicalAIInfra/actions/workflows/gate.yml/badge.svg)](https://github.com/davore/PhysicalAIInfra/actions/workflows/gate.yml)

Robot CI：Incident → Scenario → Replay → Assert → Report → Gate。不做可视化、不做采集、不做硬件。

CLI 已落地：`validate` / `schema` / `extract`（LeRobot v3）/ `replay` / `assert` / `eval` / `diff` / `gate`。

## 安装

需要 Python 3.11（本机系统 Python 是 3.9，用 conda）：

```bash
conda create -n robogate python=3.11 -y
conda activate robogate
pip install -e ".[dev]"
```

真实 ACT 回放另装可选 extra（通常在远端 GPU）。RTX 5090 必须先装 torch cu128，再 `pip install --no-deps lerobot`；不要直接 `pip install -e ".[lerobot]"`。`scripts/remote/setup_env.sh` 按这个顺序装。远端 Hugging Face 走 `HF_ENDPOINT=https://hf-mirror.com`。

## Incident → Gate

用公开数据把一次失败冻成测试，再用门禁拦住坏权重。

**1. 切事故窗**（不必等自家 Foxglove；这里是 coffee 示教上发布权重解错空间的 5 秒）：

```bash
robogate extract lerobot lerobot/aloha_static_coffee \
  --episode 0 --from 413 --to 663 \
  --id coffee-published-action-space-mismatch \
  --out scenarios/real --slice-dir slices \
  --revision b144896feb1f37398a862927b22cd3abdf005a6b \
  --checkpoint gozdebaydogmus/act-aloha-static-coffee-test \
  --target-revision 646846823c8473712f689f59bdd03f798a688f6a
robogate validate scenarios/real/coffee-published-action-space-mismatch.yaml
```

**2. 回放**（本机 mock，或不重放、直接用已提交的 fixture）：

```bash
robogate replay scenarios/real/coffee-published-action-space-mismatch.yaml \
  --adapter mock --slice slices/coffee-published-action-space-mismatch
robogate eval scenarios/real --runs runs --eval-id candidate
```

**3. 门禁**（基线绿不得变红；`blocking` 必须过）：

```bash
robogate gate scenarios/real \
  --baseline fixtures/gate/baseline.parquet \
  --candidate fixtures/gate/candidate.parquet
```

把 `fixtures/gate/known-bad/published-b.parquet` 当成 candidate 必须红。GitHub Action 在每个 PR 上跑这两次判定。

`assert` 要求 `run.meta.scenario_id` 与 scenario `id` 一致。缺列或维度不对是 `error`（失败）。适配器没记的 latency / confidence，以及开环 run 上的闭环断言，标 `skipped`，不算失败。

## 远端 GPU 回放

```bash
bash scripts/remote/sync.sh push
ssh microbo-gpu 'bash /root/work/PhysicalAIInfra/scripts/remote/wrap.sh replay \
  scenarios/real/coffee-published-action-space-mismatch.yaml --device cuda \
  --slice slices/coffee-published-action-space-mismatch'
bash scripts/remote/sync.sh pull runs/
robogate eval scenarios/real --runs runs --version-contains act-coffee-30000 --eval-id Bp
```

## 开发

```bash
conda activate robogate
pytest
```

计划见 [PLAN.md](PLAN.md)。首个门禁记录见 [docs/notes/2026-09-16-first-gate.md](docs/notes/2026-09-16-first-gate.md)。

### Bench 附录

`scripts/remote/train_act.py` 只为门禁 bench 冻过一个会跟踪的基线 B'，不再重训。已知答案测试见 `docs/notes/2026-09-15-gate-bench-v2.md`。
