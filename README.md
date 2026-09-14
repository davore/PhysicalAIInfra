# robogate

**把一次现场失败变成一条永久的回归测试，让测试卡住发布。**

Robot CI：Incident → Scenario → Replay → Assert → Report → Gate。不做可视化、不做采集、不做硬件。

已落地：`scenario` schema v0、Run 目录、`validate` / `schema` / `assert` / `extract`（LeRobot v3）/ `replay`（mock 或 lerobot ACT）。`eval` / `diff` / `gate` 尚未实现。

## 安装

需要 Python 3.11（本机系统 Python 是 3.9，用 conda）：

```bash
conda create -n robogate python=3.11 -y
conda activate robogate
pip install -e ".[dev]"
```

真实 ACT 回放另装可选 extra（通常在远端 GPU）。RTX 5090 必须先装 torch cu128，再 `pip install --no-deps lerobot`；不要直接 `pip install -e ".[lerobot]"`，否则会把 torch 降到不支持 sm_120 的版本。`scripts/remote/setup_env.sh` 按这个顺序装。

远端访问 Hugging Face 走 IPv4 镜像（`HF_ENDPOINT=https://hf-mirror.com`），`wrap.sh` 已默认设置。

## 用法

本机切一份公开真机数据，并校验 scenario：

```bash
robogate extract lerobot lerobot/aloha_static_coffee \
  --episode 0 --from 0 --to 1100 \
  --id aloha-static-coffee-ep0 \
  --out scenarios/real --slice-dir slices \
  --revision b144896feb1f37398a862927b22cd3abdf005a6b \
  --checkpoint gozdebaydogmus/act-aloha-static-coffee-test \
  --target-revision 646846823c8473712f689f59bdd03f798a688f6a
robogate validate scenarios/real/aloha-static-coffee-ep0.yaml
robogate schema
```

本机用 mock 适配器验证流水线（不需要 GPU / lerobot）：

```bash
robogate replay scenarios/real/aloha-static-coffee-ep0.yaml --adapter mock --slice slices/aloha-static-coffee-ep0
robogate assert scenarios/real/aloha-static-coffee-ep0.yaml runs/<run-id>
```

远端 GPU 跑真实 ACT（RTX 5090 需要 torch cu128）：

```bash
bash scripts/remote/sync.sh push
ssh microbo-gpu 'bash /root/work/PhysicalAIInfra/scripts/remote/setup_env.sh'
ssh microbo-gpu 'bash /root/work/PhysicalAIInfra/scripts/remote/wrap.sh replay \
  scenarios/real/aloha-static-coffee-ep0.yaml --device cuda --slice slices/aloha-static-coffee-ep0'
bash scripts/remote/sync.sh pull runs/
robogate assert scenarios/real/aloha-static-coffee-ep0.yaml runs/<run-id>
```

`assert` 要求 `run.meta.scenario_id` 与 scenario `id` 一致。适配器没记的 latency / confidence 会标 `skipped`，不算失败。缺 `value` 列或维度对不上是 `error`，算失败。

开环 run 遇到闭环断言（抓取成功、无碰撞等）也会标为 `skipped`。

## 开发

```bash
conda activate robogate
pytest
```

计划见 [PLAN.md](PLAN.md)。真机卡点见 [docs/notes/2026-09-13-real-data-blockers.md](docs/notes/2026-09-13-real-data-blockers.md)。
