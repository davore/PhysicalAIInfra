# robogate

**把一次现场失败变成一条永久的回归测试，让测试卡住发布。**

Robot CI：Incident → Scenario → Replay → Assert → Report → Gate。不做可视化、不做采集、不做硬件。

M0 已落地：`scenario` schema v0、Run 目录格式、`validate` / `schema` / `assert`。`extract` / `replay` / `eval` / `diff` / `gate` 尚未实现。

## 安装

需要 Python 3.11（本机系统 Python 是 3.9，用 conda）：

```bash
conda create -n robogate python=3.11 -y
conda activate robogate
pip install -e ".[dev]"
```

## 用法

```bash
robogate validate scenarios/examples/lerobot-open-loop.yaml
robogate schema
robogate assert scenarios/examples/lerobot-open-loop.yaml runs/examples/pass
robogate assert scenarios/examples/lerobot-open-loop.yaml runs/examples/fail
robogate assert --json scenarios/examples/mixed-closed-loop.yaml runs/examples/pass
```

开环 run 遇到闭环断言（抓取成功、无碰撞等）会标为 `skipped`，不算失败。

## 开发

```bash
conda activate robogate
pytest
```

计划见 [PLAN.md](PLAN.md)。
