# 门禁灵敏度阶梯（2026-09-17）

从本地 B'（`act-coffee-30000`）11 条干净 run 离线合成突变，对现行 `scenarios/real` 阈值做 eval，再对 `fixtures/gate/baseline.parquet` 做 gate。纯 CPU，不改 scenario / 阈值 / fixtures。数字是「每类错误的最小检出量」：该组阶梯里第一个让 gate 变红的级别；整组都绿则写未检出。

## 最小检出量

| 错误类型 | 最小检出量 | 红 scenario | 首个 fail 断言 |
| --- | --- | --- | --- |
| 全维偏置 | 0.05 | 3/11 | action_bounds×3 |
| 单维偏置 d0 | 0.05 | 2/11 | action_bounds×2 |
| 单维偏置 d6（夹爪） | 0.1 | 7/11 | action_bounds×7 |
| 增益缩小 | 未检出 | 0/11 | — |
| 增益放大 | 1.1 | 4/11 | action_bounds×4 |
| 高斯噪声 σ | 0.02 | 2/11 | action_bounds×2 |
| 滞后（action_lag 断言） | 5 | 2/11 | action_lag×2 |
| 滞后（pred_shift，max_shift=2） | 5 | 2/11 | action_lag×2 |
| 统计量错位 action→state | 检出 | 9/11 | action_bounds×9 |
| 死模型（帧 0 常值） | 未检出 | 0/11 | — |

## 读法

- 这组阶梯里，gate 变红几乎全靠 `action_bounds`（示教包络 +15%）。
  `action_deviation` 的 `max_l2=2.232` 从没当过首个 fail。
- 增益缩到 0.5、把预测钉在第 0 帧，11/11 仍绿。
  门禁分不出「幅度不够」和「停住不动」。
- 偏置 0.02、噪声 0.01、滞后 2 帧也绿。
  「新模型差 10% 你能看出来吗」——这套门槛看不出来。
- `pred_shift` 在 lag=2 时测到 2，但 `max_shift=2` 含等于，从 5 帧起红。

## B' ep0 recorded d3 / d4

d3 std = 0.1815，d4 std = 0.1812。std 不小，ep0 上这两维是真反向，正对照也不干净。

来源 run：`aloha-static-coffee-ep0__act-coffee-30000__20260914T144519Z`。

## 各级明细

| 级别 | 红 | gate | 首个 fail |
| --- | --- | --- | --- |
| bias-0.01 | 0/11 | 绿 | — |
| bias-0.02 | 0/11 | 绿 | — |
| bias-0.05 | 3/11 | 红 | action_bounds×3 |
| bias-0.1 | 10/11 | 红 | action_bounds×10 |
| dim_bias-d0-0.05 | 2/11 | 红 | action_bounds×2 |
| dim_bias-d0-0.1 | 6/11 | 红 | action_bounds×6 |
| dim_bias-d0-0.2 | 11/11 | 红 | action_bounds×11 |
| dim_bias-d0-0.5 | 11/11 | 红 | action_bounds×11 |
| dim_bias-d6-0.05 | 0/11 | 绿 | — |
| dim_bias-d6-0.1 | 7/11 | 红 | action_bounds×7 |
| dim_bias-d6-0.2 | 10/11 | 红 | action_bounds×10 |
| dim_bias-d6-0.5 | 11/11 | 红 | action_bounds×11 |
| gain-0.9 | 0/11 | 绿 | — |
| gain-0.8 | 0/11 | 绿 | — |
| gain-0.5 | 0/11 | 绿 | — |
| gain-1.1 | 4/11 | 红 | action_bounds×4 |
| gain-1.2 | 10/11 | 红 | action_bounds×10 |
| noise-0.01 | 0/11 | 绿 | — |
| noise-0.02 | 2/11 | 红 | action_bounds×2 |
| noise-0.05 | 11/11 | 红 | action_bounds×11 |
| lag-2 | 0/11 | 绿 | — shift=[2] |
| lag-5 | 2/11 | 红 | action_lag×2 shift=[5] |
| lag-10 | 5/11 | 红 | action_lag×5 shift=[10] |
| lag-15 | 7/11 | 红 | action_lag×7 shift=[15] |
| stats_swap | 9/11 | 红 | action_bounds×9 |
| constant | 0/11 | 绿 | — |

JSON：[2026-09-17-sensitivity-ladder.json](./2026-09-17-sensitivity-ladder.json)。

## 怎么复跑

```bash
conda run -n robogate python scripts/bench/sensitivity_ladder.py
```
