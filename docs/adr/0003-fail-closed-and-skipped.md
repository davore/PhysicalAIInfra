# ADR 0003: ERROR fail-closed，不可测指标 skipped，scenario_id 硬校验

- 状态：接受
- 日期：2026-09-13

## 决定

1. 断言无法求值（缺文件、缺 `value` 列、动作维度不一致）记为 `error`，整体失败。CLI 不得因 `evaluate()` 抛异常而 traceback。
2. 适配器没有记录的指标记为 `skipped`，不算失败。ACT 没有置信度，`confidence_floor` 必须 skip，而不是假绿或崩溃。
3. `run.meta.scenario_id` 必须等于当前 scenario 的 `id`，否则 `assert` 直接退出 1。`scenario_hash` 不一致仍只警告。

## 理由

M0 的 `assert` 只看 run 目录。真机 scenario 配 4 维手写 fixture 会 PASS。缺列时 traceback 让门禁变成「工具坏了」而不是红绿。ACT 类策略没有 confidence，要求这一列等于强迫适配器造假数。

## 后果

`overall_ok` 把 `fail` 和 `error` 视为失败，`skipped` 不是。开环 ACT 回放默认跳过 `confidence_floor`。换 scenario 必须重跑 replay，不能复用别的 run。
