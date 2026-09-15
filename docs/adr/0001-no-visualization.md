# ADR 0001: 不做自己的可视化

- 状态：接受
- 日期：2026-09-11

## 决定

robogate 不实现日志 / 轨迹 / 图像的可视化 UI。scenario 只保存 Foxglove（或同类工具）的深链接。

## 理由

Foxglove 与 Rerun 已经覆盖回放与对比。自建 UI 会把单人带宽从「把失败变成门禁」挪到「再做一个查看器」，也容易变成和上游抢同一块市场。护城河是 scenario 格式、适配器和 CI 路径，不是画面。

## 后果

报告用 parquet + 最简 HTML 表格；需要看原始信号时点开 `foxglove_url`。不承诺 Comparison Mode 或 3D 场景。
