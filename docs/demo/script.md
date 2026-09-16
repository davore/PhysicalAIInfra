# 3 分钟 demo 脚本

画面按 README「Incident → Gate」三步走，最后用 PR #2 证明门禁真的卡住发布。

## 0:00–0:20 一句话

把一次现场失败冻成回归测试，让测试卡住发布。不做可视化、不做采集、不做硬件。

## 0:20–1:10 切事故窗

公开 coffee 数据上，发布 ACT 把 action 解到 `observation.state` 空间。把 L2 最高的 5 秒冻成 scenario：

```bash
robogate extract lerobot lerobot/aloha_static_coffee \
  --episode 0 --from 413 --to 663 \
  --id coffee-published-action-space-mismatch \
  --checkpoint gozdebaydogmus/act-aloha-static-coffee-test \
  --target-revision 646846823c8473712f689f59bdd03f798a688f6a
robogate validate scenarios/real/coffee-published-action-space-mismatch.yaml
```

给观众看 yaml：`id`、`blocking: true`、`action_deviation` / `action_bounds`。

## 1:10–2:00 回放与判定

本机 mock 只演示 CLI；真实数字用已提交的 fixtures：

```bash
robogate eval scenarios/real --runs runs --eval-id candidate
robogate gate scenarios/real \
  --baseline fixtures/gate/baseline.parquet \
  --candidate fixtures/gate/candidate.parquet
```

绿：B'（自训 30k，会跟踪）。再跑：

```bash
robogate gate scenarios/real \
  --baseline fixtures/gate/baseline.parquet \
  --candidate fixtures/gate/known-bad/published-b.parquet
```

必须红。一句话：工具对会跟踪的模型给绿，对解错空间的发布权重给红。

## 2:00–2:40 CI 卡住发布

打开 https://github.com/davore/PhysicalAIInfra/pull/2  
标题 `release: promote published coffee ACT`。`gate` 红，main 分支保护把它挡在合并之外。

对照：https://github.com/davore/PhysicalAIInfra/pull/1 和 #4 绿，合进了 main。

## 2:40–3:00 边界

开环动作回归，不是「会不会泡咖啡」。B' 阈值是自参照标定；非循环证据是误报 0、T2k 红 / T10k 绿、发布 B 红、突变体全红。第二个已知答案见 `docs/notes/2026-09-17-second-known-answer.md`。
