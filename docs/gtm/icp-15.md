# 15 家 ICP（LeRobot 生态中小商业团队优先）

按 PLAN：前 12 个月只服务「已经在用 LeRobot 或即将把策略放进 CI 的中小团队」。头部大厂（PI / Figure / Tesla）不做首轮。每行是公开信息 + 猜测；联系前再核一次。

| # | 团队 | 产品 / 场景 | 数据栈（猜测） | 发布流程 | 入口 |
| --- | --- | --- | --- | --- | --- |
| 1 | Phospho | SO-100 控制与数据工具 | LeRobot 数据集、云端 replay | 有产品发布节奏 | phospho.ai / Discord |
| 2 | TheRobotStudio | SO-ARM / 开源臂套件 | 示教 + LeRobot 导出 | 硬件+软件版本 | therobotstudio.com |
| 3 | Wowrobo | SO-ARM101 量产套件 | 买家自采 LeRobot | 套件发货，软件弱 | wowrobo.com |
| 4 | Seeed Studio | reComputer + 机器人套件 | ROS / LeRobot 示例 | 有产品线发布 | wiki.seeedstudio.com |
| 5 | AgileX / Cobot Magic | 移动底盘 + ALOHA 类双臂 | ROS 2、可能 MCAP | 固件/镜像发布 | agilex.ai |
| 6 | Galaxea | 双臂操作平台 | 自有采集 + 公开 LeRobot 子集 | 模型卡 / 内部权重 | galaxea-ai.com |
| 7 | Fourier Intelligence | 人形 + 操作数据 | 混合；部分公开 | 内部 OTA | fftai.com |
| 8 | Robotera | 人形 / 操作 | 自有 + 开源策略 | 早期，发布流程未公开 | robotera.com |
| 9 | AgiBot | 人形与工业数据 | 大规模自有；LeRobot 兼容导出是切入点 | 有内部评测 | agibot.com |
| 10 | Standard Bots | 工作单元机械臂 | 示教 + 云策略 | 云端推模型到机器人 | standardbots.com |
| 11 | Sereact | 拣选 / 视觉策略 | 自有；ROS 周边 | 客户现场更新 | sereact.ai |
| 12 | Micropsi Industries | 力控示教策略 | 示教轨迹，非 LeRobot | 工业发布闸 | micropsi-industries.com |
| 13 | Plus One Robotics | 物流拣选 | 自有视觉模型 | 现场软件发布 | plusonerobotics.com |
| 14 | Path Robotics | 焊接单元 | 自有；闭环更强 | 工厂软件版本 | path-robotics.com |
| 15 | Enactic / ACT 周边团队 | 开源 ACT 后续 | LeRobot ACT checkpoint | 研究权重 Hugging Face | LeRobot Discord / HF |

## 接触顺序

先 1–5（已经在 LeRobot/SO-100 栈上，接入成本最低），再 6–9（有数据、发布闸可能自建），11–14 等 MCAP / `ros2-node` 有客户数据再谈。

## 第一封信要问的三件事

1. 失败案例现在存在哪里（LeRobot repo、MCAP、Foxglove、内部表格）？
2. 发新权重要过几道人工闸，有没有 CI？
3. 能不能拿出 1 条已经复现过的失败，冻成 scenario？

MCAP extract / `ros2-node` 等第一家给出格式再做，不预建。
