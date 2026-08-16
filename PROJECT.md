# PROJECT.md — 项目整体说明

> 版本：2026-08-16 修订。

## 1. 摘要

本项目用自博弈 PPO 模拟「皇帝牌」博弈中的有限理性学习动力学，比较三类均衡：

1. **理性均衡**：完整 5 轮博弈的递归混合策略纳什均衡，`solver.py` 验证 V₄=-0.2、首轮 p=q=0.2、胜率 0.8。
2. **PPO 动力学落点**：在 min-ent 0.5 条件下，自博弈 PPO 收敛到均衡邻域（run06 identity 角 p=0.207、q=0.253、胜率 0.800、期望 -0.198）。
3. **心理均衡**：前景理论效用层（λ、τ、α/β）对落点只有弱次级效应；主要偏差 q-p≈+0.05 与效用层无关（run06_curv 真 identity 仍存在）。

**当前证据边界**：无人类数据；“现实均衡”仍为模拟侧假设。

## 2. 游戏规则

- 双方各 1 张王牌 + 4 张平民，每轮同时出牌。
- A-A → 奴隶胜（皇帝 -5）；A-C / C-A → 皇帝胜（+1）；C-C → 平局进入下一轮；第 5 轮强制 A-A。

皇帝视角收益矩阵：

| 皇帝 \ 奴隶 | A | C |
| :---: | :---: | :---: |
| **A** | -5 | +1 |
| **C** | +1 | 0 |

## 3. 方法

| 组件 | 文件 | 说明 |
| :--- | :--- | :--- |
| 环境 | `env.py` | 5 维状态；客观奖励 +1/-5/0；非法动作 fail loud |
| 精确解 | `solver.py` | 递归 2×2 零和矩阵求解，内置验证门 |
| 效用层 | `utility.py` | identity / prospect（λ、α/β、γ）/ tilt |
| 学习算法 | `ppo.py` | 双 agent、actor-critic 64 维、GAE、clip、熵正则、min-ent hinge |
| 训练入口 | `train.py` | 阶段 3 单配置训练 |
| 扫描 | `sweep.py` | λ×τ 网格、预测核验、相图、收敛诊断 |

## 4. 主要结果（截至 2026-08-16）

- run06（min-ent 0.5, 400k, 60 格）：58/60 弱平稳；λ↑→p↓ 单调（Δ=-0.023）；λ 次级效应 q↓（Δ=-0.014）；τ 无位置效应；q-p 均值 +0.048（60/60 为正）。
- run06_curv（α=β=1.0, 2 seeds）：q-p=+0.063，说明 q-p 不是曲率所致；但 n=2，属方向性证据。
- run05 及更早的“收敛”结论全部作废（旧判据对极限环失明）。

## 5. 运行

```bash
python -m pytest test_env.py -q
python solver.py
python train.py --quick
python train.py --steps 200000
python sweep.py --demo
python sweep.py --steps 400000 --min-ent 0.5
```

Windows 一键训练脚本：`train_all_next.bat`（先跑验证门，再顺序执行 run07_sym/run08_ref/run09_asym）。

补充实验（代码已支持，命令见 results.md §7）：`--reward-loss -1`（对称收益消融）、
`--slave-lam 2.25`（非对称 λ）、`--weight-mode ref`（反馈环隔离）、
`--predictions-file`（运行专用预测）。

## 6. 诚实边界

- 阶段 2/3/4 的初版代码已由项目所有者凭记忆重写，并通过同一套验证门。
- 模拟结论仅覆盖 λ×τ×αβ 网格、5 seeds、400k 预算、min-ent 0.5、window 权重模式。
- 旧 README 数字是历史基线，不是当前结论。
