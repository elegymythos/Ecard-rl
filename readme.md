# Ecard-rl：皇帝牌自博弈 PPO 研究项目

> 当前版本：2026-08-17。本文件是项目入口；详细研究计划见 `direction.md`，
> 假设与推导见 `Question.md`，数据结论与证据边界见 `results.md`，
> 综合分析见 `ANALYSIS.md`。

## 1. 项目一句话

用自博弈 PPO 模拟赌徒玩「皇帝牌」（+1/-5 不对称收益），比较理性均衡、PPO 动力学落点、心理效用扰动三者之间的关系。

## 2. 当前进度

| 阶段 | 状态 |
| :--- | :--- |
| 0–3：问题、环境、solver、PPO 基线 | ✅ 完成 |
| 4：网格与消融 run06–run19 | ✅ 全部完成 |
| 5：人类对局数据 | ⬜ 未做 |
| 6：结果定稿 | ✅ `results.md` 已更新 |

正式运行：run06（λ×τ 主网格）、run06_curv、run07_sym、run08_ref、run09_asym、run10_sym/ref、run11_identity、run11_minent、run12_roleswap、run13_shared、run14_identity_lambda、run15_same_init、run16_update_order、run17_shared_trunk、run18_advnorm、run19_long。

## 3. 当前核心结论

1. **理性均衡**：V₄=-0.2、首轮 p=q=0.2、胜率 0.8、逐层 p_k=1/(k+1)，`solver.py` 验证通过。
2. **主观 Nash 不变性**：在当前效用层实现（λ、α/β、ref 常数权重）下，主观 Nash 均衡仍为 p_k=q_k=1/(k+1)，与心理参数无关。
3. **PPO 落点是“min-ent=0.5 下的弱平稳中心”**：run06 在 400k 步、min-ent 0.5 下 58/60 弱平稳，落点在均衡邻域；但 min-ent 敏感性（run11_minent）显示 0.55/0.60 下 q 与 q-p 大幅上升，说明该落点不是正则化无关的固有动力学。
4. **λ↑→p↓ 是梯度尺度效应**：run06/run14 观测到 p 随 λ 下降；run18 在 adv-norm 开启后该效应消失（p 差≈+0.003），支持“优化尺度”解释。
5. **τ 对 q 无位置效应**：window 与 ref 模式均 null；但 τ 理论上应先影响皇帝 p，run06 存在弱 τ→p 正效应（+0.0044，5/5 seed 同号），仍需正式核验。
6. **q-p≈+0.05 机制未完全解决**：不是曲率、不是收益量级、不是相同初始化、不是共享躯干；随机更新顺序的最终 checkpoint 接近 0，更新相位成为新的候选机制。
7. **400k 不是明显瞬态**：run19_long（1M 步）identity 窗口落点与 400k 一致。
8. **旧 README 基线**（p=0.294、q=0.456、胜率 0.797）是历史记录，不是当前结论。

## 4. 文件结构

- `Question.md`：研究假设、主观 Nash 推导、下一步问题。
- `direction.md`：研究计划、实验纪律、阶段状态。
- `results.md`：所有实验结论、预测核验、证据边界。
- `ANALYSIS.md`：全面分析报告（方法、统计、问题、建议）。
- `env.py` / `solver.py` / `utility.py` / `ppo.py` / `train.py` / `sweep.py`：核心代码。
- `analyze.py`：合并统计、seed 聚类、置换检验。
- `evaluate.py`：逐层策略与 exploitability 评估。
- `train_all_next.bat`：Windows 一键训练脚本（历史）。
- `train_all_next.fish`：Linux/macOS fish 一键训练脚本。
- `debug/`：逐日修复与审计记录。
- `data/runs/`：正式运行归档。

## 5. 快速开始

```bash
# 验证门
python -m pytest test_env.py -q
python solver.py

# 快速训练
python train.py --quick

# 分析已有结果
python analyze.py --all
python evaluate.py --dir data/runs/run11_identity --swap-eval --save
```

## 6. 证据边界

- 所有模拟结论限定在：5 seeds、400k 步、min-ent 0.5、window/off 权重模式。
- min-ent 敏感性显示结论强烈依赖正则化强度。
- 无人类数据；“现实均衡”仍是模拟侧假设。
- 数据目录 `data/runs/` 不在 git 版本控制内，复现需自行保留/同步。
