# PROJECT.md — 项目整体说明

> 版本：2026-08-16 修订。

## 1. 摘要

本项目用自博弈 PPO 模拟「皇帝牌」博弈中的有限理性学习动力学，比较三类均衡：

1. **理性均衡**：完整 5 轮博弈的递归混合策略纳什均衡，`solver.py` 验证 V₄=-0.2、首轮 p=q=0.2、胜率 0.8。
2. **PPO 动力学落点**：在 min-ent 0.5 条件下，自博弈 PPO 收敛到均衡邻域（run06 identity 角 p=0.207、q=0.253、胜率 0.800、期望 -0.198）。
3. **心理均衡**：前景理论效用层（λ、τ、α/β）对落点只有弱次级效应；主要偏差 q-p≈+0.05 在 run06_curv 真 identity 下仍存在（n=2，方向性），对称收益消融下也仍为正（5 seeds，但仅 2/5 弱平稳）。

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
| 扫描 | `sweep.py` | λ×τ 网格、预测核验、相图、收敛诊断；`--swap-roles`/`--shared-policy`/`--save-agents` |
| 合并统计 | `analyze.py` | run 目录合并、seed 聚类 CI、置换/符号检验 |
| 策略评估 | `evaluate.py` | 逐层 p_k/q_k、最优反应值、exploitability |

## 4. 主要结果（截至 2026-08-16）

- run06（min-ent 0.5, 400k, 60 格）：58/60 弱平稳；λ↑→p↓ 单调（Δ=-0.023）；λ 次级效应 q↓（Δ=-0.014）；τ 无位置效应；q-p 均值 +0.048（5/5 seed 为正，run 级 60/60）。
- run06_curv（α=β=1.0, 2 seeds）：q-p=+0.063；n=2，仅方向性证据。
- run07_sym + run10_sym：对称收益下 q-p 5/5 为正（+0.049），但仅 2/5 弱平稳。
- run08_ref + run10_ref：ref 模式下 τ 无位置效应（5 seeds，配对 p≈0.37）。
- run11_identity：真 identity 下 q-p=+0.052（5/5 为正），逐层接近均衡。
- run12_roleswap：训练式消融失败（0/5 收敛）；事后 `--swap-eval` 反号是恒等式，不作为机制证据。
- run13_shared：共享网络 p=q≈0.50，5/5 弱平稳但不收敛到均衡。
- run14_identity_lambda：真 identity 下 λ↑→p↓ 方向为负但单独不显著（🟡）。
- run11_minent：**min-ent 0.5 不是中性稳定器**；0.55/0.60 时 λ=1 的 q 升至 0.358/0.407，
  q-p 升至 0.159/0.205。
- run15_same_init：窗口 q-p=+0.033，仍为正。
- run16_update_order：slave_first q-p=+0.039；random 窗口 q-p=+0.051 但 checkpoint 近 0。
- run17_shared_trunk：共享躯干 q-p=+0.060，不消除 q-p。
- run18_advnorm：**adv-norm 下 λ↑→p↓ 消失**，支持 λ 效应为梯度尺度效应。
- run19_long：1M 步与 400k 窗口落点一致。
- run05 及更早的“收敛”结论全部作废（旧判据对极限环失明）。
- 主观 Nash 不变性：当前效用层下均衡策略与 λ/α/β/ref 常数权重无关，见 results.md §2.1。

## 5. 运行

```bash
python -m pytest test_env.py -q
python solver.py
python train.py --quick
python train.py --steps 200000
python sweep.py --demo
python sweep.py --steps 400000 --min-ent 0.5
```

训练脚本：
- Windows：`train_all_next.bat [steps]`（历史脚本，已跑完 run11–run14 与 run11_minent）。
- Linux/macOS（fish）：`fish train_all_next.fish [steps]`（已跑完 run15–run19；
  重跑会自动跳过已存在的 (λ,τ,seed)）。

补充实验（代码已支持）：`--reward-loss -1`（对称收益消融）、`--slave-lam 2.25`（非对称 λ）、
`--weight-mode ref`（反馈环隔离）、`--swap-roles`（角色交换消融）、
`--shared-policy`（共享网络消融）、`--save-agents`（保存 checkpoint 供 evaluate.py）。

## 6. 诚实边界

- 阶段 2/3/4 的初版代码已由项目所有者凭记忆重写，并通过同一套验证门。
- 模拟结论仅覆盖 λ×τ×αβ 网格、5 seeds、400k 预算、min-ent 0.5、window 权重模式。
- run11_identity/run11_minent/run13_shared/run14_identity_lambda 与
  run15_same_init/run16_update_order/run17_shared_trunk/run18_advnorm/run19_long
  均已完成并分析；run12_roleswap 训练式消融失败，`--swap-eval` 不作为机制证据。
- min-ent 敏感性显示落点强烈依赖 min-ent 强度，因此所有“PPO 落点”结论必须限定
  在 min-ent=0.5；主观 Nash 不变性说明心理参数不移动均衡策略（见 results.md §2.1）；
  adv-norm 使 λ↑→p↓ 消失，支持“梯度尺度效应”解释。
- 旧 README 数字是历史基线，不是当前结论。
