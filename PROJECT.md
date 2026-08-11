# Ecard-rl：皇帝牌博弈 × 前景理论 × PPO

> 项目整体说明（2026-08-09）。分阶段计划见 direction.md；研究问题见 Question.md；旧实现基线见 readme.md。

## 一句话

不是复现 PPO，是用 PPO 模拟赌徒：比较理性均衡、梯度下降的吸引子、赌徒的心理均衡。

## 游戏规则

- 双方各 1 张王牌 + 4 张平民，每轮同时出牌；
- A-A → 奴隶胜（皇帝 -5）；A-C / C-A → 皇帝胜（+1）；C-C → 平局，进入下一轮；
- 第 5 轮强制 A-A（奴隶胜）。

皇帝视角收益矩阵：

| 皇帝 \ 奴隶 | 出王牌 A | 出平民 C |
| :---: | :---: | :---: |
| **出王牌 A** | -5（奴隶胜） | +1（皇帝胜） |
| **出平民 C** | +1（皇帝胜） | 0（平局） |

## 三种均衡

1. **理性均衡**（数学解）：solver.py 已验证——完整 5 轮博弈值 V_4=-0.2，首轮 p=q=0.2，均衡胜率 80%，每轮闭式 p_k=1/(k+1)。README 里的 1/7、+1/7 只是单轮静态近似。
2. **PPO 原始动力学**（裸奖励）：旧实现（readme.md）停在 p=0.294、q=0.456、胜率 79.7%、期望 -0.217——优势归一化是头号嫌疑。阶段 3 初版可以复现这条研究线（data/stage3_*.json，单 seed 非结论）。
3. **心理均衡**（前景理论效用）：阶段 4 扫 λ×τ 网格，输出 (λ, τ) 相图。这张图是项目「现实均衡」的第一版答案。

## 当前进度

| 阶段 | 交付物 | 状态 |
| :--- | :--- | :--- |
| 0 | Question.md | ✅ 完成 |
| 1 | env.py + test_env.py | ✅ 完成（8/8 测试通过） |
| 2 | solver.py | ✅ 完成（欠账：AI 写的，待凭记忆重写） |
| 3 | utility.py + ppo.py + train.py | 🟡 初版完成（AI 写，待你重写） |
| 4 | sweep.py + (λ, τ) 相图 | 🟡 初版完成（AI 写，待你重写） |
| 5 | 人类对局数据 | ⬜ 待做 |
| 6 | results.md | ⬜ 待做 |

## 文件地图

| 文件 | 作用 |
| :--- | :--- |
| direction.md | 分阶段计划与实验纪律 |
| Question.md | 阶段 0 研究问题（三个可证伪假设）+ 推导附录 |
| env.py | 最小环境：5 维状态，客观奖励 +1/-5/0，非法动作 fail loud |
| test_env.py | 环境规则测试（8 项） |
| solver.py | 阶段 2 精确解：递归价值求解，验证门内建于断言 |
| utility.py | 效用层：identity / prospect（λ、α、β、γ）/ tilt |
| ppo.py | PPO 核心：actor-critic 64 维、GAE、clip、熵正则，双 agent |
| train.py | 训练入口：identity 冒烟/短跑，双日志，验证门输出 |
| sweep.py | 阶段 4：λ×τ 网格扫描 + 预测核验 + 相图 |
| debug/ | 修复记录与调试日志（2026-08-11 起） |
| data/ | 实验结果：stage3_*.json、sweep/（grid.json、predictions.json、phase_diagram.png） |
| readme.md | 旧实现基线（历史记录，不是真相） |

## 运行

```bash
conda activate torch_env
python -m pytest test_env.py -q        # 阶段 1 验证门
python solver.py                       # 阶段 2 验证门
python train.py --quick                # 阶段 3 冒烟
python train.py --steps 200000         # 阶段 3 短跑
python sweep.py --demo                 # 阶段 4 冒烟（3 次运行）
python sweep.py --steps 40000          # 阶段 4 正式网格（60 次运行，CPU 约 25 分钟）
```

## 实验纪律

1. 预测写在实验之前。实验后改预测 = 讲故事，不是做研究。
2. 单 seed 不说收敛，单次运行不写结论。
3. 优势归一化默认关，做成 flag——它是旧结果的头号嫌疑。
4. 心理写在效用层，不污染环境。
5. AI 写过的代码看完要凭记忆重写，跑同一套验证门。

## 诚实边界

- 阶段 3、4 的初版都是 AI 写的——是练习参考答案，不是结论。
- 冒烟数据只验证管道，不代表任何行为规律。
- 没有人类数据之前，「现实均衡」只能是假设（阶段 5 待做）。
- readme.md 的旧数字是起点，不是真相。

## 学习资料

完整列表在 Question.md §4：博弈论（Game Theory 101 / Osborne / von Neumann minimax）、强化学习（Spinning Up / Sutton & Barto / CleanRL）、前景理论（Kahneman & Tversky 1979 / Tversky & Kahneman 1992）、自博弈（MARL book / AlphaZero）。
