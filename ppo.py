#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《赌博默示录》E卡博弈 —— 原版规则 + PPO + 对手建模（轮次条件概率）
- 状态维度 17，包含对手在每一轮的历史出牌概率
- 奖励（皇帝视角）：皇帝胜 +1，皇帝负 -5，平局 0（奴隶视角为对称相反数）
- 平民牌有限（4张）
"""

import argparse
import dataclasses
import math
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("[警告] swanlab 未安装，跳过追踪。")

from dataclasses import dataclass

# ============================== 配置 ==============================
# frozen=True：实例创建后不可修改（替代只读常量类），可安全 hash 并被
# 保存进检查点（dataclasses.asdict）作为训练配置存档。
@dataclass(frozen=True)
class Config:
    """E卡博弈 PPO 训练的全局配置（唯一配置入口，命令行参数可覆盖部分字段）。

    字段按用途分为五组：
    [训练规模]  episodes / steps_per_update / batch_size / epochs_per_update
    [优化器]    lr / lr_schedule / lr_schedule_type / lr_end /
                gamma / gae_lambda / clip_epsilon /
                entropy_coef / entropy_schedule / entropy_coef_end
    [网络]      state_dim / action_dim / hidden_dim / device
    [环境与奖励] max_rounds / reward_emperor_win / reward_emperor_loss
    [持久化]    checkpoint_interval / checkpoint_dir / output_dir /
                swanlab_mode / export_charts / chart_dir
    """
    episodes: int = 6000000                    # 训练回合数（可适当减少）
    state_dim: int = 17                        # 7 + 5 + 5（基础特征 + 双方各5维轮次概率）
    action_dim: int = 2                        # 0=出平民(C)，1=出王牌(A)
    # ---- 优化器与退火 ----
    lr: float = 1e-4
    lr_schedule: bool = True                  # 学习率退火开关：lr → lr_end（按训练进度）
    lr_schedule_type: str = "cosine"          # 退火曲线：linear 线性 / cosine 余弦（末期斜率→0，收敛更稳）
    lr_end: float = 1e-5                      # 退火终值（lr_schedule=True 时生效；0 会在末期锁死策略）
    gamma: float = 0.99                        # 折扣因子：衡量未来奖励的价值（越小越短视）
    gae_lambda: float = 0.95                   # GAE 折衷系数：λ=0 等价 TD(0)，λ=1 等价 MC，中间值平衡方差与偏差
    clip_epsilon: float = 0.2                  # PPO 裁剪阈值：限制新旧策略比值，防止单次更新过大
    epochs_per_update: int = 4                # 每次采集后利用同一批数据更新的轮数（自博弈中过高的 epochs 会过拟合最近批次导致策略漂移）
    # ---- 熵正则（探索/利用平衡） ----
    entropy_coef: float = 0.05                # 熵正则权重（0.01 过弱，探索不足易陷入平台期）
    entropy_schedule: bool = True             # 熵退火开关：前期保探索，后期保收敛
    entropy_coef_end: float = 0.005           # 熵退火终值（0.01 → 0.005：进一步抑制末期随机漂移）
    # ---- 训练节奏 ----
    batch_size: int = 32                       # 小批量大小（每批做一次梯度下降）
    max_rounds: int = 5                        # 每局最大轮数（第5轮强制双方出王牌）
    seed: int = 42                             # 随机种子（固定后可复现训练）
    device: str = "auto"                      # auto=实测对比 CPU/CUDA 前向耗时后选择（小网络 CUDA 通常更慢）
    hidden_dim: int = 64                       # 共享特征提取器的隐藏层维度
    steps_per_update: int = 4096               # 每采集多少步（回合步数累计）执行一次双方策略更新
    # ---- 环境奖励（皇帝视角） ----
    reward_emperor_win: float = 1.0            # 皇帝胜 +1
    # 皇帝视角的"奴隶胜"奖励：奴隶获胜时皇帝得到 -5（同一结果在奴隶视角为 +5）
    reward_emperor_loss: float = -5.0
    # ---- 持久化 ----
    checkpoint_interval: int = 5000            # 每 N 回合保存一次检查点（0 = 关闭）
    checkpoint_dir: str = "./data/checkpoints" # 检查点目录（--resume 时从此加载）
    output_dir: str = "./data"                 # 结果图输出目录（ppo_opponent_modeling.png）
    swanlab_mode: str = "local"               # SwanLab 运行模式：local 仅本地记录（离线，无云端重试风暴）/
                                              # online 同步云端 / offline 本地暂存稍后同步 / disabled 不记录
    export_charts: bool = True                # 训练结束后自动导出训练曲线（PNG+CSV）
    chart_dir: str = "./swanlog/charts"       # 曲线导出目录（与 swanlab 本地日志同根）


# ============================== 全局统计器（对手建模） ==============================
class RoundStats:
    """逐轮出王牌概率（hazard 率）的全局统计器，是对手建模机制的实现核心。

    概念：hazard 率 = 在"到达第 r 轮且此前双方都未出王牌"的条件下，恰在第 r 轮
    出王牌的（条件）概率。这一统计量近似"对手在给定状态下出王牌的风险"，
    以 5 维向量注入状态空间（见 SequentialECardEnv._get_state 维度 7~16），
    使双方策略能感知对手的轮次出牌倾向。

    数据结构（各为长度 max_rounds=5 的数组）：
      totals[r] = 到达第 r 轮且此前未出王牌的局数（含删失：出平民进入下一轮，
                  或因对方出王牌终结——无论哪种，本局都"活到了第 r 轮"，是可观测的）
      counts[r] = 其中恰在第 r 轮出王牌的局数
    于是 hazard[r] = counts[r] / totals[r]。
    """

    def __init__(self, max_rounds: int):
        # 每方各维护 counts/totals 两组计数，互不干扰
        self.max_rounds = max_rounds
        self.emperor_counts = np.zeros(max_rounds, dtype=np.int64)
        self.emperor_totals = np.zeros(max_rounds, dtype=np.int64)
        self.slave_counts = np.zeros(max_rounds, dtype=np.int64)
        self.slave_totals = np.zeros(max_rounds, dtype=np.int64)

    def update(self, emperor_hist, slave_hist):
        """按逐轮观测更新 hazard 率（对手建模核心）。

        参数:
            emperor_hist / slave_hist: 本局双方的实际出牌序列，元素为
                0（出平民）或 1（出王牌），长度 ≤ max_rounds。

        语义说明（2026-08-08 修正版）：
          totals[r] = 到达第 r 轮且此前未出王牌的局数（出平民或对方终结均计入，含删失）；
          counts[r] = 其中恰在第 r 轮出王牌的局数。
        修正前以 pos（王牌出现位置）推导：只有出王牌一侧有观测，未出王牌方
        （如 C-A 局中的皇帝、A-C 局中的奴隶）的信息全部丢失，样本仅剩 2~12%
        且系统性偏向"双方僵持同轮"场景。现改为按动作历史逐轮累计，数据利用率近 100%。
        注意：一方出王牌即终局，该方后续轮次不再有观测，故 break 提前结束。
        """
        # 皇帝侧：逐轮扫描出牌历史，第 r 轮必有一次观测（能走到这轮 = 之前没出王牌）
        for r, a in enumerate(emperor_hist):
            self.emperor_totals[r] += 1
            if a == 1:
                self.emperor_counts[r] += 1
                break  # 出王牌即终局，该方不再有后续轮次观测
        # 奴隶侧：同上，双方各自独立统计
        for r, a in enumerate(slave_hist):
            self.slave_totals[r] += 1
            if a == 1:
                self.slave_counts[r] += 1
                break

    def get_prob(self, counts, totals):
        """安全除法：totals=0（该轮无观测）时概率记为 0，避免除零与 NaN。"""
        with np.errstate(divide='ignore', invalid='ignore'):
            prob = np.divide(counts, totals, out=np.zeros_like(counts, dtype=np.float32), where=totals!=0)
        return prob

    def get_emperor_probs(self):
        """返回皇帝在 5 个轮次的 hazard 率向量（float32，注入状态用）。"""
        return self.get_prob(self.emperor_counts, self.emperor_totals)

    def get_slave_probs(self):
        """返回奴隶在 5 个轮次的 hazard 率向量（float32，注入状态用）。"""
        return self.get_prob(self.slave_counts, self.slave_totals)

    def state_dict(self):
        """导出统计状态（保存进检查点，--resume 时恢复对手建模历史）。"""
        return {
            "emperor_counts": self.emperor_counts,
            "emperor_totals": self.emperor_totals,
            "slave_counts": self.slave_counts,
            "slave_totals": self.slave_totals,
        }

    def load_state_dict(self, state):
        """从检查点恢复统计状态（与 state_dict 一一对应）。"""
        self.emperor_counts = state["emperor_counts"]
        self.emperor_totals = state["emperor_totals"]
        self.slave_counts = state["slave_counts"]
        self.slave_totals = state["slave_totals"]


# ============================== 原版环境（修改 _get_state） ==============================
class SequentialECardEnv:
    """E卡博弈环境（原版规则 + 对手建模特征注入）。

    规则要点：
      - 双方各有 1 张王牌 + 4 张平民牌，每轮同时亮牌；
      - 王牌相遇（A-A）→ 奴隶胜；任何一方王牌对平民 → 皇帝胜；平民对平民 → 平局继续；
      - 第 5 轮（max_rounds）强制双方出王牌，A-A → 奴隶胜；
      - 出过的牌不可再用：出王牌后该方后续轮次只能出平民，平民耗尽后只能出王牌。

    动作编码（与 get_legal_actions 一致）：0 = 出平民(C)，1 = 出王牌(A)。
    奖励编码（皇帝视角，奴隶视角由训练循环取相反数）：见 step 返回值说明。
    """

    def __init__(self, max_rounds: int = 5, reward_emperor_win: float = 1.0, reward_emperor_loss: float = -5.0):
        self.max_rounds = max_rounds
        self.reward_emperor_win = reward_emperor_win
        self.reward_emperor_loss = reward_emperor_loss
        # ---- 对局内部状态 ----
        self.current_round = 0           # 当前轮次（0 基；到达 max_rounds 时强制收尾）
        self.emperor_played = False      # 皇帝是否已出过王牌（出过则只能出平民）
        self.slave_played = False        # 奴隶是否已出过王牌
        self.emperor_civilian_left = 4   # 皇帝剩余平民牌数（耗尽后强制出王牌）
        self.slave_civilian_left = 4     # 奴隶剩余平民牌数
        self.final_emperor_pos = -1      # 皇帝王牌出现轮次（-1 = 未出过王牌）
        self.final_slave_pos = -1        # 奴隶王牌出现轮次
        self.winner = None               # 本局胜者：'emperor' / 'slave' / None（未分胜负）
        self.is_initial = True           # 是否为初始状态（首轮，用于记录初始策略）
        self.emperor_history = []        # 皇帝本局出牌序列（0/1，供 RoundStats 更新）
        self.slave_history = []          # 奴隶本局出牌序列
        # 对手建模：外部注入的轮次概率
        # 初始设为 0.5（无历史时的中性先验），训练循环每局结束后用
        # RoundStats 的全局统计更新（set_round_probs）
        self.emperor_round_probs = np.full(max_rounds, 0.5, dtype=np.float32)
        self.slave_round_probs = np.full(max_rounds, 0.5, dtype=np.float32)

    def set_round_probs(self, emperor_probs, slave_probs):
        """注入双方各轮次 hazard 率（由 RoundStats 计算，随训练动态演化）。

        参数:
            emperor_probs / slave_probs: 长度 max_rounds 的概率向量（float 数组），
                会转成 float32 供状态拼接。
        """
        self.emperor_round_probs = emperor_probs.astype(np.float32)
        self.slave_round_probs = slave_probs.astype(np.float32)

    def reset(self) -> np.ndarray:
        """重置一局，返回初始状态向量（17 维，见 _get_state）。"""
        self.current_round = 0
        self.emperor_played = False
        self.slave_played = False
        self.emperor_civilian_left = 4
        self.slave_civilian_left = 4
        self.final_emperor_pos = -1
        self.final_slave_pos = -1
        self.winner = None
        self.is_initial = True
        self.emperor_history = []
        self.slave_history = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """构造 17 维状态向量（状态空间定义，供策略网络输入）。

        维度布局（0~6 为对局基础特征，7~16 为对手建模特征）：
          [0]  current_round / max_rounds     —— 当前轮次归一化（进度信号）
          [1]  皇帝是否已出王牌 (0/1)
          [2]  奴隶是否已出王牌 (0/1)
          [3]  奴隶历史出王牌频率（本局内均值；无历史时取 0.5 中性先验）
          [4]  皇帝历史出王牌频率（同上）
          [5]  皇帝剩余平民牌数 / 4（0~1，耗尽=0 → 只能出王牌）
          [6]  奴隶剩余平民牌数 / 4
          [7:12]  皇帝在 5 个轮次的 hazard 率（RoundStats 全局统计，对手建模）
          [12:17] 奴隶在 5 个轮次的 hazard 率（同上）
        """
        # 本局内已出王牌频率（无历史时 0.5），反映当前局双方的真实出牌节奏
        slave_ace_freq = np.mean(self.slave_history) if self.slave_history else 0.5
        emperor_ace_freq = np.mean(self.emperor_history) if self.emperor_history else 0.5
        base = np.array([
            self.current_round / self.max_rounds,
            float(self.emperor_played),
            float(self.slave_played),
            slave_ace_freq,
            emperor_ace_freq,
            self.emperor_civilian_left / 4.0,
            self.slave_civilian_left / 4.0
        ], dtype=np.float32)
        # 拼接轮次概率（对手建模）：全局 hazard 率向量，使策略能感知对手的出牌倾向
        return np.concatenate([base, self.emperor_round_probs, self.slave_round_probs])

    def get_legal_actions(self, player: str) -> list:
        """返回指定玩家的合法动作列表（0=出平民，1=出王牌）。

        规则约束：
          - 已出过王牌 → 只能出平民 [0]；
          - 平民牌耗尽 → 只能出王牌 [1]；
          - 否则两种动作都合法。
        """
        if player == 'emperor':
            if self.emperor_played:
                return [0]
            if self.emperor_civilian_left == 0:
                return [1]
            return [0, 1]
        else:
            if self.slave_played:
                return [0]
            if self.slave_civilian_left == 0:
                return [1]
            return [0, 1]

    def step(self, emperor_action: int, slave_action: int):
        """执行双方各一个动作，推进一局。

        参数:
            emperor_action / slave_action: 0=出平民，1=出王牌（可能不在合法集，
                将被替换为合法动作——防御非法采样）。

        返回:
            (next_state, reward, done, info)
            reward: 皇帝视角的即时奖励——
                A-A  → reward_emperor_loss（-5，奴隶胜）
                E-C / C-E → reward_emperor_win（+1，皇帝胜）
                C-C  → 0（平局，继续下一轮；第 5 轮强制 A-A 后必分胜负）
            done: 本局是否结束（任何王牌相遇，或到达 max_rounds）
            info: 字典，含 result('emperor_win'/'slave_win'/'draw')、winner、
                双方实际动作与王牌出现轮次（供统计联合分布与胜负计数）。
        """
        self.is_initial = False
        # ---- 动作合法性防护：非法动作替换为合法集首个 ----
        legal_e = self.get_legal_actions('emperor')
        if emperor_action not in legal_e:
            emperor_action = legal_e[0]
        legal_s = self.get_legal_actions('slave')
        if slave_action not in legal_s:
            slave_action = legal_s[0]

        # 实际执行动作：已出过王牌的一方本轮只能出平民（强制改写为 0）
        executed_e = 0 if self.emperor_played else emperor_action
        executed_s = 0 if self.slave_played else slave_action

        # ---- 第 5 轮强制规则：到达最后一轮时，未出王牌方强制出王牌（A-A → 奴隶胜） ----
        if self.current_round == self.max_rounds - 1:
            if not self.emperor_played:
                executed_e = 1
            if not self.slave_played:
                executed_s = 1

        # ---- 消耗平民牌并记录出牌历史 ----
        if executed_e == 0:
            self.emperor_civilian_left -= 1
        if executed_s == 0:
            self.slave_civilian_left -= 1

        self.emperor_history.append(executed_e)
        self.slave_history.append(executed_s)

        # ---- 记录王牌首次出现的轮次（供联合分布统计） ----
        if executed_e == 1 and not self.emperor_played:
            self.final_emperor_pos = self.current_round
        if executed_s == 1 and not self.slave_played:
            self.final_slave_pos = self.current_round

        self.emperor_played = self.emperor_played or (executed_e == 1)
        self.slave_played = self.slave_played or (executed_s == 1)

        # ---- 胜负判定与奖励（皇帝视角） ----
        # 收益矩阵：           奴隶出王牌(A)  奴隶出平民(C)
        #           皇帝出王牌(A)    -5(奴胜)      +1(皇帝胜)
        #           皇帝出平民(C)    +1(皇帝胜)     0(平局)
        reward = 0.0
        done = False
        result = 'draw'

        if executed_e == 1 and executed_s == 1:
            reward = self.reward_emperor_loss
            done = True
            result = 'slave_win'
            self.winner = 'slave'
        elif executed_e == 1 and executed_s == 0:
            reward = self.reward_emperor_win
            done = True
            result = 'emperor_win'
            self.winner = 'emperor'
        elif executed_e == 0 and executed_s == 1:
            reward = self.reward_emperor_win
            done = True
            result = 'emperor_win'
            self.winner = 'emperor'
        else:
            reward = 0.0
            done = False
            result = 'draw'

        self.current_round += 1

        # ---- 轮次耗尽兜底：5 轮全平不可能（第 5 轮强制 A-A），此分支仅防御性保留 ----
        if self.current_round >= self.max_rounds and not done:
            done = True
            reward = 0.0
            result = 'draw'

        next_state = self._get_state()
        info = {
            'result': result,
            'winner': self.winner,
            'emperor_action': executed_e,
            'slave_action': executed_s,
            'emperor_pos': self.final_emperor_pos,
            'slave_pos': self.final_slave_pos,
        }
        return next_state, reward, done, info


# ============================== PPO 网络（不变） ==============================
class PPOActorCritic(nn.Module):
    """Actor-Critic 共享特征提取网络。

    结构：输入状态(17) → ReLU(64) → ReLU(64) →
          actor 头(→2 logits，动作概率) 与 critic 头(→1 标量，状态价值 V(s))。
    共享隐藏层让价值估计与策略共享低级特征，参数更少、训练更稳；
    两头的输出独立，actor 用 logits（softmax 后采样），critic 用原始标量。
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # 共享特征提取器：两层全连接 + ReLU
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 策略头：输出每个动作的 logits（未归一化的对数概率）
        self.actor = nn.Linear(hidden_dim, action_dim)
        # 价值头：输出状态价值估计 V(s)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """前向传播。返回 (actor_logits, critic_value)。

        x: 批量状态 (batch, state_dim)。
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action(self, state: torch.Tensor, legal_mask: torch.Tensor = None):
        """从策略中采样一个动作（训练/评估共用）。

        参数:
            state: 单样本状态 (1, state_dim) 或批量 (N, state_dim)。
            legal_mask: 合法动作掩码（同形状，1=合法）；None 表示全部合法。

        返回:
            (action, log_prob, value)：采样动作的标量值、其对数概率、状态价值。

        实现要点：
          - 非法动作 logits 减 1e8 置为 -∞，softmax 后概率≈0，从数学上屏蔽非法动作；
          - logits clamp 到 [-20, 20]：防止极端 logits 导致 softmax 数值溢出/NaN；
          - 用 Categorical 分布采样（随机策略），天然支持混合策略学习。
        """
        logits, value = self.forward(state)
        if legal_mask is not None:
            logits = logits + (1 - legal_mask) * (-1e8)
        logits = torch.clamp(logits, -20, 20)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(-1)


# ============================== PPO 智能体（不变） ==============================
class PPOAgent:
    """单方 PPO 智能体（皇帝/奴隶各一个实例，结构对称、互不共享参数）。

    职责：
      - 与环境交互采样（act / store_transition），缓存一条轨迹；
      - 用 GAE 计算优势并执行 clipped PPO 更新（update）；
      - 维护初始策略统计（argmax 频率，作为可解释的策略摘要指标）；
      - 记录最近一次 update 的损失/KL/防御性跳过计数（供指标上报与诊断）。
    """

    def __init__(self, config: Config, device: torch.device, name: str = "agent"):
        self.config = config
        self.device = device
        self.name = name
        self.net = PPOActorCritic(config.state_dim, config.action_dim, config.hidden_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr)
        self.entropy_coef = config.entropy_coef   # 随训练进度退火（训练循环更新）

        # ---- 轨迹缓冲区（每次 update 后清空） ----
        self.states = []        # 状态序列（每步 np 数组）
        self.actions = []       # 采样动作序列
        self.log_probs = []     # 采样动作的旧策略对数概率（用于 importance ratio）
        self.rewards = []       # 奖励序列
        self.dones = []         # 终止标志序列（GAE 中用于截断跨局回报）
        self.values = []        # critic 估计的状态价值序列
        self.legal_masks = []   # 每步的合法动作掩码（torch tensor）

        # ---- 初始策略统计：首轮 argmax 动作计数 ----
        # 用"贪心动作频率"作为可解释的策略摘要（不受采样噪声影响），
        # 供指标上报（emperor_initial_p1 等）与收敛性诊断使用
        self.initial_action_counts = np.zeros(config.action_dim)
        self.episode_rewards = []
        # 最近一次 update 的统计（供每 100 回合指标采样；update 之间保持不变）
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        self.last_kl = 0.0
        # 静默跳过计数器（最近一次 update 内各防御分支的触发次数）
        self.skip_flat_adv = 0             # 整段轨迹优势平坦/过小被跳过
        self.skip_batch_std = 0            # 批内优势标准差过小被跳过
        self.skip_non_finite_probs = 0     # 概率非有限被跳过
        self.skip_non_finite_loss = 0      # 损失非有限被跳过

    def reset_trajectory(self):
        """清空轨迹缓冲区（update 结束或防御性跳过时调用）。"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.legal_masks = []

    def store_transition(self, state, action, log_prob, reward, done, value, legal_mask):
        """缓存一步经验（在训练循环中每步调用）。"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.legal_masks.append(legal_mask)

    def act(self, state: np.ndarray, legal_actions: list = None):
        """根据当前策略采样一个动作（数据采集阶段，计算梯度会通过缓存 log_prob 保留）。

        参数:
            state: 单样本状态向量 (state_dim,)。
            legal_actions: 合法动作列表（如 [0,1]）；None 表示全部合法。

        返回:
            (action, log_prob, value)：动作标量、对数概率、状态价值估计（均转为 Python 标量）。
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # 构造合法动作掩码（1=合法）；合法集随对局状态变化（出过王牌/平民耗尽）
        if legal_actions is not None:
            mask = torch.zeros(self.config.action_dim).to(self.device)
            mask[legal_actions] = 1.0
        else:
            mask = None
        logits, value = self.net(state_t)
        if mask is not None:
            logits = logits + (1 - mask) * (-1e8)
        logits = torch.clamp(logits, -20, 20)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.squeeze(-1).item()

    def record_initial_action(self, state: np.ndarray, is_initial: bool):
        """在每局初始状态记录贪心（argmax）动作，累计成初始策略统计。

        为什么用 argmax 而非采样：采样含探索噪声，argmax 频率更稳定地反映
        策略的"确定倾向"，作为收敛性诊断中的可解释指标（如首轮王牌率 p1）。
        """
        if is_initial:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                logits, _ = self.net(state_t)
                logits = torch.clamp(logits, -20, 20)
                probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
                action = int(np.argmax(probs))
            self.initial_action_counts[action] += 1

    def get_initial_strategy(self):
        """返回初始状态贪心动作频率（[平民率, 王牌率]）；无样本时均匀分布兜底。"""
        total = self.initial_action_counts.sum()
        if total == 0:
            return np.ones(self.config.action_dim) / self.config.action_dim
        return self.initial_action_counts / total

    def update(self):
        """PPO 核心更新：用缓存的轨迹计算 GAE 优势，执行 clipped PPO 若干轮。

        算法流程（详见各段注释）：
          1) 数据上设备并转张量；轨迹未终结时用 critic 引导 bootstrap 末值；
          2) GAE（广义优势估计）递推计算优势，clip 到 [-100,100] 防极端值；
          3) 防御分支：优势过平坦（标准差 < 1e-6）说明本段轨迹无学习信号，跳过；
          4) 多轮小批量更新：importance ratio → clipped 目标 → 熵正则 → 梯度裁剪。
        """
        if len(self.states) == 0:
            return

        # 每次 update 开始时重置跳过计数器（随指标上报的窗口周期）
        self.skip_flat_adv = 0
        self.skip_batch_std = 0
        self.skip_non_finite_probs = 0
        self.skip_non_finite_loss = 0

        # ---- 数据上设备：全部转成批量张量 ----
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        legal_masks = [m.to(self.device) if m is not None else torch.ones(self.config.action_dim).to(self.device)
                    for m in self.legal_masks]
        legal_masks = torch.stack(legal_masks)

        # ---- GAE 优势计算（torch.no_grad：仅用旧网络估计，不反向传播） ----
        with torch.no_grad():
            # 轨迹末值 bootstrap：若最后一步未终结，用 critic 估计"未来收益的延续值"；
            # 若已终结则末值为 0（终结后不再有收益）
            if not self.dones[-1]:
                last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
                _, last_value = self.net(last_state)
                last_value = last_value.squeeze(-1).item()
            else:
                last_value = 0.0

            # GAE 递推（自后向前）：
            #   delta_t = r_t + γ·V(s_{t+1})·(1-done) − V(s_t)   （TD 残差）
            #   A_t = delta_t + γ·λ·(1-done)·A_{t+1}             （残差向后传播）
            # γ=0.99 折衷近期/远期收益；λ=0.95 平衡方差（MC 风格）与偏差（TD 风格）
            advantages = []
            gae = 0.0
            for t in reversed(range(len(self.rewards))):
                if t == len(self.rewards) - 1:
                    next_value = last_value
                else:
                    next_value = old_values[t + 1].item()
                delta = rewards[t].item() + self.config.gamma * next_value * (1 - dones[t].item()) - old_values[t].item()
                delta = np.clip(delta, -100.0, 100.0)
                gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t].item()) * gae
                gae = np.clip(gae, -100.0, 100.0)
                advantages.append(gae)

            # 倒序计算后需反转回时间正序；returns = 优势 + 旧价值（critic 回归目标）
            advantages = torch.FloatTensor(np.array(advantages[::-1])).to(self.device)
            returns = advantages + old_values

            # 数值防御：清理任何 NaN/Inf（理论上不应出现，防崩溃保训练）
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1.0, neginf=-1.0)
            returns = torch.nan_to_num(returns, nan=0.0, posinf=1.0, neginf=-1.0)

            # 防御分支 1：整段优势平坦 → 无学习信号（如长串平局），跳过本段
            if advantages.numel() <= 1 or advantages.std(unbiased=False).item() < 1e-6:
                self.skip_flat_adv += 1
                self.reset_trajectory()
                return

        # ---- PPO 更新（含统计累计） ----
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_kl = 0.0
        num_batches = 0

        dataset_size = len(self.states)
        for _ in range(self.config.epochs_per_update):
            # 每轮重新洗牌 → 小批量随机梯度下降（同一批数据用 epochs 轮）
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                if len(batch_indices) == 0:
                    continue

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_legal_masks = legal_masks[batch_indices]

                batch_advantages = torch.nan_to_num(batch_advantages, nan=0.0, posinf=1.0, neginf=-1.0)
                batch_returns = torch.nan_to_num(batch_returns, nan=0.0, posinf=1.0, neginf=-1.0)

                # 防御分支 2：批内优势标准差过小 → 本批无有效梯度信号，跳过
                if batch_advantages.numel() <= 1 or batch_advantages.std(unbiased=False).item() < 1e-6:
                    self.skip_batch_std += 1
                    continue

                # 前向：新策略 logits + 价值（对批次前向即反向传播的起点）
                logits, values = self.net(batch_states)
                logits = logits + (1 - batch_legal_masks) * (-1e8)
                logits = torch.clamp(logits, -20, 20)
                probs = F.softmax(logits, dim=-1)

                # 防御分支 3：softmax 后概率非有限 → 数值异常，跳过本批
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    self.skip_non_finite_probs += 1
                    continue

                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO 重要性采样比值：π_new(a|s) / π_old(a|s)
                # clamp 到 [0, 10] 防极端比值（数值稳定性）
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                ratio = torch.clamp(ratio, 0, 10)

                # 优势归一化：减均值除标准差，稳定梯度尺度（-5 大惩罚在此被稀释，
                # 这是"皇帝学出赢得多但期望负"现象的关键成因之一，见 README 分析）
                adv = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std(unbiased=False) + 1e-8)
                adv = torch.nan_to_num(adv, nan=0.0, posinf=1.0, neginf=-1.0)
                adv = torch.clamp(adv, -10.0, 10.0)

                # ---- PPO clipped 目标函数 ----
                # surr1 = ratio × A（无约束的策略梯度项）
                # surr2 = clip(ratio, 1±ε) × A（裁剪到 1±0.2，限制单步更新幅度）
                # actor_loss = −min(surr1, surr2)（保守取小：超限更新被截断）
                # critic_loss = MSE(V(s), returns)（价值回归）
                # 总损失 = actor − 0.5·critic − entropy_coef·H（熵正则鼓励探索）
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                # 防御分支 4：总损失非有限 → 跳过本批
                if torch.isnan(loss) or torch.isinf(loss):
                    self.skip_non_finite_loss += 1
                    continue

                # 累计统计（每批均值，最后除以批数得到本次 update 的平均值）
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_kl += (new_log_probs - batch_old_log_probs).mean().item()
                num_batches += 1

                # ---- 梯度更新：反向传播 + 梯度裁剪 + Adam 步进 ----
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()

        # ---- 记录本次 update 的平均指标（供指标上报） ----
        if num_batches > 0:
            self.last_actor_loss = total_actor_loss / num_batches
            self.last_critic_loss = total_critic_loss / num_batches
            self.last_kl = total_kl / num_batches
        else:
            self.last_actor_loss = 0.0
            self.last_critic_loss = 0.0
            self.last_kl = 0.0

        self.reset_trajectory()

# ============================== 检查点（保存 / 恢复） ==============================
CHECKPOINT_FILENAME = "ppo_checkpoint.pt"
CHECKPOINT_FORMAT_VERSION = 1


def checkpoint_path(config: Config) -> str:
    """返回检查点文件路径（checkpoint_dir/ppo_checkpoint.pt）。"""
    return os.path.join(config.checkpoint_dir, CHECKPOINT_FILENAME)


def save_checkpoint(config, episode, total_steps, emperor_agent, slave_agent,
                    round_stats, joint_counts, emperor_wins, slave_wins, episode_rounds,
                    joint_missing=0):
    """保存训练状态（网络、优化器、统计、RNG），仅影响持久化，不改变算法行为。

    参数:
        config: 当前训练配置（dataclasses.asdict 序列化为纯字典存档）。
        episode: 已完成回合数（续训时从此值继续，而不会从 0 重来）。
        total_steps: 当前 update 周期内累计步数（续训后保留，避免丢失半个周期）。
        emperor_agent / slave_agent: 双方智能体（网络权重 + 优化器状态 + 初始策略计数）。
        round_stats: 对手建模 hazard 率统计（续训后无需重学对手模型）。
        joint_counts: 双方王牌出现轮次的联合分布计数（结果统计）。
        emperor_wins / slave_wins / episode_rounds: 胜负计数与回合长度序列。
        joint_missing: 未记录到联合分布的对局数（防御性统计）。
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    # 三套 RNG 状态全部入档：恢复后随机序列与未中断运行完全一致
    rng = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": dataclasses.asdict(config),
        "episode": episode,                      # 已完成回合数，续训从此值继续
        "total_steps": total_steps,
        "emperor_net": emperor_agent.net.state_dict(),
        "slave_net": slave_agent.net.state_dict(),
        "emperor_optimizer": emperor_agent.optimizer.state_dict(),
        "slave_optimizer": slave_agent.optimizer.state_dict(),
        "emperor_initial_counts": emperor_agent.initial_action_counts,
        "slave_initial_counts": slave_agent.initial_action_counts,
        "round_stats": round_stats.state_dict(),
        "joint_counts": joint_counts,
        "joint_missing": joint_missing,
        "emperor_wins": emperor_wins,
        "slave_wins": slave_wins,
        "episode_rounds": episode_rounds,
        "rng": rng,
    }
    path = checkpoint_path(config)
    torch.save(payload, path)
    print(f"[检查点] 已保存 回合 {episode} -> {path}")


def load_checkpoint(config: Config):
    """加载最新检查点；不存在时返回 None。本地自产文件，允许非张量负载。"""
    path = checkpoint_path(config)
    if not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容不支持 weights_only 的旧版 torch
        payload = torch.load(path, map_location="cpu")
    print(f"[恢复] 从检查点加载: {path}（回合 {payload['episode']}）")
    return payload


def apply_checkpoint(config, resumed, emperor_agent, slave_agent, round_stats, env):
    """将检查点负载写回训练状态；返回 (episode, total_steps, 各统计量)。

    参数:
        resumed: load_checkpoint 返回的检查点字典。
        emperor_agent / slave_agent: 需要被恢复权重的新建智能体实例。
        round_stats: 需要被恢复统计的对手建模统计器。
        env: 环境实例（恢复后立即注入对手建模概率，保证后续状态一致）。

    返回:
        (episode, total_steps, joint_counts, joint_missing, emperor_wins, slave_wins, episode_rounds)
    """
    # 网络与优化器权重恢复
    emperor_agent.net.load_state_dict(resumed["emperor_net"])
    slave_agent.net.load_state_dict(resumed["slave_net"])
    emperor_agent.optimizer.load_state_dict(resumed["emperor_optimizer"])
    slave_agent.optimizer.load_state_dict(resumed["slave_optimizer"])
    emperor_agent.initial_action_counts = np.asarray(resumed["emperor_initial_counts"])
    slave_agent.initial_action_counts = np.asarray(resumed["slave_initial_counts"])
    round_stats.load_state_dict(resumed["round_stats"])

    # RNG 恢复：保证"未中断运行"与"中断后 --resume"的随机序列一致（可复现性）
    rng = resumed.get("rng", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "python" in rng:
        random.setstate(rng["python"])
    if "torch_cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])

    # 恢复后的对手建模概率重新注入环境
    env.set_round_probs(round_stats.get_emperor_probs(), round_stats.get_slave_probs())

    return (
        int(resumed["episode"]),
        int(resumed.get("total_steps", 0)),
        np.asarray(resumed["joint_counts"]),
        int(resumed.get("joint_missing", 0)),
        int(resumed["emperor_wins"]),
        int(resumed["slave_wins"]),
        list(resumed.get("episode_rounds", [])),
    )


# ============================== 训练主函数 ==============================
def _schedule(schedule_type: str, progress: float, start: float, end: float) -> float:
    """退火调度：progress ∈ [0,1] → [start, end]。

    参数:
        schedule_type: "cosine" 或 "linear"。
        progress: 训练进度（0=开始，1=结束）。
        start / end: 调度起点 / 终点值。

    linear  线性：末期斜率恒定，策略仍在匀速移动；
    cosine  余弦：末期斜率→0，策略趋于冻结，收敛更稳定（自博弈末期防漂移）。
    """
    if schedule_type == "cosine":
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
    return start + (end - start) * progress


def _pick_device(config: Config) -> torch.device:
    """设备选择：auto 时对 CPU/CUDA 各测 200 次前向耗时，选明显更快者。

    小网络下 CUDA 的每步 CPU→GPU 同步开销可能超过 GPU 计算收益，
    实测本任务 CUDA 约 100 回合/秒，CPU 约 300-500 回合/秒（快 3-7 倍）。

    参数:
        config: 训练配置（device 字段：auto / cpu / cuda）。

    返回:
        选定的 torch.device（auto 模式下实测后确定）。
    """
    if config.device != "auto":
        return torch.device(config.device)
    if not torch.cuda.is_available():
        print("[信息] CUDA 不可用，使用 CPU")
        return torch.device("cpu")
    import time
    probe = PPOActorCritic(config.state_dim, config.action_dim, config.hidden_dim)
    state = torch.randn(1, config.state_dim)

    def _bench(device):
        probe.to(device)
        s = state.to(device)
        for _ in range(5):  # warmup
            probe(s)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200):
            probe(s)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / 200

    t_cpu = _bench(torch.device("cpu"))
    t_cuda = _bench(torch.device("cuda"))
    chosen = "cuda" if t_cuda < t_cpu * 0.9 else "cpu"  # 需明显更快才用 CUDA
    print(f"[信息] 前向测速: CPU {t_cpu*1e3:.2f}ms/步, CUDA {t_cuda*1e3:.2f}ms/步 -> 选择 {chosen}")
    return torch.device(chosen)


def train_ppo_original(config: Config, tracking: bool = True, resume: bool = False,
                       loss_log=None, skip_log=None, final_callback=None):
    """主训练循环：自博弈 PPO（皇帝 vs 奴隶），含退火、检查点、指标采集与收敛性诊断。

    自博弈结构：同一局中双方各由一个独立 PPOAgent 决策，奖励互为相反数
    （slave_reward = -reward），互为非平稳对手——这是研究"循环博弈中 PPO
    是否收敛"的核心设定。

    参数:
        config: 训练配置（Config 实例）。
        tracking: 是否启用 SwanLab 追踪（--no-tracking 可关闭）。
        resume: 是否从最新检查点恢复（--resume）。
        loss_log: 若提供（list），每个指标采样点追加皇帝 actor loss（--quick 用）。
        skip_log: 若提供（dict），累计双方 update 防御性跳过总次数（--quick 用）。
        final_callback: 训练结束回调（若提供，传 emperor_agent, slave_agent）。

    返回:
        (final_emperor_strat, final_slave_strat, joint_prob,
         emperor_rate, slave_rate, overall_avg_rounds)
    """
    # ---- 初始化：设备选择 + 三套 RNG 播种（固定种子可复现） ----
    device = _pick_device(config)
    print(f"[信息] 使用设备: {device}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---- SwanLab 初始化（可选；失败仅警告，不影响训练） ----
    run = None
    if SWANLAB_AVAILABLE and tracking and config.swanlab_mode != "disabled":
        try:
            run = swanlab.init(
                project="ECard-PPO-Original",
                mode=config.swanlab_mode,  # 默认 local：仅本地记录，离线运行不产生云端连接重试
                config={  # 记录关键超参数到实验面板（便于复盘对比不同 run）
                    "episodes": config.episodes,
                    "lr": config.lr,
                    "lr_schedule": config.lr_schedule,
                    "lr_schedule_type": config.lr_schedule_type,
                    "lr_end": config.lr_end,
                    "gamma": config.gamma,
                    "gae_lambda": config.gae_lambda,
                    "clip_epsilon": config.clip_epsilon,
                    "epochs_per_update": config.epochs_per_update,
                    "entropy_coef": config.entropy_coef,
                    "entropy_schedule": config.entropy_schedule,
                    "entropy_coef_end": config.entropy_coef_end,
                    "device": config.device,
                    "swanlab_mode": config.swanlab_mode,
                    "steps_per_update": config.steps_per_update,
                    "reward_emperor_win": config.reward_emperor_win,
                    "reward_emperor_loss": config.reward_emperor_loss,
                    "state_dim": config.state_dim,
                    "max_rounds": config.max_rounds,
                    "平民牌限制": "每方4张平民牌，用完强制出王牌",
                    "对手建模": "轮次条件概率（双方各5维）"
                },
                experiment_name="PPO_Opponent_Modeling"
            )
        except Exception as e:
            print(f"[警告] SwanLab 初始化失败: {e}")

    # ---- 环境与双方智能体（结构对称、参数独立） ----
    env = SequentialECardEnv(
        max_rounds=config.max_rounds,
        reward_emperor_win=config.reward_emperor_win,
        reward_emperor_loss=config.reward_emperor_loss,
    )
    emperor_agent = PPOAgent(config, device, "emperor")
    slave_agent = PPOAgent(config, device, "slave")

    # 对手建模统计器：全局累计双方 hazard 率
    round_stats = RoundStats(config.max_rounds)

    # ---- 全局统计（保存进检查点，恢复后继续累计） ----
    joint_counts = np.zeros((config.max_rounds, config.max_rounds), dtype=np.int64)
    # 双方王牌出现轮次的联合分布（5×5），用于最终热力图
    joint_missing = 0
    emperor_wins = 0          # 皇帝胜局数（累计）
    slave_wins = 0            # 奴隶胜局数（= 累积 A-A 局数，注意非对称奖励下的关键指标）
    episode_rounds = []       # 每局结束轮次（avg_rounds 统计源）

    episode = 0
    total_steps = 0           # 当前 update 周期累计步数（达到 steps_per_update 触发更新）
    metrics_history = []  # 每 100 回合累积指标，训练结束自动导出曲线（swanlog）
    # 初始对手建模概率：无历史时取 0.5 中性先验
    init_probs = np.full(config.max_rounds, 0.5, dtype=np.float32)
    env.set_round_probs(init_probs, init_probs)

    # 从检查点恢复（若启用且存在）
    if resume:
        resumed = load_checkpoint(config)
        if resumed is not None:
            (episode, total_steps, joint_counts, joint_missing, emperor_wins, slave_wins,
             episode_rounds) = apply_checkpoint(config, resumed, emperor_agent, slave_agent,
                                                round_stats, env)
            print(f"[恢复] 从回合 {episode} 继续训练（共 {config.episodes} 回合）")

    # ===================== 自博弈主循环（一局一迭代） =====================
    while episode < config.episodes:
        state = env.reset()
        done = False
        ep_reward_e = 0.0     # 本局皇帝累计奖励（皇帝视角）
        ep_reward_s = 0.0     # 本局奴隶累计奖励（= -ep_reward_e）
        step_count = 0        # 本局步数防护（正常最多 max_rounds 步）

        # 每局初始状态记录贪心动作（初始策略统计，收敛性诊断的数据源）
        emperor_agent.record_initial_action(state, env.is_initial)
        slave_agent.record_initial_action(state, env.is_initial)

        # ---- 局内步进循环：双方各决策一次 = 一步 ----
        while not done and step_count < 100:
            legal_e = env.get_legal_actions('emperor')
            legal_s = env.get_legal_actions('slave')

            # 双方基于同一状态各自采样动作（同时决策，互不先见）
            e_action, e_log_prob, e_value = emperor_agent.act(state, legal_e)
            s_action, s_log_prob, s_value = slave_agent.act(state, legal_s)

            next_state, reward, done, info = env.step(e_action, s_action)
            slave_reward = -reward  # 零和博弈：奴隶奖励 = 皇帝奖励的相反数

            # 双方各自缓存轨迹（带合法动作掩码，更新时屏蔽非法动作）
            emperor_agent.store_transition(
                state, e_action, e_log_prob, reward, done, e_value,
                torch.tensor([1 if a in legal_e else 0 for a in range(config.action_dim)])
            )
            slave_agent.store_transition(
                state, s_action, s_log_prob, slave_reward, done, s_value,
                torch.tensor([1 if a in legal_s else 0 for a in range(config.action_dim)])
            )

            ep_reward_e += reward
            ep_reward_s += slave_reward
            state = next_state
            step_count += 1
            total_steps += 1

        # ---- 局末统计 ----
        episode_rounds.append(env.current_round)
        if info['winner'] == 'emperor':
            emperor_wins += 1
        elif info['winner'] == 'slave':
            slave_wins += 1

        # 对手建模：按逐轮动作历史更新两侧 hazard 率；
        # 修正前要求双方同时出王牌才更新，会丢弃 C-A / A-C 局的单侧信息
        round_stats.update(env.emperor_history, env.slave_history)
        # 联合分布统计：双方王牌出现轮次（未出王牌方记 -1，整局全平则缺失）
        if env.final_emperor_pos >= 0 and env.final_slave_pos >= 0:
            joint_counts[env.final_emperor_pos, env.final_slave_pos] += 1
        else:
            joint_missing += 1

        # 更新对手建模概率：最新全局 hazard 率注入环境（下一局生效）
        emperor_probs = round_stats.get_emperor_probs()
        slave_probs = round_stats.get_slave_probs()
        env.set_round_probs(emperor_probs, slave_probs)

        emperor_agent.episode_rewards.append(ep_reward_e)
        slave_agent.episode_rewards.append(ep_reward_s)

        episode += 1

        # ---- 策略更新触发：步数达标时执行退火 + 双方 PPO 更新 ----
        if total_steps >= config.steps_per_update:
            # 学习率与熵系数退火（按训练进度，从检查点恢复后自动延续；
            # 默认余弦曲线，末期变化率→0，抑制自博弈末期策略漂移）
            if config.lr_schedule or config.entropy_schedule:
                progress = min(episode / max(1, config.episodes), 1.0)
                if config.lr_schedule:
                    current_lr = _schedule(config.lr_schedule_type, progress, config.lr, config.lr_end)
                else:
                    current_lr = config.lr
                if config.entropy_schedule:
                    current_entropy = _schedule(config.lr_schedule_type, progress,
                                                config.entropy_coef, config.entropy_coef_end)
                else:
                    current_entropy = config.entropy_coef
                for agent in (emperor_agent, slave_agent):
                    for group in agent.optimizer.param_groups:
                        group["lr"] = current_lr
                    agent.entropy_coef = current_entropy
            # 双方随机顺序更新：固定先皇帝后奴隶会给一方系统性"信息优势"，
            # 随机化保持自博弈对称性（RNG 已入检查点，恢复后可复现）
            for agent in random.sample([emperor_agent, slave_agent], 2):
                agent.update()
            total_steps = 0
            # 防御性跳过计数累计（--quick 验证用）
            if skip_log is not None:
                skip_log["emperor_total"] = skip_log.get("emperor_total", 0) + (
                    emperor_agent.skip_flat_adv + emperor_agent.skip_batch_std
                    + emperor_agent.skip_non_finite_probs + emperor_agent.skip_non_finite_loss)
                skip_log["slave_total"] = skip_log.get("slave_total", 0) + (
                    slave_agent.skip_flat_adv + slave_agent.skip_batch_std
                    + slave_agent.skip_non_finite_probs + slave_agent.skip_non_finite_loss)

        # 周期性保存检查点（含配置与回合标识）
        if config.checkpoint_interval > 0 and episode > 0 and episode % config.checkpoint_interval == 0:
            save_checkpoint(config, episode, total_steps, emperor_agent, slave_agent,
                            round_stats, joint_counts, emperor_wins, slave_wins, episode_rounds,
                            joint_missing)

        # ---- 指标采集（每 100 回合采样一次，供 SwanLab / CSV / 诊断使用） ----
        if episode % 100 == 0:
            e_init = emperor_agent.get_initial_strategy()
            s_init = slave_agent.get_initial_strategy()
            avg_rounds = np.mean(episode_rounds[-100:]) if episode_rounds else 0

            # ---- O‑06: 获取损失和 KL ----
            emperor_actor_loss = emperor_agent.last_actor_loss
            emperor_critic_loss = emperor_agent.last_critic_loss
            emperor_kl = emperor_agent.last_kl
            slave_actor_loss = slave_agent.last_actor_loss
            slave_critic_loss = slave_agent.last_critic_loss
            slave_kl = slave_agent.last_kl

            if loss_log is not None:
                loss_log.append(emperor_agent.last_actor_loss)

            # ---- O‑02: 获取轮次概率（每 1000 回合记录） ----
            round_probs_metrics = {}
            if episode % 1000 == 0:
                emp_probs = round_stats.get_emperor_probs()
                slv_probs = round_stats.get_slave_probs()
                for r in range(config.max_rounds):
                    round_probs_metrics[f"emperor_round_{r+1}_prob"] = emp_probs[r]
                    round_probs_metrics[f"slave_round_{r+1}_prob"] = slv_probs[r]

            metrics = {
                "emperor_win_rate": emperor_wins / episode,
                "slave_win_rate": slave_wins / episode,
                # 皇帝期望收益 = 胜率×(+1) + 负率×(−5)（非对称奖励下的客观均衡指标）
                # 注：奴隶胜局 ≈ 累积 A-A 局，故 E ≈ 皇帝胜率 − 5×(1−皇帝胜率)
                "emperor_expected_reward": (emperor_wins / episode) * config.reward_emperor_win
                                           + (slave_wins / episode) * config.reward_emperor_loss,
                "emperor_initial_p0": e_init[0],
                "emperor_initial_p1": e_init[1],   # 皇帝首轮王牌率（收敛性诊断核心指标）
                "slave_initial_p0": s_init[0],
                "slave_initial_p1": s_init[1],     # 奴隶首轮王牌率
                "avg_rounds": avg_rounds,
                "episode": episode,
                # ---- O‑06: 损失与 KL ----
                "emperor_actor_loss": emperor_actor_loss,
                "emperor_critic_loss": emperor_critic_loss,
                "emperor_kl": emperor_kl,
                "slave_actor_loss": slave_actor_loss,
                "slave_critic_loss": slave_critic_loss,
                "slave_kl": slave_kl,
                # ---- O‑07: 静默跳过计数（最近一次 update 窗口） ----
                "emperor_skip_flat_adv": emperor_agent.skip_flat_adv,
                "emperor_skip_batch_std": emperor_agent.skip_batch_std,
                "emperor_skip_non_finite_probs": emperor_agent.skip_non_finite_probs,
                "emperor_skip_non_finite_loss": emperor_agent.skip_non_finite_loss,
                "slave_skip_flat_adv": slave_agent.skip_flat_adv,
                "slave_skip_batch_std": slave_agent.skip_batch_std,
                "slave_skip_non_finite_probs": slave_agent.skip_non_finite_probs,
                "slave_skip_non_finite_loss": slave_agent.skip_non_finite_loss,
                # ---- O‑02: 轮次概率 ----
                **round_probs_metrics,
            }
            # 离线曲线导出：只保留绘图所需核心字段（丢弃 10 维轮次概率，控制内存）
            metrics_history.append({k: metrics[k] for k in CHART_METRIC_KEYS})
            metrics_history[-1]["episode"] = episode

            if run:
                try:
                    swanlab.log(metrics)
                except Exception:
                    pass

        # ---- 终端进度输出（每 2000 回合） ----
        if episode % 2000 == 0:
            e_init = emperor_agent.get_initial_strategy()
            s_init = slave_agent.get_initial_strategy()
            avg_rounds = np.mean(episode_rounds[-2000:]) if episode_rounds else 0
            skip_e = (emperor_agent.skip_flat_adv + emperor_agent.skip_batch_std
                      + emperor_agent.skip_non_finite_probs + emperor_agent.skip_non_finite_loss)
            skip_s = (slave_agent.skip_flat_adv + slave_agent.skip_batch_std
                      + slave_agent.skip_non_finite_probs + slave_agent.skip_non_finite_loss)
            print(f"回合 {episode:>6}: 皇帝胜率={emperor_wins/episode:.3f}, "
                  f"皇初始出王牌概率={e_init[1]:.3f}, 奴初始出王牌概率={s_init[1]:.3f}, "
                  f"平均回合={avg_rounds:.2f}, 跳过计数(皇/奴)={skip_e}/{skip_s}")

    # ===================== 训练结束：汇总结果 =====================
    final_emperor_strat = emperor_agent.get_initial_strategy()
    final_slave_strat = slave_agent.get_initial_strategy()
    joint_prob = joint_counts / (joint_counts.sum() + 1e-8)  # 归一化为联合分布（热力图数据）
    emperor_rate = emperor_wins / config.episodes
    slave_rate = slave_wins / config.episodes
    overall_avg_rounds = np.mean(episode_rounds)

    print("\n[训练完成]")
    print(f"  皇帝初始策略（平民, 王牌）: {np.round(final_emperor_strat, 3)}")
    print(f"  奴隶初始策略: {np.round(final_slave_strat, 3)}")
    print(f"  皇帝实际胜率: {emperor_rate:.3f}")
    print(f"  奴隶实际胜率: {slave_rate:.3f}")
    print(f"  平均结束回合: {overall_avg_rounds:.2f}")
    print(f"  联合分布有效样本数: {config.episodes - joint_missing} / {config.episodes}")
    # 客观收敛性摘要（与 diagnosis txt 同源，以博弈论均衡为参照，不偏向任何一方）
    # 判定逻辑：取最后 5% 采样点为"末段"，与其前相邻 5% 窗口比较均值漂移；
    # 双方漂移均 ≤ 0.005 判定收敛（阈值与诊断文件一致）
    if len(metrics_history) >= 40:
        eq_p1 = 1 / 7  # 静态混合均衡（无第5轮强制规则近似）下的首轮王牌率
        _p = np.array([m.get("emperor_initial_p1", np.nan) for m in metrics_history], dtype=float)
        _q = np.array([m.get("slave_initial_p1", np.nan) for m in metrics_history], dtype=float)
        _ev = np.array([m.get("emperor_expected_reward", np.nan) for m in metrics_history], dtype=float)
        _w = max(1, len(metrics_history) // 20)  # 末段窗口：最后 5% 采样点
        p_tail, p_prev = float(np.nanmean(_p[-_w:])), float(np.nanmean(_p[-2 * _w:-_w]))
        q_tail, q_prev = float(np.nanmean(_q[-_w:])), float(np.nanmean(_q[-2 * _w:-_w]))
        ev_tail = float(np.nanmean(_ev[-_w:]))
        print(f"  皇帝期望收益: {ev_tail:+.3f}（静态均衡近似 +1/7 ≈ {eq_p1:.3f}）")
        print(f"  首轮王牌率: 皇帝 {p_tail:.3f} / 奴隶 {q_tail:.3f}（均衡 1/7 ≈ {eq_p1:.3f}）")
        print(f"  末段漂移(皇/奴): {p_tail - p_prev:+.4f} / {q_tail - q_prev:+.4f}"
              f"（{'未收敛' if abs(p_tail - p_prev) > 0.005 or abs(q_tail - q_prev) > 0.005 else '已稳定'}，阈值 0.005）")

    if final_callback is not None:
        final_callback(emperor_agent, slave_agent)

    export_charts(metrics_history, config)

    if run:
        try:
            swanlab.finish()
        except Exception:
            pass

    return final_emperor_strat, final_slave_strat, joint_prob, emperor_rate, slave_rate, overall_avg_rounds


# ============================== 训练曲线自动导出（swanlog） ==============================
CHART_METRIC_KEYS = (
    "emperor_win_rate", "slave_win_rate", "emperor_expected_reward", "avg_rounds",
    "emperor_initial_p1", "slave_initial_p1",
    "emperor_actor_loss", "emperor_critic_loss", "slave_actor_loss", "slave_critic_loss",
    "emperor_kl", "slave_kl",
    "emperor_skip_flat_adv", "emperor_skip_batch_std",
    "emperor_skip_non_finite_probs", "emperor_skip_non_finite_loss",
    "slave_skip_flat_adv", "slave_skip_batch_std",
    "slave_skip_non_finite_probs", "slave_skip_non_finite_loss",
)


def _smooth_series(x, y, window):
    """滑动平均（valid 卷积），x/y 同步截断；窗口不足时返回原始序列。"""
    if window <= 1 or len(y) < 2 * window:
        return x, y
    kernel = np.ones(window) / window
    y_s = np.convolve(y, kernel, mode="valid")
    return x[:len(y_s)], y_s


def _write_convergence_diagnosis(path, metrics_history, config: Config):
    """生成客观收敛性诊断文件（txt），写入 path。

    功能：以博弈论均衡为参照，如实记录策略位置与漂移——
    不偏向任何一方：仅报告 期望收益 vs 静态均衡近似、双方王牌率偏差、末段漂移，
    用于判断"是否收敛"而非"谁赢得多"。

    收敛判定逻辑（与终端摘要一致，阈值 0.005）：
      - 取最后 5% 采样点均值（末段）与其前相邻 5% 窗口均值之差为"末段漂移"；
      - |漂移| ≤ 0.005 判定收敛（策略在训练末期保持稳定）；
      - 注意：偏离静态均衡 1/7 本身不构成"不收敛"证据（第 5 轮强制规则
        使深轮次均衡王牌率更高，见文件内的说明段）。

    参数:
        path: 诊断文件输出路径（如 swanlog/charts/diagnosis_*.txt）。
        metrics_history: 每 100 回合的指标记录列表（含 emperor_initial_p1 等）。
        config: 训练配置（保留参数以兼容扩展，当前未直接使用）。
    """
    import time
    # 从指标历史中抽取关键序列（np.nan 兜底缺失字段）
    ep = np.array([m["episode"] for m in metrics_history], dtype=float)
    p = np.array([m.get("emperor_initial_p1", np.nan) for m in metrics_history], dtype=float)
    q = np.array([m.get("slave_initial_p1", np.nan) for m in metrics_history], dtype=float)
    ev = np.array([m.get("emperor_expected_reward", np.nan) for m in metrics_history], dtype=float)
    n = len(ep)
    # 博弈论参考基准：静态混合均衡（无第5轮强制规则近似）下双方首轮王牌率与皇帝期望收益均为 1/7
    eq_p1, eq_ev = 1 / 7, 1 / 7
    lines = []
    lines.append(f"E卡博弈 训练收敛性诊断（{time.strftime('%Y-%m-%d %H:%M:%S')}）")
    lines.append(f"回合数: {int(ep[-1])}，采样点数: {n}")
    lines.append("")
    lines.append("== 参考基准（博弈论） ==")
    lines.append(f"  静态混合均衡（无第5轮强制规则近似）: 双方首轮王牌率 = 1/7 ≈ {eq_p1:.4f}，"
                 f"皇帝期望收益 = +1/7 ≈ {eq_ev:.4f}")
    lines.append(f"  注意: 第5轮强制 A-A → 奴隶胜，使均衡对前4轮近似成立；"
                 f"深轮次（第4轮）均衡王牌率会更高（约 0.5），"
                 f"因此实际训练值偏离 1/7 不一定意味着不收敛。")
    lines.append("")
    if n >= 40:  # 采样点足够（至少 4000 回合）才做统计评估
        w = max(1, n // 20)             # 对比窗口：最后 5% 采样点
        p_tail, p_prev = np.nanmean(p[-w:]), np.nanmean(p[-2 * w:-w])
        q_tail, q_prev = np.nanmean(q[-w:]), np.nanmean(q[-2 * w:-w])
        ev_tail = np.nanmean(ev[-w:])
        lines.append("== 实际训练结果 ==")
        lines.append(f"  皇帝首轮王牌率 p: {p_tail:.4f}（偏差 {p_tail - eq_p1:+.4f}）")
        lines.append(f"  奴隶首轮王牌率 q: {q_tail:.4f}（偏差 {q_tail - eq_p1:+.4f}）")
        lines.append(f"  皇帝期望收益: {ev_tail:+.4f}（均衡近似 {eq_ev:+.4f}）")
        lines.append(f"  末段漂移（最后 {w} 采样点 vs 前一相邻窗口）:")
        lines.append(f"    皇帝 Δp: {p_tail - p_prev:+.4f}  奴隶 Δq: {q_tail - q_prev:+.4f}")
        lines.append("")
        lines.append("== 收敛判定（客观标准） ==")
        lines.append(f"  漂移阈值 0.005: 皇帝 {'未收敛（仍在漂移）' if abs(p_tail - p_prev) > 0.005 else '收敛（稳定）'}"
                     f"，奴隶 {'未收敛（仍在漂移）' if abs(q_tail - q_prev) > 0.005 else '收敛（稳定）'}")
        # 期望收益与均衡的比较：仅作客观参考，不构成收敛判定（±0.02 容差）
        lines.append(f"  期望收益 {'达到/超过均衡近似' if ev_tail >= eq_ev - 0.02 else '低于均衡近似'} "
                     f"（皇帝视角）")
    else:
        lines.append("采样点不足（<40），跳过统计评估")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[诊断] 收敛性诊断已写入 {path}")


def export_charts(metrics_history, config: Config):
    """训练结束后自动导出训练曲线（PNG + CSV）到 chart_dir。

    与 SwanLab UI 使用同一批每 100 回合的指标数据，全程离线可生成，
    适合云端不可用时在本地归档训练曲线。

    导出内容（文件名带时间戳 chart_{stamp}_{name}.png）：
      1) win_rate_avg_rounds   胜率 + 平均结束回合
      2) initial_ace_prob      首轮王牌率（含静态均衡线 1/7）
      3) loss                  Actor / Critic 损失
      4) kl                    更新 KL 散度
      5) skip_count            update 静默跳过累计计数
      6) expected_reward       皇帝期望收益（含均衡线 + 零线 + 末段均值标注）
      7) convergence           收敛性面板（距均衡偏差 + 策略漂移率）
      8) metrics_{stamp}.csv   全量指标数据备份（二次分析用）
      9) diagnosis_{stamp}.txt 客观收敛性诊断

    参数:
        metrics_history: 每 100 回合采样的指标列表（字段见 CHART_METRIC_KEYS）。
        config: 训练配置（export_charts / chart_dir 控制开关与输出目录）。
    """
    if not config.export_charts or not metrics_history:
        return
    os.makedirs(config.chart_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境安全绘图
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import time
    # 中文字体回退（标题含中文，缺字体时渲染为方框）
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ("CJK", "Noto Sans SC", "WenQuanYi", "Source Han", "Hei")):
            plt.rcParams["font.sans-serif"] = [f.name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    episodes = np.array([m["episode"] for m in metrics_history], dtype=float)
    series = {k: np.array([m.get(k, np.nan) for m in metrics_history], dtype=float)
              for k in CHART_METRIC_KEYS}
    window = max(1, len(episodes) // 200)  # 平滑窗口 ≈ 数据点数的 0.5%
    stamp = time.strftime("%Y%m%d_%H%M%S")
    saved = []

    def _plot(fig, name):
        path = os.path.join(config.chart_dir, f"chart_{stamp}_{name}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # 1) 胜率与平均回合
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for key, label, color in (("emperor_win_rate", "皇帝胜率", "#e74c3c"),
                              ("slave_win_rate", "奴隶胜率", "#3498db")):
        axes[0].plot(episodes, series[key], color=color, alpha=0.3, lw=0.7)
        axes[0].plot(*_smooth_series(episodes, series[key], window),
                     color=color, lw=1.8, label=label)
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("win rate")
    axes[0].set_title("胜率曲线"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(episodes, series["avg_rounds"], color="#2ecc71", alpha=0.3, lw=0.7)
    axes[1].plot(*_smooth_series(episodes, series["avg_rounds"], window), color="#2ecc71", lw=1.8)
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("avg rounds")
    axes[1].set_title("平均结束回合"); axes[1].grid(alpha=0.3)
    _plot(fig, "win_rate_avg_rounds")

    # 2) 首轮出王牌概率（对比静态均衡 1/7）
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for key, label, color in (("emperor_initial_p1", "皇帝首轮王牌率", "#e74c3c"),
                              ("slave_initial_p1", "奴隶首轮王牌率", "#3498db")):
        ax.plot(episodes, series[key], color=color, alpha=0.3, lw=0.7)
        ax.plot(*_smooth_series(episodes, series[key], window), color=color, lw=1.8, label=label)
    ax.axhline(1 / 7, color="gray", ls="--", lw=1, label="静态均衡 1/7")
    ax.set_xlabel("episode"); ax.set_ylabel("probability")
    ax.set_title("首轮出王牌概率（初始策略 p1）"); ax.legend(); ax.grid(alpha=0.3)
    _plot(fig, "initial_ace_prob")

    # 3) 损失
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for key, label, color in (("emperor_actor_loss", "皇帝 actor", "#e74c3c"),
                              ("slave_actor_loss", "奴隶 actor", "#3498db"),
                              ("emperor_critic_loss", "皇帝 critic", "#e67e22"),
                              ("slave_critic_loss", "奴隶 critic", "#9b59b6")):
        ax = axes[0] if "actor" in key else axes[1]
        ax.plot(episodes, series[key], color=color, alpha=0.3, lw=0.7)
        ax.plot(*_smooth_series(episodes, series[key], window), color=color, lw=1.6, label=label)
        ax.set_xlabel("episode"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_title("Actor 损失"); axes[1].set_title("Critic 损失")
    _plot(fig, "loss")

    # 4) KL 散度
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for key, label, color in (("emperor_kl", "皇帝", "#e74c3c"), ("slave_kl", "奴隶", "#3498db")):
        ax.plot(episodes, series[key], color=color, alpha=0.3, lw=0.7)
        ax.plot(*_smooth_series(episodes, series[key], window), color=color, lw=1.6, label=label)
    ax.set_xlabel("episode"); ax.set_ylabel("KL"); ax.set_title("更新 KL 散度")
    ax.legend(); ax.grid(alpha=0.3)
    _plot(fig, "kl")

    # 5) update 静默跳过计数（累计）
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for side, color in (("emperor", "#e74c3c"), ("slave", "#3498db")):
        total = np.nansum([series[f"{side}_skip_flat_adv"], series[f"{side}_skip_batch_std"],
                           series[f"{side}_skip_non_finite_probs"], series[f"{side}_skip_non_finite_loss"]],
                          axis=0)
        ax.plot(episodes, np.cumsum(np.nan_to_num(total)), color=color, lw=1.6, label=f"{side} 累计")
    ax.set_xlabel("episode"); ax.set_ylabel("count")
    ax.set_title("update 静默跳过累计计数"); ax.legend(); ax.grid(alpha=0.3)
    _plot(fig, "skip_count")

    # 6) 皇帝期望收益（非对称奖励下的客观均衡指标）
    #    E = 皇帝胜率 × reward_win + 奴隶胜率 × reward_loss（皇帝视角）
    eq_reward = 1 / 7  # 静态混合均衡下的皇帝期望收益 ≈ +0.143（无第5轮强制规则的近似）
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ev = series["emperor_expected_reward"]
    ax.plot(episodes, ev, color="#8e44ad", alpha=0.3, lw=0.7)
    ax.plot(*_smooth_series(episodes, ev, window), color="#8e44ad", lw=1.8,
            label="皇帝期望收益")
    ax.axhline(eq_reward, color="gray", ls="--", lw=1, label=f"静态均衡近似 {eq_reward:.3f}")
    ax.axhline(0.0, color="black", ls=":", lw=0.8, label="零收益线")
    if len(ev) >= 20:  # 末段均值标注（最后 5% 采样点）
        tail_mean = float(np.nanmean(ev[-max(1, len(ev) // 20):]))
        ax.axhline(tail_mean, color="#8e44ad", ls=":", lw=1)
        ax.text(0.02, 0.95, f"末段均值 {tail_mean:+.3f}（均衡 {eq_reward:+.3f}）",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="#8e44ad", alpha=0.8))
    ax.set_xlabel("episode"); ax.set_ylabel("expected reward (emperor)")
    ax.set_title("皇帝期望收益（胜率×1 + 负率×(−5)）"); ax.legend(); ax.grid(alpha=0.3)
    _plot(fig, "expected_reward")

    # 7) 收敛性面板：距均衡偏差 + 策略漂移率
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    # 左：首轮王牌率距静态均衡 1/7 的绝对偏差（下降 = 向均衡收敛）
    eq_p1 = 1 / 7
    for key, label, color in (("emperor_initial_p1", "皇帝 |p−1/7|", "#e74c3c"),
                              ("slave_initial_p1", "奴隶 |q−1/7|", "#3498db")):
        dev = np.abs(series[key] - eq_p1)
        ax = axes[0]
        ax.plot(episodes, dev, color=color, alpha=0.3, lw=0.7)
        ax.plot(*_smooth_series(episodes, dev, window), color=color, lw=1.8, label=label)
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("|p − 1/7|")
    axes[0].set_title("首轮王牌率距均衡偏差（收敛性）"); axes[0].legend(); axes[0].grid(alpha=0.3)
    # 右：策略漂移率 = 王牌率的一阶差分绝对值（滑动平均，下降趋 0 = 策略趋于稳定）
    for key, label, color in (("emperor_initial_p1", "皇帝", "#e74c3c"),
                              ("slave_initial_p1", "奴隶", "#3498db")):
        diff = np.abs(np.diff(np.nan_to_num(series[key])))
        drift_x, drift_y = _smooth_series(episodes[1:], diff, window)
        axes[1].plot(drift_x, drift_y, color=color, lw=1.6, label=f"{label} |Δp|")
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("|Δp| per 100 episodes")
    axes[1].set_title("策略漂移率（|Δ王牌率|，下降 = 稳定）"); axes[1].legend(); axes[1].grid(alpha=0.3)
    _plot(fig, "convergence")

    # CSV 数据备份（便于后续二次分析）
    csv_path = os.path.join(config.chart_dir, f"metrics_{stamp}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ["episode"] + list(CHART_METRIC_KEYS)
        f.write(",".join(cols) + "\n")
        for i in range(len(episodes)):
            row = [f"{episodes[i]:.0f}"]
            row += [f"{series[k][i]:.6g}" if np.isfinite(series[k][i]) else ""
                    for k in CHART_METRIC_KEYS]
            f.write(",".join(row) + "\n")
    saved.append(csv_path)

    # 客观收敛性诊断（txt）：期望收益 vs 均衡、王牌率偏差、末段漂移
    diag_path = os.path.join(config.chart_dir, f"diagnosis_{stamp}.txt")
    try:
        _write_convergence_diagnosis(diag_path, metrics_history, config)
        saved.append(diag_path)
    except Exception as e:
        print(f"[警告] 收敛性诊断写入失败: {e}")

    print(f"[图表] 训练曲线已导出 {len(saved)} 个文件 -> {config.chart_dir}")


# ============================== 绘图 ==============================
def plot_results(joint_prob, emp_strat, slv_strat, emp_rate, slv_rate, avg_rounds, output_dir):
    """绘制最终结果图（三合一）：联合分布热力图 + 双方初始策略条形图。

    参数:
        joint_prob: 双方王牌出现轮次的联合分布（5×5，行=皇帝 E1~E5，列=奴隶 S1~S5）。
        emp_strat / slv_strat: 双方初始贪心策略概率向量 [平民率, 王牌率]。
        emp_rate / slv_rate: 双方实际胜率。
        avg_rounds: 全局平均结束回合。
        output_dir: 保存目录（文件名为 ppo_opponent_modeling.png）。
    """
    fig = plt.figure(figsize=(16, 6))

    # ---- 子图 1：联合分布热力图（体现双方出王牌轮次的博弈结构） ----
    ax1 = plt.subplot(1, 3, 1)
    im = ax1.imshow(joint_prob, cmap='Blues', aspect='auto', vmin=0, vmax=0.5)
    ax1.set_xticks(range(joint_prob.shape[1]))
    ax1.set_yticks(range(joint_prob.shape[0]))
    ax1.set_xticklabels([f'S{i+1}' for i in range(joint_prob.shape[1])], fontsize=12)
    ax1.set_yticklabels([f'E{i+1}' for i in range(joint_prob.shape[0])], fontsize=12)
    ax1.set_xlabel('Slave Position', fontsize=14)
    ax1.set_ylabel('Emperor Position', fontsize=14)
    ax1.set_title(f'Joint Distribution\n(avg rounds: {avg_rounds:.2f}, samples: {joint_prob.sum():.0f}%)', fontsize=13)

    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            val = joint_prob[i, j]
            if val > 0.01:
                color = 'white' if val > 0.25 else 'black'
                ax1.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(1, 3, 2)
    bars = ax2.bar(['Civilian', 'Ace'], emp_strat, color=['#3498db', '#e74c3c'], edgecolor='black')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'Emperor Initial Policy\nWin Rate: {emp_rate:.3f}', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, emp_strat):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', ha='center', fontsize=11)

    ax3 = plt.subplot(1, 3, 3)
    bars = ax3.bar(['Civilian', 'Ace'], slv_strat, color=['#3498db', '#e74c3c'], edgecolor='black')
    ax3.set_ylim(0, 1)
    ax3.set_title(f'Slave Initial Policy\nWin Rate: {slv_rate:.3f}', fontsize=13)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, slv_strat):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', ha='center', fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "ppo_opponent_modeling.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[保存] 结果图: {save_path}")


# ============================== 短时验证（--quick） ==============================
def _quick_check(condition: bool, message: str) -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
    return bool(condition)


def run_quick_validation(config: Config) -> bool:
    """短时验证：奖励符号 / 环境步进 / 网络前向 / 短循环收敛方向。

    固定种子（--quick 用 quick_cfg 覆盖为 CPU、300 回合、无检查点/图表/SwanLab），
    覆盖四类检查项：
      1) 奖励符号：A-A=-5(奴胜)、E-C=+1(皇帝胜)、C-E=+1(皇帝胜)、C-C=0(平局)；
      2) 环境步进：状态维度=17、第5轮强制 A-A → slave_win、状态数值有限；
      3) 网络前向：actor 输出 (1,2)、critic 输出 (1,1)、数值有限；
      4) 短循环收敛方向：300 回合内 actor loss 前段→后段下降、胜率 ∈[0,1]、
         平均回合有限、跳过计数器存在且非负。

    参数:
        config: 基础训练配置（用于环境奖励与网络维度；循环部分内部覆盖为短时配置）。

    返回:
        True = 全部检查项通过（退出码 0）；False = 存在失败项（退出码 1）。
    """
    print("=" * 60)
    print("短时验证（--quick）")
    print("=" * 60)
    checks = []

    # ---- 1) 奖励符号（皇帝视角） ----
    print("[1/4] 奖励符号")
    env = SequentialECardEnv(
        max_rounds=config.max_rounds,
        reward_emperor_win=config.reward_emperor_win,
        reward_emperor_loss=config.reward_emperor_loss,
    )
    env.reset()
    _, r_aa, done_aa, info_aa = env.step(1, 1)   # A-A → 奴隶胜
    checks.append(_quick_check(
        r_aa == config.reward_emperor_loss and r_aa < 0,
        f"A-A 奴隶胜: 奖励={r_aa}（期望 {config.reward_emperor_loss}，负）"))
    checks.append(_quick_check(done_aa and info_aa["result"] == "slave_win",
                               "A-A 相遇立即终止且 result=slave_win"))
    env.reset()
    _, r_ec, done_ec, info_ec = env.step(1, 0)   # E-C → 皇帝胜
    checks.append(_quick_check(
        r_ec == config.reward_emperor_win and r_ec > 0,
        f"E-C 皇帝胜: 奖励={r_ec}（期望 {config.reward_emperor_win}，正）"))
    checks.append(_quick_check(done_ec and info_ec["result"] == "emperor_win",
                               "E-C 相遇立即终止且 result=emperor_win"))
    env.reset()
    _, r_ce, done_ce, info_ce = env.step(0, 1)   # C-E → 皇帝胜
    checks.append(_quick_check(
        r_ce == config.reward_emperor_win and r_ce > 0,
        f"C-E 皇帝胜: 奖励={r_ce}（期望 {config.reward_emperor_win}，正）"))
    checks.append(_quick_check(done_ce and info_ce["result"] == "emperor_win",
                               "C-E 相遇立即终止且 result=emperor_win"))
    env.reset()
    _, r_cc, done_cc, info_cc = env.step(0, 0)   # C-C → 平局
    checks.append(_quick_check(r_cc == 0.0 and not done_cc and info_cc["result"] == "draw",
                               "C-C 平局: 奖励=0，不终止"))

    # ---- 2) 环境步进 ----
    print("[2/4] 环境步进")
    env = SequentialECardEnv(
        max_rounds=config.max_rounds,
        reward_emperor_win=config.reward_emperor_win,
        reward_emperor_loss=config.reward_emperor_loss,
    )
    state = env.reset()
    checks.append(_quick_check(
        state.shape == (config.state_dim,),
        f"reset 状态维度: {state.shape[0]}（期望 {config.state_dim}）"))
    env = SequentialECardEnv(
        max_rounds=config.max_rounds,
        reward_emperor_win=config.reward_emperor_win,
        reward_emperor_loss=config.reward_emperor_loss,
    )
    env.reset()
    final_state = None
    final_info = None
    for _ in range(config.max_rounds):
        final_state, r_forced, done_forced, final_info = env.step(0, 0)
    checks.append(_quick_check(
        done_forced and final_info["result"] == "slave_win" and r_forced == config.reward_emperor_loss,
        f"第 {config.max_rounds} 轮强制 A-A → slave_win，奖励={r_forced}"))
    checks.append(_quick_check(np.isfinite(final_state).all(), "步进后状态数值有限"))

    # ---- 3) 网络前向 ----
    print("[3/4] 网络前向")
    net = PPOActorCritic(config.state_dim, config.action_dim, config.hidden_dim)
    logits, value = net(torch.zeros(1, config.state_dim))
    checks.append(_quick_check(tuple(logits.shape) == (1, config.action_dim),
                               f"actor 输出形状: {tuple(logits.shape)}（期望 (1, {config.action_dim})）"))
    checks.append(_quick_check(tuple(value.shape) == (1, 1),
                               f"critic 输出形状: {tuple(value.shape)}（期望 (1, 1)）"))
    checks.append(_quick_check(
        bool(torch.isfinite(logits).all() and torch.isfinite(value).all()),
        "网络前向输出数值有限"))

    # ---- 4) 短循环收敛方向（固定种子，可复现） ----
    print("[4/4] 短循环收敛方向")
    quick_cfg = dataclasses.replace(
        config,
        episodes=300,
        steps_per_update=64,
        epochs_per_update=5,
        checkpoint_interval=0,
        export_charts=False,  # 短时验证不导出曲线
        device="cpu",         # 短时验证固定 CPU，跳过测速开销
        swanlab_mode="disabled",  # 短时验证不记录 SwanLab
    )
    random.seed(quick_cfg.seed)
    np.random.seed(quick_cfg.seed)
    torch.manual_seed(quick_cfg.seed)
    losses = []
    skip_log = {}
    _, _, _, emp_rate, slv_rate, avg_rounds = train_ppo_original(
        quick_cfg, tracking=False, loss_log=losses, skip_log=skip_log)
    skip_e = skip_log.get("emperor_total", 0)
    skip_s = skip_log.get("slave_total", 0)
    print(f"  跳过计数（短循环累计，皇/奴）: {skip_e}/{skip_s}")
    checks.append(_quick_check(skip_e >= 0 and skip_s >= 0, "跳过计数器存在且非负"))
    checks.append(_quick_check(0.0 <= emp_rate <= 1.0 and 0.0 <= slv_rate <= 1.0,
                               f"胜率在 [0,1]: 皇帝={emp_rate:.3f}, 奴隶={slv_rate:.3f}"))
    checks.append(_quick_check(np.isfinite(avg_rounds) and avg_rounds > 0,
                               f"平均回合有限: {avg_rounds:.2f}"))
    checks.append(_quick_check(len(losses) >= 3 and all(np.isfinite(l) for l in losses),
                               f"actor loss 采样点有限: {len(losses)} 个"))
    if len(losses) >= 3:
        head = np.mean(losses[:max(1, len(losses) // 3)])
        tail = np.mean(losses[-max(1, len(losses) // 3):])
        checks.append(_quick_check(tail < head,
                                   f"actor loss 收敛方向: 前段={head:.4f} → 后段={tail:.4f}（下降）"))
    ok = all(checks)
    print("=" * 60)
    print(f"短时验证结果: {'全部通过' if ok else '存在失败项'}（{sum(checks)}/{len(checks)}）")
    print("=" * 60)
    return ok


# ============================== 主入口 ==============================
def main():
    """命令行入口：解析参数 → 构造 Config → 运行短时验证或完整训练。

    命令行参数设计为"覆盖式"：--episodes / --checkpoint-interval / --seed /
    --device / --swanlab-mode / --no-charts 仅在显式给出时覆盖 Config 默认值，
    未给出则保持默认（Config 是唯一配置源，检查点存档也基于它）。
    """
    parser = argparse.ArgumentParser(description="E卡博弈 PPO 训练（检查点 + 短时验证）")
    parser.add_argument("--quick", action="store_true",
                        help="运行短时验证：奖励符号/环境步进/网络前向/收敛方向")
    parser.add_argument("--episodes", type=int, default=None, help="覆盖训练回合数")
    parser.add_argument("--resume", action="store_true", help="从最新检查点恢复训练")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="覆盖检查点保存间隔（回合数，0=关闭）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    parser.add_argument("--no-tracking", action="store_true", help="禁用 SwanLab 追踪")
    parser.add_argument("--swanlab-mode", choices=["online", "local", "offline", "disabled"], default=None,
                        help="SwanLab 运行模式（默认 local 离线；online 同步云端 / offline 本地暂存 / disabled 不记录）")
    parser.add_argument("--no-charts", action="store_true", help="禁用训练结束后的曲线自动导出（默认导出到 ./swanlog/charts）")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                        help="计算设备（auto=实测对比 CPU/CUDA 前向耗时后选择，默认）")
    args = parser.parse_args()

    # 收集显式覆盖字段（None 表示未指定，保持 Config 默认值）
    fields = {}
    if args.episodes is not None:
        fields["episodes"] = args.episodes
    if args.checkpoint_interval is not None:
        fields["checkpoint_interval"] = args.checkpoint_interval
    if args.seed is not None:
        fields["seed"] = args.seed
    if args.no_charts:
        fields["export_charts"] = False
    if args.device is not None:
        fields["device"] = args.device
    if args.swanlab_mode is not None:
        fields["swanlab_mode"] = args.swanlab_mode
    config = Config(**fields)
    os.makedirs(config.output_dir, exist_ok=True)

    if args.quick:
        ok = run_quick_validation(config)
        sys.exit(0 if ok else 1)

    try:
        emp_strat, slv_strat, joint_prob, emp_rate, slv_rate, avg_rounds = train_ppo_original(
            config, tracking=not args.no_tracking, resume=args.resume)
        plot_results(joint_prob, emp_strat, slv_strat, emp_rate, slv_rate, avg_rounds, config.output_dir)

        print("\n" + "="*50)
        print("最终结论（原版 E 卡博弈 + 对手建模）")
        print("="*50)
        print(f"皇帝初始出王牌概率 (p): {emp_strat[1]:.3f}")
        print(f"奴隶初始出王牌概率 (q): {slv_strat[1]:.3f}")
        print(f"皇帝实际胜率: {emp_rate:.3f}")
        print(f"奴隶实际胜率: {slv_rate:.3f}")
        print(f"平均结束回合: {avg_rounds:.2f}")
        print("奖励设置（皇帝视角）：皇帝胜 +1，皇帝负 -5（奴隶胜），平局 0")
        print("平民牌限制：每方 4 张平民牌")
        print("对手建模：包含每一轮的历史出牌概率（5维）")

    except Exception as e:
        print(f"[错误] 训练异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()