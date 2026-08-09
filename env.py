"""皇帝牌（E卡）环境：最小可运行版本。

设计原则：
1. 状态 = 5 维最小马尔可夫状态，不含任何全局统计或对手建模。
2. 环境是确定性的：随机性全部来自策略，不在环境里。
3. 非法动作直接抛错（fail loud），不做静默替换——bug 应该炸出来。
4. 奖励保持客观（+1/-5/0）；心理效用放在 utility 层，不污染环境。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 动作编码
PLAY_CIVILIAN = 0  # 出平民
PLAY_ACE = 1       # 出王牌

MAX_ROUNDS = 5
CIVILIANS = 4


@dataclass(frozen=True)
class EnvConfig:
    """环境配置：只放规则参数，不放训练超参。"""
    max_rounds: int = MAX_ROUNDS
    reward_emperor_win: float = 1.0    # 皇帝胜（E-C / C-E）
    reward_emperor_loss: float = -5.0  # 皇帝负（A-A → 奴隶胜）


class ECardEnv:
    def __init__(self, config: EnvConfig | None = None) -> None:
        self.cfg = config or EnvConfig()
        self.reset()

    # ---------- 状态 ----------
    @property
    def state(self) -> np.ndarray:
        """5 维状态（float32）：
        [轮次/最大轮次, 皇帝已出王牌?, 奴隶已出王牌?,
         皇帝剩余平民/4, 奴隶剩余平民/4]
        """
        return np.array([
            self.round / self.cfg.max_rounds,
            float(self.e_played),
            float(self.s_played),
            self.e_civ_left / CIVILIANS,
            self.s_civ_left / CIVILIANS,
        ], dtype=np.float32)

    @property
    def done(self) -> bool:
        return self.winner is not None

    # ---------- 生命周期 ----------
    def reset(self) -> np.ndarray:
        self.round = 0
        self.e_played = False
        self.s_played = False
        self.e_civ_left = CIVILIANS
        self.s_civ_left = CIVILIANS
        self.e_ace_round = -1  # 王牌出现的轮次，-1 = 未出
        self.s_ace_round = -1
        self.winner = None     # 'emperor' | 'slave' | None
        return self.state

    def legal_actions(self, player: str) -> list[int]:
        """该玩家当前合法的动作列表。"""
        if player == "emperor":
            ace_played, civ_left = self.e_played, self.e_civ_left
        else:
            ace_played, civ_left = self.s_played, self.s_civ_left
        if ace_played:
            return [PLAY_CIVILIAN]
        if civ_left == 0 or self.round == self.cfg.max_rounds - 1:
            # 平民耗尽 或 最后一轮：只能出王牌（两者实际等价）
            return [PLAY_ACE]
        return [PLAY_CIVILIAN, PLAY_ACE]

    def step(self, action_e: int, action_s: int) -> tuple[np.ndarray, float, bool, dict]:
        """双方同时出牌，返回 (next_state, reward_emperor, done, info)。"""
        if self.done:
            raise RuntimeError("step 在终局后调用")
        self._check_action(action_e, self.legal_actions("emperor"))
        self._check_action(action_s, self.legal_actions("slave"))

        # 执行：消耗平民 / 记录王牌轮次
        if action_e == PLAY_CIVILIAN:
            self.e_civ_left -= 1
        else:
            self.e_played = True
            self.e_ace_round = self.round
        if action_s == PLAY_CIVILIAN:
            self.s_civ_left -= 1
        else:
            self.s_played = True
            self.s_ace_round = self.round

        # 胜负判定（皇帝视角）：任何王牌出现即终局
        reward = 0.0
        if action_e == PLAY_ACE and action_s == PLAY_ACE:
            self.winner, reward = "slave", self.cfg.reward_emperor_loss
        elif action_e == PLAY_ACE or action_s == PLAY_ACE:
            self.winner, reward = "emperor", self.cfg.reward_emperor_win
        # 平民对平民：继续，reward 保持 0

        self.round += 1
        done = self.winner is not None or self.round >= self.cfg.max_rounds
        info = {
            "winner": self.winner,
            "action_e": action_e,
            "action_s": action_s,
            "e_ace_round": self.e_ace_round,
            "s_ace_round": self.s_ace_round,
        }
        return self.state, reward, done, info

    def _check_action(self, action: int, legal: list[int]) -> None:
        if action not in legal:
            raise ValueError(f"非法动作 {action}，合法集合 {legal}")