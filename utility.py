"""阶段 3 效用层：把客观奖励映射成 agent 最大化的主观效用。

接口统一：utility(reward, context) -> float
context 由 train.py 传入，至少包含 streak_losses（该 agent 当前连败数）。

设计顺序（direction.md 阶段 3）：identity 先跑通 → prospect → tilt。
阶段 4 扫描会用到 prospect 的概率权重（use_weighting=True）。
心理只在这里，不污染 env.py。
"""
from __future__ import annotations

from collections import deque


def identity(reward, context=None):
    """无心理：主观效用 = 客观奖励。用来验证 PPO 本身。"""
    return float(reward)


def _value(x: float, alpha: float, beta: float, lam: float) -> float:
    """前景理论价值函数：收益 x**alpha，损失 -λ*(-x)**beta。"""
    if x >= 0:
        return x ** alpha
    return -lam * (-x) ** beta


def weight(p: float, gamma: float) -> float:
    """Tversky & Kahneman 1992 概率权重函数，典型 γ≈0.61。"""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return p ** gamma / (p ** gamma + (1 - p) ** gamma) ** (1.0 / gamma)


def make_prospect(alpha: float = 0.88, beta: float = 0.88,
                  lam: float = 2.25, gamma: float = 0.61,
                  use_weighting: bool | str = False, window: int = 64,
                  ref_prob: float = 0.2):
    """前景理论效用（Kahneman & Tversky 1979/1992 典型参数）。

    use_weighting=False：只做价值函数（收益 x**α，损失 -λ*(-x)**β）。
    use_weighting="window"（或 True）：对滑动窗口内的胜负频率做概率权重 w(p, γ)，
    每个事件按 w(p)/p 缩放，使期望主观效用 ≈ w(p_win)·v(+|x|) + w(p_loss)·v(-|x|)。
    这是把 CPT 决策权重装进逐回合奖励的一种实现——重写时你可以换成别的。
    注意：窗口频率由 agent 当前策略自己造成，形成反馈环；要隔离它，
    用 use_weighting="ref"——固定参考概率 ref_prob（默认 0.2 = 均衡胜率），
    乘子 = w(p_ref)/p_ref 或 w(1-p_ref)/(1-p_ref)，与策略无关。

    gamma 范围：T&K 单参数权重函数在 γ<~0.3 时退化——典型概率下 w(p)≈0，
    奖励信号被抹平（run02 实测 γ=0.1：+5 的主观值只剩 +1.11，subj_std≈0.01）。
    因此启用概率权重时强制 γ ∈ [0.5, 1.0]；要更小的 γ，请换 Prelec 双参数函数。
    """
    mode = "window" if use_weighting is True else (use_weighting or False)
    if mode and not (0.5 <= gamma <= 1.0):
        raise ValueError(
            f"gamma={gamma} 会让 T&K 单参数权重函数退化（γ<0.5 时小概率被低权、"
            f"γ=0.1 时 w(p)≈0）。启用概率权重时请用 [0.5, 1.0]，"
            f"或改用 Prelec 双参数权重函数。"
        )
    history = deque(maxlen=window) if mode == "window" else None

    def utility(reward, context=None):
        x = float(reward)
        if x == 0:
            return 0.0
        if mode:
            mult = 1.0
            if mode == "window":
                history.append(1 if x > 0 else 0)  # 只统计决胜回合，平局不进分布
                n = len(history)
                if n >= 2:
                    p_win = min(max(sum(history) / n, 1.0 / n), 1.0 - 1.0 / n)
                    if x > 0:
                        mult = weight(p_win, gamma) / p_win
                    else:
                        mult = weight(1.0 - p_win, gamma) / (1.0 - p_win)
            else:  # "ref"：固定参考概率，切断策略反馈环
                p_win = ref_prob
                if x > 0:
                    mult = weight(p_win, gamma) / p_win
                else:
                    mult = weight(1.0 - p_win, gamma) / (1.0 - p_win)
            return _value(x, alpha, beta, lam) * mult
        return _value(x, alpha, beta, lam)
    return utility


def make_tilt(value_fn=None, threshold: int = 1, rate: float = 0.5):
    """连败后加码追回本：streak >= threshold 时，赌注乘 1 + rate*(streak-threshold+1)。"""
    def utility(reward, context=None):
        streak = (context or {}).get("streak_losses", 0)
        mult = 1.0
        if streak >= threshold:
            mult = 1.0 + rate * (streak - threshold + 1)
        base = value_fn(reward, context) if value_fn is not None else float(reward)
        return base * mult
    return utility


def make_utility(name: str, **params):
    """工厂：train.py 只认名字。"""
    if name == "identity":
        return identity
    if name == "prospect":
        return make_prospect(**params)
    if name == "tilt":
        return make_tilt(**params)
    raise ValueError(f"未知效用: {name}")
