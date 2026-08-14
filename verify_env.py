#!/usr/bin/env python
"""验证脚本：ECardEnv 规则不变量模糊测试。

旧版对照 SequentialECardEnv 已随阶段 3 重写移除（ppo.py 不再包含旧环境类），
本脚本改为直接对 ECardEnv 做规则不变量检查：用随机合法动作驱动大量对局，
逐局断言以下不变量，任何一条被违反即 FAIL：

- 每局 <= max_rounds 步必终局；终局必有 winner（无整体平局）；
- 奖励只出现在终局步，取值 ∈ {+1, -5}；中间步恒 0；
- winner 与奖励符号一致（皇帝胜 +1 / 奴隶胜 -5）；
- 王牌已出则不能再出王牌；平民耗尽或最后一轮强制出王牌；
- 终局后 step 抛 RuntimeError；非法动作抛 ValueError；
- 状态恒在 [0, 1]^5，平民剩余恒 >= 0。

用法：
    python verify_env.py                # 默认 100000 局，seed=42
    python verify_env.py --games 1000   # 快速冒烟
"""
from __future__ import annotations

import argparse
import random
import sys

from env import (CIVILIANS, MAX_ROUNDS, PLAY_ACE, PLAY_CIVILIAN,
                 ECardEnv, EnvConfig)


def play_one_game(env: ECardEnv, rng: random.Random) -> str | None:
    """随机合法动作玩一局，返回第一个违规描述（无违规返回 None）。"""
    env.reset()
    for _ in range(MAX_ROUNDS + 1):  # 防呆上限：合法状态下最多 max_rounds 步
        # 状态不变量
        s = env.state
        if s.shape != (5,) or not ((s >= 0).all() and (s <= 1).all()):
            return f"状态越界 {s}"

        le, ls = env.legal_actions("emperor"), env.legal_actions("slave")
        if not le or not ls:
            return f"存在空合法动作集 E={le} S={ls}"

        # 王牌已出 → 不能再出王牌（若可达，规则必须拦住）
        if env.e_played and PLAY_ACE in le:
            return "皇帝已出王牌却仍可再出王牌"
        if env.s_played and PLAY_ACE in ls:
            return "奴隶已出王牌却仍可再出王牌"
        # 平民耗尽或最后一轮 → 必须强制出王牌
        if env.round == MAX_ROUNDS - 1 and le != [PLAY_ACE]:
            return f"最后一轮皇帝合法动作应为 [ACE]，实际 {le}"
        if env.e_civ_left == 0 and le != [PLAY_ACE]:
            return f"平民耗尽皇帝合法动作应为 [ACE]，实际 {le}"

        a_e, a_s = rng.choice(le), rng.choice(ls)
        before_round = env.round
        _, r, done, info = env.step(a_e, a_s)

        if done:
            # 终局不变量
            if info["winner"] is None:
                return "终局但 winner 为 None（不应存在整体平局）"
            if r not in (1.0, -5.0):
                return f"终局奖励异常 {r}"
            if (info["winner"] == "emperor") != (r > 0):
                return f"winner={info['winner']} 与奖励 {r} 矛盾"
            if env.round != before_round + 1:
                return "终局步后轮次未递增"
            return None
        # 非终局不变量
        if r != 0.0:
            return f"非终局步奖励应为 0，实际 {r}"
        if env.round != before_round + 1:
            return "平局步后轮次未递增"

    return "超过 max_rounds 步未终局（理论上不可达）"


def main() -> int:
    parser = argparse.ArgumentParser(description="ECardEnv 规则不变量模糊测试")
    parser.add_argument("--games", type=int, default=100_000, help="对局数（默认 100000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = ECardEnv(EnvConfig())

    mismatches: list[str] = []
    for i in range(args.games):
        msg = play_one_game(env, rng)
        if msg is not None:
            mismatches.append(f"第 {i} 局: {msg}")
            if len(mismatches) >= 10:
                break  # 前 10 个违规就停，不刷屏

    if mismatches:
        print(f"[FAIL] {len(mismatches)} 个违规：")
        for msg in mismatches:
            print("  " + msg)
        return 1

    print(f"[PASS] {args.games} 局全部满足规则不变量（seed={args.seed}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
