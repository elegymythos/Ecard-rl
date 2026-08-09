#!/usr/bin/env python
"""验证脚本：新环境 ECardEnv 与旧环境 SequentialECardEnv 的行为一致性。

方法：用同一个随机策略同时驱动两个环境，逐局比较
winner / 总奖励 / 局数 / 王牌轮次。10 万局全部一致 = 规则没有改坏。

用法：
    python verify_env.py                # 默认 100000 局，seed=42
    python verify_env.py --games 1000   # 快速冒烟
"""
from __future__ import annotations

import argparse
import random
import sys

from env import ECardEnv, EnvConfig
from ppo import SequentialECardEnv  # 旧环境：只作参照，不作学习对象


def play_one_game(old_env: SequentialECardEnv, new_env: ECardEnv, rng: random.Random) -> dict:
    """用随机合法动作驱动两个环境各玩一局，返回本局结果（或第一个不一致点）。"""
    old_env.reset()
    new_env.reset()
    old_reward = 0.0
    new_reward = 0.0

    for _ in range(50):  # 规则保证 <=5 步终局；50 只是防呆上限
        # 每步先断言：两个环境对「合法动作」的理解必须一致
        old_legal = (
            old_env.get_legal_actions("emperor"),
            old_env.get_legal_actions("slave"),
        )
        new_legal = (
            new_env.legal_actions("emperor"),
            new_env.legal_actions("slave"),
        )
        if old_legal != new_legal:
            return {"mismatch": f"合法动作不一致 旧={old_legal} 新={new_legal} 轮次={old_env.current_round}"}

        a_e = rng.choice(old_legal[0])
        a_s = rng.choice(old_legal[1])

        _, r_old, done_old, info_old = old_env.step(a_e, a_s)
        _, r_new, done_new, info_new = new_env.step(a_e, a_s)
        old_reward += r_old
        new_reward += r_new

        if done_old != done_new:
            return {"mismatch": f"done 不一致 旧={done_old} 新={done_new} 轮次={old_env.current_round}"}
        if done_old:
            break
    else:
        return {"mismatch": "超过 50 步未终局（理论上不可达）"}

    # 局末逐字段对比
    fields = {
        "winner": (info_old["winner"], info_new["winner"]),
        "reward": (old_reward, new_reward),
        "rounds": (old_env.current_round, new_env.round),
        "emperor_pos": (info_old["emperor_pos"], info_new["e_ace_round"]),
        "slave_pos": (info_old["slave_pos"], info_new["s_ace_round"]),
    }
    for name, (a, b) in fields.items():
        if a != b:
            return {"mismatch": f"{name} 不一致 旧={a} 新={b}"}
    return {"mismatch": None}


def main() -> int:
    parser = argparse.ArgumentParser(description="新旧环境一致性验证")
    parser.add_argument("--games", type=int, default=100_000, help="对局数（默认 100000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    old_env = SequentialECardEnv()
    new_env = ECardEnv(EnvConfig())

    mismatches: list[str] = []
    for i in range(args.games):
        result = play_one_game(old_env, new_env, rng)
        if result["mismatch"] is not None:
            mismatches.append(f"第 {i} 局: {result['mismatch']}")
            if len(mismatches) >= 10:
                break  # 前 10 个不一致就停，不刷屏

    if mismatches:
        print(f"[FAIL] {len(mismatches)} 个不一致：")
        for msg in mismatches:
            print("  " + msg)
        return 1

    print(f"[PASS] {args.games} 局全部一致（seed={args.seed}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
