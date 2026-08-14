"""阶段 3 训练入口：identity 先跑通，prospect/tilt 之后再说。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from env import ECardEnv
from ppo import Agent, collect_rollout, first_round_probs, update_policy
from utility import make_utility


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--utility", default="identity", choices=["identity", "prospect", "tilt"])
    p.add_argument("--adv-norm", action="store_true", help="优势归一化（默认关）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=200_000, help="总环境步数")
    p.add_argument("--rollout", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--quick", action="store_true", help="冒烟测试：短跑、少 epoch")
    return p.parse_args()


def window_drift(series: list[float], tail: float = 0.2) -> float:
    """末段漂移 = 最后 tail 比例的均值 - 之前窗口的均值。"""
    if len(series) < 2:
        return float("nan")
    cut = max(1, int(len(series) * tail))
    return float(np.mean(series[-cut:]) - np.mean(series[:-cut]))


def _window_stats(series: list[float], tail: float = 0.2) -> tuple[float, float]:
    """末段窗口均值与标准差（修：最终指标不用单点快照，与 sweep.py 口径一致）。"""
    if not series:
        return float("nan"), float("nan")
    cut = min(len(series), max(2, int(len(series) * tail))) if len(series) >= 2 else 1
    window = series[-cut:]
    return float(np.mean(window)), float(np.std(window))


def main():
    args = parse_args()
    if args.quick:
        args.steps, args.rollout, args.epochs, args.batch = 50_000, 2048, 2, 128

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = ECardEnv()
    agents = {"emperor": Agent(), "slave": Agent()}
    optimizers = {k: torch.optim.Adam(agents[k].parameters(), lr=args.lr) for k in agents}
    utilities = {k: make_utility(args.utility) for k in agents}

    n_updates = max(1, args.steps // args.rollout)
    history = {"p": [], "q": [], "win_rate": [], "obj_return": []}
    t0 = time.time()

    for it in range(1, n_updates + 1):
        buffers, stats = collect_rollout(env, agents, utilities,
                                         args.rollout, args.gamma, args.lam)
        for name in agents:
            update_policy(agents[name], optimizers[name], buffers[name],
                          gamma=args.gamma, lam=args.lam, clip_eps=args.clip,
                          ent_coef=args.ent_coef, adv_norm=args.adv_norm,
                          epochs=args.epochs, batch_size=args.batch)
        p, q = first_round_probs(agents)
        history["p"].append(p)
        history["q"].append(q)
        history["win_rate"].append(stats["emperor_win_rate"])
        history["obj_return"].append(stats["emperor_obj_return"])
        if it == 1 or it % 5 == 0 or it == n_updates:
            print(f"iter {it:4d}/{n_updates} | win={stats['emperor_win_rate']:.3f} "
                  f"| obj_ret={stats['emperor_obj_return']:+.3f} "
                  f"| p={p:.3f} q={q:.3f} "
                  f"| subj E/S={stats['subj_mean']['emperor']:+.3f}/{stats['subj_mean']['slave']:+.3f}")

    # 修：最终指标取末段窗口均值（与 sweep.py 一致）；final_p/final_q 保留为
    # 单点快照，兼容旧 stage3 JSON 的字段。
    p, p_std = _window_stats(history["p"])
    q, q_std = _window_stats(history["q"])
    win_rate, _ = _window_stats(history["win_rate"])
    obj_return, _ = _window_stats(history["obj_return"])
    summary = {
        "utility": args.utility,
        "adv_norm": args.adv_norm,
        "seed": args.seed,
        "steps": args.steps,
        "final_p": history["p"][-1],
        "final_q": history["q"][-1],
        "p": p,
        "p_std": p_std,
        "q": q,
        "q_std": q_std,
        "equilibrium_p": 0.2,
        "equilibrium_q": 0.2,
        "deviation_p": p - 0.2,
        "deviation_q": q - 0.2,
        "win_rate": win_rate,
        "obj_return": obj_return,
        "drift_p": window_drift(history["p"]),
        "drift_q": window_drift(history["q"]),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print("\n[阶段 3 验证门：单 seed 记录，不下结论]")
    print(f"  首轮 p̂={p:.4f}±{p_std:.4f}  vs 均衡 0.2  偏差 {summary['deviation_p']:+.4f}")
    print(f"  首轮 q̂={q:.4f}±{q_std:.4f}  vs 均衡 0.2  偏差 {summary['deviation_q']:+.4f}")
    print(f"  皇帝胜率={win_rate:.4f}  vs 均衡 0.8")
    print(f"  末段漂移 Δp={summary['drift_p']:+.4f}  Δq={summary['drift_q']:+.4f}")
    print(f"  用时 {summary['elapsed_s']}s（{args.utility}, adv_norm={'on' if args.adv_norm else 'off'}）")

    out = Path("data") / f"stage3_{args.utility}_seed{args.seed}_advnorm{int(args.adv_norm)}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已存 {out}")


if __name__ == "__main__":
    main()
