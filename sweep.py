"""阶段 4：心理参数扫描（初版，AI 写，待你重写）。

网格：λ ∈ {1, 2, 3, 5} × τ ∈ {0.1, 0.5, 1.0} × 5 seeds = 75 次自博弈训练。
- 皇帝效用：prospect(λ)，α=β=0.88，无概率权重——隔离损失厌恶效应；
- 奴隶效用：prospect(τ)，γ=τ，λ=1，滑动窗口概率权重——隔离概率扭曲效应；
- 每个配置记录：首轮 p̂、q̂、皇帝胜率、客观期望、主观效用均值、末段漂移。
- 预测先写（data/sweep/predictions.json）：
    λ↑ → 皇帝 p↓；τ↑ → 策略逼近均匀随机。
- 输出：data/sweep/grid.json + data/sweep/phase_diagram.png。

用法：
  python sweep.py --demo          # 冒烟：少量格子 × 1 seed（验证管道）
  python sweep.py --steps 40000   # 正式扫描：75 次训练，CPU 约 1-2 小时
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from env import ECardEnv
from ppo import Agent, collect_rollout, first_round_probs, update_policy
from utility import make_prospect

LAMBDAS = [1.0, 2.0, 3.0, 5.0]
TAUS = [0.1, 0.5, 1.0]
SEEDS = [42, 43, 44, 45, 46]

PREDICTIONS = {
    "lambda_up_emperor_p_down": "λ↑ → 皇帝首轮 p↓（损失厌恶让他躲 A-A）",
    "tau_up_toward_uniform": "τ↑ → 策略逼近均匀随机",
}


def _drift(series: list[float], tail: float = 0.2) -> float:
    if len(series) < 2:
        return float("nan")
    cut = max(1, int(len(series) * tail))
    return float(np.mean(series[-cut:]) - np.mean(series[:-cut]))


def run_config(lam: float, tau: float, seed: int, *, steps: int,
               rollout: int, epochs: int, batch: int, lr: float) -> dict:
    """单格子单 seed 的自博弈训练，返回最终指标。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = ECardEnv()
    agents = {"emperor": Agent(), "slave": Agent()}
    optimizers = {k: torch.optim.Adam(agents[k].parameters(), lr=lr) for k in agents}
    utilities = {
        # 皇帝只变 λ，奴隶只变 τ——隔离效应，重写时别混掉
        "emperor": make_prospect(lam=lam, gamma=1.0, use_weighting=False),
        "slave": make_prospect(lam=1.0, gamma=tau, use_weighting=True),
    }
    n_updates = max(1, steps // rollout)
    p_hist, q_hist = [], []
    last_stats = None
    for _ in range(n_updates):
        buffers, stats = collect_rollout(env, agents, utilities, rollout,
                                         gamma=0.99, lam=0.95)
        for name in agents:
            update_policy(agents[name], optimizers[name], buffers[name],
                          gamma=0.99, lam=0.95, clip_eps=0.2, ent_coef=0.01,
                          adv_norm=False, epochs=epochs, batch_size=batch)
        p, q = first_round_probs(agents)
        p_hist.append(p)
        q_hist.append(q)
        last_stats = stats
    return {
        "lambda": lam, "tau": tau, "seed": seed,
        "final_p": p_hist[-1], "final_q": q_hist[-1],
        "win_rate": last_stats["emperor_win_rate"],
        "obj_return": last_stats["emperor_obj_return"],
        "subj_mean": last_stats["subj_mean"],
        "drift_p": _drift(p_hist), "drift_q": _drift(q_hist),
    }


def plot_phase_diagram(cells: list[dict], out_png: Path) -> None:
    """相图：横轴 λ、纵轴 τ、颜色=皇帝客观期望收益（左）/ 首轮 p̂（右）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lambdas = sorted({c["lambda"] for c in cells})
    taus = sorted({c["tau"] for c in cells})
    by_cell = {(c["lambda"], c["tau"]): c for c in cells}
    z_obj = np.array([[by_cell[(l, t)]["obj_return"] for l in lambdas] for t in taus])
    z_p = np.array([[by_cell[(l, t)]["final_p"] for l in lambdas] for t in taus])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, z, title in zip(axes, (z_obj, z_p),
                            ("emperor obj return", "emperor first-round p_hat")):
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="RdYlBu_r")
        ax.set_xticks(range(len(lambdas)), [f"lambda={l:g}" for l in lambdas])
        ax.set_yticks(range(len(taus)), [f"tau={t:g}" for t in taus])
        ax.set_title(title)
        for i in range(len(taus)):
            for j in range(len(lambdas)):
                ax.text(j, i, f"{z[i, j]:+.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="冒烟：少量格子 × 1 seed")
    ap.add_argument("--steps", type=int, default=40_000, help="每个配置的环境步数")
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    out_dir = Path("data/sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(
        json.dumps(PREDICTIONS, ensure_ascii=False, indent=2))

    lambdas, taus, seeds = LAMBDAS, TAUS, SEEDS
    if args.demo:
        lambdas, taus, seeds = [1.0, 3.0, 5.0], [0.5], [42]
        args.steps = min(args.steps, 8192)  # 冒烟只验管道，不做结论

    runs = []
    t0 = time.time()
    for lam in lambdas:
        for tau in taus:
            for seed in seeds:
                rec = run_config(lam, tau, seed, steps=args.steps, rollout=args.rollout,
                                 epochs=args.epochs, batch=args.batch, lr=args.lr)
                runs.append(rec)
                print(f"λ={lam:g} τ={tau:g} seed={seed} | p={rec['final_p']:.3f} "
                      f"q={rec['final_q']:.3f} win={rec['win_rate']:.3f} "
                      f"ret={rec['obj_return']:+.3f}")

    grouped: dict[tuple[float, float], list[dict]] = {}
    for r in runs:
        grouped.setdefault((r["lambda"], r["tau"]), []).append(r)
    cells = []
    for (lam, tau), rs in sorted(grouped.items()):
        cells.append({
            "lambda": lam, "tau": tau, "n_seeds": len(rs),
            "final_p": float(np.mean([r["final_p"] for r in rs])),
            "final_q": float(np.mean([r["final_q"] for r in rs])),
            "win_rate": float(np.mean([r["win_rate"] for r in rs])),
            "obj_return": float(np.mean([r["obj_return"] for r in rs])),
        })

    # 预测核验（预测写在跑之前，这里是事后对照）
    p_min_l = np.mean([c["final_p"] for c in cells if c["lambda"] == min(lambdas)])
    p_max_l = np.mean([c["final_p"] for c in cells if c["lambda"] == max(lambdas)])
    q_min_t = np.mean([c["final_q"] for c in cells if c["tau"] == min(taus)])
    q_max_t = np.mean([c["final_q"] for c in cells if c["tau"] == max(taus)])
    d_min_t = np.mean([abs(c["final_q"] - 0.5) for c in cells if c["tau"] == min(taus)])
    d_max_t = np.mean([abs(c["final_q"] - 0.5) for c in cells if c["tau"] == max(taus)])
    summary = {
        "predictions": PREDICTIONS,
        "check_lambda": {
            "p_at_lambda_min": round(float(p_min_l), 4),
            "p_at_lambda_max": round(float(p_max_l), 4),
            "direction": "λ↑→p↓ 成立" if p_max_l < p_min_l else "λ↑→p↓ 不成立",
        },
        "check_tau": {
            "q_at_tau_min": round(float(q_min_t), 4),
            "q_at_tau_max": round(float(q_max_t), 4),
            "dist_to_uniform_min_tau": round(float(d_min_t), 4),
            "dist_to_uniform_max_tau": round(float(d_max_t), 4),
            "direction": "τ↑→均匀 成立" if d_max_t < d_min_t else "τ↑→均匀 不成立",
        },
        "cells": cells,
        "runs": runs,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "grid.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    png = out_dir / "phase_diagram.png"
    if len(cells) >= 4:
        plot_phase_diagram(cells, png)
        print(f"相图已存 {png}")
    print(f"网格汇总已存 {out_dir / 'grid.json'}（{len(runs)} 次运行，"
          f"用时 {summary['elapsed_s']}s）")


if __name__ == "__main__":
    main()
