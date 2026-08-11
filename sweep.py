"""阶段 4：心理参数扫描（初版，AI 写，待你重写）。

网格：λ ∈ {1, 2, 3, 5} × τ ∈ {0.5, 0.75, 1.0} × 5 seeds = 60 次自博弈训练。
- τ 取 0.5/0.75/1.0，不用 0.1：T&K 单参数权重函数在 γ<~0.3 退化（w(p)≈0，
  奖励信号被抹平，熵正则主导——run02 的 τ=0.1 列就是这样被污染的）。
- 皇帝效用：prospect(λ)，α=β=0.88，无概率权重——隔离损失厌恶效应；
- 奴隶效用：prospect(τ)，γ=τ，λ=1，滑动窗口概率权重——隔离概率扭曲效应；
- 每个配置记录：首轮 p̂、q̂、皇帝胜率、客观期望、主观效用均值、末段漂移。
- 预测先写（data/sweep/predictions.json）：
    λ↑ → 皇帝 p↓；τ↑ → 策略逼近均匀随机。
- 输出：data/sweep/grid.json + data/sweep/phase_diagram.png。

用法：
  python sweep.py --demo          # 冒烟：少量格子 × 1 seed（验证管道）
  python sweep.py --steps 40000 --name run02   # 正式扫描并归档
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from env import ECardEnv
from ppo import Agent, collect_rollout, first_round_stats, update_policy
from utility import make_prospect

LAMBDAS = [1.0, 2.0, 3.0, 5.0]
TAUS = [0.5, 0.75, 1.0]  # 修：0.1 已废弃，权重函数在 γ<0.5 退化
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


def _window_stats(series: list[float], tail: float = 0.2) -> tuple[float, float]:
    """最后 tail 比例的均值与标准差（修：最终指标不再用单点快照）。"""
    if not series:
        return float("nan"), float("nan")
    cut = min(len(series), max(2, int(len(series) * tail))) if len(series) >= 2 else 1
    window = series[-cut:]
    return float(np.mean(window)), float(np.std(window))


def run_config(lam: float, tau: float, seed: int, *, steps: int,
               rollout: int, epochs: int, batch: int, lr: float,
               ent_coef: float, ret_norm: bool, min_ent: float,
               weight_mode: str, conv_tol: float) -> dict:
    """单格子单 seed 的自博弈训练，返回最终指标。"""
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = ECardEnv()
    agents = {"emperor": Agent(), "slave": Agent()}
    optimizers = {k: torch.optim.Adam(agents[k].parameters(), lr=lr) for k in agents}
    slave_weighting = {"off": False, "window": True, "ref": "ref"}[weight_mode]
    utilities = {
        # 皇帝只变 λ，奴隶只变 τ——隔离效应，重写时别混掉
        "emperor": make_prospect(lam=lam, gamma=1.0, use_weighting=False),
        "slave": make_prospect(lam=1.0, gamma=tau, use_weighting=slave_weighting),
    }
    n_updates = max(1, steps // rollout)
    hist = {k: [] for k in ["p", "q", "ent_e", "ent_s", "win", "ret",
                            "subj_std_e", "subj_std_s"]}
    for _ in range(n_updates):
        buffers, stats = collect_rollout(env, agents, utilities, rollout,
                                         gamma=0.99, lam=0.95)
        for name in agents:
            update_policy(agents[name], optimizers[name], buffers[name],
                          gamma=0.99, lam=0.95, clip_eps=0.2, ent_coef=ent_coef,
                          adv_norm=False, epochs=epochs, batch_size=batch,
                          ret_norm=ret_norm, min_ent=min_ent)
        p, q, ent_e, ent_s = first_round_stats(agents)
        hist["p"].append(p)
        hist["q"].append(q)
        hist["ent_e"].append(ent_e)
        hist["ent_s"].append(ent_s)
        hist["win"].append(stats["emperor_win_rate"])
        hist["ret"].append(stats["emperor_obj_return"])
        hist["subj_std_e"].append(stats["subj_std"]["emperor"])
        hist["subj_std_s"].append(stats["subj_std"]["slave"])

    p_mean, p_std = _window_stats(hist["p"])
    q_mean, q_std = _window_stats(hist["q"])
    win_mean, _ = _window_stats(hist["win"])
    ret_mean, ret_std = _window_stats(hist["ret"])
    ent_e_mean, _ = _window_stats(hist["ent_e"])
    ent_s_mean, _ = _window_stats(hist["ent_s"])
    subj_std_e_mean, _ = _window_stats(hist["subj_std_e"])
    subj_std_s_mean, _ = _window_stats(hist["subj_std_s"])
    drift_p = _drift(hist["p"])
    drift_q = _drift(hist["q"])
    return {
        "lambda": lam, "tau": tau, "seed": seed,
        "final_p": hist["p"][-1], "final_q": hist["q"][-1],  # 与 run01 兼容的旧字段
        "p": p_mean, "q": q_mean, "p_std": p_std, "q_std": q_std,
        "win_rate": win_mean, "obj_return": ret_mean, "obj_return_std": ret_std,
        "ent_e_init": ent_e_mean, "ent_s_init": ent_s_mean,
        "subj_std": {"emperor": subj_std_e_mean, "slave": subj_std_s_mean},
        "drift_p": drift_p, "drift_q": drift_q,
        "converged": abs(drift_p) < conv_tol and abs(drift_q) < conv_tol,
        "elapsed_s": round(time.time() - t0, 2),
        "p_series": hist["p"], "q_series": hist["q"],  # 完整序列进 runs.jsonl
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


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _git_dirty() -> bool:
    """工作区是否有未提交改动（写进 meta，防止 run 的版本对不上）。"""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                             text=True, check=True).stdout.strip()
        return bool(out)
    except Exception:
        return True  # 拿不到 git 状态就按「不确定」处理


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="冒烟：少量格子 × 1 seed")
    ap.add_argument("--name", default=None, help="运行名：归档到 data/runs/<name>")
    ap.add_argument("--steps", type=int, default=40_000, help="每个配置的环境步数")
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    ap.add_argument("--min-ent", type=float, default=0.0,
                    help="最小熵约束（>0 时加 hinge 惩罚，防策略塌缩）")
    ap.add_argument("--ret-norm", action="store_true",
                    help="优势+回报标准化（心理运行奖励尺度大时的稳定性开关）")
    ap.add_argument("--weight-mode", choices=["window", "ref", "off"], default="window",
                    help="奴隶概率权重：window=滑动窗口（策略反馈环）/ ref=固定参考概率 / off=无")
    ap.add_argument("--conv-tol", type=float, default=0.05,
                    help="收敛门：|末段漂移| 低于此值才算站住")
    args = ap.parse_args()

    sweep_dir = Path("data/sweep")
    out_dir = sweep_dir if args.name is None else Path("data/runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    predictions_text = json.dumps(PREDICTIONS, ensure_ascii=False, indent=2)
    (out_dir / "predictions.json").write_text(predictions_text, encoding="utf-8")
    if out_dir != sweep_dir:
        (sweep_dir / "predictions.json").write_text(predictions_text, encoding="utf-8")

    lambdas, taus, seeds = LAMBDAS, TAUS, SEEDS
    if args.demo:
        lambdas, taus, seeds = [1.0, 3.0, 5.0], [0.5], [42]
        args.steps = min(args.steps, 8192)  # 冒烟只验管道，不做结论

    started_at = datetime.now().isoformat(timespec="seconds")
    runs = []
    runs_jsonl = out_dir / "runs.jsonl"
    t0 = time.time()
    for lam in lambdas:
        for tau in taus:
            for seed in seeds:
                rec = run_config(lam, tau, seed, steps=args.steps, rollout=args.rollout,
                                 epochs=args.epochs, batch=args.batch, lr=args.lr,
                                 ent_coef=args.ent_coef, ret_norm=args.ret_norm,
                                 min_ent=args.min_ent, weight_mode=args.weight_mode,
                                 conv_tol=args.conv_tol)
                rec_log = {k: v for k, v in rec.items()
                           if k not in ("p_series", "q_series")}
                runs.append(rec_log)
                with runs_jsonl.open("a", encoding="utf-8") as f:  # 每跑完一条立即落盘
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"λ={lam:g} τ={tau:g} seed={seed} | p={rec['p']:.3f}±{rec['p_std']:.3f} "
                      f"q={rec['q']:.3f}±{rec['q_std']:.3f} win={rec['win_rate']:.3f} "
                      f"ret={rec['obj_return']:+.3f}")

    grouped: dict[tuple[float, float], list[dict]] = {}
    for r in runs:
        grouped.setdefault((r["lambda"], r["tau"]), []).append(r)
    cells = []
    for (lam, tau), rs in sorted(grouped.items()):
        cells.append({
            "lambda": lam, "tau": tau, "n_seeds": len(rs),
            # 修：聚合用的是末段窗口均值（r["p"]/r["q"]），不再是单点快照
            "final_p": float(np.mean([r["p"] for r in rs])),
            "final_q": float(np.mean([r["q"] for r in rs])),
            "final_p_std": float(np.std([r["p"] for r in rs])),
            "final_q_std": float(np.std([r["q"] for r in rs])),
            "final_p_median": float(np.median([r["p"] for r in rs])),
            "final_q_median": float(np.median([r["q"] for r in rs])),
            "win_rate": float(np.mean([r["win_rate"] for r in rs])),
            "obj_return": float(np.mean([r["obj_return"] for r in rs])),
            "obj_return_std": float(np.std([r["obj_return"] for r in rs])),
            "ent_s_init_mean": float(np.mean([r["ent_s_init"] for r in rs])),
            "ent_s_init_median": float(np.median([r["ent_s_init"] for r in rs])),
        })

    # 预测核验（预测写在跑之前，这里是事后对照）
    p_min_l = np.mean([c["final_p"] for c in cells if c["lambda"] == min(lambdas)])
    p_max_l = np.mean([c["final_p"] for c in cells if c["lambda"] == max(lambdas)])
    q_min_t = np.mean([c["final_q"] for c in cells if c["tau"] == min(taus)])
    q_max_t = np.mean([c["final_q"] for c in cells if c["tau"] == max(taus)])
    d_min_t = np.mean([abs(c["final_q"] - 0.5) for c in cells if c["tau"] == min(taus)])
    d_max_t = np.mean([abs(c["final_q"] - 0.5) for c in cells if c["tau"] == max(taus)])
    # 中位数版核验：均值被双峰 seed 污染，中位数是更稳的对照
    p_med_min = np.median([c["final_p_median"] for c in cells if c["lambda"] == min(lambdas)])
    p_med_max = np.median([c["final_p_median"] for c in cells if c["lambda"] == max(lambdas)])
    q_med_min = np.median([c["final_q_median"] for c in cells if c["tau"] == min(taus)])
    q_med_max = np.median([c["final_q_median"] for c in cells if c["tau"] == max(taus)])
    d_med_min = np.median([abs(c["final_q_median"] - 0.5) for c in cells if c["tau"] == min(taus)])
    d_med_max = np.median([abs(c["final_q_median"] - 0.5) for c in cells if c["tau"] == max(taus)])
    # 修：均匀随机的度量改用初始状态策略熵（均匀 = ln2 ≈ 0.6931）
    ln2 = math.log(2.0)
    e_med_min = np.median([ln2 - c["ent_s_init_median"] for c in cells if c["tau"] == min(taus)])
    e_med_max = np.median([ln2 - c["ent_s_init_median"] for c in cells if c["tau"] == max(taus)])
    summary = {
        "predictions": PREDICTIONS,
        "meta": {
            "run_name": args.name,
            "steps_per_run": args.steps,
            "rollout": args.rollout,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "lr": args.lr,
            "ent_coef": args.ent_coef,
            "min_ent": args.min_ent,
            "ret_norm": args.ret_norm,
            "adv_norm": False,
            "weight_mode": args.weight_mode,
            "weight_ref_prob": 0.2,
            "conv_tol": args.conv_tol,
            "git_dirty": _git_dirty(),
            "seeds": seeds,
            "started_at": started_at,
            "git_commit": _git_head(),
            "sweep_py_sha256": _file_sha(Path(__file__)),
            "encoding": "utf-8",
        },
        "check_lambda": {
            "p_at_lambda_min": round(float(p_min_l), 4),
            "p_at_lambda_max": round(float(p_max_l), 4),
            "direction": "λ↑→p↓ 成立" if p_max_l < p_min_l else "λ↑→p↓ 不成立",
        },
        "check_lambda_median": {
            "p_median_at_lambda_min": round(float(p_med_min), 4),
            "p_median_at_lambda_max": round(float(p_med_max), 4),
            "direction": "λ↑→p↓ 成立" if p_med_max < p_med_min else "λ↑→p↓ 不成立",
        },
        "check_tau": {
            "q_at_tau_min": round(float(q_min_t), 4),
            "q_at_tau_max": round(float(q_max_t), 4),
            "dist_to_uniform_min_tau": round(float(d_min_t), 4),
            "dist_to_uniform_max_tau": round(float(d_max_t), 4),
            "direction": "τ↑→均匀 成立" if d_max_t < d_min_t else "τ↑→均匀 不成立",
        },
        "check_tau_median": {
            "q_median_at_tau_min": round(float(q_med_min), 4),
            "q_median_at_tau_max": round(float(q_med_max), 4),
            "dist_to_uniform_min_tau": round(float(d_med_min), 4),
            "dist_to_uniform_max_tau": round(float(d_med_max), 4),
            "direction": "τ↑→均匀 成立" if d_med_max < d_med_min else "τ↑→均匀 不成立",
        },
        "check_tau_entropy": {
            "note": "均匀随机 = 初始状态策略熵 ln2≈0.6931；距离 = ln2 - 熵",
            "dist_uniform_median_at_tau_min": round(float(e_med_min), 4),
            "dist_uniform_median_at_tau_max": round(float(e_med_max), 4),
            "direction": "τ↑→均匀 成立" if e_med_max < e_med_min else "τ↑→均匀 不成立",
        },
        "cells": cells,
        "runs": runs,
        "n_converged": sum(1 for r in runs if r.get("converged")),
        "elapsed_s": round(time.time() - t0, 1),
    }
    grid_text = json.dumps(summary, ensure_ascii=False, indent=2)
    (out_dir / "grid.json").write_text(grid_text, encoding="utf-8")
    if out_dir != sweep_dir:
        (sweep_dir / "grid.json").write_text(grid_text, encoding="utf-8")
    print(f"收敛门（tol={args.conv_tol}）：{summary['n_converged']}/{len(runs)} 次运行站住")
    if summary["n_converged"] == 0:
        print("警告：没有任何运行在容差内收敛——结果只是有限预算快照，不是均衡。")

    png = out_dir / "phase_diagram.png"
    if len(cells) >= 4:
        plot_phase_diagram(cells, png)
        if out_dir != sweep_dir:
            import shutil
            shutil.copyfile(png, sweep_dir / "phase_diagram.png")
        print(f"相图已存 {png}")
    print(f"网格汇总已存 {out_dir / 'grid.json'}（{len(runs)} 次运行，"
          f"用时 {summary['elapsed_s']}s，UTF-8 + 元数据）")


if __name__ == "__main__":
    main()
