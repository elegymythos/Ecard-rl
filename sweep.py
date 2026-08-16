"""阶段 4：心理参数扫描与自博弈动力学诊断。

主网格：λ ∈ {1, 2, 3, 5} × τ ∈ {0.5, 0.75, 1.0} × 5 seeds。
- τ 不用 0.1：T&K 单参数权重函数在 γ<0.5 时退化（run02 已证明）。
- 皇帝效用：prospect(λ)，无概率权重——隔离损失厌恶；
- 奴隶效用：prospect(γ=τ, λ=1)，窗口/ref/off 概率权重——隔离概率扭曲；
- 每个配置记录：首轮 p̂、q̂、胜率、客观期望、末段漂移、末段 std、末段 peak-to-peak。
- 预测先写；支持 `--predictions-file` 加载运行专用预测。
- 支持 `--reward-loss`（对称收益消融）与 `--slave-lam`（非对称 λ 扩展）。

收敛诊断：
- 主判据：|Δp|、|Δq| < conv_tol 且末段窗口 p_std、q_std < conv_std；
- grid.json 额外输出 `convergence_by_std`（不同 std 阈值的敏感性）。
- 术语约定：满足主判据称为「弱平稳」；固定点收敛需更严格阈值与更长预算支持。

用法：
  python sweep.py --demo
  python sweep.py --steps 400000 --min-ent 0.5
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

from env import ECardEnv, EnvConfig
from ppo import Agent, collect_rollout, first_round_stats, update_policy
from utility import make_prospect

LAMBDAS = [1.0, 2.0, 3.0, 5.0]
TAUS = [0.5, 0.75, 1.0]  # 修：0.1 已废弃，权重函数在 γ<0.5 退化
SEEDS = [42, 43, 44, 45, 46]

PREDICTIONS = {
    "lambda_up_emperor_p_down": "λ↑ → 皇帝首轮 p↓（损失厌恶让他躲 A-A）",
    "tau_up_toward_uniform": "τ↑ → 策略逼近均匀随机",
}


def _load_predictions(path: Path | None) -> dict:
    """读取运行专用预测；没有则用默认 PREDICTIONS。"""
    if path is None:
        return dict(PREDICTIONS)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"预测文件必须是 JSON 对象: {path}")
    return data


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


def _tail_range(series: list[float], tail: float = 0.2) -> float:
    """末段窗口 peak-to-peak 幅度（用于报告弱平稳下的残余振荡）。"""
    if not series:
        return float("nan")
    cut = min(len(series), max(2, int(len(series) * tail))) if len(series) >= 2 else 1
    window = series[-cut:]
    return float(np.max(window) - np.min(window))


def run_config(lam: float, tau: float, seed: int, *, steps: int,
               rollout: int, epochs: int, batch: int, lr: float,
               ent_coef: float, ret_norm: bool, min_ent: float,
               weight_mode: str, conv_tol: float, conv_std: float,
               alpha: float, beta: float, reward_loss: float = -5.0,
               slave_lam: float = 1.0) -> dict:
    """单格子单 seed 的自博弈训练，返回最终指标。

    reward_loss：A-A 时皇帝的客观损失（默认 -5；设为 -1 即对称收益消融）。
    slave_lam：奴隶前景理论 λ（默认 1.0；用于非对称 λ 扩展）。
    """
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = ECardEnv(EnvConfig(reward_emperor_loss=reward_loss))
    agents = {"emperor": Agent(), "slave": Agent()}
    optimizers = {k: torch.optim.Adam(agents[k].parameters(), lr=lr) for k in agents}
    slave_weighting = {"off": False, "window": True, "ref": "ref"}[weight_mode]
    utilities = {
        # 皇帝变 λ（损失厌恶）；奴隶默认 λ=1，只变 τ——隔离效应；
        # --slave-lam 可开启非对称 λ 扩展。
        "emperor": make_prospect(lam=lam, alpha=alpha, beta=beta,
                                 gamma=1.0, use_weighting=False),
        "slave": make_prospect(lam=slave_lam, alpha=alpha, beta=beta,
                               gamma=tau, use_weighting=slave_weighting),
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
    p_tail_range = _tail_range(hist["p"])
    q_tail_range = _tail_range(hist["q"])
    return {
        "lambda": lam, "tau": tau, "seed": seed,
        "reward_loss": reward_loss, "slave_lam": slave_lam,
        "final_p": hist["p"][-1], "final_q": hist["q"][-1],  # 与 run01 兼容的旧字段
        "p": p_mean, "q": q_mean, "p_std": p_std, "q_std": q_std,
        "p_tail_range": p_tail_range, "q_tail_range": q_tail_range,
        "win_rate": win_mean, "obj_return": ret_mean, "obj_return_std": ret_std,
        "ent_e_init": ent_e_mean, "ent_s_init": ent_s_mean,
        "subj_std": {"emperor": subj_std_e_mean, "slave": subj_std_s_mean},
        "drift_p": drift_p, "drift_q": drift_q,
        # 修（2026-08-15）：弱平稳必须同时满足「漂移小」和「末段窗口 std 小」。
        # 旧判据只看 |Δ|，对周期振荡失明——run05 的 23/60「收敛」全部是假阳性
        # （q 全程 std 0.25+ 的极限环，前后窗口均值巧合相等）。
        "converged": (abs(drift_p) < conv_tol and abs(drift_q) < conv_tol
                      and p_std < conv_std and q_std < conv_std),
        "weakly_stationary": (abs(drift_p) < conv_tol and abs(drift_q) < conv_tol
                              and p_std < conv_std and q_std < conv_std),
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
    # 归一化换行再算哈希：git autocrlf 会把工作区文件转成 CRLF，
    # 直接按原始字节算哈希会导致跨平台对不上（run02/run03 已踩过）。
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:12]


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
    ap.add_argument("--alpha", type=float, default=0.88,
                    help="前景理论收益曲率 α（=β；α=β=1 时为真 identity，用于曲率消融）")
    ap.add_argument("--beta", type=float, default=0.88,
                    help="前景理论损失曲率 β（=α）")
    ap.add_argument("--conv-tol", type=float, default=0.05,
                    help="弱平稳门：|末段漂移| 低于此值才算弱平稳")
    ap.add_argument("--conv-std", type=float, default=0.1,
                    help="弱平稳门（修）：末段窗口 std 低于此值——防极限环假收敛"
                         "（旧判据只查 |Δ|，对周期振荡失明，run05 的 23/60 全为假阳性）")
    ap.add_argument("--cells", default=None,
                    help="只跑指定格子，如 '1-0.5,5-1.0'（λ-τ 对，逗号分隔）")
    ap.add_argument("--seeds", default=None, help="只跑指定种子，如 '42,43'")
    ap.add_argument("--reward-loss", type=float, default=-5.0,
                    help="A-A 时皇帝的客观损失（默认 -5；设为 -1 即对称收益消融）")
    ap.add_argument("--slave-lam", type=float, default=1.0,
                    help="奴隶前景理论 λ（默认 1.0；非对称 λ 扩展用）")
    ap.add_argument("--predictions-file", default=None,
                    help="运行专用预测 JSON 文件（默认写内置 PREDICTIONS）")
    args = ap.parse_args()

    sweep_dir = Path("data/sweep")
    out_dir = sweep_dir if args.name is None else Path("data/runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    predictions = _load_predictions(Path(args.predictions_file) if args.predictions_file else None)
    predictions_text = json.dumps(predictions, ensure_ascii=False, indent=2)
    (out_dir / "predictions.json").write_text(predictions_text, encoding="utf-8")
    if out_dir != sweep_dir:
        (sweep_dir / "predictions.json").write_text(predictions_text, encoding="utf-8")

    lambdas, taus, seeds = LAMBDAS, TAUS, SEEDS
    if args.cells:
        grid_pairs = [(float(a), float(b)) for pair in args.cells.split(",")
                      for a, b in [pair.split("-")]]
    else:
        grid_pairs = [(l, t) for l in lambdas for t in taus]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    if args.demo and not args.cells:
        lambdas, taus, seeds = [1.0, 3.0, 5.0], [0.5], [42]
        grid_pairs = [(l, t) for l in lambdas for t in taus]
        args.steps = min(args.steps, 8192)  # 冒烟只验管道，不做结论

    started_at = datetime.now().isoformat(timespec="seconds")
    # 溯源必须在 run 开始时记录一次：运行中若脚本被修改，meta 才不会把
    # “跑完后的文件”误写成“实际执行的文件”（run06_curv 的 NaN 教训）。
    started_git = _git_head()
    started_dirty = _git_dirty()
    started_sha = _file_sha(Path(__file__))
    runs = []
    runs_jsonl = out_dir / "runs.jsonl"
    # 防重：append 模式不查重会重复记录（run05/run06_curv 已踩过）。
    existing_keys: set[tuple[float, float, int]] = set()
    if runs_jsonl.exists():
        for line in runs_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                existing_keys.add((float(row["lambda"]), float(row["tau"]), int(row["seed"])))
            except Exception:
                pass
    t0 = time.time()
    for lam, tau in grid_pairs:
        for seed in seeds:
            key = (lam, tau, seed)
            if key in existing_keys:
                print(f"跳过重复记录 λ={lam:g} τ={tau:g} seed={seed}（runs.jsonl 已存在）")
                continue
            rec = run_config(lam, tau, seed, steps=args.steps, rollout=args.rollout,
                             epochs=args.epochs, batch=args.batch, lr=args.lr,
                             ent_coef=args.ent_coef, ret_norm=args.ret_norm,
                             min_ent=args.min_ent, weight_mode=args.weight_mode,
                             conv_tol=args.conv_tol, conv_std=args.conv_std,
                             alpha=args.alpha, beta=args.beta,
                             reward_loss=args.reward_loss, slave_lam=args.slave_lam)
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
    # 修（2026-08-16）：--cells 部分网格只覆盖部分 λ/τ 时，min/max 必须用
    # 实际出现的值，否则空切片产生 NaN（Mean of empty slice 警告）。
    present_l = sorted({c["lambda"] for c in cells})
    present_t = sorted({c["tau"] for c in cells})
    p_min_l = np.mean([c["final_p"] for c in cells if c["lambda"] == present_l[0]])
    p_max_l = np.mean([c["final_p"] for c in cells if c["lambda"] == present_l[-1]])
    q_min_t = np.mean([c["final_q"] for c in cells if c["tau"] == present_t[0]])
    q_max_t = np.mean([c["final_q"] for c in cells if c["tau"] == present_t[-1]])
    d_min_t = np.mean([abs(c["final_q"] - 0.5) for c in cells if c["tau"] == present_t[0]])
    d_max_t = np.mean([abs(c["final_q"] - 0.5) for c in cells if c["tau"] == present_t[-1]])
    # 中位数版核验：均值被双峰 seed 污染，中位数是更稳的对照
    p_med_min = np.median([c["final_p_median"] for c in cells if c["lambda"] == present_l[0]])
    p_med_max = np.median([c["final_p_median"] for c in cells if c["lambda"] == present_l[-1]])
    q_med_min = np.median([c["final_q_median"] for c in cells if c["tau"] == present_t[0]])
    q_med_max = np.median([c["final_q_median"] for c in cells if c["tau"] == present_t[-1]])
    d_med_min = np.median([abs(c["final_q_median"] - 0.5) for c in cells if c["tau"] == present_t[0]])
    d_med_max = np.median([abs(c["final_q_median"] - 0.5) for c in cells if c["tau"] == present_t[-1]])
    # 修：均匀随机的度量改用初始状态策略熵（均匀 = ln2 ≈ 0.6931）
    ln2 = math.log(2.0)
    e_med_min = np.median([ln2 - c["ent_s_init_median"] for c in cells if c["tau"] == present_t[0]])
    e_med_max = np.median([ln2 - c["ent_s_init_median"] for c in cells if c["tau"] == present_t[-1]])
    ended_git = _git_head()
    ended_dirty = _git_dirty()
    ended_sha = _file_sha(Path(__file__))
    summary = {
        "predictions": predictions,
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
            "conv_std": args.conv_std,
            "alpha": args.alpha,
            "beta": args.beta,
            "reward_loss": args.reward_loss,
            "slave_lam": args.slave_lam,
            "seeds": seeds,
            "started_at": started_at,
            "git_commit_start": started_git,
            "git_dirty_start": started_dirty,
            "sweep_py_sha256_start": started_sha,
            "git_commit_end": ended_git,
            "git_dirty_end": ended_dirty,
            "sweep_py_sha256_end": ended_sha,
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
        "n_weakly_stationary": sum(1 for r in runs if r.get("weakly_stationary")),
        "convergence_by_std": {
            str(std_thr): sum(
                1 for r in runs
                if abs(r.get("drift_p", 1)) < args.conv_tol
                and abs(r.get("drift_q", 1)) < args.conv_tol
                and r.get("p_std", 1) < std_thr
                and r.get("q_std", 1) < std_thr
            )
            for std_thr in [0.15, 0.12, 0.10, 0.08, 0.06, 0.05]
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    grid_text = json.dumps(summary, ensure_ascii=False, indent=2)
    (out_dir / "grid.json").write_text(grid_text, encoding="utf-8")
    if out_dir != sweep_dir:
        (sweep_dir / "grid.json").write_text(grid_text, encoding="utf-8")
    print(f"弱平稳门（|Δ|<{args.conv_tol} 且 末段std<{args.conv_std}）："
          f"{summary['n_weakly_stationary']}/{len(runs)} 次运行弱平稳")
    if summary["n_weakly_stationary"] == 0:
        print("警告：没有任何运行在弱平稳容差内通过——结果只是有限预算快照，不是均衡。")

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
