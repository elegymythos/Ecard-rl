"""逐层策略与 exploitability 分析（需要 sweep.py --save-agents 产出的 checkpoint）。

用法：
  python evaluate.py --dir data/runs/run11_identity
  python evaluate.py --dir data/runs/run12_roleswap --save

对每个 checkpoint：
- 输出 k=4..1 的逐层 p_k/q_k（k = 双方剩余平民数）；
- 用动态规划精确计算联合策略值 V_actual；
- 计算皇帝最优反应值 BR_E、奴隶最优反应值 BR_S；
- exploitability = max(0, BR_E - V_actual) + max(0, V_actual - BR_S)。

说明：这个评估不需要重训，只读 checkpoint；因此可以在训练机上直接跑。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env import ECardEnv, EnvConfig, PLAY_CIVILIAN
from ppo import Agent, policy_probs_at


def load_checkpoint(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    agents = {"emperor": Agent(), "slave": Agent()}
    for name in agents:
        agents[name].load_state_dict(data["agents"][name])
        agents[name].eval()
    return agents, data.get("meta", {})


def per_state_probs(agents, reward_loss: float):
    """返回 [(k, p_k, q_k), ...] for k=4..1。"""
    env = ECardEnv(EnvConfig(reward_emperor_loss=reward_loss))
    obs = env.reset()
    out = []
    for k in range(4, 0, -1):
        p, q, _, _ = policy_probs_at(agents, obs, legal_actions=[0, 1])
        out.append((k, p, q))
        if k > 1:
            obs, _, done, _ = env.step(PLAY_CIVILIAN, PLAY_CIVILIAN)
            assert not done
    return out


def joint_value(p_by_k, q_by_k, reward_loss: float) -> float:
    """按 k=1..4 递推：V_0 = reward_loss（双方被迫 A-A）。"""
    v = reward_loss
    for k in range(1, 5):
        p = p_by_k[k]
        q = q_by_k[k]
        v = (
            p * q * reward_loss
            + p * (1 - q) * 1.0
            + (1 - p) * q * 1.0
            + (1 - p) * (1 - q) * v
        )
    return float(v)


def best_response_emperor(q_by_k, reward_loss: float) -> float:
    """皇帝在每层选择 p∈{0,1} 最大化期望值。"""
    v = reward_loss
    for k in range(1, 5):
        q = q_by_k[k]
        v_if_c = q * 1.0 + (1 - q) * v          # 皇帝出平民
        v_if_a = q * reward_loss + (1 - q) * 1.0  # 皇帝出王牌
        v = max(v_if_c, v_if_a)
    return float(v)


def best_response_slave(p_by_k, reward_loss: float) -> float:
    """奴隶在每层选择 q∈{0,1} 最小化皇帝期望值（零和）。"""
    v = reward_loss
    for k in range(1, 5):
        p = p_by_k[k]
        v_if_c = p * 1.0 + (1 - p) * v          # 奴隶出平民
        v_if_a = p * reward_loss + (1 - p) * 1.0  # 奴隶出王牌
        v = min(v_if_c, v_if_a)
    return float(v)


def analyze_checkpoint(path: Path, swap_eval: bool = False) -> dict:
    agents, meta = load_checkpoint(path)
    if swap_eval:
        # 事后交换角色：用奴隶网络当皇帝、皇帝网络当奴隶。
        # 如果 q-p 是“网络身份”造成的，交换后 q-p 应反号；
        # 如果 q-p 是“角色”造成的，交换后 q-p 应保持为正。
        agents = {"emperor": agents["slave"], "slave": agents["emperor"]}
    reward_loss = float(meta.get("reward_loss", -5.0))
    states = per_state_probs(agents, reward_loss)
    p_by_k = {k: p for k, p, _ in states}
    q_by_k = {k: q for k, _, q in states}
    v_actual = joint_value(p_by_k, q_by_k, reward_loss)
    br_e = best_response_emperor(q_by_k, reward_loss)
    br_s = best_response_slave(p_by_k, reward_loss)
    emp_exp = max(0.0, br_e - v_actual)
    slave_exp = max(0.0, v_actual - br_s)
    return {
        "checkpoint": path.name,
        "meta": meta,
        "k": [{"k": k, "p": p, "q": q} for k, p, q in states],
        "v_actual": v_actual,
        "br_emperor": br_e,
        "br_slave": br_s,
        "exploit_emperor": emp_exp,
        "exploit_slave": slave_exp,
        "nash_conv": emp_exp + slave_exp,
        "q_minus_p_first": states[0][2] - states[0][1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run 目录，内含 checkpoints/*.pt")
    ap.add_argument("--pattern", default="lam*.pt", help="checkpoint 文件名模式，默认 lam*.pt")
    ap.add_argument("--swap-eval", action="store_true",
                    help="事后交换皇帝/奴隶网络再做一次评估，用于区分 q-p 来自网络身份还是角色")
    ap.add_argument("--save", action="store_true", help="把结果写到 <dir>/evaluation.json")
    args = ap.parse_args()

    run_dir = Path(args.dir)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"未找到 {ckpt_dir}；请用 sweep.py --save-agents 重新运行")
    paths = sorted(ckpt_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"{ckpt_dir} 下没有匹配 {args.pattern} 的 checkpoint")

    results = [analyze_checkpoint(p) for p in paths]
    swapped = [analyze_checkpoint(p, swap_eval=True) for p in paths] if args.swap_eval else []
    for r, sw in zip(results, swapped or [None] * len(results)):
        meta = r["meta"]
        print(f"\n=== {r['checkpoint']} ===")
        print(f"meta: lam={meta.get('lambda')} tau={meta.get('tau')} seed={meta.get('seed')} "
              f"reward_loss={meta.get('reward_loss')} min_ent={meta.get('min_ent')} "
              f"swap_roles={meta.get('swap_roles')} shared_policy={meta.get('shared_policy')}")
        print("k :     p_k     q_k")
        for row in r["k"]:
            print(f"{row['k']} : {row['p']:.4f}  {row['q']:.4f}")
        print(f"V_actual={r['v_actual']:.4f}  BR_E={r['br_emperor']:.4f}  "
              f"BR_S={r['br_slave']:.4f}  exploit_E={r['exploit_emperor']:.4f}  "
              f"exploit_S={r['exploit_slave']:.4f}  nash_conv={r['nash_conv']:.4f}")
        if sw is not None:
            print("--- swapped roles (eval only) ---")
            for row in sw["k"]:
                print(f"{row['k']} : {row['p']:.4f}  {row['q']:.4f}")
            print(f"swapped q-p(first)={sw['q_minus_p_first']:.4f}  "
                  f"nash_conv={sw['nash_conv']:.4f}")

    if len(results) > 1:
        print("\n=== aggregate ===")
        for metric in ["v_actual", "br_emperor", "br_slave", "exploit_emperor",
                       "exploit_slave", "nash_conv", "q_minus_p_first"]:
            vals = [r[metric] for r in results]
            print(f"{metric}: mean={np.mean(vals):.4f} sd={np.std(vals):.4f} "
                  f"min={min(vals):.4f} max={max(vals):.4f}")
        if swapped:
            vals = [r["q_minus_p_first"] for r in swapped]
            print(f"swapped q-p(first): mean={np.mean(vals):.4f} sd={np.std(vals):.4f} "
                  f"min={min(vals):.4f} max={max(vals):.4f}")
        print("per-k mean p/q:")
        for k in (4, 3, 2, 1):
            ps = [next(row["p"] for row in r["k"] if row["k"] == k) for r in results]
            qs = [next(row["q"] for row in r["k"] if row["k"] == k) for r in results]
            print(f"k={k}: p={np.mean(ps):.4f} q={np.mean(qs):.4f}")

    if args.save:
        payload = {"normal": results}
        if swapped:
            payload["swapped_roles"] = swapped
        out = run_dir / "evaluation.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存 {out}")


if __name__ == "__main__":
    main()
