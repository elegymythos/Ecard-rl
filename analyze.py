"""合并运行目录 + seed 聚类统计（只读分析，不训练）。

用法：
  python analyze.py --all
  python analyze.py --run06
  python analyze.py --sym
  python analyze.py --ref
  python analyze.py --compare-sym-curv

设计要点：
- run06 的 60 个 run 只复用 5 个 seed；所有“显著性”都要同时看 seed 级聚合，
  不能把 60 个 run 当作 60 个独立样本。
- 不依赖 scipy；p 值用置换检验/符号检验/聚类 bootstrap 近似。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "runs"


def load_runs(name: str) -> list[dict]:
    path = DATA / name / "runs.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def has_run(name: str) -> bool:
    return (DATA / name / "runs.jsonl").exists()


def merge_runs(names: list[str], keys=("lambda", "tau", "seed")) -> list[dict]:
    seen = set()
    out = []
    for name in names:
        for r in load_runs(name):
            k = tuple(r.get(k) for k in keys)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    return out


def weak(r: dict) -> bool:
    return (
        abs(r.get("drift_p", 1.0)) < 0.05
        and abs(r.get("drift_q", 1.0)) < 0.05
        and r.get("p_std", 1.0) < 0.1
        and r.get("q_std", 1.0) < 0.1
    )


def mean(xs):
    return st.mean(xs)


def stdev(xs):
    return st.stdev(xs) if len(xs) > 1 else float("nan")


def cluster_bootstrap_ci(rows, stat, group_key="seed", n=20000, seed=0):
    """按 seed 聚类 bootstrap：重抽 seed，保留该 seed 的全部 run。"""
    rng = random.Random(seed)
    groups = sorted({r.get(group_key) for r in rows})
    if len(groups) < 2:
        raise ValueError("cluster bootstrap needs at least two groups")
    vals = []
    for _ in range(n):
        chosen = set(rng.choices(groups, k=len(groups)))
        sample = [r for r in rows if r.get(group_key) in chosen]
        vals.append(stat(sample))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def permutation_p(a, b, n=200000, seed=0):
    """两独立样本均值差的置换检验（双尾近似 p）。"""
    rng = random.Random(seed)
    allv = list(a) + list(b)
    na = len(a)
    obs = abs(mean(a) - mean(b))
    cnt = 0
    for _ in range(n):
        rng.shuffle(allv)
        aa = allv[:na]
        bb = allv[na:]
        if abs(mean(aa) - mean(bb)) >= obs - 1e-12:
            cnt += 1
    return max(1.0, cnt) / n


def paired_permutation_p(a, b, n=200000, seed=0):
    """配对样本均值差的符号置换检验（双尾近似 p）。"""
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    obs = abs(mean(diffs))
    cnt = 0
    for _ in range(n):
        signs = [1 if rng.random() < 0.5 else -1 for _ in diffs]
        d = [s * v for s, v in zip(signs, diffs)]
        if abs(mean(d)) >= obs - 1e-12:
            cnt += 1
    return max(1.0, cnt) / n


def sign_test_p(n_pos: int, n_total: int, two_sided=True) -> float:
    """二项符号检验。

    n_pos = 正向个数；返回在 H0:p=0.5 下观察到至少 max(n_pos, n_total-n_pos)
    个同号的概率。two_sided=True 时按常见保守做法取单侧的两倍（上限 1）。
    """
    if n_total <= 0:
        return 1.0
    k = max(n_pos, n_total - n_pos)
    one = sum(math.comb(n_total, i) for i in range(k, n_total + 1)) * (0.5 ** n_total)
    if two_sided:
        return min(1.0, 2.0 * one)
    return one


def _fmt(x: float, nd=4) -> str:
    return f"{x:.{nd}f}"


def report_run06() -> None:
    if not has_run("run06"):
        print("\n[run06] 未运行/未复制到本机，跳过")
        return
    rows = load_runs("run06")
    print("=" * 70)
    print("run06 合并统计（run 级 + seed 聚类）")
    print("=" * 70)
    print(f"runs={len(rows)}, seeds={sorted({r['seed'] for r in rows})}")

    def p_effect(sample):
        a = [r["p"] for r in sample if r["lambda"] == 1]
        b = [r["p"] for r in sample if r["lambda"] == 5]
        return mean(b) - mean(a)

    def q_effect(sample):
        a = [r["q"] for r in sample if r["lambda"] == 1]
        b = [r["q"] for r in sample if r["lambda"] == 5]
        return mean(b) - mean(a)

    def tau_effect(sample):
        a = [r["q"] for r in sample if r["tau"] == 0.5]
        b = [r["q"] for r in sample if r["tau"] == 1.0]
        return mean(b) - mean(a)

    def qp_mean(sample):
        return mean([r["q"] - r["p"] for r in sample])

    print(f"\n[λ↑ p↓] run 级点估计={_fmt(p_effect(rows))}, "
          f"seed 聚类 95% CI={cluster_bootstrap_ci(rows, p_effect)}")
    print(f"[λ↑ q↓] run 级点估计={_fmt(q_effect(rows))}, "
          f"seed 聚类 95% CI={cluster_bootstrap_ci(rows, q_effect)}")
    print(f"[τ↑ q] run 级点估计={_fmt(tau_effect(rows))}, "
          f"seed 聚类 95% CI={cluster_bootstrap_ci(rows, tau_effect)}")
    print(f"[q-p 均值] run 级={_fmt(qp_mean(rows))}, "
          f"seed 聚类 95% CI={cluster_bootstrap_ci(rows, qp_mean)}")

    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r["q"] - r["p"])
    n_all_pos = sum(1 for s in by_seed if all(v > 0 for v in by_seed[s]))
    n_seed = len(by_seed)
    print(f"\n[q-p 正号] run 级 {sum(1 for r in rows if r['q'] > r['p'])}/{len(rows)} 为正；"
          f"seed 级 {n_all_pos}/{n_seed} 个 seed 在其全部格子上为正，"
          f"单侧符号检验 p={sign_test_p(n_all_pos, n_seed, two_sided=False):.4f}")


def report_sym() -> None:
    names = [n for n in ["run07_sym", "run10_sym"] if has_run(n)]
    if not names:
        print("\n[run07_sym/run10_sym] 未运行/未复制到本机，跳过")
        return
    rows = merge_runs(names)
    vals = [r["q"] - r["p"] for r in rows]
    print("\n" + "=" * 70)
    print(" + ".join(names) + "（对称收益，q-p）")
    print("=" * 70)
    print(f"n={len(rows)}, weak={sum(weak(r) for r in rows)}")
    print(f"q-p={[_fmt(v) for v in vals]}")
    print(f"mean={_fmt(mean(vals))}, sd={_fmt(stdev(vals))}, "
          f"min={_fmt(min(vals))}, max={_fmt(max(vals))}")
    n_pos = sum(1 for v in vals if v > 0)
    seeds = sorted({r["seed"] for r in rows})
    seed_all_pos = sum(
        1 for s in seeds
        if all(v > 0 for v in [r["q"] - r["p"] for r in rows if r["seed"] == s])
    )
    print(f"run 级正号 {n_pos}/{len(vals)}；seed 级正号 {seed_all_pos}/{len(seeds)}")
    print(f"单侧符号检验 p（按 {len(seeds)} seeds）≈{sign_test_p(seed_all_pos, len(seeds), two_sided=False):.4f}")
    print(f"置换检验 vs 0（run 级，双尾）≈{permutation_p(vals, [0.0] * len(vals), n=50000):.4f}")


def report_ref() -> None:
    names = [n for n in ["run08_ref", "run10_ref"] if has_run(n)]
    if not names:
        print("\n[run08_ref/run10_ref] 未运行/未复制到本机，跳过")
        return
    rows = merge_runs(names)
    print("\n" + "=" * 70)
    print(" + ".join(names) + "（ref 模式，τ 效应）")
    print("=" * 70)
    seeds = sorted({r["seed"] for r in rows})
    q05 = [mean([r["q"] for r in rows if r["seed"] == s and r["tau"] == 0.5]) for s in seeds]
    q10 = [mean([r["q"] for r in rows if r["seed"] == s and r["tau"] == 1.0]) for s in seeds]
    print(f"seeds={seeds}")
    print(f"q(τ=0.5) per seed={[_fmt(v) for v in q05]}")
    print(f"q(τ=1.0) per seed={[_fmt(v) for v in q10]}")
    diffs = [b - a for a, b in zip(q05, q10)]
    print(f"q(1.0)-q(0.5) per seed={[_fmt(v) for v in diffs]}, mean={_fmt(mean(diffs))}")
    print(f"配对符号置换检验 p≈{paired_permutation_p(q10, q05, n=50000):.4f}")
    for tau in (0.5, 1.0):
        rs = [r for r in rows if r["tau"] == tau]
        print(f"τ={tau}: n={len(rs)}, p={_fmt(mean([r['p'] for r in rs]))}, "
              f"q={_fmt(mean([r['q'] for r in rs]))}, weak={sum(weak(r) for r in rs)}")


def report_compare_sym_curv() -> None:
    sym_names = [n for n in ["run07_sym", "run10_sym"] if has_run(n)]
    if not sym_names or not has_run("run06_curv"):
        print("\n[sym vs run06_curv] 数据不全，跳过")
        return
    sym = merge_runs(sym_names)
    curv = load_runs("run06_curv")
    a = [r["q"] - r["p"] for r in sym]
    b = [r["q"] - r["p"] for r in curv]
    print("\n" + "=" * 70)
    print("sym q-p vs run06_curv q-p")
    print("=" * 70)
    print(f"sym n={len(a)}, mean={_fmt(mean(a))}, sd={_fmt(stdev(a))}")
    print(f"curv n={len(b)}, mean={_fmt(mean(b))}, sd={_fmt(stdev(b))}")
    print(f"置换检验（双尾）p≈{permutation_p(a, b, n=100000):.4f}")


def report_identity() -> None:
    if not has_run("run11_identity"):
        print("\n[run11_identity] 未运行，跳过")
        return
    rows = load_runs("run11_identity")
    vals = [r["q"] - r["p"] for r in rows]
    print("\n" + "=" * 70)
    print("run11_identity（真 identity α=β=1.0，λ1τ1）")
    print("=" * 70)
    print(f"n={len(rows)}, weak={sum(weak(r) for r in rows)}")
    print(f"p={_fmt(mean([r['p'] for r in rows]))}, q={_fmt(mean([r['q'] for r in rows]))}, "
          f"q-p={_fmt(mean(vals))}, sd={_fmt(stdev(vals))}")
    print(f"q-p per seed={[_fmt(v) for v in vals]}")


def report_roleswap() -> None:
    if not has_run("run12_roleswap"):
        print("\n[run12_roleswap] 未运行，跳过")
        return
    rows = load_runs("run12_roleswap")
    vals = [r["q"] - r["p"] for r in rows]
    print("\n" + "=" * 70)
    print("run12_roleswap（真 identity，每轮交换角色网络）")
    print("=" * 70)
    print(f"n={len(rows)}, weak={sum(weak(r) for r in rows)}")
    print(f"p={_fmt(mean([r['p'] for r in rows]))}, q={_fmt(mean([r['q'] for r in rows]))}, "
          f"q-p={_fmt(mean(vals))}, sd={_fmt(stdev(vals))}")
    print(f"q-p per seed={[_fmt(v) for v in vals]}")


def report_shared() -> None:
    if not has_run("run13_shared"):
        print("\n[run13_shared] 未运行，跳过")
        return
    rows = load_runs("run13_shared")
    vals = [r["q"] - r["p"] for r in rows]
    print("\n" + "=" * 70)
    print("run13_shared（共享策略网络）")
    print("=" * 70)
    print(f"n={len(rows)}, weak={sum(weak(r) for r in rows)}")
    print(f"p={_fmt(mean([r['p'] for r in rows]))}, q={_fmt(mean([r['q'] for r in rows]))}, "
          f"q-p={_fmt(mean(vals))}, sd={_fmt(stdev(vals))}")
    print(f"q-p per seed={[_fmt(v) for v in vals]}")


def report_identity_lambda() -> None:
    if not has_run("run14_identity_lambda"):
        print("\n[run14_identity_lambda] 未运行，跳过")
        return
    rows = load_runs("run14_identity_lambda")
    print("\n" + "=" * 70)
    print("run14_identity_lambda（真 identity λ 轴）")
    print("=" * 70)
    for lam in sorted({r["lambda"] for r in rows}):
        rs = [r for r in rows if r["lambda"] == lam]
        print(f"λ={lam:g}: n={len(rs)}, p={_fmt(mean([r['p'] for r in rs]))}, "
              f"q={_fmt(mean([r['q'] for r in rs]))}, weak={sum(weak(r) for r in rs)}")
    l1 = [r["p"] for r in rows if r["lambda"] == 1]
    l5 = [r["p"] for r in rows if r["lambda"] == 5]
    if l1 and l5:
        print(f"λ=1→5 p 差={_fmt(mean(l5) - mean(l1))}, q 差="
              f"{_fmt(mean([r['q'] for r in rows if r['lambda']==5]) - mean([r['q'] for r in rows if r['lambda']==1]))}")


def report_minent() -> None:
    levels = [("0.45", "run11_minent_m0.45"), ("0.55", "run11_minent_m0.55"), ("0.60", "run11_minent_m0.60")]
    found = False
    if has_run("run06"):
        base = load_runs("run06")
        print("\n" + "-" * 70)
        print("run06 基线（min-ent 0.5）")
        for lam in (1.0, 5.0):
            rs = [r for r in base if r["lambda"] == lam and r["tau"] == 1.0]
            if rs:
                print(f"  λ={lam:g}: n={len(rs)}, p={_fmt(mean([r['p'] for r in rs]))}, "
                      f"q={_fmt(mean([r['q'] for r in rs]))}, q-p={_fmt(mean([r['q']-r['p'] for r in rs]))}, "
                      f"weak={sum(weak(r) for r in rs)}")
    for label, name in levels:
        if not has_run(name):
            continue
        found = True
        rows = load_runs(name)
        print("\n" + "-" * 70)
        print(f"run11_minent_{label}（min-ent {label}）")
        for lam in (1.0, 5.0):
            rs = [r for r in rows if r["lambda"] == lam and r["tau"] == 1.0]
            if rs:
                print(f"  λ={lam:g}: n={len(rs)}, p={_fmt(mean([r['p'] for r in rs]))}, "
                      f"q={_fmt(mean([r['q'] for r in rs]))}, q-p={_fmt(mean([r['q']-r['p'] for r in rs]))}, "
                      f"weak={sum(weak(r) for r in rs)}")
        l1 = [r["p"] for r in rows if r["lambda"] == 1 and r["tau"] == 1.0]
        l5 = [r["p"] for r in rows if r["lambda"] == 5 and r["tau"] == 1.0]
        if l1 and l5:
            print(f"  λ=1→5 p 差={_fmt(mean(l5) - mean(l1))}")
    if not found:
        print("\n[run11_minent] 未运行，跳过")


def report_remaining() -> None:
    names = [
        "run15_same_init",
        "run16_update_order_slave_first",
        "run16_update_order_random",
        "run17_shared_trunk",
        "run18_advnorm_lambda",
        "run19_long",
    ]
    for name in names:
        if not has_run(name):
            continue
        rows = load_runs(name)
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        print(f"n={len(rows)}, weak={sum(weak(r) for r in rows)}")
        if len({r["lambda"] for r in rows}) > 1:
            for lam in sorted({r["lambda"] for r in rows}):
                rs = [r for r in rows if r["lambda"] == lam]
                print(f"λ={lam:g}: p={_fmt(mean([r['p'] for r in rs]))}, "
                      f"q={_fmt(mean([r['q'] for r in rs]))}, "
                      f"q-p={_fmt(mean([r['q']-r['p'] for r in rs]))}, "
                      f"weak={sum(weak(r) for r in rs)}")
        else:
            print(f"p={_fmt(mean([r['p'] for r in rows]))}, "
                  f"q={_fmt(mean([r['q'] for r in rows]))}, "
                  f"q-p={_fmt(mean([r['q']-r['p'] for r in rows]))}, "
                  f"win={_fmt(mean([r['win_rate'] for r in rows]))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run06", action="store_true")
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--compare-sym-curv", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--roleswap", action="store_true")
    ap.add_argument("--shared", action="store_true")
    ap.add_argument("--identity-lambda", action="store_true")
    ap.add_argument("--minent", action="store_true")
    ap.add_argument("--remaining", action="store_true")
    args = ap.parse_args()

    if not (args.all or args.run06 or args.sym or args.ref or args.compare_sym_curv
            or args.identity or args.roleswap or args.shared
            or args.identity_lambda or args.minent or args.remaining):
        ap.error("请至少指定一个分析项，或使用 --all")

    if args.all or args.run06:
        report_run06()
    if args.all or args.sym:
        report_sym()
    if args.all or args.ref:
        report_ref()
    if args.all or args.compare_sym_curv:
        report_compare_sym_curv()
    if args.all or args.identity:
        report_identity()
    if args.all or args.roleswap:
        report_roleswap()
    if args.all or args.shared:
        report_shared()
    if args.all or args.identity_lambda:
        report_identity_lambda()
    if args.all or args.minent:
        report_minent()
    if args.all or args.remaining:
        report_remaining()


if __name__ == "__main__":
    main()
