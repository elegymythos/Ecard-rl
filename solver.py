"""阶段 2：E卡精确解 —— 递归价值求解（约 100 行）。

验证门（direction.md 阶段 2）：
1. 单轮静态 2×2 矩阵 [[-5, 1], [1, 0]] 的混合纳什均衡 = p=q=1/7、值 +1/7；
2. 完整 5 轮游戏：V_4 = -0.2、首轮 p=q=0.2，每轮满足闭式 p_k = 1/(k+1)；
3. 均衡总胜率 W = 0.8（由 V_4 = 6W - 5 推出）。

状态坍缩：任何王牌出现即终局，所以非终局状态只剩「双方都没出过王牌、
各剩 k 张平民」。平局后状态完全对称，状态空间坍缩为 k = 0..4 共 5 个值。
"""
from __future__ import annotations

from env import CIVILIANS


def solve_2x2(m: list[list[float]]) -> tuple[float, float, float]:
    """2×2 零和矩阵博弈（行=皇帝，列=奴隶）。

    返回 (p, q, v)：p = 皇帝出动作 0 的概率，q = 奴隶出动作 0 的概率，
    v = 皇帝视角博弈值。先查纯策略鞍点，没有再解混合均衡（无差异方程）。
    """
    # 纯策略鞍点：某格同时是行的最小值和列的最大值
    for r in range(2):
        for c in range(2):
            if m[r][c] == min(m[r]) and m[r][c] == max(m[0][c], m[1][c]):
                return float(r == 0), float(c == 0), float(m[r][c])

    # 混合均衡：令双方各自对两个纯策略无差异
    a, b, c_, d = m[0][0], m[0][1], m[1][0], m[1][1]
    denom = a + d - b - c_
    q = (d - b) / denom  # 皇帝无差异 → 奴隶的混合概率
    p = (d - c_) / denom  # 奴隶无差异 → 皇帝的混合概率
    assert 0 <= p <= 1 and 0 <= q <= 1, "混合解出界：矩阵退化，需边界处理"
    v = a * q + b * (1 - q)
    return p, q, v


def static_equilibrium() -> tuple[float, float, float]:
    """单轮静态情形（README 的 1/7）：C-C 直接得 0，没有延续轮。"""
    return solve_2x2([[-5.0, 1.0], [1.0, 0.0]])


def solve_game(max_k: int = CIVILIANS) -> tuple[list[float], list[tuple[float, float, float]]]:
    """完整序贯博弈：从终局 V_0 = -5 往回递推 max_k 层。

    返回 (values, rounds)：
    values[k] = V_k（双方各剩 k 张平民时的博弈值）；
    rounds[k-1] = (p_k, q_k, v_k)，即第 k 层的均衡解。
    """
    values = [-5.0]  # V_0：无平民，被迫 A-A，奴隶胜
    rounds: list[tuple[float, float, float]] = []
    for _ in range(max_k):
        p, q, v = solve_2x2([[-5.0, 1.0], [1.0, values[-1]]])
        rounds.append((p, q, v))
        values.append(v)
    return values, rounds


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    # 验证门 1：单轮静态必须给出 1/7 和 +1/7
    p, q, v = static_equilibrium()
    seventh = 1.0 / 7.0
    assert _close(p, seventh) and _close(q, seventh), f"静态均衡出错: p={p}, q={q}"
    assert _close(v, seventh), f"静态值出错: v={v}"
    print(f"验证门 1  单轮静态：p=q={p:.6f}（1/7≈{seventh:.6f}），V={v:.6f}（+1/7）")

    # 验证门 2：完整 5 轮，每轮必须满足闭式 p_k = 1/(k+1)
    values, rounds = solve_game()
    for k, (p_k, q_k, v_k) in enumerate(rounds, start=1):
        closed = 1.0 / (k + 1)
        assert _close(p_k, closed) and _close(q_k, closed), f"k={k}: p={p_k}, q={q_k}"
        assert _close(v_k, values[k]), f"k={k}: 值不自洽 {v_k} vs {values[k]}"
        print(f"         k={k}   V={values[k]:+.4f}   p=q={p_k:.4f}（闭式 {closed:.4f}）")
    assert _close(values[4], -0.2), f"V_4 应为 -0.2，实际 {values[4]}"
    assert _close(rounds[3][0], 0.2) and _close(rounds[3][1], 0.2), "首轮 p=q 应为 0.2"
    print("验证门 2  完整 5 轮：V_4 = -0.2，首轮 p=q=0.2，闭式逐层通过")

    # 验证门 3：均衡胜率 W = 0.8（V_4 = 6W - 5，游戏无整体平局）
    win_rate = (values[-1] + 5.0) / 6.0
    assert _close(win_rate, 0.8), f"胜率应为 0.8，实际 {win_rate}"
    print(f"验证门 3  均衡胜率：皇帝 {win_rate:.4f} / 奴隶 {1 - win_rate:.4f}")

    print("\n阶段 2 全部验证门通过：solver 与理论闭式一致。")


if __name__ == "__main__":
    main()
