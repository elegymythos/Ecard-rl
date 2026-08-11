# 2026-08-11 代码修复记录（debug）

## 背景

run01（60 次自博弈扫描）数据分析发现 8 个运行/训练逻辑问题。本轮全部修正并验证。run01 数据保持修复前状态，归档于 `data/runs/run01/`（编码已转 UTF-8，缺失元数据已记入 run_manifest.json）。

## 问题与修复对照

| # | 问题 | 修复 | 文件 | 验证 |
|---|------|------|------|------|
| 1 | 最终 p/q 是单点快照，末段漂移巨大（\|Δq\| 均值 0.215、最大 0.777） | 新增末段 20% 窗口均值±std（`p`/`p_std`/`q`/`q_std`）；完整 p/q 序列写入 runs.jsonl | sweep.py | demo 输出 `p=0.070±0.000`（4 个更新窗口只有 2 点；正式网格 ≥10 点后 std 才有意义） |
| 2 | 预测核验用均值，被双峰 seed 污染（p std 高达 0.40） | 新增中位数核验 `check_lambda_median`/`check_tau_median`；cells 增加 median/std 字段 | sweep.py | demo 输出通过 |
| 3 | 概率权重乘子尺度爆炸（τ=0.1、p=1/64 时乘子 ≈25×，主观奖励 ±100 vs ±10） | 记录每 agent 主观奖励 std（`subj_std`）；新增 `--ret-norm` 开关（优势+回报标准化，独立于 adv-norm） | ppo.py、utility.py、sweep.py | demo：subj_std E=1.67 / S=2.26 |
| 4 | win_rate/obj_return 只取最后一个 rollout（约 700 局） | 改为末段窗口均值，obj_return 附 std | sweep.py | demo 输出通过 |
| 5 | 概率权重窗口是策略反馈环（窗口胜率由当前策略造成） | `--weight-mode window\|ref\|off`；ref 模式用固定参考概率 p=0.2，切断反馈环 | utility.py、sweep.py | 正式对照实验待跑 |
| 6 | 熵正则 0.01 下皇帝 p 中位数 ~0.01，接近塌缩 | `--ent-coef` 参数化；新增 `--min-ent` 最小熵 hinge 约束；记录 ent_e/ent_s | ppo.py、sweep.py | demo：ent_s_init=0.478 |
| 7 | 跑 2 小时一次崩溃全丢 | runs.jsonl 每跑完一条立即落盘（上一轮已修，本轮验证） | sweep.py | 3 runs → 3 行 JSONL |
| 8 | 「τ↑→均匀」只用首轮 q 度量 | 新增初始状态策略熵度量（均匀 = ln2≈0.6931），`check_tau_entropy` | ppo.py、sweep.py | demo 输出通过 |
| 9 | GBK 编码（Windows locale 根因） | 所有 JSON 写入显式 `encoding="utf-8"`（上一轮已修） | sweep.py、train.py | `json.load(encoding='utf-8')` 通过 |
| 10 | 无超参/时间/版本元数据 | grid.json 增加 meta：steps、rollout、epochs、batch、lr、ent_coef、min_ent、ret_norm、adv_norm、weight_mode、seeds、started_at、git_commit、脚本哈希 | sweep.py | demo meta 完整 |

## 文件变更摘要

- `utility.py`：`make_prospect` 支持 `use_weighting=False | "window" | "ref"`，ref 模式用固定 `ref_prob=0.2`。
- `ppo.py`：`Buffer.get` 增加 `ret_norm`；`update_policy` 增加 `ret_norm`/`min_ent`；`collect_rollout` 输出 `subj_std`；新增 `first_round_stats`（p、q、双方初始策略熵）。
- `sweep.py`：窗口统计、中位数/熵核验、`--name`/`--ret-norm`/`--min-ent`/`--ent-coef`/`--weight-mode`、meta、runs.jsonl 增量落盘、UTF-8。
- `train.py`：JSON 写入 UTF-8。

## 验证（2026-08-11 18:39）

- `python -m pytest test_env.py -q`：8 passed。
- `python sweep.py --demo --name fix_verify`：3 次运行完成，grid.json UTF-8 + meta + 全部核验字段；验证目录已清理。
- `python train.py --steps 4096 --rollout 2048 --epochs 1 --batch 256 --seed 9`：正常运行；验证产生的 stage3 JSON 已清理。
- 验证产物不留存，避免污染 data/。

## 兼容性说明

- run 记录保留 `final_p`/`final_q`（单点）作为 run01 兼容字段；新分析应使用 `p`/`q`（末段窗口均值）。
- **run01 与修复后的运行不可直接比较**：run01 是单点快照 + 均值核验 + window 权重，修复后是窗口均值 + 中位数/熵核验。
- 建议后续正式实验：`--weight-mode ref` 对照（隔离反馈环）、`--ret-norm` 对照（稳定奖励尺度）、`--min-ent 0.3` 对照（防塌缩），并用 `--name` 归档。

## 相关文件

- 本记录：`debug/2026-08-11_fix_record.md`
- 修改：`utility.py`、`ppo.py`、`sweep.py`、`train.py`
- 数据：`data/runs/run01/`（修复前基线，含 run_manifest.json）
- 当前 git HEAD：`33e7fe3`（建议修复后单独 commit）
