# run06_curv 数据说明

- 本目录的 `runs.jsonl` 已去重（原始 4 行 → 2 行，2 个唯一 seed）。
- `grid.json` 的 `check_lambda` / `check_tau` 字段含 NaN，原因是该次运行过程中
  `sweep.py` 被修改/提交（git log 5d3ff20），`_file_sha` 记录的是运行结束后的文件，
  而不是实际执行的代码。核验请以 `runs.jsonl` 为准。
- 正确核验值（由当前代码逻辑从 cells 重算）：
  - check_lambda: p_at_lambda_min = p_at_lambda_max = 0.2051（单 λ，无方向性）。
  - check_tau: q_at_tau_min = q_at_tau_max = 0.2677（单 τ，无方向性）。
