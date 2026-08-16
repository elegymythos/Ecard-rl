@echo off
REM ============================================================
REM  Next-step training: high-value experiments after run10.
REM
REM  Experiments:
REM    run11_identity        : true identity alpha=beta=1.0, lambda=1, tau=1, 5 seeds
REM    run12_roleswap        : swap emperor/slave networks every update (identity)
REM    run13_shared          : shared policy network (identity)
REM    run14_identity_lambda : true identity lambda axis (1,2,3,5)
REM    run11_minent_m0.45/55/60 : min-ent sensitivity around the chosen 0.5
REM
REM  After training runs analyze.py (merged/seed-cluster stats) and
REM  evaluate.py (per-state policy + exploitability).
REM
REM  Usage:   train_all_next.bat [steps]
REM  Default: 400000 steps per run
REM ============================================================
setlocal EnableDelayedExpansion
cd /d %~dp0

set STEPS=400000
if not "%~1"=="" set STEPS=%~1

REM --- Activate the conda environment (adjust name if needed) ---
call conda activate Ecard-rl
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env 'Ecard-rl'.
    echo         If conda is not in PATH, run: conda init cmd.exe
    echo         Or edit this line to: call C:\path\to\conda activate Ecard-rl
    exit /b 1
)

REM --- Verification gates (fail fast) ---
echo [verify] Running environment tests...
python -m pytest test_env.py -q -p no:cacheprovider
if errorlevel 1 exit /b 1

echo [verify] Running solver verification...
python solver.py
if errorlevel 1 exit /b 1

REM --- High-value experiments ----------------------------------------
echo [1] run11_identity: true identity, alpha=beta=1.0, seeds 42-46
python sweep.py --name run11_identity --cells "1-1.0" --seeds "42,43,44,45,46" ^
    --steps %STEPS% --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off ^
    --save-agents --predictions-file predictions/run11_identity.json
if errorlevel 1 exit /b 1

echo [2] run12_roleswap: identity + swap emperor/slave networks every update
python sweep.py --name run12_roleswap --cells "1-1.0" --seeds "42,43,44,45,46" ^
    --steps %STEPS% --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off ^
    --swap-roles --save-agents --predictions-file predictions/run12_roleswap.json
if errorlevel 1 exit /b 1

echo [3] run13_shared: identity + shared policy network
python sweep.py --name run13_shared --cells "1-1.0" --seeds "42,43,44,45,46" ^
    --steps %STEPS% --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off ^
    --shared-policy --save-agents --predictions-file predictions/run13_shared.json
if errorlevel 1 exit /b 1

echo [4] run14_identity_lambda: true identity lambda axis
python sweep.py --name run14_identity_lambda --cells "1-1.0,2-1.0,3-1.0,5-1.0" ^
    --seeds "42,43,44,45,46" --steps %STEPS% --min-ent 0.5 --alpha 1.0 --beta 1.0 ^
    --weight-mode off --save-agents --predictions-file predictions/run14_identity_lambda.json
if errorlevel 1 exit /b 1

echo [5] run11_minent: min-ent sensitivity (0.45 / 0.55 / 0.60)
for %%M in (0.45 0.55 0.60) do (
    echo   -- min-ent %%M (lambda=1 and lambda=5, tau=1)
    python sweep.py --name run11_minent_m%%M --cells "1-1.0,5-1.0" --seeds "42,43,44,45,46" --steps %STEPS% --min-ent %%M --alpha 0.88 --beta 0.88 --weight-mode window --save-agents --predictions-file predictions/run11_minent.json
    if errorlevel 1 exit /b 1
)

REM --- Post-run analysis ---------------------------------------------
echo [6] Running merged/seed-cluster analysis...
python analyze.py --all
if errorlevel 1 exit /b 1

echo [7] Running per-state / exploitability evaluation...
for %%D in (run11_identity run12_roleswap run13_shared run14_identity_lambda run11_minent_m0.45 run11_minent_m0.55 run11_minent_m0.60) do (
    echo   -- evaluate %%D
    python evaluate.py --dir data/runs/%%D --save
    if errorlevel 1 exit /b 1
)

echo.
echo [DONE] All next-step experiments completed.
echo Remember to commit the updated results.md / analyze outputs.
endlocal
