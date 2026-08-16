@echo off
REM ============================================================
REM  One-click training for the three follow-up experiments:
REM    run07_sym : symmetric-payoff ablation (-1), true identity
REM    run08_ref : probability-weighting ref mode
REM    run09_asym: asymmetric lambda (emperor 5, slave 2.25)
REM
REM  Usage:   train_all_next.bat [steps]
REM  Default: 400000 steps per run
REM ============================================================
setlocal
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
echo [1/5] Running environment tests...
python -m pytest test_env.py -q -p no:cacheprovider
if errorlevel 1 exit /b 1

echo [2/5] Running solver verification...
python solver.py
if errorlevel 1 exit /b 1

REM --- Follow-up experiments ---
echo [3/5] run07_sym: symmetric payoff ablation (A-A loss = -1, alpha=beta=1.0)
python sweep.py --name run07_sym --cells "1-1.0" --seeds "42,43,44" --steps %STEPS% --min-ent 0.5 --alpha 1.0 --beta 1.0 --reward-loss -1 --predictions-file predictions/run07_sym.json
if errorlevel 1 exit /b 1

echo [4/5] run08_ref: probability-weighting ref mode (feedback loop isolated)
python sweep.py --name run08_ref --cells "1-0.5,1-1.0" --seeds "42,43" --steps %STEPS% --min-ent 0.5 --weight-mode ref --predictions-file predictions/run08_ref.json
if errorlevel 1 exit /b 1

echo [5/5] run09_asym: asymmetric lambda (emperor lambda=5, slave lambda=2.25)
python sweep.py --name run09_asym --cells "5-1.0" --seeds "42,43" --steps %STEPS% --min-ent 0.5 --alpha 0.88 --beta 0.88 --slave-lam 2.25 --predictions-file predictions/run09_asym.json
if errorlevel 1 exit /b 1

echo.
echo [DONE] All experiments completed.
echo Results are in:
echo   data\runs\run07_sym
echo   data\runs\run08_ref
echo   data\runs\run09_asym
endlocal
