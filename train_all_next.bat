@echo off
REM ============================================================
REM  Next-step training: fill seeds for run10_sym and run10_ref
REM    run10_sym : symmetric-payoff ablation, seeds 45,46
REM    run10_ref : ref weight mode, seeds 44,45,46
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
echo [1/4] Running environment tests...
python -m pytest test_env.py -q -p no:cacheprovider
if errorlevel 1 exit /b 1

echo [2/4] Running solver verification...
python solver.py
if errorlevel 1 exit /b 1

REM --- Seed-filling experiments ---
echo [3/4] run10_sym: symmetric payoff ablation, seeds 45,46
python sweep.py --name run10_sym --cells "1-1.0" --seeds "45,46" --steps %STEPS% --min-ent 0.5 --alpha 1.0 --beta 1.0 --reward-loss -1 --predictions-file predictions/run10_sym.json
if errorlevel 1 exit /b 1

echo [4/4] run10_ref: ref weight mode, seeds 44,45,46
python sweep.py --name run10_ref --cells "1-0.5,1-1.0" --seeds "44,45,46" --steps %STEPS% --min-ent 0.5 --weight-mode ref --predictions-file predictions/run10_ref.json
if errorlevel 1 exit /b 1

echo.
echo [DONE] Seed-filling experiments completed.
echo Merge with previous results:
echo   run10_sym + run07_sym = 5 seeds for symmetric payoff ablation
echo   run10_ref + run08_ref = 5 seeds for ref weight mode
endlocal
