#!/usr/bin/env fish
# ============================================================
#  Next-step training: high-value experiments after run10.
#
#  Experiments:
#    run11_identity        : true identity alpha=beta=1.0, lambda=1, tau=1, 5 seeds
#    run12_roleswap        : swap emperor/slave networks every update (identity)
#    run13_shared          : shared policy network (identity)
#    run14_identity_lambda : true identity lambda axis (1,2,3,5)
#    run11_minent_m0.45/55/60 : min-ent sensitivity around the chosen 0.5
#
#  After training runs analyze.py (merged/seed-cluster stats) and
#  evaluate.py (per-state policy + exploitability).
#
#  Usage:   ./train_all_next.fish [steps]
#  Default: 400000 steps per run
# ============================================================

# Exit on any command failure (by checking with `or exit` after each command)
# Fish does not have `set -e` as shell option; we manually handle errors.

# Set default steps
set -l steps 400000
if count $argv > /dev/null
    set steps $argv[1]
end

# Change to script directory
cd (dirname (status -f))

# --- Activate conda environment (torch_env) ---
# Ensure conda is available
if not command -v conda > /dev/null
    echo "[ERROR] conda not found in PATH. Please install conda or add it to PATH."
    exit 1
end

# Activate environment (assumes conda init fish has been run)
if not conda activate torch_env
    echo "[ERROR] Failed to activate conda env 'torch_env'."
    echo "         Please create it with: conda create -n torch_env python=3.x ..."
    exit 1
end

# --- Verification gates (fail fast) ---
echo "[verify] Running environment tests..."
python -m pytest test_env.py -q -p no:cacheprovider
or exit 1

echo "[verify] Running solver verification..."
python solver.py
or exit 1

# --- High-value experiments ----------------------------------------
echo "[1] run11_identity: true identity, alpha=beta=1.0, seeds 42-46"
python sweep.py --name run11_identity --cells "1-1.0" --seeds "42,43,44,45,46" \
    --steps $steps --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off \
    --save-agents --predictions-file predictions/run11_identity.json
or exit 1

echo "[2] run12_roleswap: identity + swap emperor/slave networks every update"
python sweep.py --name run12_roleswap --cells "1-1.0" --seeds "42,43,44,45,46" \
    --steps $steps --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off \
    --swap-roles --save-agents --predictions-file predictions/run12_roleswap.json
or exit 1

echo "[3] run13_shared: identity + shared policy network"
python sweep.py --name run13_shared --cells "1-1.0" --seeds "42,43,44,45,46" \
    --steps $steps --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off \
    --shared-policy --save-agents --predictions-file predictions/run13_shared.json
or exit 1

echo "[4] run14_identity_lambda: true identity lambda axis"
python sweep.py --name run14_identity_lambda --cells "1-1.0,2-1.0,3-1.0,5-1.0" \
    --seeds "42,43,44,45,46" --steps $steps --min-ent 0.5 --alpha 1.0 --beta 1.0 \
    --weight-mode off --save-agents --predictions-file predictions/run14_identity_lambda.json
or exit 1

echo "[5] run11_minent: min-ent sensitivity (0.45 / 0.55 / 0.60)"
for M in 0.45 0.55 0.60
    echo "  -- min-ent $M (lambda=1 and lambda=5, tau=1)"
    python sweep.py --name "run11_minent_m$M" --cells "1-1.0,5-1.0" --seeds "42,43,44,45,46" \
        --steps $steps --min-ent $M --alpha 0.88 --beta 0.88 --weight-mode window \
        --save-agents --predictions-file predictions/run11_minent.json
    or exit 1
end

# --- Post-run analysis ---------------------------------------------
echo "[6] Running merged/seed-cluster analysis..."
python analyze.py --all
or exit 1

echo "[7] Running per-state / exploitability evaluation..."
for D in run11_identity run12_roleswap run13_shared run14_identity_lambda \
         run11_minent_m0.45 run11_minent_m0.55 run11_minent_m0.60
    echo "  -- evaluate $D"
    python evaluate.py --dir "data/runs/$D" --save
    or exit 1
end

echo
echo "[DONE] All next-step experiments completed."
echo "Remember to commit the updated results.md / analyze outputs."