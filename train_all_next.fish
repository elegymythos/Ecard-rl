#!/usr/bin/env fish
# ============================================================
# Next-step experiments (fish version, after switching to Linux/macOS).
#
# Experiments:
#   run15_same_init          : independent networks, same initial weights
#   run16_update_order_*     : slave-first / random update order
#   run17_shared_trunk       : shared trunk + independent actor/critic heads
#   run18_advnorm_lambda     : adv-norm on, lambda=1 and lambda=5
#   run19_long               : longer budget (1M steps) identity corner
#
# Usage:
#   fish train_all_next.fish [steps]
#   fish train_all_next.fish 200000
#
# Default STEPS=400000 except run19_long which uses 1000000.
# ============================================================

set STEPS 400000
if test (count $argv) -gt 0
    set STEPS $argv[1]
end

set LONG_STEPS 1000000

# Go to script directory
cd (dirname (status -f))

# Activate conda env (run `conda init fish` once if needed)
conda activate Ecard-rl
or begin
    echo "Failed to activate conda env 'Ecard-rl'"
    exit 1
end

# --- Verification gates (fail fast) ---
echo "[verify] Running environment tests..."
python -m pytest test_env.py -q -p no:cacheprovider
or exit 1

echo "[verify] Running solver verification..."
python solver.py
or exit 1

# --- q-p / dynamics mechanism experiments ------------------------
echo "[1] run15_same_init: same initial weights, independent networks"
python sweep.py --name run15_same_init --cells "1-1.0" --seeds "42,43,44,45,46" --steps $STEPS --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off --same-init --save-agents --predictions-file predictions/run15_same_init.json
or exit 1

echo "[2] run16_update_order_slave_first"
python sweep.py --name run16_update_order_slave_first --cells "1-1.0" --seeds "42,43,44,45,46" --steps $STEPS --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off --update-order slave_first --save-agents --predictions-file predictions/run16_update_order.json
or exit 1

echo "[3] run16_update_order_random"
python sweep.py --name run16_update_order_random --cells "1-1.0" --seeds "42,43,44,45,46" --steps $STEPS --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off --update-order random --save-agents --predictions-file predictions/run16_update_order.json
or exit 1

echo "[4] run17_shared_trunk: shared trunk + independent heads"
python sweep.py --name run17_shared_trunk --cells "1-1.0" --seeds "42,43,44,45,46" --steps $STEPS --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off --shared-trunk --save-agents --predictions-file predictions/run17_shared_trunk.json
or exit 1

echo "[5] run18_advnorm_lambda: adv-norm on, lambda=1 and 5"
python sweep.py --name run18_advnorm_lambda --cells "1-1.0,5-1.0" --seeds "42,43,44,45,46" --steps $STEPS --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off --adv-norm --save-agents --predictions-file predictions/run18_advnorm.json
or exit 1

echo "[6] run19_long: 1M steps identity corner (3 seeds)"
python sweep.py --name run19_long --cells "1-1.0" --seeds "42,43,44" --steps $LONG_STEPS --min-ent 0.5 --alpha 1.0 --beta 1.0 --weight-mode off --save-agents --predictions-file predictions/run19_long.json
or exit 1

# --- Post-run analysis -------------------------------------------
echo "[7] Running merged/seed-cluster analysis..."
python analyze.py --all
or exit 1

echo "[8] Running per-state / exploitability evaluation..."
for d in run15_same_init run16_update_order_slave_first run16_update_order_random run17_shared_trunk run18_advnorm_lambda run19_long
    echo "  -- evaluate $d"
    python evaluate.py --dir data/runs/$d --swap-eval --save
    or exit 1
end

echo "[DONE] All remaining experiments completed."
