#!/bin/bash
# run_gate2.sh — Gate 2 test runner: Loop Closure
# Run from /workspace/dna-compute-engine/ after Gate 1 passes
# Requires: ANTHROPIC_API_KEY set, renders from Gate 1

cd /workspace/dna-compute-engine

echo "[FORGE] === GATE 2: LOOP CLOSURE ==="

python3 sim_loop.py \
    --goal AND_gate \
    --cascade half_adder \
    --max-iter 3 \
    --workers 1 \
    --variants 2 \
    --config config.vastai.yaml \
    2>&1 | tee logs/gate2_loop.log | sed 's/^/[FORGE] /'
T1=$?
echo "[FORGE] sim_loop exit: $T1"

if [ -f "renders/sim_loop/loop_history.json" ]; then
    echo "[FORGE] Loop history:"
    python3 -c "
import json
history = json.load(open('renders/sim_loop/loop_history.json'))
for h in history:
    print(f\"[FORGE] Iter {h['iter']}: mean={h['evaluation']['_mean']:.3f} best={h['evaluation'].get('_best')} converged={h['evaluation']['_converged']}\")
print(f'[FORGE] Total iterations: {len(history)}')
" 2>&1
    echo "[FORGE] *** GATE 2 PASSED ***"
else
    echo "[FORGE] *** GATE 2 FAILED — no loop_history.json ***"
fi

echo "[FORGE] $(date -Iseconds)"