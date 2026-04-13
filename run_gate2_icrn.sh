#!/bin/bash
# run_gate2_icrn.sh — Gate 2 with ICRN backend
# Run on vast.ai after vastai_setup.sh
set -e

echo "[FORGE] ============================================"
echo "[FORGE] Gate 2: Loop Closure (ICRN Backend)"
echo "[FORGE] ============================================"

cd /workspace/dna-compute-engine

# Verify ICRN
echo "[FORGE] Verifying ICRN..."
python3 -c "from icrn_bridge import simulate_gate; s,_,_ = simulate_gate('AND', time=10.0, sample_num=5); print(f'[FORGE] ICRN AND gate: {s:.2f} nM')"

# Gate 2 Test 1: AND gate, 3 iterations, 4 variants
echo ""
echo "[FORGE] Test 1: AND gate loop (3 iterations)"
python3 sim_loop.py \
    --backend icrn \
    --gate AND \
    --goal AND_gate \
    --max-iter 3 \
    --variants 4 \
    --sim-time 100.0 \
    --config config.vastai.yaml \
    2>&1 | tee /workspace/logs/gate2_and.log

# Gate 2 Test 2: NOT gate
echo ""
echo "[FORGE] Test 2: NOT gate loop (3 iterations)"
python3 sim_loop.py \
    --backend icrn \
    --gate NOT \
    --goal maximize_any \
    --max-iter 3 \
    --variants 4 \
    --sim-time 200.0 \
    --config config.vastai.yaml \
    2>&1 | tee /workspace/logs/gate2_not.log

# Gate 2 Test 3: THRESHOLD gate
echo ""
echo "[FORGE] Test 3: THRESHOLD gate loop (3 iterations)"
python3 sim_loop.py \
    --backend icrn \
    --gate THRESHOLD \
    --goal threshold \
    --max-iter 3 \
    --variants 4 \
    --sim-time 100.0 \
    --config config.vastai.yaml \
    2>&1 | tee /workspace/logs/gate2_threshold.log

echo ""
echo "[FORGE] ============================================"
echo "[FORGE] Gate 2 Results"
echo "[FORGE] ============================================"

# Check loop_history for signal progression
if [ -f renders/sim_loop/loop_history.json ]; then
    echo "[FORGE] Loop history:"
    python3 -c "
import json
h = json.loads(open('renders/sim_loop/loop_history.json').read())
for entry in h:
    sigs = entry.get('signals_raw', entry.get('signals', {}))
    ev = entry.get('evaluation', {})
    print(f'  Iter {entry[\"iter\"]}: mean={ev.get(\"_mean\",0):.4f} best={ev.get(\"_best\",\"?\")} converged={ev.get(\"_converged\",False)}')
    # Check signal differentiation
    vals = [v for k,v in sigs.items() if not k.startswith('_')]
    if vals:
        print(f'    signals: min={min(vals):.2f} max={max(vals):.2f} range={max(vals)-min(vals):.2f}')
"
fi

echo ""
echo "[FORGE] Gate 2 criteria check:"
echo "[FORGE]   [ ] ICRN package found and integrated as sim backend"
echo "[FORGE]   [ ] icrn_bridge.py translates gate specs to CRN reactions"
echo "[FORGE]   [ ] sim_loop.py completes 3 iterations with ICRN"
echo "[FORGE]   [ ] Each iteration's concentration signals differ measurably"
echo "[FORGE]   [ ] LLM refinement produces genuinely different circuit descriptions"
echo "[FORGE]   [ ] loop_history.json shows signal progression"
echo "[FORGE] ============================================"
