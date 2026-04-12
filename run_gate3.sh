#!/bin/bash
# run_gate3.sh -- Gate 3: Biological Validation
# Run from /workspace/dna-compute-engine/ after Gate 2 passes
# Requires: ANTHROPIC_API_KEY set, Gate 2 confirmed

cd /workspace/dna-compute-engine

echo "[FORGE] === GATE 3: BIOLOGICAL VALIDATION ==="

# Test with sample expression data first
echo "[FORGE] Step 1: Genomic input parsing..."
python3 genomic_input.py --rnaseq demo_data/sample_expression.csv --genes BRCA1 TP53 KRAS --info 2>&1 | sed 's/^/[FORGE] /'
G1=$?
echo "[FORGE] Genomic parse exit: $G1"

# Run sim_loop with genomic input
echo "[FORGE] Step 2: Genomic sim_loop (3 iterations)..."
python3 sim_loop.py \
    --genomic demo_data/sample_expression.csv \
    --genes BRCA1 TP53 KRAS \
    --goal AND_gate \
    --max-iter 3 \
    --workers 1 \
    --variants 2 \
    --frame-end 24 \
    --config config.vastai.yaml \
    2>&1 | tee logs/gate3_genomic.log | sed 's/^/[FORGE] /'
G2=$?
echo "[FORGE] Genomic loop exit: $G2"

if [ -f "renders/sim_loop/loop_history.json" ]; then
    echo "[FORGE] Loop history:"
    python3 << 'PYEOF'
import json
history = json.load(open('renders/sim_loop/loop_history.json'))
for h in history:
    i = h['iter']
    m = h['evaluation']['_mean']
    b = h['evaluation'].get('_best', 'none')
    c = h['evaluation']['_converged']
    print(f'[FORGE] Iter {i}: mean={m:.3f} best={b} converged={c}')
print(f'[FORGE] Total iterations: {len(history)}')
PYEOF
fi

PNG_COUNT=$(find renders/ -name '*.png' 2>/dev/null | wc -l)
echo "[FORGE] Total PNGs: $PNG_COUNT"

echo "[FORGE] === GATE 3 RESULTS ==="
echo "[FORGE] Genomic parse: $G1"
echo "[FORGE] Genomic loop:  $G2"
echo "[FORGE] PNGs: $PNG_COUNT"
echo "[FORGE] $(date -Iseconds)"