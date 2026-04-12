#!/bin/bash
# run_gate1.sh — Gate 1 test runner for DNA Compute Engine
# Run from /workspace/dna-compute-engine/ after setup
# Requires: ANTHROPIC_API_KEY set, Blender installed, repo cloned

cd /workspace/dna-compute-engine

echo "[FORGE] === GATE 1: FIRST RENDER ==="

# Test 1: Biophysics mapping
echo "[FORGE] Test 1: Biophysics params..."
python3 dna_compute.py --sequence GCGCATCGATGCGC --info > logs/gate1_info.log 2>&1
T1=$?
echo "[FORGE] Test 1 (biophysics): exit $T1"

# Test 2: LLM matrix generation (dry run — single API call)
echo "[FORGE] Test 2: LLM dry run..."
python3 dna_compute.py --sequence GCGCATCGATGCGC --variants 2 --frame-end 24 \
    --dry-run --config config.vastai.yaml 2>&1 | tee logs/gate1_dryrun.log | sed 's/^/[FORGE] /'
T2=$?
echo "[FORGE] Test 2 (dry run): exit $T2"

MATRIX="renders/dna_compute/dna_matrix.json"
if [ ! -f "$MATRIX" ]; then
    echo "[FORGE] FATAL: No matrix generated"
    exit 1
fi

# Test 3: Render from saved matrix (no second API call)
echo "[FORGE] Test 3: Render (Cycles + CUDA)..."
python3 parallel_render.py --matrix "$MATRIX" --config config.vastai.yaml \
    --workers 1 2>&1 | tee logs/gate1_render.log | sed 's/^/[FORGE] /'
T3=$?
echo "[FORGE] Test 3 (render): exit $T3"

PNG_COUNT=$(find renders/ -name '*.png' 2>/dev/null | wc -l)
echo "[FORGE] PNG count: $PNG_COUNT"

if [ $PNG_COUNT -gt 0 ]; then
    echo "[FORGE] *** GATE 1 PASSED ***"
    find renders/ -name '*.png' | head -5 | while read f; do
        echo "[FORGE] FRAME: $f ($(stat -c%s "$f") bytes)"
    done

    # Comparison viewer
    REPORT=$(find renders/ -name 'render_report.json' 2>/dev/null | head -1)
    if [ -n "$REPORT" ]; then
        python3 compare_viewer.py --report "$REPORT" --output renders/dna_compare.html 2>&1
        echo "[FORGE] Viewer: renders/dna_compare.html"
    fi
else
    echo "[FORGE] *** GATE 1 FAILED — 0 PNGs ***"
fi

echo "[FORGE] === GATE 1 RESULTS ==="
echo "[FORGE] Test 1 (biophysics): $T1"
echo "[FORGE] Test 2 (dry run):    $T2"
echo "[FORGE] Test 3 (render):     $T3"
echo "[FORGE] PNGs: $PNG_COUNT"
echo "[FORGE] $(date -Iseconds)"