#!/bin/bash
set -e

echo "============================================"
echo "DNA Compute Engine -- Vast.ai Setup"
echo "============================================"

# --- System packages ---
echo "[1/7] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    vim curl wget unzip git ffmpeg \
    libglu1-mesa-dev libxi6 libxrender1 \
    libfontconfig1 libxxf86vm-dev libxfixes-dev \
    libgl1-mesa-glx libxkbcommon0 libsm6 \
    python3 python3-pip python3-venv \
    > /dev/null 2>&1
echo "    Done."

# --- Blender ---
BLENDER_VERSION="4.2.0"
BLENDER_URL="https://download.blender.org/release/Blender4.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_DIR="/opt/blender"

if [ ! -d "$BLENDER_DIR" ]; then
    echo "[2/7] Downloading Blender ${BLENDER_VERSION}..."
    cd /tmp
    wget -q "$BLENDER_URL" -O blender.tar.xz
    echo "    Extracting..."
    mkdir -p "$BLENDER_DIR"
    tar xf blender.tar.xz --strip-components=1 -C "$BLENDER_DIR"
    rm blender.tar.xz
    ln -sf "$BLENDER_DIR/blender" /usr/local/bin/blender
    echo "    Done."
else
    echo "[2/7] Blender already installed, skipping."
fi

# --- Python packages ---
echo "[3/7] Installing Python dependencies..."
pip3 install -q numpy Pillow PyYAML requests anthropic > /dev/null 2>&1
echo "    Done."

# --- GPU CUDA enablement script for Blender ---
echo "[4/7] Creating Blender GPU enablement script..."
cat > /opt/blender/gpu_enable.py << 'GPUPY'
"""
gpu_enable.py -- Force Blender to use CUDA GPU for Cycles rendering.
Run via: blender --background --python gpu_enable.py
"""
import bpy

prefs = bpy.context.preferences
cycles_prefs = prefs.addons['cycles'].preferences
cycles_prefs.compute_device_type = 'CUDA'
cycles_prefs.get_devices()

for device in cycles_prefs.devices:
    if device.type == 'CUDA':
        device.use = True
        print(f"[gpu] Enabled: {device.name}")
    elif device.type == 'CPU':
        device.use = False

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
print("[gpu] CUDA GPU rendering enabled for Cycles.")
GPUPY
echo "    Done."

# --- Workspace ---
echo "[5/7] Setting up workspace..."
WORKSPACE="/workspace/dna-compute-engine"
mkdir -p "$WORKSPACE/renders" "$WORKSPACE/logs"
echo "    Done."

# --- Config ---
echo "[6/7] Checking config..."
if [ -f "$WORKSPACE/config.vastai.yaml" ]; then
    echo "    config.vastai.yaml found."
else
    echo "    WARNING: config.vastai.yaml not found in $WORKSPACE"
    echo "    Clone the repo first: git clone https://github.com/Cloud-Eye-Prime/dna-compute-engine.git $WORKSPACE"
fi
echo "    Done."

# --- Validation ---
echo "[7/7] Validating installation..."

echo -n "    Blender: "
blender --version 2>/dev/null | head -1 || echo "FAILED"

echo -n "    NVIDIA GPU: "
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "FAILED"

echo -n "    CUDA: "
nvcc --version 2>/dev/null | grep "release" || echo "FAILED (nvcc not found, may still work)"

echo -n "    Python: "
python3 --version 2>/dev/null || echo "FAILED"

echo -n "    NumPy: "
python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "FAILED"

echo -n "    Pillow: "
python3 -c "import PIL; print(PIL.__version__)" 2>/dev/null || echo "FAILED"

echo -n "    ffmpeg: "
ffmpeg -version 2>/dev/null | head -1 || echo "FAILED"

# GPU render test
echo ""
echo "    Running GPU render test..."
blender --background --python /opt/blender/gpu_enable.py \
    --python-expr "
import bpy
bpy.ops.mesh.primitive_ico_sphere_add(radius=1)
bpy.context.scene.render.filepath = '/tmp/gpu_test.png'
bpy.context.scene.render.resolution_x = 320
bpy.context.scene.render.resolution_y = 240
bpy.context.scene.cycles.samples = 16
bpy.ops.render.render(write_still=True)
print('[gpu_test] Render complete.')
" 2>&1 | tail -5

if [ -f /tmp/gpu_test.png ]; then
    echo "    GPU RENDER TEST: PASSED"
    rm /tmp/gpu_test.png
else
    echo "    GPU RENDER TEST: FAILED"
    echo "    Check NVIDIA drivers and CUDA installation."
fi

echo ""
echo "============================================"
echo "Setup complete."
echo "Workspace: $WORKSPACE"
echo ""
echo "Next steps:"
echo "  1. Clone repo: git clone https://github.com/Cloud-Eye-Prime/dna-compute-engine.git $WORKSPACE"
echo "  2. Set API key: export ANTHROPIC_API_KEY=sk-ant-..."
echo "  3. Dry run:     cd $WORKSPACE && python3 dna_compute.py --sequence GCGCATCGATGCGC --dry-run"
echo "  4. Render:      python3 dna_compute.py --sequence GCGCATCGATGCGC --variants 4 --workers 2"
echo "============================================"
