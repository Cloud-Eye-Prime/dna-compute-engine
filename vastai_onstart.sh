#!/bin/bash
# vastai_onstart.sh — Proven onstart script for vast.ai DNA Compute Engine
# Use with: forge_launch(custom_onstart=<this script>, offer_id=<Quebec/US host>)
#
# Requirements:
#   - Docker image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
#   - Host: Non-China (needs api.anthropic.com access)
#   - Host: Quebec CA (25436523) or Montana US (33173224) confirmed working
#   - ANTHROPIC_API_KEY must be set in this script or via env
#
# Proven configuration (Gate 1 passed 2026-04-12):
#   - Quebec 2x RTX 3090, offer 25436523, $0.15/hr
#   - Blender 4.2.0 LTS, Cycles + CUDA
#   - Sonnet 4.6 via Anthropic API (300s timeout)
#   - Two-step: dry-run matrix gen -> parallel_render from saved matrix

export FORGE_STATUS="/workspace/forge_status.json"
export DEBIAN_FRONTEND=noninteractive

echo "[FORGE] === DNA COMPUTE ENGINE ==="
echo "[FORGE] $(date -Iseconds)"
echo '{"phase": "setup"}' > $FORGE_STATUS

# System deps
apt-get update -qq
apt-get install -y -qq vim curl wget unzip git ffmpeg \
    libglu1-mesa-dev libxi6 libxrender1 libfontconfig1 \
    libxxf86vm-dev libxfixes-dev libgl1-mesa-glx libxkbcommon0 libsm6 \
    > /dev/null 2>&1
echo "[FORGE] System deps done"

# Blender 4.2 LTS
cd /tmp
wget -q "https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz" -O blender.tar.xz
mkdir -p /opt/blender
tar xf blender.tar.xz --strip-components=1 -C /opt/blender
rm blender.tar.xz
ln -sf /opt/blender/blender /usr/local/bin/blender
echo "[FORGE] Blender: $(blender --version 2>/dev/null | head -1)"

# Python deps
pip install -q numpy Pillow PyYAML requests anthropic > /dev/null 2>&1
echo "[FORGE] Python deps done"
echo "[FORGE] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

# Clone repo
cd /workspace
git clone https://github.com/Cloud-Eye-Prime/dna-compute-engine.git 2>&1 | tail -2
cd /workspace/dna-compute-engine
mkdir -p renders logs
echo "[FORGE] Repo: $(git log --oneline -1)"

# API key — SET THIS before deploying
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-SET_YOUR_KEY_HERE}"

echo '{"phase": "running"}' > $FORGE_STATUS
echo "[FORGE] === SETUP COMPLETE ==="