#!/data/data/com.termux/files/usr/bin/bash
# bootstrap.sh — S26 Termux setup for Blender LLM Pipeline
set -e
echo "=== Cloud-Eye Blender LLM Pipeline Bootstrap ==="

pkg update -y && pkg upgrade -y
pkg install -y proot-distro wget curl python

echo "[1/4] Installing Debian container..."
proot-distro install debian 2>/dev/null || echo "Already installed."

echo "[2/4] Setting up Debian..."
proot-distro login debian -- bash -c "
  apt-get update -qq && apt-get install -y blender ffmpeg python3 python3-pip python3-yaml -qq
  pip3 install requests pyyaml --quiet
  curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null || true
  echo Done.
"

echo "[3/4] Creating proot launcher..."
DEST=\$(cd -- "\$(dirname -- "\$0")" && pwd)
cat > "\$DEST/run.sh" << 'RUN'
#!/data/data/com.termux/files/usr/bin/bash
# Usage: ./run.sh [--provider X] [--model Y] "scene description..."
SCRIPT_DIR=\$(cd -- "\$(dirname -- "\$0")" && pwd)
proot-distro login debian -- bash -c "
  ollama serve > /tmp/ollama.log 2>&1 &
  sleep 2
  cd \$SCRIPT_DIR
  python3 scene_pipeline.py \$*
"
RUN
chmod +x "\$DEST/run.sh"

echo "[4/4] Pull a model (edit to taste):"
echo "  proot-distro login debian -- ollama pull qwen2.5-coder:7b"
echo ""
echo "=== Done. Quick start ==="
echo "  ./run.sh \"cinematic asteroid field\""
echo "  proot-distro login debian -- python3 stitch.py"
echo ""
echo "Switch providers: edit config.yaml -> active: ollama|openai_compat|anthropic|lxr5|hf"
