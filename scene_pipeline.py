"""
scene_pipeline.py — Main orchestrator: LLM -> bpy code -> blender headless -> frames

Usage:
    python scene_pipeline.py                           # reads scenes.yaml
    python scene_pipeline.py "asteroid field" "dusk"   # inline prompts
    python scene_pipeline.py --provider ollama --model qwen2.5-coder:14b "scene"
    python scene_pipeline.py --dry-run "scene"         # generate code only
"""
import sys, os, subprocess, tempfile, pathlib, yaml, argparse, time
from llm_bridge import LLMBridge, load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("prompts", nargs="*",
                   help="Scene descriptions. Empty = read scenes.yaml")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--provider", default=None, help="Override provider")
    p.add_argument("--model",    default=None, help="Override model")
    p.add_argument("--dry-run",  action="store_true",
                   help="Generate bpy code but skip rendering")
    return p.parse_args()


def load_scenes(cfg, inline_prompts):
    if inline_prompts:
        return [{"name": f"scene_{i:03d}", "prompt": p}
                for i, p in enumerate(inline_prompts)]
    with open(cfg["pipeline"]["scenes_file"]) as f:
        return yaml.safe_load(f)["scenes"]


def render_scene(blender_bin, exec_script, code_path, scene_name, out_dir):
    cmd = [
        blender_bin, "--background",
        "--python", exec_script,
        "--",
        "--code", code_path
    ]
    print(f"[render] {scene_name}: {" ".join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"[render] FAILED in {elapsed:.1f}s")
        print(result.stderr[-1500:])
        return False
    print(f"[render] OK in {elapsed:.1f}s")
    return True


def main():
    args = parse_args()
    cfg = load_config(args.config)

    bridge = LLMBridge(args.config)
    if args.provider:
        bridge.set_provider(args.provider)

    scenes = load_scenes(cfg, args.prompts)
    bcfg = cfg["blender"]
    out_dir = pathlib.Path(bcfg["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    exec_script = str(pathlib.Path(__file__).parent / "blender_scene_exec.py")

    for i, scene in enumerate(scenes):
        name = scene.get("name", f"scene_{i:03d}")
        prompt = scene["prompt"]
        model_label = args.model or bridge.pcfg.get("model", "?")
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(scenes)}] {name}  |  {bridge.provider}/{model_label}")
        print(f"Prompt: {prompt}")

        frame_dir = out_dir / name
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = str(frame_dir / "frame_####")

        code = bridge.generate_bpy_code(prompt, frame_path, model=args.model)
        code_file = out_dir / f"{name}.py"
        code_file.write_text(code)
        print(f"[pipeline] bpy code -> {code_file}  ({len(code)} chars)")

        if args.dry_run:
            print("[pipeline] --dry-run: skipping render")
            continue

        render_scene(
            bcfg.get("executable", "blender"),
            exec_script,
            str(code_file),
            name,
            out_dir
        )

    if not args.dry_run:
        print("\n[pipeline] Done. Run: python stitch.py")


if __name__ == "__main__":
    main()
