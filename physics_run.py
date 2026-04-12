"""
physics_run.py — One-shot command: LLM matrix -> parallel render -> HTML comparison

Usage:
    python physics_run.py "cloth sphere falling on rigid spikes" --type cloth --variants 6
    python physics_run.py "fluid pouring" --type fluid --variants 4 --workers 2
    python physics_run.py "shattered glass" --type rigid_body --variants 8 --frame-end 96
"""
import argparse, pathlib, subprocess, sys
from llm_bridge import LLMBridge, load_config
from physics_matrix import PhysicsMatrix

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("description", help="Scene + physics description")
    p.add_argument("--type",      default="rigid_body",
                   choices=["rigid_body","soft_body","cloth","fluid",
                            "particles","geometry_nodes"])
    p.add_argument("--variants",  type=int, default=4)
    p.add_argument("--frame-end", type=int, default=72)
    p.add_argument("--workers",   type=int, default=2)
    p.add_argument("--config",    default="config.yaml")
    p.add_argument("--provider",  default=None)
    p.add_argument("--model",     default=None)
    p.add_argument("--frame",     type=int, default=24,
                   help="Which frame to show in comparison viewer")
    p.add_argument("--dry-run",   action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = pathlib.Path(cfg["blender"]["output_dir"]).expanduser() / "physics_matrix"

    # Step 1: Generate matrix
    print(f"\n[1/3] Generating {args.variants}-variant {args.type} matrix via LLM...")
    pm = PhysicsMatrix(args.config)
    if args.provider:
        pm.bridge.set_provider(args.provider)

    matrix = pm.generate(
        args.description,
        physics_type=args.type,
        n_variants=args.variants,
        frame_end=args.frame_end,
        model=args.model
    )
    matrix_path = out_dir / "matrix.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix.save(str(matrix_path))

    if args.dry_run:
        print("[physics_run] --dry-run: prepared matrix files, skipping render.")
        matrix.prepare_variant_files(out_dir)
        return

    # Step 2: Parallel render
    print(f"\n[2/3] Parallel rendering {args.variants} variants (workers={args.workers})...")
    subprocess.run([
        sys.executable, "parallel_render.py",
        "--matrix", str(matrix_path),
        "--config", args.config,
        "--workers", str(args.workers),
    ], check=False)

    # Step 3: HTML comparison
    report = out_dir / "render_report.json"
    viewer = out_dir / "matrix_compare.html"
    print(f"\n[3/3] Generating HTML comparison viewer...")
    subprocess.run([
        sys.executable, "compare_viewer.py",
        "--report", str(report),
        "--frame",  str(args.frame),
        "--output", str(viewer),
    ], check=False)
    print(f"\n[physics_run] Done.")
    print(f"  Viewer: {viewer}")
    print(f"  Matrix: {matrix_path}")

if __name__ == "__main__":
    main()
