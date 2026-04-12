"""
parallel_render.py — Launch N Blender headless instances in parallel.

Each variant from the physics matrix runs in its own subprocess.
Concurrency is capped by max_workers to respect S26 RAM budget.

Usage:
    python parallel_render.py --matrix my_sim_matrix.json
    python parallel_render.py --matrix my_sim_matrix.json --workers 2
    python parallel_render.py --matrix my_sim_matrix.json --variants v001 v003
"""
import subprocess, time, pathlib, argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_bridge import load_config
from physics_matrix import MatrixResult

EXEC_SCRIPT = str(pathlib.Path(__file__).parent / "physics_exec.py")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix",   required=True, help="Path to matrix JSON")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--workers",  type=int, default=None,
                   help="Max parallel Blender instances (default: auto from RAM budget)")
    p.add_argument("--variants", nargs="*", default=None,
                   help="Render only these variant IDs (default: all)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print commands without executing")
    return p.parse_args()


def auto_workers(cfg):
    """Estimate safe parallel count from RAM budget in config."""
    ram_mb = cfg.get("hardware", {}).get("ram_budget_mb", 6000)
    blender_mb = cfg.get("hardware", {}).get("blender_ram_per_instance_mb", 2000)
    return max(1, ram_mb // blender_mb)


def run_variant(blender_bin, variant_id, code_path, output_dir,
                quality_label, dry_run=False):
    cmd = [
        blender_bin, "--background",
        "--python", EXEC_SCRIPT,
        "--",
        "--code", code_path
    ]
    tag = f"[{variant_id}] {quality_label}"
    if dry_run:
        print(f"  DRY-RUN {tag}: {' '.join(cmd)}")
        return variant_id, True, 0.0, []
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else "FAIL"
    print(f"  {status} {tag}  ({elapsed:.1f}s)")
    if not ok:
        print(result.stderr[-800:])
    # collect rendered frames
    frames = sorted(pathlib.Path(output_dir).glob("frame_*.png"))
    return variant_id, ok, elapsed, [str(f) for f in frames]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    blender_bin = cfg["blender"].get("executable", "blender")

    matrix = MatrixResult.load(args.matrix)
    out_dir = pathlib.Path(cfg["blender"]["output_dir"]).expanduser() / "physics_matrix"
    variant_files = matrix.prepare_variant_files(out_dir)

    # Filter variants if requested
    if args.variants:
        variant_files = [v for v in variant_files if v[0] in args.variants]

    max_w = args.workers or auto_workers(cfg)
    print(f"\n{'='*60}")
    print(f"Physics Matrix: {matrix.physics_type}")
    print(f"Prompt: {matrix.prompt}")
    print(f"Variants: {len(variant_files)}  |  Workers: {max_w}")
    print(f"Output: {out_dir}")
    print("="*60)

    results = {}
    with ThreadPoolExecutor(max_workers=max_w) as pool:
        futures = {
            pool.submit(run_variant, blender_bin,
                        vid, code_path, str(vdir), qlabel, args.dry_run): vid
            for vid, code_path, vdir, qlabel, _ in variant_files
        }
        for future in as_completed(futures):
            vid, ok, elapsed, frames = future.result()
            results[vid] = {"ok": ok, "elapsed": elapsed, "frames": frames}

    # Summary
    ok_count = sum(1 for r in results.values() if r["ok"])
    print(f"\n[parallel] {ok_count}/{len(results)} variants rendered successfully.")
    report = {
        "matrix_file": str(args.matrix),
        "physics_type": matrix.physics_type,
        "variants": {
            v[0]: {
                "quality_label": v[3],
                "description": v[4],
                **results.get(v[0], {"ok": False, "frames": []})
            }
            for v in variant_files
        }
    }
    report_path = out_dir / "render_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[parallel] Report -> {report_path}")
    print(f"[parallel] View:    python compare_viewer.py --report {report_path}")

if __name__ == "__main__":
    main()
