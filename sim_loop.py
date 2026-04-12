"""
sim_loop.py -- Closed LLM feedback loop: simulate -> readout -> refine -> repeat.

Usage:
    python sim_loop.py --goal AND_gate --cascade half_adder --max-iter 3
    python sim_loop.py --genomic demo_data/sample_expression.csv --genes BRCA1 TP53 KRAS
    python sim_loop.py --dry-run --max-iter 3
"""
import argparse, json, pathlib, subprocess, sys, shutil
from llm_bridge import LLMBridge, load_config

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

REFINE_PROMPT = (
    "You are a DNA circuit engineer and Blender physics expert.\n"
    "You receive a circuit spec, simulation readout signals per variant (0=off 1=on),\n"
    "and a goal. Output ONLY an improved circuit description as plain text.\n"
    "State what you are changing and why, then give the new circuit spec."
)

READOUT_PROMPT = (
    "Estimate output signal (0.0-1.0) per variant from the render report.\n"
    "Output ONLY valid JSON: {\"variant_id\": signal_float, ...}"
)


def extract_readout_pixels(report_path, output_region=None):
    if not HAS_PIL or not report_path.exists():
        return {}
    report = json.loads(report_path.read_text())
    signals = {}
    for vid, info in report.get("variants", {}).items():
        frames = info.get("frames", [])
        if not frames:
            signals[vid] = 0.0
            continue
        fp = pathlib.Path(frames[-1])
        if not fp.exists():
            signals[vid] = 0.0
            continue
        img = Image.open(fp).convert("L")
        w, h = img.size
        region = output_region or (w//2, h//2, w, h)
        arr = np.array(img.crop(region), dtype=float)
        signals[vid] = round(float(np.mean(arr > 200)), 4)
    return signals


def extract_readout_llm(bridge, report_path):
    import re
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text())
    variants = report.get("variants", {})
    if not variants:
        return {}
    summary = json.dumps(
        {vid: {k: v for k, v in info.items() if k != "frames"}
         for vid, info in variants.items()},
        indent=2
    )
    raw = bridge.ask("Simulation report:\n" + summary + "\nEstimate output signal 0.0-1.0 per variant.")
    raw = re.sub(r"^```[\w]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {vid: 0.5 for vid in variants}


def evaluate(signals, goal_type="maximize_any"):
    if not signals:
        return {"_best": None, "_mean": 0.0, "_converged": False}
    best_vid  = max(signals, key=signals.get)
    mean_sig  = sum(signals.values()) / len(signals)
    converged = False
    if goal_type == "maximize_any":
        converged = max(signals.values()) > 0.8
    elif goal_type == "AND_gate":
        and_v = [v for v in signals if "both" in v.lower() or "and" in v.lower()]
        not_v = [v for v in signals if v not in and_v]
        converged = (max((signals[v] for v in and_v), default=0) > 0.7 and
                     max((signals[v] for v in not_v), default=1) < 0.2)
    elif goal_type == "threshold":
        above = [v for v in signals if "high" in v.lower() or "above" in v.lower()]
        below = [v for v in signals if "low" in v.lower() or "below" in v.lower()]
        converged = (max((signals[v] for v in above), default=0) > 0.6 and
                     max((signals[v] for v in below), default=1) < 0.3)
    result = dict(signals)
    result["_best"]      = best_vid
    result["_mean"]      = round(mean_sig, 4)
    result["_converged"] = converged
    return result


class SimLoop:
    def __init__(self, config_path="config.yaml"):
        self.cfg         = load_config(config_path)
        self.bridge      = LLMBridge(config_path)
        self.config_path = config_path
        self.history     = []

    def run(self, initial_prompt, max_iter=4, workers=2, goal_type="maximize_any",
            physics_type="geometry_nodes", n_variants=4, frame_end=72, dry_run=False):
        out_dir = pathlib.Path(self.cfg["blender"]["output_dir"]).expanduser() / "sim_loop"
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = initial_prompt
        best_signals = {}
        evaluation = {}

        print("\n" + "="*60)
        print("SIM LOOP: goal=" + goal_type + "  max_iter=" + str(max_iter))
        print("="*60)

        for i in range(max_iter):
            iter_dir = out_dir / ("iter_" + str(i+1).zfill(2))
            iter_dir.mkdir(parents=True, exist_ok=True)
            matrix_path = iter_dir / "matrix.json"
            report_path = iter_dir / "render_report.json"

            print("\n[iter " + str(i+1) + "/" + str(max_iter) + "] Generating...")

            cmd = [
                sys.executable, "physics_run.py", prompt,
                "--type", physics_type,
                "--variants", str(n_variants),
                "--frame-end", str(frame_end),
                "--workers", str(workers),
                "--config", self.config_path,
            ]
            if dry_run:
                cmd.append("--dry-run")
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.stderr:
                print("[sim_loop] STDERR:", res.stderr[-1000:])
            if res.stdout:
                print(res.stdout[-2000:])

            # Copy outputs to iter dir
            src_dir = pathlib.Path(self.cfg["blender"]["output_dir"]).expanduser() / "physics_matrix"
            for fname in ["matrix.json", "render_report.json"]:
                src_file = src_dir / fname
                dst_file = iter_dir / fname
                if src_file.exists():
                    shutil.copy(src_file, dst_file)

            # Readout
            signals = {}
            if not dry_run and report_path.exists():
                signals = extract_readout_pixels(report_path) if HAS_PIL else {}
                # Fall back to LLM readout if pixel signals show no variation
                if not signals:
                    signals = extract_readout_llm(self.bridge, report_path)
                elif len(set(round(v, 3) for v in signals.values())) <= 1:
                    print("[readout] Pixel signals identical -- falling back to LLM readout")
                    llm_signals = extract_readout_llm(self.bridge, report_path)
                    if llm_signals:
                        signals = llm_signals
            if dry_run:
                import random; random.seed(i * 37)
                signals = {"v" + str(j+1).zfill(3): round(random.uniform(0.1, 0.9), 3)
                           for j in range(n_variants)}
            print("[signals] " + json.dumps(signals))

            # Evaluate
            evaluation = evaluate(signals, goal_type)
            self.history.append({"iter": i+1, "prompt_len": len(prompt),
                                  "signals": signals, "evaluation": evaluation})
            print("[eval] mean=" + str(evaluation["_mean"]) +
                  "  best=" + str(evaluation.get("_best")) +
                  "  converged=" + str(evaluation["_converged"]))

            if evaluation["_converged"]:
                print("\n[loop] CONVERGED at iter " + str(i+1))
                best_signals = signals
                break

            # LLM refinement
            if i < max_iter - 1:
                import llm_bridge as lb
                old_sp = lb.SYSTEM_PROMPT
                lb.SYSTEM_PROMPT = REFINE_PROMPT
                refine_input = (
                    "Goal: " + goal_type + "\n"
                    "Current spec (summary):\n" + prompt[:500] + "\n\n"
                    "Signals: " + json.dumps(signals) + "\n"
                    "Best: " + str(evaluation.get("_best")) + "  Mean: " + str(evaluation["_mean"])
                )
                prompt = self.bridge.ask(refine_input)
                lb.SYSTEM_PROMPT = old_sp
                print("[refine] new prompt length: " + str(len(prompt)))

        # Save history
        hist_path = out_dir / "loop_history.json"
        hist_path.write_text(json.dumps(self.history, indent=2))
        print("\n[loop] History -> " + str(hist_path))
        return best_signals, evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goal",     default="maximize_any",
                   choices=["maximize_any", "AND_gate", "threshold"])
    p.add_argument("--prompt",   default=None)
    p.add_argument("--genomic",  default=None)
    p.add_argument("--genes",    nargs="*", default=["BRCA1","TP53","KRAS"])
    p.add_argument("--cascade",  default=None)
    p.add_argument("--max-iter", type=int, default=3)
    p.add_argument("--workers",  type=int, default=2)
    p.add_argument("--frame-end", type=int, default=24)
    p.add_argument("--variants", type=int, default=4)
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--provider", default=None)
    p.add_argument("--dry-run",  action="store_true")
    args = p.parse_args()

    loop = SimLoop(args.config)
    if args.provider:
        loop.bridge.set_provider(args.provider)

    if args.genomic:
        from genomic_input import GenomicInput
        gi = GenomicInput()
        gi.load_rnaseq(args.genomic)
        prompt = gi.to_diagnostic_circuit_prompt(args.genes)
    elif args.cascade:
        from dna_gates import DNACascade
        prompt = DNACascade.preset(args.cascade).to_llm_prompt()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = ("DNA AND gate: activate output only when BRCA1 AND TP53 "
                  "strand concentrations are both above 200 nM threshold.")

    signals, evaluation = loop.run(
        initial_prompt=prompt,
        max_iter=args.max_iter,
        workers=args.workers,
        goal_type=args.goal,
        n_variants=args.variants,
        frame_end=args.frame_end,
        dry_run=args.dry_run,
    )
    print("\nFinal signals: " + json.dumps(signals, indent=2))
    print("Converged: " + str(evaluation.get("_converged")))


if __name__ == "__main__":
    main()