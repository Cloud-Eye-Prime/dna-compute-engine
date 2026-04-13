"""
sim_loop.py -- Closed LLM feedback loop: simulate -> readout -> refine -> repeat.

Supports two backends:
  - blender: LLM generates bpy code, Blender renders, pixel readout (original)
  - icrn:    ICRN mass-action kinetics simulation, concentration readout (new)

Usage:
    # ICRN backend (recommended — fast, reliable, differentiable)
    python sim_loop.py --backend icrn --goal AND_gate --gate AND --max-iter 3
    python sim_loop.py --backend icrn --cascade half_adder --max-iter 3

    # Blender backend (original — visualization, slower, less reliable)
    python sim_loop.py --backend blender --goal AND_gate --cascade half_adder --max-iter 3

    # Genomic input (works with either backend)
    python sim_loop.py --backend icrn --genomic demo_data/sample_expression.csv --genes BRCA1 TP53 KRAS

    # Dry run
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

try:
    from icrn_bridge import (
        gate_to_icrn, simulate_gate, simulate_cascade,
        cascade_to_icrn, GATE_BUILDERS, DEFAULT_EXP_PARAMS, SJDict,
    )
    import jax.numpy as jnp
    HAS_ICRN = True
except ImportError:
    HAS_ICRN = False

REFINE_PROMPT_BLENDER = (
    "You are a DNA circuit engineer and Blender physics expert.\n"
    "You receive a circuit spec, simulation readout signals per variant (0=off 1=on),\n"
    "and a goal. Output ONLY an improved circuit description as plain text.\n"
    "State what you are changing and why, then give the new circuit spec."
)

REFINE_PROMPT_ICRN = (
    "You are a DNA circuit engineer specializing in strand displacement circuits.\n"
    "You receive a circuit spec with ICRN concentration readouts (in nM) per variant,\n"
    "and a design goal. The simulation uses mass-action kinetics.\n\n"
    "For each variant, you see the output concentration at simulation end.\n"
    "Higher output = stronger signal. A good AND gate shows high output when both\n"
    "inputs are present and near-zero when either is absent.\n\n"
    "Output ONLY an improved circuit description as plain text.\n"
    "You may adjust: toehold lengths (4-10nt), sequences, salt concentration (50-500mM),\n"
    "temperature (25-45C), input concentrations (10-1000nM), gate concentrations.\n"
    "State what you are changing and why, then give the new parameters as JSON."
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
    def __init__(self, config_path="config.yaml", backend="blender"):
        self.cfg         = load_config(config_path)
        self.bridge      = LLMBridge(config_path)
        self.config_path = config_path
        self.backend     = backend
        self.history     = []

        if backend == "icrn" and not HAS_ICRN:
            print("[WARN] ICRN backend requested but icrn_bridge not available.")
            print("  Install: pip install git+https://github.com/SwissChardLeaf/icrn.git")
            print("  Dry-run mode will still work. Live simulation will fail.")

    def _run_icrn_variants(self, gate_type, n_variants, iter_num,
                           time=100.0, sample_num=50, **gate_kwargs):
        """
        Generate N variants by sweeping toehold length and concentration.
        Returns signals dict {variant_id: output_concentration_nM}.
        """
        signals = {}
        variant_params = []

        # Variant generation: sweep toehold length and input concentration
        base_toehold = gate_kwargs.get("toehold_len", 6)
        base_conc = 100.0
        for v in range(n_variants):
            # Vary toehold: base +/- 2nt across variants
            toehold = max(3, base_toehold - 1 + (v % 4))
            # Vary concentration: 50-200nM
            conc_scale = 0.5 + (v / max(n_variants - 1, 1)) * 1.5
            input_conc = base_conc * conc_scale

            kw = dict(gate_kwargs)
            kw["toehold_len"] = toehold

            vid = f"v{str(v+1).zfill(3)}_th{toehold}_c{int(input_conc)}"
            try:
                # Set input concentrations based on gate type
                if gate_type.upper() in ("AND",):
                    input_overrides = {"input_A": input_conc, "input_B": input_conc}
                elif gate_type.upper() in ("NOT",):
                    input_overrides = {"input": input_conc}
                elif gate_type.upper() in ("THRESHOLD",):
                    input_overrides = {"input": input_conc * 2}  # above threshold
                else:
                    input_overrides = {}

                signal, traj, smap = simulate_gate(
                    gate_type, input_concs=input_overrides,
                    time=time, sample_num=sample_num, **kw
                )
                signals[vid] = round(signal, 4)
                variant_params.append({
                    "id": vid, "toehold_len": toehold,
                    "input_conc_nM": round(input_conc, 1),
                    "output_nM": round(signal, 4),
                })
            except Exception as e:
                print(f"[ICRN] Variant {vid} failed: {e}")
                signals[vid] = 0.0
                variant_params.append({"id": vid, "error": str(e)})

        return signals, variant_params

    def run(self, initial_prompt, max_iter=4, workers=2, goal_type="maximize_any",
            physics_type="geometry_nodes", n_variants=4, frame_end=72, dry_run=False,
            gate_type=None, sim_time=100.0, **gate_kwargs):
        out_dir = pathlib.Path(self.cfg["blender"]["output_dir"]).expanduser() / "sim_loop"
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = initial_prompt
        best_signals = {}
        evaluation = {}

        print("\n" + "="*60)
        print(f"SIM LOOP: backend={self.backend}  goal={goal_type}  max_iter={max_iter}")
        if self.backend == "icrn" and gate_type:
            print(f"  gate={gate_type}  sim_time={sim_time}s  variants={n_variants}")
        print("="*60)

        for i in range(max_iter):
            iter_dir = out_dir / ("iter_" + str(i+1).zfill(2))
            iter_dir.mkdir(parents=True, exist_ok=True)

            print("\n[iter " + str(i+1) + "/" + str(max_iter) + "] Simulating...")

            signals = {}
            variant_details = []

            if dry_run:
                import random; random.seed(i * 37)
                signals = {"v" + str(j+1).zfill(3): round(random.uniform(0.1, 0.9), 3)
                           for j in range(n_variants)}

            elif self.backend == "icrn":
                # ── ICRN backend: direct concentration simulation ──
                if gate_type is None:
                    gate_type = "AND"  # default
                signals, variant_details = self._run_icrn_variants(
                    gate_type, n_variants, i,
                    time=sim_time, sample_num=50, **gate_kwargs
                )
                # Save variant details
                details_path = iter_dir / "icrn_variants.json"
                details_path.write_text(json.dumps(variant_details, indent=2))

            else:
                # ── Blender backend: original render pipeline ──
                matrix_path = iter_dir / "matrix.json"
                report_path = iter_dir / "render_report.json"

                cmd = [
                    sys.executable, "physics_run.py", prompt,
                    "--type", physics_type,
                    "--variants", str(n_variants),
                    "--frame-end", str(frame_end),
                    "--workers", str(workers),
                    "--config", self.config_path,
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.stderr:
                    print("[sim_loop] STDERR:", res.stderr[-1000:])
                if res.stdout:
                    print(res.stdout[-2000:])

                src_dir = pathlib.Path(self.cfg["blender"]["output_dir"]).expanduser() / "physics_matrix"
                for fname in ["matrix.json", "render_report.json"]:
                    src_file = src_dir / fname
                    dst_file = iter_dir / fname
                    if src_file.exists():
                        shutil.copy(src_file, dst_file)

                if report_path.exists():
                    signals = extract_readout_pixels(report_path) if HAS_PIL else {}
                    if not signals:
                        signals = extract_readout_llm(self.bridge, report_path)
                    elif len(set(round(v, 3) for v in signals.values())) <= 1:
                        print("[readout] Pixel signals identical -- falling back to LLM readout")
                        llm_signals = extract_readout_llm(self.bridge, report_path)
                        if llm_signals:
                            signals = llm_signals

            print("[signals] " + json.dumps(signals))

            # Evaluate (normalize ICRN signals to 0-1 for evaluation)
            if self.backend == "icrn" and signals:
                max_signal = max(signals.values()) if max(signals.values()) > 0 else 1.0
                eval_signals = {k: round(v / max(max_signal, 1.0), 4) for k, v in signals.items()}
            else:
                eval_signals = signals

            evaluation = evaluate(eval_signals, goal_type)
            self.history.append({
                "iter": i+1, "backend": self.backend,
                "gate_type": gate_type if self.backend == "icrn" else None,
                "signals_raw": signals,
                "signals_normalized": eval_signals,
                "evaluation": evaluation,
                "variant_details": variant_details if self.backend == "icrn" else [],
            })
            print("[eval] mean=" + str(evaluation["_mean"]) +
                  "  best=" + str(evaluation.get("_best")) +
                  "  converged=" + str(evaluation["_converged"]))

            if evaluation["_converged"]:
                print("\n[loop] CONVERGED at iter " + str(i+1))
                best_signals = signals
                break

            # LLM refinement (skip in dry-run: no LLM available)
            if dry_run:
                print("[refine] dry-run: skipping LLM refinement")
            elif i < max_iter - 1:
                import llm_bridge as lb
                old_sp = lb.SYSTEM_PROMPT
                if self.backend == "icrn":
                    lb.SYSTEM_PROMPT = REFINE_PROMPT_ICRN
                    refine_input = (
                        f"Goal: {goal_type}\n"
                        f"Gate type: {gate_type}\n"
                        f"Backend: ICRN (mass-action kinetics)\n\n"
                        f"Variant results (output concentration in nM):\n"
                        + json.dumps(variant_details, indent=2) + "\n\n"
                        f"Best: {evaluation.get('_best')}  "
                        f"Mean output: {evaluation['_mean']:.4f}\n"
                        f"Suggest parameter changes to improve the circuit."
                    )
                else:
                    lb.SYSTEM_PROMPT = REFINE_PROMPT_BLENDER
                    refine_input = (
                        "Goal: " + goal_type + "\n"
                        "Current spec (summary):\n" + prompt[:500] + "\n\n"
                        "Signals: " + json.dumps(signals) + "\n"
                        "Best: " + str(evaluation.get("_best")) +
                        "  Mean: " + str(evaluation["_mean"])
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
    p = argparse.ArgumentParser(description="DNA Compute sim loop (Blender or ICRN backend)")
    p.add_argument("--backend",  default="blender", choices=["blender", "icrn"],
                   help="Simulation backend: blender (render) or icrn (mass-action kinetics)")
    p.add_argument("--goal",     default="maximize_any",
                   choices=["maximize_any", "AND_gate", "threshold"])
    p.add_argument("--gate",     default=None,
                   choices=["AND", "OR", "NOT", "THRESHOLD", "CATALYST", "SEESAW", "HAIRPIN"],
                   help="Gate type for ICRN backend")
    p.add_argument("--prompt",   default=None)
    p.add_argument("--genomic",  default=None)
    p.add_argument("--genes",    nargs="*", default=["BRCA1","TP53","KRAS"])
    p.add_argument("--cascade",  default=None)
    p.add_argument("--max-iter", type=int, default=3)
    p.add_argument("--workers",  type=int, default=2)
    p.add_argument("--frame-end", type=int, default=24)
    p.add_argument("--variants", type=int, default=4)
    p.add_argument("--sim-time", type=float, default=100.0,
                   help="ICRN simulation time in seconds")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--provider", default=None)
    p.add_argument("--dry-run",  action="store_true")
    args = p.parse_args()

    loop = SimLoop(args.config, backend=args.backend)
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

    # Determine gate type for ICRN backend
    gate_type = args.gate
    if args.backend == "icrn" and gate_type is None:
        # Infer from goal
        if args.goal == "AND_gate":
            gate_type = "AND"
        elif args.goal == "threshold":
            gate_type = "THRESHOLD"
        else:
            gate_type = "AND"  # default
        print(f"[ICRN] Inferred gate type: {gate_type}")

    signals, evaluation = loop.run(
        initial_prompt=prompt,
        max_iter=args.max_iter,
        workers=args.workers,
        goal_type=args.goal,
        n_variants=args.variants,
        frame_end=args.frame_end,
        dry_run=args.dry_run,
        gate_type=gate_type,
        sim_time=args.sim_time,
    )
    print("\nFinal signals: " + json.dumps(signals, indent=2))
    print("Converged: " + str(evaluation.get("_converged")))


if __name__ == "__main__":
    main()