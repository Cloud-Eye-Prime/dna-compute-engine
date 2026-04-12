"""
dna_compute.py — DNA computation engine: sequences → physics matrices → Blender renders.

Usage:
    python dna_compute.py "AND gate cascade" --cascade half_adder --variants 4
    python dna_compute.py --sequence GCGCATCGATGCGC --variants 6
    python dna_compute.py --gate CATALYST --dry-run
    python dna_compute.py --sequence ATCGATCG --info
"""
import argparse, json, pathlib, subprocess, sys
from dna_physics_map import DNAtoBlender
from dna_gates import DNAGate, DNACascade
from physics_matrix import PhysicsMatrix
from llm_bridge import load_config

DNA_SYSTEM_PROMPT = (
    "You are a Blender Python expert AND a DNA biophysics specialist.\n"
    "Output ONLY valid JSON matching the physics matrix structure.\n\n"
    "You are simulating DNA COMPUTATION ENGINES in Blender.\n"
    "Each variant represents a distinct molecular configuration / circuit state.\n\n"
    "DNA physics -> Blender mappings:\n"
    "  ssDNA: soft_body goal_stiffness 0.05-0.15 (flexible, Lp ~1-3nm)\n"
    "  dsDNA: soft_body goal_stiffness 0.6-0.85 (semi-rigid, Lp ~50nm)\n"
    "  Toehold binding: particle collision triggers soft_body state change\n"
    "  k_on (10^4-10^6 M-1 s-1): maps to geo_nodes force_scale\n"
    "  High salt: smaller collision_radius (Debye screening)\n"
    "  High temp: higher brownian_factor, lower goal_stiffness\n\n"
    "Visual language:\n"
    "  ssDNA strand: thin flexible tube (soft body), color=cyan\n"
    "  dsDNA duplex: thick cylinder (rigid body), color=gold\n"
    "  Toehold domain: glowing emissive section at strand end\n"
    "  Free nucleotides: small icosphere particles (brownian motion)\n"
    "  Gate activation: material emission pulse (white flash)\n"
    "  Signal output: magenta particle burst\n\n"
    "Always bake simulation before rendering."
)


class DNAComputeMatrix(PhysicsMatrix):
    def __init__(self, config_path="config.yaml"):
        import llm_bridge
        llm_bridge.SYSTEM_PROMPT = DNA_SYSTEM_PROMPT
        super().__init__(config_path)

    def from_sequence(self, sequence, n_variants=4, frame_end=96, model=None):
        bio = DNAtoBlender(sequence)
        s = bio.summary()
        bp = s["biophysics"]
        bl = s["blender"]
        prompt = (
            "DNA sequence: " + sequence + " (" + str(len(sequence)) + "nt)\n"
            "GC content: " + str(round(bp["gc_content"] * 100)) + "%\n"
            "Melting temperature: " + str(bp["melting_temp_C"]) + " C\n"
            "Persistence length: " + str(bp["persistence_nm"]) + " nm\n"
            "k_on toehold: " + str(bp["toehold_k_on"]) + "\n\n"
            "Base Blender params:\n" + json.dumps(bl, indent=2) + "\n\n"
            "Create " + str(n_variants) + " variants exploring:\n"
            "  - Different temperatures (25C, 37C, 50C, 65C)\n"
            "  - Different salt concentrations (10mM, 150mM, 500mM)\n"
            "  - Hybridized vs single-stranded state\n"
            "  - Different strand concentrations (1nM, 10nM, 100nM, 1uM)\n"
            "Each variant visually encodes a distinct molecular state."
        )
        return self.generate(prompt, physics_type="geometry_nodes",
                            n_variants=n_variants, frame_end=frame_end, model=model)

    def from_gate(self, gate, n_variants=4, frame_end=96, model=None):
        gd = gate.to_dict()
        prompt = (
            "DNA Gate: " + gate.gate_type + "\n"
            "Description: " + gd["description"] + "\n"
            "Visual hint: " + gd["visual_hint"] + "\n"
            "Base physics params:\n" + json.dumps(gd["params"], indent=2) + "\n\n"
            "Create " + str(n_variants) + " variants exploring:\n"
            "  - Different toehold lengths (3, 5, 7, 10nt)\n"
            "  - Different reaction rates (slow leak to fast cascade)\n"
            "  - Different signal amplitudes (weak/strong concentration)\n"
            "  - Gate with noise vs clean signal\n"
        )
        return self.generate(prompt, physics_type="geometry_nodes",
                            n_variants=n_variants, frame_end=frame_end, model=model)

    def from_cascade(self, cascade, n_variants=6, frame_end=120, model=None):
        prompt = cascade.to_llm_prompt()
        return self.generate(prompt, physics_type="geometry_nodes",
                            n_variants=n_variants, frame_end=frame_end, model=model)


def parse_args():
    p = argparse.ArgumentParser(description="DNA Computation Engine -> Blender Physics Matrix")
    p.add_argument("description", nargs="?", default=None)
    p.add_argument("--sequence",  default=None, help="DNA sequence string")
    p.add_argument("--gate",      default=None,
                   choices=["AND","OR","NOT","THRESHOLD","CATALYST","SEESAW","HAIRPIN"])
    p.add_argument("--cascade",   default=None,
                   choices=["half_adder","amplifier","memory_latch","analog_threshold"])
    p.add_argument("--variants",  type=int, default=4)
    p.add_argument("--frame-end", type=int, default=96)
    p.add_argument("--workers",   type=int, default=2)
    p.add_argument("--config",    default="config.yaml")
    p.add_argument("--provider",  default=None)
    p.add_argument("--model",     default=None)
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--info",      action="store_true",
                   help="Print biophysics params for --sequence and exit")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = pathlib.Path(cfg["blender"]["output_dir"]).expanduser() / "dna_compute"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.info and args.sequence:
        bio = DNAtoBlender(args.sequence)
        print(json.dumps(bio.summary(), indent=2))
        return

    engine = DNAComputeMatrix(args.config)
    if args.provider:
        engine.bridge.set_provider(args.provider)

    print("\n" + "="*60)
    print("DNA Computation Engine -- Cloud-Eye Blender Pipeline")
    print("="*60)

    if args.cascade:
        print("[dna] Cascade preset: " + args.cascade)
        cascade = DNACascade.preset(args.cascade)
        matrix = engine.from_cascade(cascade, args.variants, args.frame_end, args.model)
    elif args.gate:
        print("[dna] Gate: " + args.gate)
        gate = DNAGate(args.gate)
        matrix = engine.from_gate(gate, args.variants, args.frame_end, args.model)
    elif args.sequence:
        bio = DNAtoBlender(args.sequence)
        print("[dna] Sequence: " + args.sequence)
        print("      Tm=" + str(round(bio.Tm,1)) + "C  GC=" +
              str(round(bio.gc*100)) + "%  Lp=" + str(round(bio.Lp,1)) + "nm")
        matrix = engine.from_sequence(args.sequence, args.variants, args.frame_end, args.model)
    elif args.description:
        print("[dna] Description: " + args.description)
        matrix = engine.generate(args.description, physics_type="geometry_nodes",
                                n_variants=args.variants, frame_end=args.frame_end,
                                model=args.model)
    else:
        print("[dna] No input. Examples:")
        print("  python dna_compute.py --cascade half_adder")
        print("  python dna_compute.py --sequence GCGCATCGATGCGC --variants 6")
        print("  python dna_compute.py 'strand displacement wave in water'")
        return

    matrix_path = out_dir / "dna_matrix.json"
    matrix.save(str(matrix_path))

    if not args.dry_run:
        print("\n[dna] Launching parallel render (workers=" + str(args.workers) + ")...")
        subprocess.run([
            sys.executable, "parallel_render.py",
            "--matrix", str(matrix_path),
            "--config", args.config,
            "--workers", str(args.workers),
        ], check=False)
        report = out_dir / "render_report.json"
        viewer = out_dir / "dna_compare.html"
        subprocess.run([
            sys.executable, "compare_viewer.py",
            "--report", str(report),
            "--frame", "24",
            "--output", str(viewer),
        ], check=False)
        print("\n[dna] Viewer: " + str(viewer))
    else:
        print("[dna] --dry-run: matrix prepared, render skipped.")
        print("[dna] Matrix: " + str(matrix_path))
        if matrix.variants:
            print("[dna] " + str(len(matrix.variants)) + " variants:")
            for v in matrix.variants:
                print("  " + v["variant_id"] + "  " + v.get("quality_label",""))


if __name__ == "__main__":
    main()
