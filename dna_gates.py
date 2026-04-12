"""
dna_gates.py — DNA computing gate library mapped to Blender physics.

Each gate is a named set of physics parameters + a bpy code template.
Gates implement classic DNA strand displacement circuit motifs:
  AND, OR, NOT, Threshold, Catalyst (fuel-driven), Seesaw, Hairpin Toggle

Reference: Chen et al. (2013) Nature Nanotechnology — "Programmable DNA circuits"
           Thubagere et al. (2017) Science — "A cargo-sorting DNA robot"
           Qian & Winfree (2011) Science — "Scaling up DNA computation"
"""
from dna_physics_map import DNAtoBlender


class DNAGate:
    def __init__(self, gate_type: str, **kwargs):
        self.gate_type = gate_type
        self.kwargs = kwargs
        self._build()

    def _build(self):
        builders = {
            "AND":       self._and_gate,
            "OR":        self._or_gate,
            "NOT":       self._not_gate,
            "THRESHOLD": self._threshold_gate,
            "CATALYST":  self._catalyst_gate,
            "SEESAW":    self._seesaw_gate,
            "HAIRPIN":   self._hairpin_gate,
        }
        fn = builders.get(self.gate_type.upper())
        if not fn:
            raise ValueError(f"Unknown gate type: {self.gate_type}")
        fn()

    # ── AND gate ──────────────────────────────────────────────────────────────
    def _and_gate(self):
        """Both input strands must bind before output releases.
           Two particle streams must collide at a convergence point."""
        d1 = DNAtoBlender(sequence="ATCGATCGATCG", toehold_len=6,
                          salt_mM=150, temp_C=37)
        d2 = DNAtoBlender(sequence="GCGCATCGATGC", toehold_len=6,
                          salt_mM=150, temp_C=37)
        self.params = {
            "input_A": d1.particle_params(),
            "input_B": d2.particle_params(),
            "gate_strand": d1.soft_body_params(),
            "collision_threshold": 2,   # both must hit
            "output_emission_delay": 5, # frames after gate activation
        }
        self.description = "AND: two input strand streams converge; output only on dual collision"
        self.visual_hint = "Two particle emitters aimed at a central gate mesh; output emitter activates on contact"

    # ── OR gate ───────────────────────────────────────────────────────────────
    def _or_gate(self):
        """Either input strand triggers output."""
        d = DNAtoBlender(sequence="ATCGATCG", toehold_len=5, salt_mM=150, temp_C=37)
        self.params = {
            "input_A": d.particle_params(),
            "input_B": d.particle_params(),
            "collision_threshold": 1,
            "output_emission_delay": 2,
        }
        self.description = "OR: either input strand activates output gate"
        self.visual_hint = "Two parallel emitters → shared gate mesh; first contact triggers output"

    # ── NOT gate ──────────────────────────────────────────────────────────────
    def _not_gate(self):
        """Input strand displaces and sequesters the output strand.
           High input → no output. No input → output present."""
        d = DNAtoBlender(sequence="GCGCGCGC", toehold_len=8, gc_toehold=0.9
                         if False else 0.75, salt_mM=150, temp_C=37)
        sb = d.soft_body_params()
        sb["goal_stiffness"] = 0.9   # output strand tightly bound until displaced
        self.params = {
            "inhibitor": d.particle_params(),
            "output_strand": sb,
            "displacement_energy": d.geo_nodes_params()["force_scale"],
        }
        self.description = "NOT: input strand sequesters output via high-affinity displacement"
        self.visual_hint = "Input particles collide with tightly coiled soft body; on contact the soft body collapses"

    # ── Threshold gate ─────────────────────────────────────────────────────────
    def _threshold_gate(self):
        """Output only activates above a concentration threshold.
           Modeled as: N particles must accumulate before emission."""
        threshold = self.kwargs.get("threshold", 10)
        d = DNAtoBlender(
            sequence=self.kwargs.get("sequence", "ATCGATCGATCG"),
            toehold_len=self.kwargs.get("toehold_len", 5),
            salt_mM=150, temp_C=37
        )
        self.params = {
            "accumulation_target": threshold,
            "input":  d.particle_params(),
            "output_emission_delay": 3,
            "leak_rate": 0.02,    # sub-threshold leakage (realistic)
        }
        self.description = f"THRESHOLD({threshold}): output only after {threshold} input strands accumulate"
        self.visual_hint = f"Particles fill a container mesh; at {threshold} contacts, output emitter fires"

    # ── Catalyst gate ──────────────────────────────────────────────────────────
    def _catalyst_gate(self):
        """Catalytic strand is NOT consumed — drives repeated displacement cycles.
           Fuel strands provide free energy. Models autocatalytic circuits."""
        d_catalyst = DNAtoBlender(
            sequence="GCATCGATGCAT", toehold_len=7,
            salt_mM=150, temp_C=37, strand_nM=10.0  # low catalyst concentration
        )
        d_fuel = DNAtoBlender(
            sequence="ATCGATCGATCG", toehold_len=4,
            salt_mM=150, temp_C=37, strand_nM=1000.0  # high fuel
        )
        self.params = {
            "catalyst": d_catalyst.particle_params(),
            "fuel":     d_fuel.particle_params(),
            "cycle_recovery_frames": 8,
            "amplification_factor": 50,  # 1 catalyst → ~50 output events
        }
        self.description = "CATALYST: single catalyst strand drives continuous output; fuel-limited amplification"
        self.visual_hint = "One persistent slow particle + fast fuel stream; catalyst particle regenerates each cycle"

    # ── Seesaw gate ────────────────────────────────────────────────────────────
    def _seesaw_gate(self):
        """Tunable threshold via relative strand concentrations.
           Implements fractional logic between 0 and 1."""
        ratio = self.kwargs.get("threshold_ratio", 0.5)  # 0=always on, 1=never on
        d_input = DNAtoBlender(sequence="GCGCATCG", toehold_len=5, salt_mM=150,
                               temp_C=37, strand_nM=100.0 * (1.0 - ratio))
        d_fuel  = DNAtoBlender(sequence="ATCGCGAT", toehold_len=5, salt_mM=150,
                               temp_C=37, strand_nM=100.0 * ratio)
        self.params = {
            "input":  d_input.particle_params(),
            "fuel":   d_fuel.particle_params(),
            "threshold_ratio": ratio,
            "gate": {**d_input.geo_nodes_params(), "force_scale": ratio * 0.5},
        }
        self.description = f"SEESAW(ratio={ratio}): continuously tunable between fuel and input dominance"
        self.visual_hint = "Two opposing particle streams balanced at a pivot mesh; visual tipping point"

    # ── Hairpin toggle ─────────────────────────────────────────────────────────
    def _hairpin_gate(self):
        """Hairpin structure toggles between open/closed on input binding.
           Models memory elements in DNA circuits."""
        d = DNAtoBlender(
            sequence=self.kwargs.get("sequence", "GCGCATCGATGCGC"),
            toehold_len=4,
            salt_mM=self.kwargs.get("salt_mM", 100.0),
            temp_C=self.kwargs.get("temp_C", 37.0),
            is_dsDNA=False
        )
        sb = d.soft_body_params()
        sb["goal_stiffness"] = 0.85  # hairpin stem is stiff
        sb["bending"] = 0.1          # loop is flexible
        self.params = {
            "strand":        sb,
            "input":         d.particle_params(),
            "toggle_energy": d.geo_nodes_params()["force_scale"],
            "memory": True,          # maintains state after input removed
        }
        self.description = "HAIRPIN: strand folds to closed state; input opens it; bistable memory"
        self.visual_hint = "Soft body strand coiled tight (closed); input particle hits toehold loop; strand unfolds to open state"

    def to_dict(self) -> dict:
        return {
            "gate_type":   self.gate_type,
            "description": self.description,
            "visual_hint": self.visual_hint,
            "params":      self.params,
        }


# ── Gate cascade builder ───────────────────────────────────────────────────────

class DNACascade:
    """
    Chains multiple DNA gates into a computation cascade.
    Output of gate N becomes input of gate N+1.
    Models: signal amplifiers, logic circuits, neural-like threshold networks.
    """
    def __init__(self, gates: list):
        self.stages = [DNAGate(g["type"], **g.get("kwargs", {})) for g in gates]

    def to_llm_prompt(self, visual_style: str = "cinematic molecular") -> str:
        """Generate a prompt for the LLM to create the Blender bpy scene."""
        stage_descriptions = [
            f"Stage {i+1} ({g.gate_type}): {g.description}. Visual: {g.visual_hint}"
            for i, g in enumerate(self.stages)
        ]
        stage_params = [
            f"Stage {i+1} physics params: {g.to_dict()['params']}"
            for i, g in enumerate(self.stages)
        ]
        return (
            f"Create a {visual_style} Blender scene of a DNA computation cascade "
            f"with {len(self.stages)} stages:\n\n"
            + "\n".join(stage_descriptions)
            + "\n\n--- Physics parameters per stage ---\n"
            + "\n".join(stage_params)
            + "\n\nArrange stages left-to-right in 3D space (2m spacing). "
            "Color code: input strands = cyan, gate strands = gold, output strands = magenta. "
            "Use particle systems for strand diffusion, soft bodies for flexible strands, "
            "rigid bodies for hybridized duplexes. Show signal propagation as a particle wave."
        )

    @classmethod
    def preset(cls, name: str) -> "DNACascade":
        """Named presets for common DNA computation circuits."""
        presets = {
            "half_adder": [
                {"type": "AND"},
                {"type": "OR"},
                {"type": "NOT"},
            ],
            "amplifier": [
                {"type": "CATALYST"},
                {"type": "THRESHOLD", "kwargs": {"threshold": 5}},
            ],
            "memory_latch": [
                {"type": "HAIRPIN"},
                {"type": "NOT"},
                {"type": "HAIRPIN", "kwargs": {"sequence": "GCATCGATCGAT"}},
            ],
            "analog_threshold": [
                {"type": "SEESAW", "kwargs": {"threshold_ratio": 0.3}},
                {"type": "SEESAW", "kwargs": {"threshold_ratio": 0.5}},
                {"type": "SEESAW", "kwargs": {"threshold_ratio": 0.7}},
                {"type": "THRESHOLD", "kwargs": {"threshold": 15}},
            ],
        }
        config = presets.get(name)
        if not config:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        return cls(config)
