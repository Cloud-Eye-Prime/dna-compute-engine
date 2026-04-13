"""
icrn_bridge.py — Translates DNA gate specs into ICRN chemical reaction networks.

Maps each DNAGate type (AND, OR, NOT, THRESHOLD, CATALYST, SEESAW, HAIRPIN)
to a set of MassActionReaction objects with biophysically grounded rate constants
derived from dna_physics_map.py.

Rate constant grounding:
  - Toehold binding k_on: Zhang & Winfree (2009), 10^4-10^6 M^-1 s^-1
  - Strand displacement k_fwd: ~10x slower than toehold binding
  - Degradation/leak: ~10^-4 s^-1 (background hydrolysis)
  - Fuel-driven catalysis: k_cat ~ k_on * [fuel]/Km

Usage:
    from icrn_bridge import gate_to_icrn, cascade_to_icrn
    crn, species_map, rate_map = gate_to_icrn("AND", sequence="ATCGATCG")
    exp = Experiment(crn, DEFAULT_EXP_PARAMS)
    result, trajectory = exp.simulate_time(concs, rates, {}, time=100.0, sample_num=50)

References:
    Zhang & Winfree (2009) JACS — toehold exchange kinetics
    Srinivas et al. (2013) NAR — biophysics of strand displacement
    Qian, Winfree & Bruck (2011) Nature — neural network in DNA
    Cherry & Qian (2018) Nature — winner-take-all DNA networks
"""
import jax.numpy as jnp
from icrn import (
    many_species, many_rate_constants,
    MassActionReaction, ICRN, Experiment, SJDict,
)
from dna_physics_map import DNAtoBlender, toehold_k_on, gc_content

rxn = MassActionReaction

# ── Default experiment parameters ────────────────────────────────────────────

DEFAULT_EXP_PARAMS = {
    "dt": 1e-4,
    "batch": False,
    "integration_method": "euler",
    "spatial_dim": None,
}

# ── Rate constant scaling ────────────────────────────────────────────────────
# ICRN uses concentration in nM and time in seconds.
# Zhang & Winfree k_on is in M^-1 s^-1; we scale to nM^-1 s^-1.
# k_on (nM) = k_on (M) * 1e-9

def _scale_k_on(k_on_per_M: float) -> float:
    """Convert M^-1 s^-1 to nM^-1 s^-1 for ICRN concentrations in nM."""
    return k_on_per_M * 1e-9

def _k_on_from_params(sequence: str = "ATCGATCG", toehold_len: int = 6,
                      gc_toehold: float = None) -> float:
    """Get scaled toehold k_on from biophysics."""
    gc = gc_toehold if gc_toehold is not None else gc_content(sequence)
    return _scale_k_on(toehold_k_on(toehold_len, gc))

# ── Leak rate ────────────────────────────────────────────────────────────────
# Background leak represents spontaneous strand displacement without toehold.
# Typically 10^4-10^5 fold slower than toehold-mediated (Zhang & Winfree 2009).
LEAK_RATE = 1e-8  # nM^-1 s^-1 — very slow background

# Degradation: spontaneous hydrolysis / nuclease degradation in lab conditions
DEGRADATION_RATE = 1e-5  # s^-1


# ═══════════════════════════════════════════════════════════════════════════════
# Gate-to-ICRN translators
# ═══════════════════════════════════════════════════════════════════════════════

def _and_gate(sequence_A="ATCGATCGATCG", sequence_B="GCGCATCGATGC",
              toehold_len=6, salt_mM=150, temp_C=37, **kwargs):
    """
    AND gate: Output only when both inputs present.

    Reactions:
        A + Gate -> A:Gate           (toehold binding, fast)
        A:Gate + B -> Output + Waste (strand displacement, requires both)
        Gate -> leak                 (background leak, very slow)

    This is the standard two-input AND from Seelig et al. (2006).
    """
    A, B, Gate, AG, Output, Waste = many_species("A, B, Gate, AG, Output, Waste")
    k_bind, k_displace, k_leak = many_rate_constants("k_bind, k_displace, k_leak")

    crn = ICRN([
        rxn(A + Gate, AG, k_bind),            # A binds gate toehold
        rxn(AG + B, Output + Waste, k_displace),  # B completes displacement
        rxn(Gate, Waste, k_leak),             # leak (no toehold)
    ])

    k_bind_val = _k_on_from_params(sequence_A, toehold_len)
    k_disp_val = k_bind_val * 0.5  # displacement ~2x slower than binding

    species_map = {
        "input_A": A, "input_B": B, "gate": Gate,
        "intermediate": AG, "output": Output, "waste": Waste,
    }
    rate_map = {k_bind: k_bind_val, k_displace: k_disp_val, k_leak: LEAK_RATE}
    default_concs = {
        A: 100.0, B: 100.0, Gate: 200.0,
        AG: 0.0, Output: 0.0, Waste: 0.0,
    }

    return crn, species_map, rate_map, default_concs


def _or_gate(sequence="ATCGATCG", toehold_len=5, **kwargs):
    """
    OR gate: Either input triggers output.

    Reactions:
        A + Gate -> Output_A + Waste_A
        B + Gate -> Output_B + Waste_B

    Two independent displacement pathways on the same gate complex.
    Output = Output_A + Output_B (additive).
    """
    A, B, Gate, OutA, OutB, WasteA, WasteB = many_species(
        "A, B, Gate, OutA, OutB, WasteA, WasteB"
    )
    k_a, k_b = many_rate_constants("k_a, k_b")

    crn = ICRN([
        rxn(A + Gate, OutA + WasteA, k_a),
        rxn(B + Gate, OutB + WasteB, k_b),
    ])

    k_val = _k_on_from_params(sequence, toehold_len)
    species_map = {
        "input_A": A, "input_B": B, "gate": Gate,
        "output_A": OutA, "output_B": OutB,
        "output": [OutA, OutB],  # total output = sum
    }
    rate_map = {k_a: k_val, k_b: k_val}
    default_concs = {
        A: 100.0, B: 0.0, Gate: 200.0,
        OutA: 0.0, OutB: 0.0, WasteA: 0.0, WasteB: 0.0,
    }

    return crn, species_map, rate_map, default_concs


def _not_gate(sequence="GCGCGCGC", toehold_len=8, **kwargs):
    """
    NOT gate: Input suppresses output.

    Reactions:
        Source -> Output              (constitutive production)
        Input + Output -> Waste       (annihilation / sequestration)
        Output -> decay               (background degradation)

    High input consumes output faster than it's produced -> low output.
    No input -> output accumulates.
    """
    Input, Output, Source, Waste = many_species("Input, Output, Source, Waste")
    k_produce, k_annihilate, k_decay = many_rate_constants(
        "k_produce, k_annihilate, k_decay"
    )

    crn = ICRN([
        rxn(Source, Source + Output, k_produce),    # constitutive production
        rxn(Input + Output, Waste, k_annihilate),   # input sequesters output
        rxn(Output, 0, k_decay),                    # background decay
    ])

    k_val = _k_on_from_params(sequence, toehold_len)
    species_map = {
        "input": Input, "output": Output, "source": Source, "waste": Waste,
    }
    rate_map = {
        k_produce: 0.01,     # slow constitutive production (s^-1)
        k_annihilate: k_val, # fast annihilation via strand displacement
        k_decay: DEGRADATION_RATE,
    }
    default_concs = {
        Input: 0.0, Output: 0.0, Source: 100.0, Waste: 0.0,
    }

    return crn, species_map, rate_map, default_concs


def _threshold_gate(sequence="ATCGATCGATCG", toehold_len=5,
                    threshold=10, **kwargs):
    """
    Threshold gate: Output only above concentration threshold.

    Reactions:
        Input + Threshold -> Waste       (threshold absorbs input)
        Input + Gate -> Output + Waste2  (excess input produces output)

    Threshold species acts as a sponge. Only after threshold is consumed
    does input reach the gate and produce output.
    Ref: Qian & Winfree (2011) seesaw-based thresholding.
    """
    Input, Thresh, Gate, Output, Waste, Waste2 = many_species(
        "Input, Thresh, Gate, Output, Waste, Waste2"
    )
    k_absorb, k_gate = many_rate_constants("k_absorb, k_gate")

    crn = ICRN([
        rxn(Input + Thresh, Waste, k_absorb),        # threshold absorbs
        rxn(Input + Gate, Output + Waste2, k_gate),   # excess -> output
    ])

    k_val = _k_on_from_params(sequence, toehold_len)
    species_map = {
        "input": Input, "threshold": Thresh, "gate": Gate,
        "output": Output,
    }
    rate_map = {
        k_absorb: k_val * 2.0,  # threshold binding slightly faster (competitive)
        k_gate: k_val,
    }
    default_concs = {
        Input: 0.0, Thresh: float(threshold) * 10.0,  # threshold in nM
        Gate: 200.0, Output: 0.0, Waste: 0.0, Waste2: 0.0,
    }

    return crn, species_map, rate_map, default_concs


def _catalyst_gate(sequence_cat="GCATCGATGCAT", sequence_fuel="ATCGATCGATCG",
                   toehold_len_cat=7, toehold_len_fuel=4,
                   strand_nM_cat=10.0, strand_nM_fuel=1000.0, **kwargs):
    """
    Catalyst gate: One catalyst strand drives repeated output cycles.

    Reactions:
        Cat + Substrate -> Cat + Output     (catalytic cycle)
        Fuel + Substrate -> Intermediate    (fuel replenishment)
        Intermediate -> Substrate           (recycling)
        Output -> decay                     (background degradation)

    The catalyst is NOT consumed. Fuel provides free energy.
    Ref: Zhang, Turberfield, Yurke & Winfree (2007) — catalytic hairpin assembly.
    """
    Cat, Sub, Output, Fuel, Inter = many_species("Cat, Sub, Output, Fuel, Inter")
    k_cat, k_fuel, k_recycle, k_decay = many_rate_constants(
        "k_cat, k_fuel, k_recycle, k_decay"
    )

    crn = ICRN([
        rxn(Cat + Sub, Cat + Output, k_cat),     # catalytic turnover
        rxn(Fuel + Sub, Inter, k_fuel),           # fuel-assisted
        rxn(Inter, Sub, k_recycle),               # substrate recycling
        rxn(Output, 0, k_decay),                  # output decay
    ])

    k_cat_val = _k_on_from_params(sequence_cat, toehold_len_cat)
    k_fuel_val = _k_on_from_params(sequence_fuel, toehold_len_fuel)

    species_map = {
        "catalyst": Cat, "substrate": Sub, "output": Output,
        "fuel": Fuel, "intermediate": Inter,
    }
    rate_map = {
        k_cat: k_cat_val,
        k_fuel: k_fuel_val * 0.1,  # fuel slower than catalyst
        k_recycle: 0.001,          # slow recycling (s^-1)
        k_decay: DEGRADATION_RATE,
    }
    default_concs = {
        Cat: strand_nM_cat, Sub: 200.0, Output: 0.0,
        Fuel: strand_nM_fuel, Inter: 0.0,
    }

    return crn, species_map, rate_map, default_concs


def _seesaw_gate(sequence_input="GCGCATCG", sequence_fuel="ATCGCGAT",
                 toehold_len=5, threshold_ratio=0.5, **kwargs):
    """
    Seesaw gate: Tunable threshold via relative strand concentrations.

    Reactions:
        Input + Gate:Fuel -> Gate:Input + Fuel    (forward seesaw)
        Fuel + Gate:Input -> Gate:Fuel + Input    (reverse seesaw)
        Gate:Input -> Output                       (readout)

    Equilibrium position determined by Input/Fuel concentration ratio.
    Ref: Qian & Winfree (2011) Science — seesaw gates for neural networks.
    """
    Input, Fuel, GF, GI, Output = many_species("Input, Fuel, GF, GI, Output")
    k_fwd, k_rev, k_read = many_rate_constants("k_fwd, k_rev, k_read")

    crn = ICRN([
        rxn(Input + GF, GI + Fuel, k_fwd),   # forward: input displaces fuel
        rxn(Fuel + GI, GF + Input, k_rev),    # reverse: fuel displaces input
        rxn(GI, Output, k_read),              # readout from gate:input state
    ])

    k_val = _k_on_from_params(sequence_input, toehold_len)
    input_conc = 100.0 * (1.0 - threshold_ratio)
    fuel_conc = 100.0 * threshold_ratio

    species_map = {
        "input": Input, "fuel": Fuel,
        "gate_fuel": GF, "gate_input": GI, "output": Output,
    }
    rate_map = {
        k_fwd: k_val,
        k_rev: k_val,       # symmetric seesaw
        k_read: 0.001,      # slow readout (s^-1)
    }
    default_concs = {
        Input: input_conc, Fuel: fuel_conc, GF: 100.0,
        GI: 0.0, Output: 0.0,
    }

    return crn, species_map, rate_map, default_concs


def _hairpin_gate(sequence="GCGCATCGATGCGC", toehold_len=4,
                  salt_mM=100.0, temp_C=37.0, **kwargs):
    """
    Hairpin gate: Bistable memory element.

    Reactions:
        Input + Closed -> Open + Waste     (opening reaction)
        Open -> Closed                      (spontaneous refolding)
        Open -> Output                      (readout from open state)

    The hairpin stem is thermodynamically stable (high Tm, high GC).
    Opening requires input strand to invade the stem via toehold.
    Memory: refolding rate << opening rate means state persists.
    """
    Input, Closed, Open, Output, Waste = many_species(
        "Input, Closed, Open, Output, Waste"
    )
    k_open, k_refold, k_read = many_rate_constants("k_open, k_refold, k_read")

    crn = ICRN([
        rxn(Input + Closed, Open + Waste, k_open),   # toehold-mediated opening
        rxn(Open, Closed, k_refold),                  # spontaneous refolding
        rxn(Open, Open + Output, k_read),             # readout (non-consuming)
    ])

    k_open_val = _k_on_from_params(sequence, toehold_len)
    # Refolding is slow for stable hairpins — this gives memory
    # Higher GC, higher salt -> more stable stem -> slower refolding
    gc = gc_content(sequence)
    k_refold_val = 1e-4 * (1.0 - gc)  # GC-rich = very slow refold

    species_map = {
        "input": Input, "closed": Closed, "open": Open, "output": Output,
    }
    rate_map = {
        k_open: k_open_val,
        k_refold: k_refold_val,
        k_read: 0.005,  # moderate readout rate (s^-1)
    }
    default_concs = {
        Input: 0.0, Closed: 100.0, Open: 0.0, Output: 0.0, Waste: 0.0,
    }

    return crn, species_map, rate_map, default_concs


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

GATE_BUILDERS = {
    "AND":       _and_gate,
    "OR":        _or_gate,
    "NOT":       _not_gate,
    "THRESHOLD": _threshold_gate,
    "CATALYST":  _catalyst_gate,
    "SEESAW":    _seesaw_gate,
    "HAIRPIN":   _hairpin_gate,
}


def gate_to_icrn(gate_type: str, **kwargs):
    """
    Convert a gate type string to an ICRN simulation.

    Returns:
        crn:           ICRN object (ready for Experiment)
        species_map:   dict mapping role names -> Species objects
        rate_map:      SJDict of rate constant values
        default_concs: dict of default initial concentrations (nM)

    Example:
        crn, smap, rates, concs = gate_to_icrn("AND")
        exp = Experiment(crn, DEFAULT_EXP_PARAMS)
        conc_data = SJDict({sp: jnp.array(v) for sp, v in concs.items()})
        rate_data = SJDict(rates)
        result, traj = exp.simulate_time(conc_data, rate_data, {}, time=100.0)
        output_signal = float(result[smap["output"]])
    """
    builder = GATE_BUILDERS.get(gate_type.upper())
    if builder is None:
        raise ValueError(
            f"Unknown gate type: {gate_type}. "
            f"Available: {list(GATE_BUILDERS.keys())}"
        )
    return builder(**kwargs)


def simulate_gate(gate_type: str, input_concs: dict = None,
                  time: float = 100.0, sample_num: int = 50,
                  exp_params: dict = None, **gate_kwargs):
    """
    One-shot gate simulation. Returns output concentration and trajectory.

    Args:
        gate_type:    "AND", "OR", "NOT", etc.
        input_concs:  dict overriding default input concentrations
                      e.g. {"input_A": 200.0, "input_B": 0.0}
        time:         simulation time in seconds
        sample_num:   number of trajectory samples
        exp_params:   override DEFAULT_EXP_PARAMS
        **gate_kwargs: passed to gate builder (sequence, toehold_len, etc.)

    Returns:
        output_signal: float, final output concentration in nM
        trajectory:    list of SJDict snapshots at each sample point
        species_map:   dict mapping role names -> Species objects
    """
    crn, species_map, rate_map, default_concs = gate_to_icrn(gate_type, **gate_kwargs)

    # Apply input overrides
    if input_concs:
        for role, value in input_concs.items():
            if role in species_map:
                sp = species_map[role]
                if sp in default_concs:
                    default_concs[sp] = value

    # Build JAX arrays
    conc_data = SJDict({sp: jnp.array(float(v)) for sp, v in default_concs.items()})
    rate_data = SJDict({k: jnp.array(float(v)) for k, v in rate_map.items()})

    params = exp_params or DEFAULT_EXP_PARAMS
    exp = Experiment(crn, params)
    result, trajectory = exp.simulate_time(
        conc_data, rate_data, {}, time=time, sample_num=sample_num
    )

    # Extract output signal
    output_key = species_map.get("output")
    if isinstance(output_key, list):
        # OR gate: sum of multiple output species
        output_signal = sum(float(result[sp]) for sp in output_key)
    else:
        output_signal = float(result[output_key])

    return output_signal, trajectory, species_map


def cascade_to_icrn(cascade_spec: list):
    """
    Build a cascaded circuit from a list of gate specs.

    Each gate in the cascade feeds its output as input to the next gate.
    Returns a combined ICRN with all reactions, plus metadata for readout.

    Args:
        cascade_spec: list of dicts, e.g.
            [{"type": "AND"}, {"type": "THRESHOLD", "kwargs": {"threshold": 5}}]

    Returns:
        stages: list of (crn, species_map, rate_map, default_concs) per stage
        stage_names: list of gate type strings

    Note: True cascade wiring (output of stage N -> input of stage N+1)
    requires shared species across ICRN objects, which isn't trivial with
    the current ICRN API (each ICRN is self-contained). For now, stages
    are simulated sequentially: output concentration of stage N becomes
    input concentration of stage N+1.
    """
    stages = []
    stage_names = []
    for spec in cascade_spec:
        gate_type = spec["type"]
        gate_kwargs = spec.get("kwargs", {})
        result = gate_to_icrn(gate_type, **gate_kwargs)
        stages.append(result)
        stage_names.append(gate_type)
    return stages, stage_names


def simulate_cascade(cascade_spec: list, initial_inputs: dict = None,
                     time_per_stage: float = 100.0, sample_num: int = 50):
    """
    Simulate a multi-stage cascade sequentially.

    Output of each stage becomes input to the next.

    Args:
        cascade_spec:   list of gate specs (same as DNACascade presets)
        initial_inputs: dict of input concentrations for stage 1
        time_per_stage: simulation time per stage (seconds)
        sample_num:     trajectory samples per stage

    Returns:
        results: list of (output_signal, trajectory, species_map) per stage
        final_signal: output of the last stage
    """
    stages, stage_names = cascade_to_icrn(cascade_spec)
    results = []
    carry_signal = None

    for i, (crn, smap, rates, concs) in enumerate(stages):
        # Wire: previous output -> current input
        if carry_signal is not None:
            # Find the first input species and set its concentration
            for role in ["input", "input_A"]:
                if role in smap:
                    concs[smap[role]] = carry_signal
                    break

        # Apply initial inputs to first stage
        if i == 0 and initial_inputs:
            for role, val in initial_inputs.items():
                if role in smap and smap[role] in concs:
                    concs[smap[role]] = val

        conc_data = SJDict({sp: jnp.array(float(v)) for sp, v in concs.items()})
        rate_data = SJDict({k: jnp.array(float(v)) for k, v in rates.items()})
        exp = Experiment(crn, DEFAULT_EXP_PARAMS)
        result, traj = exp.simulate_time(
            conc_data, rate_data, {}, time=time_per_stage, sample_num=sample_num
        )

        output_key = smap.get("output")
        if isinstance(output_key, list):
            out_signal = sum(float(result[sp]) for sp in output_key)
        else:
            out_signal = float(result[output_key])

        results.append((out_signal, traj, smap))
        carry_signal = out_signal
        print(f"[cascade] Stage {i+1} ({stage_names[i]}): output = {out_signal:.2f} nM")

    return results, carry_signal


# ═══════════════════════════════════════════════════════════════════════════════
# CLI test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ICRN Bridge — Gate Translation Test")
    print("=" * 60)

    # Test each gate type
    for gate_type in GATE_BUILDERS:
        print(f"\n--- {gate_type} gate ---")
        try:
            signal, _, smap = simulate_gate(
                gate_type, time=50.0, sample_num=10
            )
            print(f"  Output signal: {signal:.4f} nM")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Test AND gate with both inputs vs one input
    print("\n--- AND gate: signal differentiation ---")
    sig_both, _, _ = simulate_gate("AND", input_concs={"input_A": 100.0, "input_B": 100.0})
    sig_one, _, _ = simulate_gate("AND", input_concs={"input_A": 100.0, "input_B": 0.0})
    print(f"  Both inputs:  {sig_both:.4f} nM")
    print(f"  One input:    {sig_one:.4f} nM")
    print(f"  Ratio:        {sig_both / max(sig_one, 1e-10):.1f}x")

    # Test cascade
    print("\n--- Half adder cascade ---")
    cascade = [{"type": "AND"}, {"type": "OR"}]
    results, final = simulate_cascade(
        cascade,
        initial_inputs={"input_A": 100.0, "input_B": 100.0},
        time_per_stage=50.0
    )
    print(f"  Final output: {final:.4f} nM")

    print("\n" + "=" * 60)
    print("ICRN Bridge tests complete.")
