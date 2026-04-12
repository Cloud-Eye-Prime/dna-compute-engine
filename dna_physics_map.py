"""
dna_physics_map.py — Maps real DNA biophysics constants to Blender physics parameters.

Grounded in actual molecular biology:
  - Persistence lengths (ssDNA ~1-3nm, dsDNA ~50nm)
  - Toehold-mediated strand displacement kinetics (k_on ~ 10^5-10^6 M-1 s-1)
  - Wallace / SantaLucia melting temperature formulas
  - Diffusion coefficients (D = kT / 6πηr)
  - GC content effects on duplex stability
  - Salt concentration / Debye screening length

References:
  - Srinivas et al. (2013): Nucleic Acids Research 41(22)
  - Zhang & Winfree (2009): JACS strand displacement kinetics
  - oxDNA coarse-grained model parameters (Sulc et al.)
"""
import math

# ── Physical constants ────────────────────────────────────────────────────────
kB = 1.380649e-23   # Boltzmann constant (J/K)
T_STANDARD = 310.15 # 37°C in Kelvin (physiological)
ETA_WATER  = 0.001  # dynamic viscosity of water at 20°C (Pa·s)

# ── Biophysics → Blender parameter mappings ───────────────────────────────────

def gc_content(sequence: str) -> float:
    """GC fraction of a DNA sequence (0.0 - 1.0)."""
    s = sequence.upper()
    gc = sum(1 for b in s if b in "GC")
    return gc / len(s) if s else 0.5

def melting_temp_wallace(sequence: str) -> float:
    """
    Wallace rule: Tm = 2*(A+T) + 4*(G+C)  [°C]
    Valid for short oligos < 14 nt in 50mM NaCl.
    """
    s = sequence.upper()
    at = sum(1 for b in s if b in "AT")
    gc = sum(1 for b in s if b in "GC")
    return 2.0 * at + 4.0 * gc

def melting_temp_santalucia(gc: float, length: int, salt_mM: float = 50.0) -> float:
    """
    Simplified SantaLucia (1998) estimate for longer strands.
    Tm (°C) = 81.5 + 16.6*log10([Na+]) + 0.41*GC% - 675/N
    """
    na_M = salt_mM / 1000.0
    return 81.5 + 16.6 * math.log10(na_M) + 41.0 * gc - 675.0 / max(length, 1)

def persistence_length(is_double_stranded: bool, gc: float = 0.5) -> float:
    """
    Persistence length in nm.
    ssDNA: 0.75-3nm (flexible, sequence-dependent)
    dsDNA: ~50nm (semi-rigid, slight GC dependence)
    """
    if is_double_stranded:
        return 45.0 + 10.0 * gc   # 45-55 nm depending on GC
    else:
        return 0.75 + 2.25 * gc   # 0.75-3 nm

def diffusion_coefficient(radius_nm: float, temp_C: float = 25.0) -> float:
    """Stokes-Einstein: D = kT / (6πηr)  [m²/s]"""
    T = temp_C + 273.15
    r = radius_nm * 1e-9
    return (kB * T) / (6 * math.pi * ETA_WATER * r)

def toehold_k_on(toehold_length: int, gc_toehold: float) -> float:
    """
    Approximate k_on for toehold binding (M^-1 s^-1).
    Based on Zhang & Winfree (2009): k_on ≈ k_max * f(toehold_len)
    GC-rich toeholds bind ~3-4 orders of magnitude faster than AT-rich.
    """
    k_max = 1e6    # upper limit ~10^6 M-1 s-1
    # Saturation curve: longer toehold → faster, plateaus at ~10nt
    length_factor = 1.0 - math.exp(-toehold_length / 5.0)
    gc_factor = 0.1 + 0.9 * gc_toehold  # AT-rich = 0.1x, GC-rich = 1.0x
    return k_max * length_factor * gc_factor

def debye_length(salt_mM: float) -> float:
    """Debye screening length κ^-1 (nm) — electrostatic range."""
    return 0.304 / math.sqrt(salt_mM / 1000.0)


# ── Blender parameter translators ─────────────────────────────────────────────

class DNAtoBlender:
    """
    Translates a DNA molecular state into Blender physics parameters.
    All Blender params normalized to [0, 1] or natural ranges.
    """
    def __init__(self, sequence: str = "ATCGATCG", toehold_len: int = 6,
                 salt_mM: float = 150.0, temp_C: float = 37.0,
                 is_dsDNA: bool = False, strand_nM: float = 100.0):
        self.seq       = sequence.upper()
        self.gc        = gc_content(sequence)
        self.toehold   = toehold_len
        self.salt_mM   = salt_mM
        self.temp_C    = temp_C
        self.is_ds     = is_dsDNA
        self.strand_nM = strand_nM
        # derived
        self.Tm     = melting_temp_wallace(sequence) if len(sequence) < 14                       else melting_temp_santalucia(self.gc, len(sequence), salt_mM)
        self.Lp     = persistence_length(is_dsDNA, self.gc)
        self.D      = diffusion_coefficient(0.5 + 0.02 * len(sequence), temp_C)
        self.k_on   = toehold_k_on(toehold_len, self.gc)
        self.kappa  = debye_length(salt_mM)

    def soft_body_params(self) -> dict:
        """Soft body settings for DNA strand flexibility."""
        # Stiffness: ssDNA flexible, dsDNA rigid
        stiff = 0.05 + 0.75 * (self.Lp / 55.0)   # 0.05 (ssDNA) → 0.8 (dsDNA)
        # Damping scales with solvent viscosity + salt
        damp = 0.1 + 0.4 * (self.salt_mM / 1000.0)
        # Thermal motion: higher temp = more brownian = lower goal weight
        goal = max(0.1, 1.0 - (self.temp_C - 20.0) / 80.0)
        return {
            "goal_stiffness":  round(stiff, 3),
            "goal_damping":    round(damp, 3),
            "goal_default":    round(goal, 3),
            "bending":         round(0.5 * stiff, 3),
            "mass":            round(0.001 + 0.0001 * len(self.seq), 5),
        }

    def particle_params(self) -> dict:
        """Particle system for nucleotide/strand diffusion."""
        # Count scales with strand concentration (nM → particle density proxy)
        count = max(50, min(5000, int(self.strand_nM * 20)))
        # Lifetime: stable duplexes live longer
        stability = self.Tm / 80.0   # normalize to ~0-1
        lifetime  = 20 + 200 * stability
        # Brownian motion: inversely proportional to size and viscosity
        brownian = max(0.01, 1.0 - stability) * (1.0 + (self.temp_C - 25.0) / 50.0)
        return {
            "count":            count,
            "lifetime":         round(lifetime, 1),
            "size":             round(0.02 + 0.01 * len(self.seq) / 20.0, 3),
            "brownian_factor":  round(brownian, 3),
            "mass":             round(0.001 + 0.0001 * len(self.seq), 5),
            "factor_gravity":   0.0,   # DNA in solution: gravity negligible
            "normal_factor":    round(self.k_on / 1e6, 3),
        }

    def geo_nodes_params(self) -> dict:
        """Geometry Nodes simulation zone params for reaction dynamics."""
        # Substeps: faster reactions need more substeps
        substeps = max(2, min(16, int(self.k_on / 5e4)))
        return {
            "simulation_substeps": substeps,
            "force_scale":         round(self.k_on / 1e6, 4),
            "damping":             round(0.1 + 0.5 * (self.salt_mM / 500.0), 3),
            "stiffness":           round(self.soft_body_params()["goal_stiffness"], 3),
            "collision_radius":    round(self.kappa / 10.0, 4),   # nm → blender units
            "noise_scale":         round(1.0 / (self.Lp + 1.0), 3),
        }

    def rigid_body_params(self) -> dict:
        """Rigid body for hybridized dsDNA segments / DNA origami tiles."""
        restitution = max(0.0, 1.0 - self.gc * 0.8)  # GC-rich = less bouncy (stable)
        friction    = 0.3 + 0.4 * self.gc
        return {
            "mass":              round(0.001 * len(self.seq), 4),
            "friction":          round(friction, 3),
            "restitution":       round(restitution, 3),
            "linear_damping":    round(0.05 + 0.3 * (self.salt_mM / 500.0), 3),
            "angular_damping":   0.1,
        }

    def summary(self) -> dict:
        """Full parameter summary with biophysics + Blender mappings."""
        return {
            "biophysics": {
                "sequence":          self.seq,
                "length_nt":         len(self.seq),
                "gc_content":        round(self.gc, 3),
                "melting_temp_C":    round(self.Tm, 1),
                "persistence_nm":    round(self.Lp, 1),
                "diffusion_m2s":     f"{self.D:.3e}",
                "toehold_k_on":      f"{self.k_on:.2e}",
                "debye_length_nm":   round(self.kappa, 2),
                "salt_mM":           self.salt_mM,
                "temp_C":            self.temp_C,
            },
            "blender": {
                "soft_body":     self.soft_body_params(),
                "particles":     self.particle_params(),
                "geo_nodes":     self.geo_nodes_params(),
                "rigid_body":    self.rigid_body_params(),
            }
        }


if __name__ == "__main__":
    import json
    # Example: toehold strand displacement setup
    # Toehold: 6nt, GC-rich, physiological conditions
    strand = DNAtoBlender(
        sequence="GCGCATCGATCGATGCGC",
        toehold_len=6,
        salt_mM=150.0,
        temp_C=37.0,
        is_dsDNA=False,
        strand_nM=100.0
    )
    print(json.dumps(strand.summary(), indent=2))
