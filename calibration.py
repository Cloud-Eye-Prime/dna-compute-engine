"""
calibration.py — Map Blender units to physical units.

Establishes the bridge between the visual simulation and measurable reality.
Without this, renders are qualitative art. With it, they become quantitative models.

Unit conventions:
  1 Blender unit (BU) = 5 nm     (tuned for DNA-scale work)
  1 Blender frame    = 50 ns     (Brownian timescale for DNA in water)
  1000 particles     = 100 nM    (in a 1 BU^3 volume)

These can be overridden via the [calibration] section of config.yaml.

Validation targets (from literature):
  - dsDNA persistence length: 50 nm  ± 5 nm         [Hagerman 1988]
  - ssDNA persistence length: 1-3 nm                 [Murphy et al. 2004]
  - k_on toehold (6nt):       ~3e5 M-1 s-1           [Zhang & Winfree 2009]
  - Diffusion coeff 100bp:    ~4.5e-12 m2/s           [Robertson et al. 2006]
  - Tm GCGCATCGATGCGC (14nt): ~48°C (150mM NaCl)     [Wallace rule]
"""
import json, math


class CalibrationLayer:
    """
    Converts between Blender simulation units and physical SI units.
    Also validates simulation parameters against known experimental values.
    """
    DEFAULTS = {
        # Spatial: 1 BU = nm
        "bu_to_nm":           5.0,
        # Temporal: 1 frame = ns
        "frame_to_ns":        50.0,
        # Concentration: N particles per BU^3 volume → nM
        "particles_per_bu3_per_nM": 10.0,
        # Energy: soft_body stiffness 1.0 = kT/nm persistence
        "stiffness_to_kT_per_nm": 4.11,   # kT at 37°C in pN·nm
    }

    # Known experimental reference values
    VALIDATION_TARGETS = {
        "dsDNA_Lp_nm":          {"value": 50.0,  "tolerance": 5.0,  "ref": "Hagerman 1988"},
        "ssDNA_Lp_nm":          {"value": 2.0,   "tolerance": 1.5,  "ref": "Murphy 2004"},
        "k_on_6nt_toehold":     {"value": 3e5,   "tolerance": 2e5,  "ref": "Zhang&Winfree 2009"},
        "diffusion_100bp_m2s":  {"value": 4.5e-12,"tolerance": 2e-12,"ref": "Robertson 2006"},
        "Tm_GCGCATCGATGCGC":    {"value": 48.0,  "tolerance": 3.0,  "ref": "Wallace rule"},
    }

    def __init__(self, overrides: dict = None):
        self.units = dict(self.DEFAULTS)
        if overrides:
            self.units.update(overrides)

    # ── Spatial conversions ───────────────────────────────────────────────────
    def bu_to_nm(self, bu: float) -> float:
        return bu * self.units["bu_to_nm"]

    def nm_to_bu(self, nm: float) -> float:
        return nm / self.units["bu_to_nm"]

    def bu_to_angstrom(self, bu: float) -> float:
        return self.bu_to_nm(bu) * 10.0

    # ── Temporal conversions ──────────────────────────────────────────────────
    def frame_to_ns(self, frame: float) -> float:
        return frame * self.units["frame_to_ns"]

    def ns_to_frame(self, ns: float) -> float:
        return ns / self.units["frame_to_ns"]

    def frame_to_us(self, frame: float) -> float:
        return self.frame_to_ns(frame) / 1000.0

    # ── Concentration conversions ─────────────────────────────────────────────
    def particles_to_nM(self, count: int, volume_bu3: float = 1.0) -> float:
        return count / (self.units["particles_per_bu3_per_nM"] * volume_bu3)

    def nM_to_particles(self, nM: float, volume_bu3: float = 1.0) -> int:
        return int(nM * self.units["particles_per_bu3_per_nM"] * volume_bu3)

    # ── Stiffness / energy ─────────────────────────────────────────────────────
    def stiffness_to_persistence_nm(self, goal_stiffness: float) -> float:
        """Convert Blender goal_stiffness (0-1) to persistence length in nm."""
        if goal_stiffness < 0.001:
            return 0.1
        return goal_stiffness * 55.0   # linear approximation: 0.9 → 49.5 nm (≈ dsDNA)

    def persistence_nm_to_stiffness(self, Lp_nm: float) -> float:
        return min(1.0, max(0.0, Lp_nm / 55.0))

    # ── Validation ────────────────────────────────────────────────────────────
    def validate_params(self, blender_params: dict) -> dict:
        """
        Cross-check Blender physics parameters against known experimental values.
        Returns dict of {check_name: {"status": pass/warn/fail, "msg": str}}
        """
        results = {}

        # Check dsDNA persistence length
        ds_stiff = blender_params.get("soft_body", {}).get("goal_stiffness", None)
        if ds_stiff is not None:
            Lp_sim = self.stiffness_to_persistence_nm(ds_stiff)
            target = self.VALIDATION_TARGETS["dsDNA_Lp_nm"]
            diff   = abs(Lp_sim - target["value"])
            status = "pass" if diff <= target["tolerance"] else                      "warn" if diff <= target["tolerance"] * 2 else "fail"
            results["dsDNA_persistence"] = {
                "status":     status,
                "simulated":  round(Lp_sim, 1),
                "target":     target["value"],
                "tolerance":  target["tolerance"],
                "ref":        target["ref"],
                "msg":        ("Lp=" + str(round(Lp_sim,1)) + "nm vs target " +
                               str(target["value"]) + "±" + str(target["tolerance"]) + "nm")
            }

        # Check toehold k_on via geo_nodes force_scale
        force_scale = blender_params.get("geo_nodes", {}).get("force_scale", None)
        if force_scale is not None:
            k_on_sim = force_scale * 1e6   # reverse of DNAtoBlender mapping
            target   = self.VALIDATION_TARGETS["k_on_6nt_toehold"]
            diff     = abs(k_on_sim - target["value"])
            status   = "pass" if diff <= target["tolerance"] else                        "warn" if diff <= target["tolerance"] * 3 else "fail"
            results["toehold_k_on"] = {
                "status":    status,
                "simulated": f"{k_on_sim:.2e}",
                "target":    f"{target['value']:.2e}",
                "ref":       target["ref"],
                "msg":       ("k_on=" + f"{k_on_sim:.2e}" + " vs target " +
                               f"{target['value']:.2e}" + " M-1 s-1")
            }

        # Check particle count vs expected concentration
        particle_count = blender_params.get("particles", {}).get("count", None)
        strand_nM      = blender_params.get("_strand_nM", None)
        if particle_count and strand_nM:
            predicted = self.nM_to_particles(strand_nM)
            ratio     = particle_count / max(predicted, 1)
            status    = "pass" if 0.5 <= ratio <= 2.0 else "warn"
            results["concentration_mapping"] = {
                "status":         status,
                "actual_count":   particle_count,
                "predicted_from_nM": predicted,
                "strand_nM":      strand_nM,
                "msg":            (str(particle_count) + " particles for " +
                                   str(round(strand_nM,1)) + " nM (predicted " +
                                   str(predicted) + ")")
            }

        return results

    def calibration_report(self, blender_params: dict) -> str:
        """Human-readable calibration summary."""
        val = self.validate_params(blender_params)
        lines = ["CALIBRATION REPORT", "="*40]
        lines.append("Unit system:")
        lines.append("  1 BU  = " + str(self.units["bu_to_nm"]) + " nm")
        lines.append("  1 frm = " + str(self.units["frame_to_ns"]) + " ns")
        lines.append("")
        lines.append("Parameter validation:")
        for name, result in val.items():
            icon = "OK" if result["status"] == "pass" else                    "!!" if result["status"] == "warn" else "XX"
            lines.append("  [" + icon + "] " + name + ": " + result["msg"])
        pass_count = sum(1 for r in val.values() if r["status"] == "pass")
        lines.append("")
        lines.append("Passed: " + str(pass_count) + "/" + str(len(val)))
        return "\n".join(lines)


if __name__ == "__main__":
    from dna_physics_map import DNAtoBlender
    cal = CalibrationLayer()
    bio = DNAtoBlender("GCGCATCGATGCGC", toehold_len=6, salt_mM=150, temp_C=37,
                       is_dsDNA=True, strand_nM=100)
    params = bio.summary()["blender"]
    params["_strand_nM"] = 100.0
    print(cal.calibration_report(params))
