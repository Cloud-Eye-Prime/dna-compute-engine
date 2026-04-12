# DNA Compute Engine

**LLM-driven simulation of DNA strand displacement circuits using Blender as the physics substrate.**

A rapid prototyping tool for DNA circuit designers — the equivalent of a breadboard for molecular logic, where the LLM is the engineer and Blender is the bench.

---

## What This Does

The pipeline closes a design loop that currently exists only in physical wet labs:

```
Natural language circuit description
  → LLM generates Blender Python (bpy) scene code
  → Blender renders N parallel circuit variants (headless, GPU-accelerated)
  → Readout extracts output signals from rendered frames
  → LLM evaluates results against design goal
  → LLM refines circuit description
  → Repeat until convergence
```

A parameter sweep that takes weeks in the wet lab and thousands of dollars in reagents completes in hours on a GPU for dollars.

**This is not a molecular dynamics engine.** It is a rapid prototyping tool where approximate physics enables fast iteration. The innovation is the closed feedback loop, not the physics fidelity.

---

## Quick Start

### On Vast.ai (primary compute target)

```bash
# 1. Provision the instance
bash vastai_setup.sh

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Dry run — verify LLM generates valid bpy code
python3 dna_compute.py --sequence GCGCATCGATGCGC --variants 2 --dry-run

# 4. Full render
python3 dna_compute.py --sequence GCGCATCGATGCGC --variants 4 --workers 2

# 5. Run the feedback loop
python3 sim_loop.py --goal AND_gate --cascade half_adder --max-iter 3
```

### Inspect biophysics parameters

```bash
python3 dna_compute.py --sequence GCGCATCGATGCGC --info
```

### Run with genomic data

```bash
python3 sim_loop.py \
  --genomic data/sample_expression.csv \
  --genes BRCA1 TP53 KRAS \
  --goal AND_gate \
  --max-iter 3
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT                                                       │
│  --sequence GCGC...  |  --gate AND  |  --cascade half_adder │
│  --genomic expression.csv                                    │
└──────────────┬──────────────────────────────────────────────┘
               ▼
┌──────────────────────────┐
│  dna_physics_map.py      │  Real molecular constants → Blender params
│  Tm, Lp, k_on, Debye     │  (Wallace/SantaLucia, Zhang & Winfree 2009)
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  dna_compute.py          │  Orchestrator: builds LLM prompt with
│  + dna_gates.py          │  biophysics context, generates variant matrix
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  llm_bridge.py           │  Provider-agnostic: Anthropic, Ollama,
│                          │  LXR-5, HuggingFace, OpenAI-compat
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  parallel_render.py      │  N workers × M variants
│  blender_scene_exec.py   │  Headless Blender + Cycles GPU
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  sim_loop.py             │  Readout → Evaluate → Refine → Repeat
│                          │  Pixel-based or LLM-based signal extraction
└──────────────┬───────────┘
               ▼
┌──────────────────────────┐
│  compare_viewer.py       │  HTML comparison of variant renders
└──────────────────────────┘
```

---

## DNA Gate Library

| Gate | Description |
|------|-------------|
| AND | Output only when both input toeholds present |
| OR | Output when either input present |
| NOT | Output suppressed by input strand |
| THRESHOLD | Output above concentration threshold |
| CATALYST | One input triggers N amplified outputs |
| SEESAW | Reversible equilibrium gate |
| HAIRPIN | Self-complementary folding trigger |

### Cascade Presets

| Preset | Gates | Purpose |
|--------|-------|---------|
| `half_adder` | AND + OR | Binary addition |
| `amplifier` | CATALYST chain | Signal amplification |
| `memory_latch` | SEESAW pair | Bistable memory |
| `analog_threshold` | THRESHOLD cascade | Analog-to-digital conversion |

---

## Biophysics Grounding

Parameters in `dna_physics_map.py` are derived from published experimental values:

| Parameter | Source | Value |
|-----------|--------|-------|
| dsDNA persistence length | Experimental consensus | ~50 nm |
| ssDNA persistence length | Experimental consensus | ~1–3 nm |
| Toehold k_on | Zhang & Winfree 2009 | 10⁴–10⁶ M⁻¹s⁻¹ |
| Melting temperature | Wallace rule / SantaLucia 1998 | Sequence-dependent |
| Debye screening | Electrostatic theory | κ⁻¹ = 0.304/√[NaCl] nm |
| Diffusion | Stokes-Einstein | D = kT/6πηr |

These map to Blender physics parameters (soft body stiffness, particle counts, collision radii, etc.) through scaling functions validated by `calibration.py`.

---

## File Structure

```
dna-compute-engine/
├── dna_compute.py          # Main DNA computation orchestrator
├── dna_gates.py            # 7 gate types + 4 cascade presets
├── dna_physics_map.py      # Molecular constants → Blender parameters
├── genomic_input.py        # RNA-seq CSV → strand concentrations
├── calibration.py          # Blender units ↔ nm/ns/nM validation
├── sim_loop.py             # Closed feedback loop
├── scene_pipeline.py       # Generic scene orchestrator
├── llm_bridge.py           # Multi-provider LLM interface
├── blender_scene_exec.py   # Runs inside blender --background
├── physics_matrix.py       # N variants × M params → scenes
├── physics_exec.py         # Physics scene executor
├── physics_run.py          # CLI for physics matrix + render
├── parallel_render.py      # N-worker parallel rendering
├── compare_viewer.py       # HTML variant comparison viewer
├── stitch.py               # ffmpeg scene stitcher
├── config.yaml             # Provider and render settings
├── scenes.yaml             # Scene sequence definitions
├── vastai_setup.sh         # Vast.ai one-command provisioning
├── bootstrap.sh            # S26/Termux setup (deferred)
└── data/
    └── sample_expression.csv   # Demo RNA-seq data
```

---

## LLM Providers

Switch with one line in `config.yaml`: `active: anthropic`

| Provider | Use Case | Config Key |
|----------|----------|------------|
| Anthropic (Sonnet 4.6) | Primary: code generation + refinement | `anthropic` |
| Ollama (qwen2.5-coder:7b) | Offline / edge deployment | `ollama` |
| LXR-5 Dragon | Cloud-Eye MoA stack | `lxr5` |
| HuggingFace | Cloud inference | `hf` |
| OpenAI-compatible | LM Studio, vLLM, etc. | `openai_compat` |

---

## Compute Targets

| Target | GPU | Render Speed | Use Case |
|--------|-----|-------------|----------|
| Vast.ai (RTX 4090) | 24 GB VRAM | ~5–10 min/iteration | Research validation |
| Vast.ai (RTX 3090) | 24 GB VRAM | ~10–15 min/iteration | Budget option |
| Samsung Galaxy S26 | CPU only | ~90 min/iteration | Sovereign offline (deferred) |

---

## Validation Status

- [x] All source files pass Python syntax check
- [x] Biophysics constants reference published experimental values
- [x] Multi-provider LLM bridge functional
- [ ] Blender renders LLM-generated bpy code without crashing
- [ ] Feedback loop converges toward improved circuit designs
- [ ] Genomic input produces differentiated tumor vs normal output
- [ ] Calibration values match experimental references

---

## Scientific Context

DNA computing is at an inflection point. Recent work includes DNA-based supervised learning (Caltech Qian lab, Nature 2025), conformational DNA computing for cancer gene regulation (Science Advances, Feb 2026), and DNA strand displacement achieving 94.7% cancer classification accuracy (PMC 2022).

The field designs circuits on paper, simulates kinetics as text, then goes directly to expensive wet lab synthesis. No visual rapid prototyping step exists. This pipeline fills that gap.

Target venue: DNA32 (International Conference on DNA Computing, Aug 2026), which explicitly calls for software tools for analysis, simulation, and design.

---

## References

- Zhang & Winfree (2009). Control of DNA strand displacement kinetics using toehold exchange. *JACS*.
- SantaLucia (1998). A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics. *PNAS*.
- Srinivas et al. (2013). On the biophysics and kinetics of toehold-mediated DNA strand displacement. *Nucleic Acids Research*.
- Šulc et al. (2012). Sequence-dependent thermodynamics of a coarse-grained DNA model. *J. Chem. Phys.* (oxDNA)

---

## Part of the Cloud-Eye ecosystem

This project is a satellite of [Cloud-Eye Prime](https://github.com/Cloud-Eye-Prime), sharing infrastructure (Librarian, Forge dispatch, LLM bridge) but operating independently as a research tool.

---

*Yang er bu yong — cultivate without forcing. The circuit designs itself; we hold the space.*
