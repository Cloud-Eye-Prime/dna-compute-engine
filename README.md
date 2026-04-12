# DNA Compute Engine

**LLM-driven simulation of DNA strand displacement circuits using Blender as the physics substrate.**

A rapid prototyping tool for DNA circuit designers â€” the equivalent of a breadboard for molecular logic, where the LLM is the engineer and Blender is the bench.

---

## What This Does

The pipeline closes a design loop that currently exists only in physical wet labs:

```
Natural language circuit description
  â†’ LLM generates Blender Python (bpy) scene code
  â†’ Blender renders N parallel circuit variants (headless, GPU-accelerated)
  â†’ Readout extracts output signals from rendered frames
  â†’ LLM evaluates results against design goal
  â†’ LLM refines circuit description
  â†’ Repeat until convergence
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

# 3. Dry run â€” verify LLM generates valid bpy code
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
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  INPUT                                                       â”‚
â”‚  --sequence GCGC...  |  --gate AND  |  --cascade half_adder â”‚
â”‚  --genomic expression.csv                                    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  dna_physics_map.py      â”‚  Real molecular constants â†’ Blender params
â”‚  Tm, Lp, k_on, Debye     â”‚  (Wallace/SantaLucia, Zhang & Winfree 2009)
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  dna_compute.py          â”‚  Orchestrator: builds LLM prompt with
â”‚  + dna_gates.py          â”‚  biophysics context, generates variant matrix
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  llm_bridge.py           â”‚  Provider-agnostic: Anthropic, Ollama,
â”‚                          â”‚  LXR-5, HuggingFace, OpenAI-compat
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  parallel_render.py      â”‚  N workers Ã— M variants
â”‚  blender_scene_exec.py   â”‚  Headless Blender + Cycles GPU
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  sim_loop.py             â”‚  Readout â†’ Evaluate â†’ Refine â†’ Repeat
â”‚                          â”‚  Pixel-based or LLM-based signal extraction
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  compare_viewer.py       â”‚  HTML comparison of variant renders
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
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
| ssDNA persistence length | Experimental consensus | ~1â€“3 nm |
| Toehold k_on | Zhang & Winfree 2009 | 10â´â€“10â¶ Mâ»Â¹sâ»Â¹ |
| Melting temperature | Wallace rule / SantaLucia 1998 | Sequence-dependent |
| Debye screening | Electrostatic theory | Îºâ»Â¹ = 0.304/âˆš[NaCl] nm |
| Diffusion | Stokes-Einstein | D = kT/6Ï€Î·r |

These map to Blender physics parameters (soft body stiffness, particle counts, collision radii, etc.) through scaling functions validated by `calibration.py`.

---

## File Structure

```
dna-compute-engine/
â”œâ”€â”€ dna_compute.py          # Main DNA computation orchestrator
â”œâ”€â”€ dna_gates.py            # 7 gate types + 4 cascade presets
â”œâ”€â”€ dna_physics_map.py      # Molecular constants â†’ Blender parameters
â”œâ”€â”€ genomic_input.py        # RNA-seq CSV â†’ strand concentrations
â”œâ”€â”€ calibration.py          # Blender units â†” nm/ns/nM validation
â”œâ”€â”€ sim_loop.py             # Closed feedback loop
â”œâ”€â”€ scene_pipeline.py       # Generic scene orchestrator
â”œâ”€â”€ llm_bridge.py           # Multi-provider LLM interface
â”œâ”€â”€ blender_scene_exec.py   # Runs inside blender --background
â”œâ”€â”€ physics_matrix.py       # N variants Ã— M params â†’ scenes
â”œâ”€â”€ physics_exec.py         # Physics scene executor
â”œâ”€â”€ physics_run.py          # CLI for physics matrix + render
â”œâ”€â”€ parallel_render.py      # N-worker parallel rendering
â”œâ”€â”€ compare_viewer.py       # HTML variant comparison viewer
â”œâ”€â”€ stitch.py               # ffmpeg scene stitcher
â”œâ”€â”€ config.yaml             # Provider and render settings
â”œâ”€â”€ scenes.yaml             # Scene sequence definitions
â”œâ”€â”€ vastai_setup.sh         # Vast.ai one-command provisioning
â”œâ”€â”€ bootstrap.sh            # S26/Termux setup (deferred)
â””â”€â”€ data/
    â””â”€â”€ sample_expression.csv   # Demo RNA-seq data
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
| Vast.ai (RTX 4090) | 24 GB VRAM | ~5â€“10 min/iteration | Research validation |
| Vast.ai (RTX 3090) | 24 GB VRAM | ~10â€“15 min/iteration | Budget option |
| Samsung Galaxy S26 | CPU only | ~90 min/iteration | Sovereign offline (deferred) |

---

## Validation Status

- [x] All source files pass Python syntax check
- [x] Biophysics constants reference published experimental values
- [x] Multi-provider LLM bridge functional
- [x] Blender renders LLM-generated bpy code (Gate 1 passed 2026-04-12)
- [ ] Feedback loop converges toward improved circuit designs (Gate 2)
- [ ] Genomic input produces differentiated tumor vs normal output (Gate 3)
- [ ] Calibration values match experimental references (Gate 4)

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
- Å ulc et al. (2012). Sequence-dependent thermodynamics of a coarse-grained DNA model. *J. Chem. Phys.* (oxDNA)

---

## Part of the Cloud-Eye ecosystem

This project is a satellite of [Cloud-Eye Prime](https://github.com/Cloud-Eye-Prime), sharing infrastructure (Librarian, Forge dispatch, LLM bridge) but operating independently as a research tool.

---

*Yang er bu yong â€” cultivate without forcing. The circuit designs itself; we hold the space.*
