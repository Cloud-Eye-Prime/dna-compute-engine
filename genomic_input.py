"""
genomic_input.py — Translate genomic data into DNA physics parameters.

Handles:
  - RNA-seq CSV: gene expression → strand concentration (nM)
  - VCF variants: SNPs → GC content change → stability change
  - Manual gene lists: HGNC symbols → known pathway membership

Output feeds directly into DNAComputeMatrix and dna_gates.py gate library.

Usage:
    from genomic_input import GenomicInput
    gi = GenomicInput()
    gi.load_rnaseq("sample_expression.csv")
    circuit = gi.to_diagnostic_circuit(target_genes=["BRCA1","TP53","KRAS"])
    circuit.run()
"""
import csv, json, math, pathlib
from typing import Optional
from dna_physics_map import DNAtoBlender, gc_content


# ── Canonical gene → DNA sequence proxies ─────────────────────────────────────
# Real sequences would come from Ensembl/NCBI API.
# These are representative promoter/miRNA sequences for prototype use.
GENE_SEQUENCES = {
    "BRCA1":  "GCGCATCGATGCGCATCGATGCGC",  # BRCA1 promoter CpG island proxy (GC-rich)
    "BRCA2":  "GCGCGCATCGCGCATCGCGCATCG",
    "TP53":   "ATCGATCGCCGGCTATCGATCGCC",  # p53 binding element (moderate GC)
    "KRAS":   "ATCGATCGATCGATCGATCGATCG",  # KRAS G-box (AT-rich)
    "EGFR":   "GCGCGCGCATCGATCGGCGCGCGC",  # EGFR promoter (very GC-rich)
    "MYC":    "GCGCACGTGCGCACGTGCGCACGT",  # E-box motif (MYC binding)
    "VEGFA":  "ATCGATCGCCCGGGATCGATCGCC",
    "HIF1A":  "GCGCATCGATCGCACGTGCGCATC",
    "PTEN":   "ATCGATCGATGCGCATCGATCGAT",
    "MDM2":   "GCGCATCGATCGATCGATGCGCAT",
    # miRNA sequences (relevant to DNA diagnostic circuits)
    "miR-21":  "UAGCUUAUCAGACUGAUGUUGA",  # oncomiR, elevated in many cancers
    "miR-155": "UUAAUGCUAAUUGUGAUAGGGGU",
    "miR-210": "CUGUGCGUGUGACAGCGGCUGA",
}

# Gene → cancer pathway membership
PATHWAY_MAP = {
    "DNA_damage_repair": ["BRCA1","BRCA2","TP53","MDM2","PTEN"],
    "RAS_MAPK":          ["KRAS","EGFR","MYC"],
    "angiogenesis":      ["VEGFA","HIF1A","MYC"],
    "tumor_suppressor":  ["TP53","BRCA1","PTEN"],
    "oncogene":          ["KRAS","EGFR","MYC","MDM2"],
}


class GenomicInput:
    def __init__(self):
        self.expression = {}      # gene → log2FC or TPM
        self.variants   = {}      # gene → list of SNP dicts
        self.metadata   = {}

    # ── Loaders ───────────────────────────────────────────────────────────────

    def load_rnaseq(self, path: str, gene_col: str = "gene",
                    value_col: str = "log2FC", sep: str = ","):
        """Load RNA-seq results CSV. Expects at minimum gene + expression columns."""
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                gene = row.get(gene_col, "").strip().upper()
                try:
                    val = float(row.get(value_col, 0))
                except ValueError:
                    continue
                if gene:
                    self.expression[gene] = val
        print("[genomic] Loaded " + str(len(self.expression)) + " genes from " + path)

    def load_vcf(self, path: str):
        """Parse minimal VCF for SNP data (CHROM, POS, REF, ALT, INFO)."""
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("	")
                if len(parts) < 5:
                    continue
                chrom, pos, vid, ref, alt = parts[:5]
                gene = self._gene_from_vcf_info(parts[7] if len(parts) > 7 else "")
                if gene:
                    self.variants.setdefault(gene, []).append({
                        "chrom": chrom, "pos": int(pos),
                        "ref": ref, "alt": alt,
                        "gc_delta": self._gc_delta(ref, alt)
                    })
        print("[genomic] Loaded variants for " + str(len(self.variants)) + " genes")

    def load_dict(self, data: dict):
        """Load expression data directly from a Python dict: {gene: value}."""
        self.expression = {k.upper(): float(v) for k, v in data.items()}

    # ── Parameter translators ─────────────────────────────────────────────────

    def gene_to_params(self, gene: str) -> dict:
        """
        Translate a gene's expression level into DNAtoBlender parameters.
        log2FC > 0 = upregulated (higher concentration, more particles)
        log2FC < 0 = downregulated (lower concentration, fewer particles)
        """
        gene = gene.upper()
        lfc  = self.expression.get(gene, 0.0)
        seq  = GENE_SEQUENCES.get(gene, "ATCGATCGATCG")

        # Expression → strand concentration (nM)
        # log2FC of 3 = 8x overexpression → ~800 nM
        base_conc = 100.0
        strand_nM = base_conc * (2.0 ** lfc)
        strand_nM = max(1.0, min(5000.0, strand_nM))

        # SNP effect: GC delta changes duplex stability
        gc_delta = sum(v["gc_delta"] for v in self.variants.get(gene, []))
        gc_adj   = max(0.2, min(0.8, gc_content(seq) + gc_delta))

        bio = DNAtoBlender(
            sequence=seq,
            toehold_len=6,
            salt_mM=150.0,     # physiological
            temp_C=37.0,
            is_dsDNA=(lfc > 0),  # upregulated → assume more duplex formation
            strand_nM=strand_nM
        )
        params = bio.summary()
        params["gene"] = gene
        params["log2FC"] = round(lfc, 3)
        params["strand_nM"] = round(strand_nM, 1)
        params["pathway"] = self._get_pathways(gene)
        return params

    def expression_matrix(self, genes: Optional[list] = None) -> list:
        """Return list of gene param dicts, sorted by |log2FC| descending."""
        genes = genes or list(self.expression.keys())
        result = [self.gene_to_params(g) for g in genes if g in self.expression or g in GENE_SEQUENCES]
        return sorted(result, key=lambda x: abs(x["log2FC"]), reverse=True)

    def top_deregulated(self, n: int = 6) -> list:
        """Return the N most deregulated genes for circuit design."""
        matrix = self.expression_matrix()
        return matrix[:n]

    # ── Circuit builders ───────────────────────────────────────────────────────

    def to_diagnostic_circuit_prompt(self, target_genes: list,
                                     condition: str = "cancer") -> str:
        """
        Build an LLM prompt for a diagnostic DNA logic circuit
        based on measured expression of target genes.
        """
        gene_data = [self.gene_to_params(g) for g in target_genes]

        lines = ["Diagnostic DNA Circuit Design for: " + condition + "\n"]
        lines.append("Input gene expression values:\n")
        for gd in gene_data:
            bp = gd["biophysics"]
            lfc = gd["log2FC"]
            direction = "OVER" if lfc > 0 else "UNDER"
            lines.append(
                "  " + gd["gene"] + ": log2FC=" + str(lfc) +
                " (" + direction + "-expressed), " +
                str(round(gd["strand_nM"])) + " nM, " +
                "Tm=" + str(bp["melting_temp_C"]) + "C, " +
                "GC=" + str(round(bp["gc_content"]*100)) + "%"
            )
        lines.append("\nPathway context:")

        for gd in gene_data:
            if gd["pathway"]:
                lines.append("  " + gd["gene"] + " → " + ", ".join(gd["pathway"]))

        lines.append("\nDesign a DNA strand displacement circuit that:")
        lines.append("  1. Uses strand concentrations proportional to expression values")
        lines.append("  2. Activates output signal ONLY when the " + condition + " signature is present")
        lines.append("  3. Uses AND/THRESHOLD gates to require multiple markers")
        lines.append("  4. Minimizes false positives via toehold specificity")
        lines.append("  5. Renders visually: cyan=input strands, gold=gate, magenta=output")
        return "\n".join(lines)

    def to_physics_variant_prompt(self, top_n: int = 4) -> str:
        """
        Generate a parallel physics matrix prompt from top deregulated genes.
        Each variant = one gene's expression state as a physics simulation.
        """
        top = self.top_deregulated(top_n)
        lines = ["Physics matrix from RNA-seq data:\n"]
        lines.append("Simulate " + str(len(top)) + " variants, one per gene:\n")
        for i, gd in enumerate(top):
            bl = gd["blender"]
            lines.append(
                "Variant v" + str(i+1).zfill(3) + " — " + gd["gene"] + " (log2FC " +
                str(gd["log2FC"]) + "):\n"
                "  soft_body params: " + json.dumps(bl["soft_body"]) + "\n"
                "  particle params:  " + json.dumps(bl["particles"]) + "\n"
                "  geo_nodes params: " + json.dumps(bl["geo_nodes"])
            )
        return "\n".join(lines)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_pathways(self, gene: str) -> list:
        return [pw for pw, genes in PATHWAY_MAP.items() if gene in genes]

    def _gc_delta(self, ref: str, alt: str) -> float:
        ref_gc = sum(1 for b in ref.upper() if b in "GC") / max(len(ref), 1)
        alt_gc = sum(1 for b in alt.upper() if b in "GC") / max(len(alt), 1)
        return alt_gc - ref_gc

    def _gene_from_vcf_info(self, info: str) -> str:
        for field in info.split(";"):
            if field.startswith("GENEINFO="):
                return field.split("=")[1].split(":")[0].upper()
        return ""
