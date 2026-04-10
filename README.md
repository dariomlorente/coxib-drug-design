# Bachelor's Thesis — Coxib Drug Design

Final Degree Project (*Trabajo de Fin de Grado*), Universidad de Zaragoza, 2025–26.

Design and computational evaluation of a focused combinatorial library of COX-2 selective inhibitors (coxibs). The *in silico* component covers virtual compound generation via multi-step heterocyclic synthesis, drug-likeness filtering, and purchasability assessment using commercially available building blocks.

## Background

Cyclooxygenase-2 (COX-2) selective inhibitors are a class of non-steroidal anti-inflammatory drugs (NSAIDs) with reduced gastrointestinal side effects compared to non-selective NSAIDs. This work focuses on the computational design of novel imidazolone-, and thiazolone-based scaffolds derived from three commercial lists of building blocks.

## Requirements

- Python 3.10+
- Conda (recommended for environment management)
- Enamine Store API credentials (optional — required for real-time compound pricing)

## Installation

```bash
conda env create -f env.yaml
conda activate coxibs
```

## Computational Pipeline

Building blocks (aldehydes, carboxylic acids, and amines) are retrieved from the Enamine commercial catalogue and filtered by price and bioavailability (Veber criteria). A hard global molecular-price cutoff is also enforced (`PriceMol <= MAX_PRICE_MOL`, currently `200.00`) in Veber filters and reaction pairing. Filtered sets are then combined through three encoded reaction pathways. The resulting products (imidazolones and thiazolones) are filtered for structural alerts (Brenk + PAINS) before export:

| Step | Reaction | Inputs | Output |
|------|----------|--------|--------|
| 1 | Erlenmeyer–Plöchl condensation | Aldehydes + Carboxylic acids | Oxazolones |
| 2 | Aminolysis (GFPc variant) | Oxazolones + Amines | Imidazolones |
| 3 | Sulphur exchange | Oxazolones | Thiazolones |

Reactions are implemented as RDKit SMARTS templates. The resulting library is re-assessed for drug-likeness before export. The `filter_BrenkPAINS` function removes compounds with structural alerts.

## Repository Structure

| Path | Description |
|------|-------------|
| `py_utils/` | Python package: reactions, filters, I/O, pricing client |
| `py_utils/_checkpoint.py` | Checkpoint management for robust resume support |
| `py_utils/_pipeline.py` | Stage paths + load/resume orchestration |
| `mol_files/` | Input SDFs (tracked) and generated outputs (gitignored) |
| `01_library_generation.ipynb` | Phase 1: Combinatorial library generation |
| `02_hit_prioritization.ipynb` | Phase 2: Hit prioritization |
| `03_activity_prediction.ipynb` | Phase 3: *In silico* activity prediction |
| `env.yaml` | Conda environment specification |

## File Naming Convention

All output files include row counts for clarity and easy identification:

| Stage | Pattern | Example |
|-------|---------|---------|
| **Reaction (raw)** | `{Stage}_raw_{N}cmpds.csv` | `Imidazolones_raw_858270cmpds.csv` |
| **Reaction checkpoint** | `.cache/{Stage}_checkpoint.json` | `mol_files/3. Oxazolones/.cache/Oxazolones_checkpoint.json` |
| **Veber filter** | `{Stage}_veber_{N}cmpds.csv` | `Oxazolones_veber_4087cmpds.csv` |
| **Brenk+PAINS filter** | `{Stage}_brenkpains_{N}cmpds.csv` | `Imidazolones_brenkpains_118151cmpds.csv` |

**Example file structure:**
```
mol_files/4. Imidazolones/
  Imidazolones_raw_858270cmpds.csv        # A-G reaction output
  Imidazolones_veber_162291cmpds.csv      # After Veber filter
  Imidazolones_brenkpains_118151cmpds.csv # Final export (Brenk+PAINS)
  .cache/
    Imidazolones_checkpoint.json          # Resume metadata
  .rejected/
```

## Checkpoint System

The pipeline now uses a robust checkpoint system for crash recovery:

- **Reaction outputs**: `{Stage}_raw_{N}cmpds.csv` (row count in filename)
- **Checkpoints**: `.cache/{Stage}_checkpoint.json` (metadata in `.cache/`)
- **Filter outputs**: `{Stage}_{filter}_{N}cmpds.csv` (row count + filter suffix)
- **Final exports**: `{Stage}_brenkpains_{N}cmpds.csv` (row count + filter suffix)

Resume integrity checks:
- **Reactions** resume by completed reactant IDs stored in checkpoint metadata.
- **Filters** are only reused when `accepted_rows + rejected_rows == input_rows`; otherwise they are recomputed.

If the kernel crashes, re-running the notebook will:
1. Detect the checkpoint JSON in `.cache/`
2. Skip already-processed aldehydes/oxazolones (tracked by IDs)
3. Resume from the last completed chunk
4. Continue until completion

## Author

Darío M. Lorente — University of Zaragoza  
[840629@unizar.es](mailto:840629@unizar.es)  
[dariomlorente@gmail.com](mailto:dariomlorente@gmail.com)
