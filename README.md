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

Building blocks (aldehydes, carboxylic acids, and amines) are retrieved from the Enamine commercial catalogue and filtered by price and bioavailability (Veber criteria). Filtered sets are then combined through three encoded reaction pathways. The resulting products (imidazolones and thiazolones) are filtered for structural alerts (Brenk + PAINS) before export:

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
| `mol_files/` | Input SDFs (tracked) and generated outputs (gitignored) |
| `01_library_generation.ipynb` | Phase 1: Combinatorial library generation |
| `02_hit_prioritization.ipynb` | Phase 2: Hit prioritization |
| `03_activity_prediction.ipynb` | Phase 3: *In silico* activity prediction |
| `env.yaml` | Conda environment specification |

## Author

Dario M. Lorente — University of Zaragoza
[840629@unizar.es](mailto:840629@unizar.es)
[dariomlorente@gmail.com](mailto:dariomlorente@gmail.com)
