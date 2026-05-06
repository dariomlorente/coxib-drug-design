# Bachelor's Thesis — Coxib Drug Design

Final Degree Project (*Trabajo de Fin de Grado*), Universidad de Zaragoza, 2025–26.

Design and computational evaluation of a focused combinatorial library of COX-2 selective inhibitors (coxibs). The *in silico* component covers virtual compound generation via multi-step heterocyclic synthesis, drug-likeness filtering, and purchasability assessment using commercially available building blocks.

## Background:

Cyclooxygenase-2 (COX-2) selective inhibitors are a class of non-steroidal anti-inflammatory drugs (NSAIDs) with reduced gastrointestinal side effects compared to non-selective NSAIDs. This work focuses on the computational design of novel imidazolone- and thiazolone-based scaffolds derived from three commercial lists of building blocks.

## Requirements:

- Python 3.10+
- Conda (recommended for environment management)
- ALMOS CLI (`almos-kit`) available for the ALMOS clustering stage
- Enamine Store API credentials (optional — required for real-time compound pricing)

## Installation:

```bash
conda env create -f synthesis.yml
conda activate coxibs
```

Additional environments (for later phases):
```bash
conda env create -f clustering.yml
conda env create -f docking.yml
```

## Computational Pipeline:

Building blocks (aldehydes, carboxylic acids, and amines) are retrieved from the Enamine commercial catalogue and filtered by price and bioavailability (Veber criteria). A hard global molecular-price cutoff is also enforced (`PriceMol <= MAX_PRICE_MOL`, currently `200.00`) in Veber filters and reaction pairing. Filtered sets are then combined through three encoded reaction pathways. The resulting products (imidazolones and thiazolones) are filtered for structural alerts (Brenk + PAINS) before export:

| Step | Reaction | Inputs | Output |
|------|----------|---------|--------|
| 1 | Erlenmeyer–Plöchl condensation | Aldehydes + Carboxylic acids | Oxazolones |
| 2 | Aminolysis (GFPc variant) | Oxazolones + Amines | Imidazolones |
| 3 | Sulphur exchange | Oxazolones | Thiazolones |

Reactions are implemented as RDKit SMARTS templates. The resulting library is reassessed for drug-likeness before export. The `filter_BrenkPAINS` function removes compounds with structural alerts.

## Repository Structure:

| Path | Description |
|------|-------------|
| `py_utils/` | Python package: reactions, filters, I/O, pricing, pipeline |
| `py_utils/_utils.py` | Hardware resources + caching helpers (private) |
| `py_utils/_smarts_catalog.py` | Brenk + PAINS structural alerts (private) |
| `mol_files/` | Input SDFs (tracked) and generated outputs (gitignored) |
| `protein_files/` | PDB structures, ChEMBL data, docking outputs |
| `00_purchased_reagents.ipynb` | Phase 0: Purchased reagents (CAS → Enamine SDF subsets) |
| `01_library_generation.ipynb` | Phase 1: Combinatorial library generation |
| `02_hit_prioritization.ipynb` | Phase 2: Hit prioritization + ALMOS clustering |
| `03_docking_validation.ipynb` | Phase 3: Docking validation |
| `py_utils/winnowing.py` | Phase 2 helper (QED + bioavailability + QSAR) |
| `py_utils/clustering.py` | Phase 2 helper (ALMOS clustering + representatives + exports) |
| `py_utils/docking.py` | Phase 3 helper (docking + pose scoring + ranking) |
| `py_utils/inventory.py` | Phase 0 helper (CAS → SMILES → SDF subset) |
| `scripts/visualize_pose.py` | Pose visualization script (PyMOL-based, HPC-safe) |
| `synthesis.yml` | Conda environment for Phase 1 (RDKit, pandas) |
| `clustering.yml` | Conda environment for Phase 2 (ALMOS, scikit-learn) |
| `docking.yml` | Conda environment for Phase 3 (PyMOL, Vina, RDKit) |

## File Naming Convention:

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

## Checkpoint System:

The pipeline uses a robust checkpoint system for crash recovery:

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

## Phase 2 Outputs (Hit Prioritization):

Notebook `02_hit_prioritization.ipynb` loads Phase 1 products (`*_brenkpains_*cmpds.csv`), adds QED, applies a composite bioavailability filter (Lipinski, Ghose, Egan, Muegge, Veber) with a **4/5 pass threshold**, exports phase-2 accepted/rejected sets under `mol_files/6. QED/`, runs an ML-QSAR winnow stage (COX2/COX1 Random Forest models + applicability domain), exports QSAR accepted/rejected sets under `mol_files/7. QSAR/`, and then runs ALMOS clustering on the QSAR-accepted sets.

Output paths:

- Accepted imidazolones: `mol_files/6. QED/Imidazolones_{N}cmpds.csv`
- Accepted thiazolones: `mol_files/6. QED/Thiazolones_{N}cmpds.csv`
- Rejected imidazolones: `mol_files/6. QED/.rejected/Imidazolones_rejected_bioavailability_{N}cmpds.csv`
- Rejected thiazolones: `mol_files/6. QED/.rejected/Thiazolones_rejected_bioavailability_{N}cmpds.csv`

The `Violation` column is inserted after `QED` and stores violated rule names.

QSAR outputs:

- Accepted imidazolones: `mol_files/7. QSAR/Imidazolones_qsar_{N}cmpds.csv`
- Accepted thiazolones: `mol_files/7. QSAR/Thiazolones_qsar_{N}cmpds.csv`
- Rejected imidazolones: `mol_files/7. QSAR/.rejected/Imidazolones_rejected_qsar_{N}cmpds.csv`
- Rejected thiazolones: `mol_files/7. QSAR/.rejected/Thiazolones_rejected_qsar_{N}cmpds.csv`

## ALMOS Clustering Outputs:

ALMOS clustering consumes the Phase 2 clustering-input files and runs independently for imidazolones and thiazolones. The canonical execution entrypoint is the ALMOS section inside `02_hit_prioritization.ipynb`, backed by `py_utils/clustering.py` (same pipeline stage).

Implementation note: ALMOS requires a unique `--name` column. If `ID` contains duplicates, the helper module auto-generates a temporary unique name column (`ALMOS_ID`) for the ALMOS run and maps cluster labels back to the original rows.

Run-level outputs are written to `mol_files/8. Clustering/ALMOS/`:

- Representatives (one per cluster):
  - `mol_files/8. Clustering/Imidazolones_{K}_samples.csv`
  - `mol_files/8. Clustering/Thiazolones_{K}_samples.csv`
- Full clustered imidazolones: `mol_files/8. Clustering/ALMOS/Imidazolones_clusters_k{K}_{N}cmpds.csv`
- Full clustered thiazolones: `mol_files/8. Clustering/ALMOS/Thiazolones_clusters_k{K}_{N}cmpds.csv`
- Representatives (with ALMOS metadata):
  - `mol_files/8. Clustering/ALMOS/Imidazolones_representatives_k{K}_{K}cmpds.csv`
  - `mol_files/8. Clustering/ALMOS/Thiazolones_representatives_k{K}_{K}cmpds.csv`
- Top-N shortlist for group discussion:
  - `mol_files/8. Clustering/ALMOS/Imidazolones_shortlist_top{T}_k{K}_{M}cmpds.csv`
  - `mol_files/8. Clustering/ALMOS/Thiazolones_shortlist_top{T}_k{K}_{M}cmpds.csv`
- Cluster summary tables:
  - `mol_files/8. Clustering/ALMOS/Imidazolones_cluster_summary_k{K}.csv`
  - `mol_files/8. Clustering/ALMOS/Thiazolones_cluster_summary_k{K}.csv`
- Run metadata JSON:
  - `mol_files/8. Clustering/ALMOS/Imidazolones_cluster_run_k{K}_{N}cmpds.json`
  - `mol_files/8. Clustering/ALMOS/Thiazolones_cluster_run_k{K}_{N}cmpds.json`

The metadata JSON stores command-line parameters, input SHA256, selected ALMOS outputs, and stdout/stderr log paths from the run folder under `mol_files/8. Clustering/ALMOS/.runs/`.

## Phase 3 (Docking Validation):

Notebook `03_docking_validation.ipynb` validates QSAR-selected cluster representatives via AutoDock Vina docking. It handles ligand preparation, docking execution, pose validation, geometric scoring, and composite ranking. QSAR modelling and IC₅₀ prediction are handled entirely within `02_hit_prioritization.ipynb`.

Representative compounds are loaded from:
- `mol_files/8. Clustering/Imidazolones_{K}_samples.csv`
- `mol_files/8. Clustering/Thiazolones_{K}_samples.csv`

QSAR predictions produced by Phase 2 are available at:
- `protein_files/Machine Learning/qsar_predictions.csv`

### Visualization:

Pose visualization uses `scripts/visualize_pose.py`, a standalone PyMOL-based script that:
- Converts ligand PDBQT → PDB via Open Babel
- Renders receptor (cartoon, light blue) and ligand (sticks, magenta)
- Outputs 1200×900 PNG at 300 DPI
- Runs headless (no GUI), suitable for HPC environments

## Author:

Darío M. Lorente — University of Zaragoza  
[840629@unizar.es](mailto:840629@unizar.es)  
[dariomlorente@gmail.com](mailto:dariomlorente@gmail.com)
