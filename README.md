![@TheAlegreGroup](01_library_mols_data/inputs/.visuals/.affiliation.png)

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
conda env create -f envs/synthesis.yml
conda activate coxibs
```

Additional environments (for later phases):
```bash
conda env create -f envs/clustering.yml
conda env create -f envs/docking.yml
```

## Flowdiagrams

### 01_library_generation.png
![1st Phase Flowdiagram](01_library_mols_data/inputs/.visuals/01_library_generation.png)
Flowchart of the Jupyter Notebook [`01_LIBRARY_GENERATION.ipynb`](03_DOCKING_GRADING.ipynb)

### 02_hit_prioritisation.png
![2nd Phase Flowdiagram](02_selected_mols_data/inputs/.visuals/02_hit_prioritisation.png)
Flowchart of the Jupyter Notebook [`02_HIT_PRIORITISATION.ipynb`](03_DOCKING_GRADING.ipynb)

### 03_docking_grading.png
![3rd Phase Flowdiagram](03_docking_pdbqts_data/inputs/.visuals/03_docking_grading.png)
Flowchart of the Jupyter Notebook [`03_DOCKING_GRADING.ipynb`](03_DOCKING_GRADING.ipynb)

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
| `.env` | Enamine API credentials (gitignored) |
| `01_LIBRARY_GENERATION.ipynb` | Phase 1: Combinatorial library generation |
| `02_HIT_PRIORISATION.ipynb` | Phase 2: Hit prioritization + ALMOS clustering |
| `03_DOCKING_GRADING.ipynb` | Phase 3: Docking grading and scoring |
| `01_library_mols_data/` | Phase 1 data: SDF building blocks, intermediates, products |
| `01_library_mols_data/modules/` | Phase 1 package: `io.py`, `filters.py`, `reactions.py`, `enamine_api.py`, `pipeline.py` |
| `02_selected_mols_data/` | Phase 2 data: IC50 datasets, QED/QSAR intermediates, cluster outputs |
| `02_selected_mols_data/modules/` | Phase 2 package: `winnowing.py`, `clustering.py`, `prediction.py` |
| `03_docking_pdbqts_data/` | Phase 3 data: PDB structures, PDBQT intermediates, docking scores |
| `03_docking_pdbqts_data/modules/` | Phase 3 package: `docking.py` |
| `figures/` | Figures and affiliation badge |
| `envs/` | Conda environment files |
| `LICENSE` | Apache 2.0 |

## File Naming Convention:

All output files include row counts for clarity and easy identification:

| Stage | Pattern | Example |
|-------|---------|---------|
| **Reaction (raw)** | `{Stage}_raw_{N}cmpds.csv` | `Imidazolones_raw_858270cmpds.csv` |
| **Reaction checkpoint** | `.cache/{Stage}_checkpoint.json` | `01_library_mols_data/.interim/02_oxazolones/.cache/Oxazolones_checkpoint.json` |
| **Veber filter** | `{Stage}_veber_{N}cmpds.csv` | `Oxazolones_veber_4087cmpds.csv` |
| **Brenk+PAINS filter** | `{Stage}_brenkpains_{N}cmpds.csv` | `Imidazolones_brenkpains_118151cmpds.csv` |

**Example file structure:**
```
01_library_mols_data/.interim/03_imidazolones/
  Imidazolones_raw_858270cmpds.csv           # A-G reaction output
  Imidazolones_veber_162291cmpds.csv         # After Veber filter
  Imidazolones_brenkpains_118151cmpds.csv    # Final export (Brenk+PAINS)
  .cache/
    Imidazolones_checkpoint.json             # Resume metadata
  .rejected/
```

## Checkpoint System:

The pipeline uses a robust checkpoint system for crash recovery:

- **Reaction outputs**: `{Stage}_raw_{N}cmpds.csv` (row count in filename)
- **Checkpoints**: `.cache/{Stage}_checkpoint.json` (metadata in `01_library_mols_data/.interim/*/.cache/`)
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

Notebook `02_HIT_PRIORISATION.ipynb` loads Phase 1 products (`*_brenkpains_*cmpds.csv`), adds QED, applies a composite bioavailability filter (Lipinski, Ghose, Egan, Muegge, Veber) with a **4/5 pass threshold**, exports phase-2 accepted/rejected sets under `02_selected_mols_data/.interim/qed/`, runs an ML-QSAR winnow stage (COX2/COX1 Random Forest models + applicability domain), exports QSAR accepted/rejected sets under `02_selected_mols_data/.interim/qsar/`, and then runs ALMOS clustering on the QSAR-accepted sets.

Output paths:

- Accepted imidazolones: `02_selected_mols_data/.interim/qed/Imidazolones_{N}cmpds.csv`
- Accepted thiazolones: `02_selected_mols_data/.interim/qed/Thiazolones_{N}cmpds.csv`
- Rejected imidazolones: `02_selected_mols_data/.interim/qed/.rejected/Imidazolones_rejected_bioavailability_{N}cmpds.csv`
- Rejected thiazolones: `02_selected_mols_data/.interim/qed/.rejected/Thiazolones_rejected_bioavailability_{N}cmpds.csv`

The `Violation` column is inserted after `QED` and stores violated rule names.

QSAR outputs:

- Accepted imidazolones: `02_selected_mols_data/.interim/qsar/Imidazolones_qsar_{N}cmpds.csv`
- Accepted thiazolones: `02_selected_mols_data/.interim/qsar/Thiazolones_qsar_{N}cmpds.csv`
- Rejected imidazolones: `02_selected_mols_data/.interim/qsar/.rejected/Imidazolones_rejected_qsar_{N}cmpds.csv`
- Rejected thiazolones: `02_selected_mols_data/.interim/qsar/.rejected/Thiazolones_rejected_qsar_{N}cmpds.csv`

## ALMOS Clustering Outputs:

ALMOS clustering consumes the Phase 2 clustering-input files and runs independently for imidazolones and thiazolones. The canonical execution entrypoint is the ALMOS section inside `02_HIT_PRIORISATION.ipynb`, backed by `02_selected_mols_data/modules/clustering.py`.

Implementation note: ALMOS requires a unique `--name` column. If `ID` contains duplicates, the helper module auto-generates a temporary unique name column (`ALMOS_ID`) for the ALMOS run and maps cluster labels back to the original rows.

Run-level outputs are written to `02_selected_mols_data/.interim/clustering/ALMOS/`:

- Representatives (one per cluster):
  - `02_selected_mols_data/outputs/Imidazolones_{K}_samples.csv`
  - `02_selected_mols_data/outputs/Thiazolones_{K}_samples.csv`
- Full clustered imidazolones: `02_selected_mols_data/.interim/clustering/ALMOS/Imidazolones_clusters_k{K}_{N}cmpds.csv`
- Full clustered thiazolones: `02_selected_mols_data/.interim/clustering/ALMOS/Thiazolones_clusters_k{K}_{N}cmpds.csv`
- Representatives (with ALMOS metadata):
  - `02_selected_mols_data/.interim/clustering/ALMOS/Imidazolones_representatives_k{K}_{K}cmpds.csv`
  - `02_selected_mols_data/.interim/clustering/ALMOS/Thiazolones_representatives_k{K}_{K}cmpds.csv`
- Top-N shortlist for group discussion:
  - `02_selected_mols_data/.interim/clustering/ALMOS/Imidazolones_shortlist_top{T}_k{K}_{M}cmpds.csv`
  - `02_selected_mols_data/.interim/clustering/ALMOS/Thiazolones_shortlist_top{T}_k{K}_{M}cmpds.csv`
- Cluster summary tables:
  - `02_selected_mols_data/.interim/clustering/ALMOS/Imidazolones_cluster_summary_k{K}.csv`
  - `02_selected_mols_data/.interim/clustering/ALMOS/Thiazolones_cluster_summary_k{K}.csv`
- Run metadata JSON:
  - `02_selected_mols_data/.interim/clustering/ALMOS/Imidazolones_cluster_run_k{K}_{N}cmpds.json`
  - `02_selected_mols_data/.interim/clustering/ALMOS/Thiazolones_cluster_run_k{K}_{N}cmpds.json`

The metadata JSON stores command-line parameters, input SHA256, selected ALMOS outputs, and stdout/stderr log paths from the run folder under `02_selected_mols_data/.interim/clustering/ALMOS/.runs/`.

## Phase 3 (Docking Grading):

Notebook `03_DOCKING_GRADING.ipynb` validates QSAR-selected cluster representatives via AutoDock Vina docking. It handles ligand preparation, docking execution, pose validation, geometric scoring, and composite ranking. QSAR modelling and IC₅₀ prediction are handled entirely within `02_HIT_PRIORISATION.ipynb`.

Logs and intermediates are stored per receptor under `03_docking_pdbqts_data/.interim/hpc/{receptor}/`. Exported poses and scores go to `03_docking_pdbqts_data/outputs/`.

Representative compounds are loaded from:
- `02_selected_mols_data/outputs/Imidazolones_{K}_samples.csv`
- `02_selected_mols_data/outputs/Thiazolones_{K}_samples.csv`

QSAR predictions produced by Phase 2 are available at:
- `02_selected_mols_data/.interim/qsar/qsar_predictions.csv`

### Visualization:

Pose visualization is handled by `render_top_poses` in `03_docking_pdbqts_data/modules/docking.py`, a PyMOL-based function that:
- Converts ligand PDBQT → PDB via Open Babel
- Renders receptor (cartoon, light blue) and ligand (sticks, magenta)
- Outputs 1200×900 PNG at 300 DPI
- Runs headless (no GUI), suitable for HPC environments

## Author:

Darío M. Lorente — University of Zaragoza  
[840629@unizar.es](mailto:840629@unizar.es)  
[dariomlorente@gmail.com](mailto:dariomlorente@gmail.com)
