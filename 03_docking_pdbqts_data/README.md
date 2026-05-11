# PHASE 3: Structure-Based Docking Validation (Bioactivity Grading)

This folder holds the data and code for Phase 3 of the project, which runs inside
`03_DOCKING_GRADING.ipynb`. The goal of this phase is to take the ~80 shortlisted
compounds from Phase 2 and test them computationally against the real 3D shape of the
COX-2 binding pocket — and that of COX-1 for comparison. A program called AutoDock Vina
attempts to fit each molecule into the pocket in the most favourable orientation, and the
results are combined with the machine-learning scores from Phase 2 to produce a final
ranked shortlist for experimental follow-up.

![Phase 3 Flowchart](inputs/.visuals/03_docking_grading.png)

---

## What this phase produces

- `outputs/docking_scores.csv` — ranked table combining QSAR scores, docking scores for
  both receptors, geometric quality scores, and the composite final ranking.
- `outputs/poses/` — the best predicted binding pose for each compound, stored as PDBQT
  files compatible with standard molecular visualisation software.
- `outputs/figures/` — rendered images of the top-ranked poses (optional step).

---

## How to run it

Phase 3 requires the [`docking`](modules/docking.yml) conda environment:

```bash
conda env create -f 03_docking_pdbqts_data/modules/docking.yml
conda activate docking
```

Then open and run `03_DOCKING_GRADING.ipynb` from the repo root. The notebook is
designed to run on a personal computer in under 20 minutes. Re-running is safe —
completed docking jobs are detected automatically and skipped.

---

## Pipeline overview

The notebook runs nine steps in sequence:

| Step | What happens |
|------|-------------|
| 17 | Load the cluster-representative compounds from Phase 2 |
| 18 | Build realistic 3D shapes for each molecule and convert them to the format Vina requires |
| 19 | Prepare the two receptor structures and define the search region around the binding site |
| 20 | Dock every compound against both receptors using AutoDock Vina |
| 21 | Check that all docking jobs completed correctly and extract the results |
| 22 | Score each predicted pose by whether it makes the expected contacts in the binding site |
| 24 | Combine QSAR scores, geometry scores, and Vina scores into a unified ranking |
| 25 | Export the ranked table and the best poses to [`outputs/`](outputs) |
| 26 | Render images of the top five poses using PyMOL (optional) |

Imidazolones and thiazolones are merged into a single dataset and processed together
from Step 17 onwards.

---

## Design decisions

### Why two receptors?

Every compound is docked against both COX-2 (the therapeutic target, crystal structure
6COX) and COX-1 (the off-target we want to avoid, crystal structure 3KK6). This lets the
final score reward compounds that bind well to COX-2 while penalising those that also
bind tightly to COX-1 — which is exactly the selectivity profile the project is looking for.

### Locating the binding site

For COX-2 the binding site is straightforward: the crystal structure already contains a
known inhibitor (SC-558) sitting in the pocket, and we define the search box around it.

For COX-1 no equivalent reference ligand is available in the right position, so the code
mathematically aligns the two protein structures by matching 29 equivalent reference points
(Cα atoms of conserved residues), then transfers the known COX-2 binding site coordinates
across into COX-1's frame of reference.

### Geometric scoring (COX-2 only)

Vina outputs a single number per pose (an estimated binding energy), but that number alone
does not tell us whether the molecule is sitting in the *right* part of the pocket or making
the contacts that matter for COX-2 selectivity. The geometric scoring step checks each pose
against three structural criteria:

- **Arg120 and Tyr355 contacts** — these two residues flank the entrance to the active
  site and are known anchor points for virtually all NSAID-class inhibitors. A pose that
  reaches both scores higher.
- **Side-pocket occupancy** — COX-2 has a small side pocket (near Val523) that COX-1 lacks
  due to a single amino-acid difference (Ile523). Coxibs owe much of their selectivity to
  filling this pocket with a bulky group. Poses that occupy it are rewarded.
- **Clash penalty** — poses where the molecule overlaps with receptor atoms are penalised.

### Final ranking

The three sources of evidence are combined into a single score:

```
final_score = 0.5 × QSAR_norm + 0.4 × geo_norm + 0.1 × Vina_norm
```

Each term is rescaled to the same 0–1 range before combining, so no single metric
dominates unfairly. The QSAR term carries the most weight because it is trained on real
experimental data; geometry adds structure-aware selectivity information; and the Vina
score serves mainly as a sanity check.

---

## Folder layout

```
03_docking_pdbqts_data/
├── inputs/               # PDB crystal structures and visual assets
│   ├── 6COX.pdb          # COX-2 (SC-558 co-crystallised)
│   ├── 3KK6.pdb          # COX-1 (flurbiprofen co-crystallised)
│   └── .visuals/         # Flowchart and figure source files
├── outputs/              # Final docking results
│   ├── docking_scores.csv
│   ├── poses/            # Best-pose PDBQT files per receptor
│   │   ├── 6COX/
│   │   └── 3KK6/
│   └── figures/          # PyMOL renderings (optional)
├── .interim/             # Generated intermediates (gitignored)
│   ├── ligands/          # 3D ligand files (SDF and PDBQT)
│   ├── receptors/        # Prepared receptor files and binding-box definitions
│   ├── docking/          # Raw Vina outputs
│   │   ├── 6COX/
│   │   │   ├── logs/     # Per-job Vina log files
│   │   │   └── poses/    # Vina output PDBQT (all modes)
│   │   └── 3KK6/
│   │       ├── logs/
│   │       └── poses/
│   ├── mapping.csv       # Full list of ligand–receptor docking tasks
│   ├── rescoring/        # MM-GBSA rescoring outputs (HPC only)
│   └── md/               # MD simulation files (HPC only)
└── modules/              # Python package used by the notebook (see below)
```

---

## Inputs

| File | Source | Role |
|------|--------|------|
| `6COX.pdb` | RCSB Protein Data Bank | COX-2 receptor structure with SC-558 co-crystallised |
| `3KK6.pdb` | RCSB Protein Data Bank | COX-1 receptor structure with flurbiprofen co-crystallised |
| `Imidazolones_*samples.csv` | Phase 2 output | Cluster representatives for docking |
| `Thiazolones_*samples.csv` | Phase 2 output | Cluster representatives for docking |

The latest `*samples.csv` files are picked up automatically based on the filename.

---

## Modules reference

The [`modules/`](modules) package is the implementation layer behind the notebook.
You should not need to edit these files directly during a normal run.

| Module | Role |
|--------|------|
| [`docking.py`](modules/docking.py) | High-level workflow functions called directly by the notebook |
| [`_ligands.py`](modules/_ligands.py) | Building 3D structures for each compound and converting to docking format |
| [`_receptors.py`](modules/_receptors.py) | Cleaning receptor PDB files, locating the binding site, generating receptor input files |
| [`_docking.py`](modules/_docking.py) | Running AutoDock Vina, writing the task mapping, creating HPC job scripts |
| [`_parsing.py`](modules/_parsing.py) | Reading Vina log files, validating that jobs completed, extracting pose data |
| [`_scoring.py`](modules/_scoring.py) | Geometric scoring, pose selection, and all ranking calculations |
| [`_hpc.py`](modules/_hpc.py) | Optional HPC scripts for MM-GBSA rescoring and short MD simulations |
| [`_visualize_pose.py`](modules/_visualize_pose.py) | Headless PyMOL rendering of receptor–ligand poses as PNG images |
| [`_paths.py`](modules/_paths.py) | Central definitions of all directory and file paths used across the package |

---

## Scoring nomenclature

| Score | Meaning |
|-------|---------|
| `QSAR_score` | From Phase 2 ML models: predicted COX-2 potency minus COX-1 potency (higher is more selective) |
| `docking_score` | Vina estimated binding energy in kcal/mol — more negative means a tighter predicted fit |
| `geometric_score` | Structure-based pose quality for COX-2 (Arg120/Tyr355 contacts, side-pocket occupancy, clashes) |
| `final_score` | Unified 0–1 ranking combining all three sources of evidence |

---

## HPC workflow (optional)

The standard notebook is designed for local execution. The `_hpc.py` module provides
tools for larger-scale deployment on an HPC cluster:

1. Generate SLURM array job scripts via `generate_docking_slurm`, `generate_rescore_slurm`, and `generate_md_slurm`.
2. Transfer the contents of `.interim/` to the cluster and submit the array jobs.
3. Copy results back and use `parse_mmgbsa_results` and `select_md_candidates` to process them.

The MM-GBSA rescoring step refines the binding energy estimates using a physics-based
implicit-solvent model. The MD step runs a short (1–2 ns) simulation to check that the
docked poses remain stable under realistic conditions.

---

## Troubleshooting

**Vina not found** — `vina` must be available on the command line. Install via conda:
`conda install -c conda-forge vina=1.2.5`.

**Meeko not available** — the ligand-to-PDBQT conversion step will fail. Install via pip:
`pip install meeko`.

**Receptor preparation fails** — the code tries three conversion tools in order
(`obabel`, `mk_prepare_receptor.py`, `prepare_receptor4.py`). At least one must be
installed. Open Babel (`obabel`) is the recommended option and is included in the
`docking` environment.

**No valid docking results** — run the validation cell (Step 21) and read the summary.
Common causes: docking was interrupted before finishing, Vina crashed on a particular
compound, or disk space ran out during the run.

**PyMOL rendering fails** — `pymol-open-source` must be installed
(`conda install -c conda-forge pymol-open-source`). On Linux, headless rendering
may require a virtual display: prefix the notebook launch command with `xvfb-run`.
