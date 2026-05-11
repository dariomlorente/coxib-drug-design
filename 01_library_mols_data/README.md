# PHASE 1: Virtual Library Generation (Imidazolones + Thiazolones)

This folder holds the data and code for Phase 1 of the project, which runs inside
`01_LIBRARY_GENERATION.ipynb`. The goal of this phase is to computationally synthesise
a large set of drug-like molecules and filter them down to a focused, purchasable library.

![Phase 1 Flowchart](inputs/.visuals/01_library_generation.png)

---

## What this phase produces

Two CSV files, one per scaffold family, exported to `01_library_mols_data/outputs/`:

- `Imidazolones_{N}cmpds.csv`
- `Thiazolones_{N}cmpds.csv`

These are the compounds that survive every filter and are ready for Phase 2 (hit prioritisation).

---

## How to run it

Phase 1 requires the `synthesis` conda environment:

```bash
conda env create -f ...coxib-drug-design/01_library_mols_data/modules/synthesis.yml
conda activate synthesis
```

Then open and run `01_LIBRARY_GENERATION.ipynb` from the repo root. The notebook
is designed to resume safely if interrupted — re-running a cell will pick up where it left off.

The two main levers at the top of the notebook are:

- **`FORCE_RECOMPUTE`** — set to `True` to ignore cached results and recompute everything from scratch.
- **`MAX_PRICE_MOL`** — the maximum cost (EUR) allowed per compound. Raising this value increases library size but also cost.

---

## Pipeline overview

The notebook runs ten steps in sequence:

| Step | What happens |
|------|-------------|
| 1 | Load building blocks from SDF files (aldehydes, carboxylic acids, amines) |
| 2 | Query the Enamine Store API and discard compounds that are too expensive |
| 3 | Apply drug-likeness filters to each building block class |
| 4 | **Erlenmeyer–Plöchl reaction**: combine aldehydes + carboxylic acids to make oxazolones |
| 5 | Filter oxazolones for drug-likeness |
| 6 | **Aminolysis (GFPc)**: combine oxazolones + amines to make imidazolones |
| 7 | **Sulphur exchange**: convert oxazolones to thiazolones |
| 8 | Filter both product families for drug-likeness (descriptors recomputed from scratch) |
| 9 | Remove compounds containing structural alerts (Brenk) or assay-interference patterns (PAINS) |
| 10 | Export final libraries to `outputs/` |

Steps 4, 6, and 7 are computationally heavy. They are automatically split into chunks and
checkpointed, so the notebook can recover from a crash without restarting from zero.

---

## Folder layout

```
01_library_mols_data/
├── inputs/          # SDF building blocks (aldehydes, carboxylics, amines)
├── outputs/         # Final product libraries exported at Step 10
├── .interim/        # Generated intermediates (gitignored)
│   ├── building_blocks/   # Price- and Veber-filtered building blocks
│   ├── oxazolones/        # EP reaction output + oxazolone filter results
│   ├── imidazolones/      # AG reaction output + product filters
│   └── thiazolones/       # SE reaction output + product filters
└── modules/         # Python package used by the notebook (see below)
```

Each `.interim/` subfolder also contains:

- `.cache/` — reaction caches and checkpoint files that make resume possible.
- `.rejected/` — compounds that failed a filter, kept for audit purposes.

> Temporary files (`.tmp_ep_results.csv`, `.tmp_ag_results.csv`, `.tmp_se_results.csv`)
> may appear inside `.cache/` during a long reaction run. They are safe to delete once
> that step is marked complete.

---

## Pricing and the Enamine API

Pricing data is fetched from the Enamine Store (a commercial supplier of research chemicals).
Credentials must be placed in a `.env` file at the repo root:

```
ENAMINE_EMAIL=your_email
ENAMINE_PASSWORD=your_password
```

The price query returns two values per compound:

- **`PriceG`** — cost in EUR per gram.
- **`PriceMol`** — cost in EUR per mole (derived from molecular weight).

Compounds with no valid price pack, or whose price exceeds the configured limits,
are dropped at this stage. Once queried, prices are cached locally so subsequent
runs do not repeat the API calls.

---

## Filters

**Drug-likeness (Veber filter):** checks molecular weight, polar surface area, lipophilicity,
rotatable bonds, and hydrogen-bond counts. Limits are tuned separately for building blocks
and for final products, and back-calculated from the target thresholds using each reaction's
expected property increments.

**Structural alerts (Brenk filter):** removes compounds containing functional groups
associated with poor metabolism, mutagenicity, or other ADMET liabilities.

**Assay interference (PAINS filter):** removes compounds likely to produce false
positives in biochemical assays, regardless of their actual potency.

Brenk and PAINS are applied to products only (imidazolones and thiazolones), not
to building blocks.

---

## Checkpointing and crash recovery

Every reaction and filter step records its progress in a small JSON file:

```
.interim/<stage>/.cache/<Stage>_checkpoint.json
```

If the kernel crashes mid-run, re-executing the notebook cell will:

1. Detect the checkpoint.
2. Skip already-processed molecules.
3. Resume from the last completed chunk.

If input parameters change (e.g. a new `MAX_PRICE_MOL`), the checkpoint detects
the mismatch via a stored hash and triggers a recompute automatically.

---

## Modules reference

The `modules/` package is the implementation layer behind the notebook.
You should not need to edit these files directly during a normal run.

| Module | Role |
|--------|------|
| `io.py` | Reads SDF files and computes RDKit molecular descriptors |
| `enamine_api.py` | Enamine Store API client, batch price queries, and price caching |
| `filters.py` | Veber drug-likeness filter and Brenk + PAINS structural alert filter |
| `reactions.py` | Orchestrates the three reactions (EP, AG, SE) with chunking and resume logic |
| `pipeline.py` | Stage paths, checkpoint management, `load_or_run`, `load_or_filter` |
| `_reaction_workers.py` | Low-level batch workers and SMARTS templates called by `reactions.py` |
| `_smarts_catalog.py` | Brenk and PAINS SMARTS pattern catalogue used by `filters.py` |
| `_utils.py` | Shared utilities: cache I/O, worker count, and canonical folder paths |
