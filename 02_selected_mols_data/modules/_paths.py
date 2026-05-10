from __future__ import annotations

PHASE_DIR = "02_selected_mols_data"
INPUTS = f"{PHASE_DIR}/inputs"
OUTPUTS = f"{PHASE_DIR}/outputs"
INTERIM = f"{PHASE_DIR}/.interim"

INPUT_IC50 = f"{INPUTS}/COX1&2_IC50.csv"

# Upstream phase outputs (Phase 1 → Phase 2)
P1_OUTPUTS = "01_library_mols_data/outputs"

# Interim dirs
QSAR_DIR = f"{INTERIM}/qsar"
QED_DIR = f"{INTERIM}/qed"
QED_CACHE = f"{QED_DIR}/.cache"
CLUSTERING_INTERIM = f"{INTERIM}/clustering"
CLUSTERING_INPUTS = f"{CLUSTERING_INTERIM}/.inputs"
CLUSTERING_REJECTED = f"{CLUSTERING_INTERIM}/.rejected"

# Interim dirs — ALMOS outputs
ALMOS_DIR = f"{INTERIM}/ALMOS"
