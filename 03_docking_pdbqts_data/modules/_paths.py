from __future__ import annotations

PHASE_DIR = "03_docking_pdbqts_data"
INPUTS = f"{PHASE_DIR}/inputs"
OUTPUTS = f"{PHASE_DIR}/outputs"
INTERIM = f"{PHASE_DIR}/.interim"

# Upstream phase outputs (Phase 2 → Phase 3)
P2_OUTPUTS = "02_selected_mols_data/outputs"

# Interim directories (flat structure)
LIGANDS_DIR = f"{INTERIM}/ligands"
RECEPTORS_DIR = f"{INTERIM}/receptors"
DOCKING_DIR = f"{INTERIM}/docking"
DOCKING_MAPPING = f"{INTERIM}/mapping.csv"

# Output files
DOCKING_SCORES = f"{OUTPUTS}/docking_scores.csv"
