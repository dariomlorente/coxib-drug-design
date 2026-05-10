from __future__ import annotations

PHASE_DIR = "01_library_mols_data"
INPUTS = f"{PHASE_DIR}/inputs"
OUTPUTS = f"{PHASE_DIR}/outputs"
INTERIM = f"{PHASE_DIR}/.interim"

BUILDING_BLOCKS = f"{INTERIM}/building_blocks"
OXAZOLONES = f"{INTERIM}/oxazolones"
IMIDAZOLONES = f"{INTERIM}/imidazolones"
THIAZOLONES = f"{INTERIM}/thiazolones"

PRICE_CACHE = f"{BUILDING_BLOCKS}/.cache/price_cache"
REACTION_CACHE = f"{OXAZOLONES}/.cache"

STAGE_REGISTRY: dict[str, str] = {
    "Aldehydes": BUILDING_BLOCKS,
    "Carboxylics": BUILDING_BLOCKS,
    "Amines": BUILDING_BLOCKS,
    "Oxazolones": OXAZOLONES,
    "Imidazolones": IMIDAZOLONES,
    "Thiazolones": THIAZOLONES,
}
