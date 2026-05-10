from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# This file re-exports from the canonical Phase 1 copy.
# Make all edits in 01_library_mols_data/modules/_utils.py (single
# source of truth). This file stays read‑only.
# ──────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

_canonical = (
    Path(__file__).resolve().parent.parent.parent
    / "01_library_mols_data/modules/_utils.py"
)
exec(compile(_canonical.read_text(), _canonical, "exec"))
