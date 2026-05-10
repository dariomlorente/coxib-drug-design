from __future__ import annotations

# Phase 1: Library Generation
# Re-exports public API from Phase 1 modules

from .io import (
    sdf_to_dataframe,
    report_df_size,
    save_dataframe_as_csv,
    add_rdkit_properties,
)

from .filters import (
    filter_Veber,
    filter_BrenkPAINS,
)

from .reactions import (
    rxn_ErlenmeyerPlochl,
    rxn_AminolysisGFPc,
    rxn_SulphurExchange,
)

from .enamine_api import (
    EnamineClient,
    add_enamine_prices,
)

from .pipeline import (
    CheckpointManager,
    stage_path,
    checkpoint_path,
    rejected_path,
    init_stage_dirs,
    load_or_run,
    load_or_filter,
    save_dataframe,
)