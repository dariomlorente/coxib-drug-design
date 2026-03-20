"""
Cheminformatics utilities for virtual screening and compound generation.
Modules: chemistry, dataframes, enamine_api, _resources, _checkpoint, _pipeline.
"""

from .chemistry import (
    rxn_ErlenmeyerPlochl,
    rxn_AminolysisGFPc,
    rxn_SulphurExchange,
)

from .dataframes import (
    sdf_to_dataframe,
    report_df_size,
    save_dataframe_as_csv,
    add_rdkit_properties,
    filter_Veber,
    filter_BrenkPAINS,
)

from .enamine_api import (
    EnamineClient,
    add_enamine_prices,
)

# _stage_cache replaced by _checkpoint (CheckpointManager)

from ._checkpoint import (
    CheckpointManager,
    _get_checkpoint,
    _get_stage_dir,
)

from ._pipeline import (
    stage_path,
    checkpoint_path,
    rejected_path,
    init_stage_dirs,
    load_or_run,
    load_or_filter,
    save_dataframe,
)

__version__ = "3.19" # Month.Day, revised on 2026-03-19
__author__ = "Dario M Lorente"

__all__ = [
    # Chemistry
    "rxn_ErlenmeyerPlochl",
    "rxn_AminolysisGFPc",
    "rxn_SulphurExchange",
    # DataFrames
    "sdf_to_dataframe",
    "report_df_size",
    "save_dataframe_as_csv",
    "add_rdkit_properties",
    "filter_Veber",
    "filter_BrenkPAINS",
    # Enamine API
    "EnamineClient",
    "add_enamine_prices",
    # Checkpoint
    "CheckpointManager",
    # Pipeline
    "stage_path",
    "checkpoint_path",
    "rejected_path",
    "init_stage_dirs",
    "load_or_run",
    "load_or_filter",
    "save_dataframe",
]
