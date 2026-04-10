"""
Cheminformatics utilities for virtual screening and compound generation.

Modules:
  - reactions  : rxn_ErlenmeyerPlochl, rxn_AminolysisGFPc, rxn_SulphurExchange
  - io        : sdf_to_dataframe, add_rdkit_properties, save helpers
  - filters   : filter_Veber, filter_BrenkPAINS
  - enamine_api: EnamineClient, add_enamine_prices
  - pipeline  : CheckpointManager, load_or_run, load_or_filter, paths
"""

from .reactions import (
    rxn_ErlenmeyerPlochl,
    rxn_AminolysisGFPc,
    rxn_SulphurExchange,
)

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

__version__ = "4.10"
__author__ = "Dario M Lorente"

__all__ = [
    # Reactions
    "rxn_ErlenmeyerPlochl",
    "rxn_AminolysisGFPc",
    "rxn_SulphurExchange",
    # I/O
    "sdf_to_dataframe",
    "report_df_size",
    "save_dataframe_as_csv",
    "add_rdkit_properties",
    # Filters
    "filter_Veber",
    "filter_BrenkPAINS",
    # Enamine API
    "EnamineClient",
    "add_enamine_prices",
    # Pipeline
    "CheckpointManager",
    "stage_path",
    "checkpoint_path",
    "rejected_path",
    "init_stage_dirs",
    "load_or_run",
    "load_or_filter",
    "save_dataframe",
]
