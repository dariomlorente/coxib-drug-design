"""
Cheminformatics utilities for virtual screening and compound generation.
Modules: chemistry, dataframes, enamine_api, _resources, _stage_cache, _pipeline.
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

from ._stage_cache import (
    StageCache,
    load_or_compute,
    save_stage,
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

__version__ = "3.18"  # Month.Day
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
    # Stage Cache
    "StageCache",
    "load_or_compute",
    "save_stage",
    # Pipeline
    "stage_path",
    "checkpoint_path",
    "rejected_path",
    "init_stage_dirs",
    "load_or_run",
    "load_or_filter",
    "save_dataframe",
]
