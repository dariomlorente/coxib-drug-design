"""
Cheminformatics utilities for virtual screening and compound generation.

Modules:
  - reactions  : rxn_ErlenmeyerPlochl, rxn_AminolysisGFPc, rxn_SulphurExchange
  - io        : sdf_to_dataframe, add_rdkit_properties, save helpers
  - filters   : filter_Veber, filter_BrenkPAINS
  - enamine_api: EnamineClient, add_enamine_prices
  - pipeline  : CheckpointManager, load_or_run, load_or_filter, paths
  - phase2_hit_prioritization: Phase 2 QED + prioritization helpers
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

from .phase2_hit_prioritization import (
    find_latest_stage_csv,
    load_generated_product_sets,
    ensure_required_bioavailability_columns,
    add_qed_column,
    load_or_compute_qed,
    filter_bioavailability,
    save_bioavailability_outputs,
    apply_price_controls,
    save_price_control_outputs,
    plot_qed_histograms,
)

__version__ = "4.17"
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
    # Phase 2
    "find_latest_stage_csv",
    "load_generated_product_sets",
    "ensure_required_bioavailability_columns",
    "add_qed_column",
    "load_or_compute_qed",
    "filter_bioavailability",
    "save_bioavailability_outputs",
    "apply_price_controls",
    "save_price_control_outputs",
    "plot_qed_histograms",
]
