from __future__ import annotations

# Cheminformatics utilities for virtual screening and compound generation.
# Modules:
#   - reactions: rxn_ErlenmeyerPlochl, rxn_AminolysisGFPc, rxn_SulphurExchange
#   - io: sdf_to_dataframe, add_rdkit_properties, save helpers
#   - filters: filter_Veber, filter_BrenkPAINS
#   - enamine_api: EnamineClient, add_enamine_prices
#   - pipeline: CheckpointManager, load_or_run, load_or_filter, paths
#   - ultrafilter: Phase 2 QED + prioritization helpers
#   - clustering: Phase 3 ALMOS clustering helpers

from .ultrafilter import (
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
    run_clustering_input_export,
)

from .clustering import (
    DEFAULT_IGNORE_COLS,
    find_latest_clustering_input_csv,
    load_phase2_clustering_input_paths,
    validate_distinct_series_inputs,
    validate_clustering_input,
    validate_clustering_input_csv,
    build_almos_cluster_command,
    run_almos_cluster,
    load_almos_clustered_dataframe,
    select_cluster_representatives,
    select_top_n_per_cluster,
    summarize_clusters,
    save_clustering_outputs,
    cluster_with_almos,
    cluster_inputs,
    run_phase3_clustering,
)

__version__ = "4.29"
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
    "run_clustering_input_export",
    # Phase 3
    "DEFAULT_IGNORE_COLS",
    "find_latest_clustering_input_csv",
    "load_phase2_clustering_input_paths",
    "validate_distinct_series_inputs",
    "validate_clustering_input",
    "validate_clustering_input_csv",
    "build_almos_cluster_command",
    "run_almos_cluster",
    "load_almos_clustered_dataframe",
    "select_cluster_representatives",
    "select_top_n_per_cluster",
    "summarize_clusters",
    "save_clustering_outputs",
    "cluster_with_almos",
    "cluster_inputs",
    "run_phase3_clustering",
]
