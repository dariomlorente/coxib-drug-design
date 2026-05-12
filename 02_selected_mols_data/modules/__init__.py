from __future__ import annotations

# Phase 2: Hit Prioritization
# Re-exports public API from Phase 2 modules

# ── I/O and file discovery ────────────────────────────────────────────
from ._io import (
    report_df_size,
    find_latest_stage_csv,
    load_generated_product_sets,
)

# ── RDKit descriptors ─────────────────────────────────────────────────
from ._descriptors import (
    DESCRIPTOR_COLUMNS,
    ensure_required_bioavailability_columns,
    DescriptorCalculator,
)

# ── QED scoring ───────────────────────────────────────────────────────
from ._qed import (
    add_qed_column,
    load_or_compute_qed,
    QEDCalculator,
)

# ── Bioavailability filtering ─────────────────────────────────────────
from ._bioavailability import (
    filter_bioavailability,
    save_bioavailability_outputs,
    plot_qed_histograms,
    BioavailabilityFilter,
)

# ── Price controls ────────────────────────────────────────────────────
from ._price_controls import (
    apply_price_controls,
    save_price_control_outputs,
    run_clustering_input_export,
    PriceController,
)

# ── QSAR scoring primitives ───────────────────────────────────────────
from ._qsar_scoring import (
    load_chembl_ic50_summary,
    add_qsar_targets,
    make_stratification_bins,
    compute_centroid_distances,
    pic50_to_ic50_nm,
    compute_selectivity_index,
    compute_qsar_score,
)

# ── QSAR model training and winnowing ─────────────────────────────────
from ._qsar_models import (
    run_qsar_winnow,
    QSARPipeline,
)

# ── Clustering (public API) ───────────────────────────────────────────
from .clustering import (
    find_latest_clustering_input_csv,
    load_phase2_clustering_input_paths,
    validate_distinct_series_inputs,
    select_cluster_representatives,
    select_top_n_per_cluster,
    summarize_clusters,
    save_clustering_outputs,
    cluster_with_almos,
    cluster_inputs,
    run_phase3_clustering,
)

# ── Clustering (core validation and execution) ────────────────────────
from ._clustering_core import (
    DEFAULT_IGNORE_COLS,
    validate_clustering_input,
    validate_clustering_input_csv,
    build_almos_cluster_command,
    run_almos_cluster,
    load_almos_clustered_dataframe,
)

# ── ChEMBL IC50 prediction utilities ─────────────────────────────────
from .prediction import (
    DEFAULT_TARGET_IDS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_BAO_LABELS,
    DEFAULT_IC50_SUMMARY,
    DEFAULT_IC50_OUTPUT_DIR,
    DEFAULT_COXIB_SMARTS,
    extract_ic50_by_target,
    merge_ic50_summary,
    merge_ic50_into_csv,
    find_chembl_ids_by_smarts,
    IC50Analyzer,
)
