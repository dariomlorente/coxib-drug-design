"""
Cheminformatics utilities for virtual screening and compound generation.
Modules: chemistry, dataframes, enamine_api, resources.
"""

from .chemistry import (
    rxn_SchottenBaumann,
    rxn_ErlenmeyerPlochl,
    rxn_DivergenceSe34,
    rxn_DivergenceSe34_chunked,
    deduplicate_chunks,
    rxn_AminolysisGFPc,
    rxn_SulphurExchange,
    _rxn_DivergenceSe34_single,
    _process_azlactone_batch,
    _process_ep_batch,
)

from .dataframes import (
    sdf_to_dataframe,
    read_smiles_csv,
    report_df_size,
    replace_price_3g_eur,
    save_dataframe_as_csv,
    save_dataframe_as_sdf,
    save_dataframe_as_smi,
    add_rdkit_properties,
    filter_Veber,
    filter_by_properties,
    filter_brenk,
    add_dicarboxylic_flag,
    digest,
    cleanup_generated_files,
    _process_digest_batch,
)

from .enamine_api import (
    EnamineClient,
    add_enamine_prices,
)

__version__ = "3.7"  # Month.Day - Phase 2b pipeline: filter_brenk, rxn_AminolysisGFPc, rxn_SulphurExchange, EnamineClient
__author__ = "Dario M Lorente"

__all__ = [
    # Chemistry
    "rxn_SchottenBaumann",
    "rxn_ErlenmeyerPlochl",
    "rxn_DivergenceSe34",
    "rxn_DivergenceSe34_chunked",
    "deduplicate_chunks",
    "rxn_AminolysisGFPc",
    "rxn_SulphurExchange",
    # DataFrames
    "sdf_to_dataframe",
    "read_smiles_csv",
    "report_df_size",
    "replace_price_3g_eur",
    "save_dataframe_as_csv",
    "save_dataframe_as_sdf",
    "save_dataframe_as_smi",
    "add_rdkit_properties",
    "filter_Veber",
    "filter_by_properties",
    "filter_brenk",
    "add_dicarboxylic_flag",
    "digest",
    "cleanup_generated_files",
    # Enamine API
    "EnamineClient",
    "add_enamine_prices",
]
