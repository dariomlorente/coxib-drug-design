from __future__ import annotations

# DATAFRAMES
# DataFrame operations: I/O, filters, and property calculations

import gzip
import hashlib
import json
import multiprocessing as mp
import os
import re
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolToSmiles, SDWriter, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from py_utils._resources import _get_n_workers


# Cache directory for persistent storage
DEFAULT_CACHE_DIR = Path("mol_files/3. Oxazolones/.cache")


def _get_cache_key(*args) -> str:
    """Generate a cache key from arguments using SHA256."""
    key_str = "|".join(str(arg) for arg in args)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _load_cache(cache_file: Path) -> dict[str, Any]:
    """Load cache from gzip-compressed JSON file."""
    if not cache_file.exists():
        return {}
    
    try:
        with gzip.open(cache_file, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (gzip.BadGzipFile, json.JSONDecodeError, IOError):
        print(f"⚠️  Cache file corrupted, starting fresh: {cache_file}")
        return {}


def _save_cache(cache_file: Path, cache: dict[str, Any]) -> None:
    """Save cache to gzip-compressed JSON file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(cache_file, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(cache, f, separators=(",", ":"))
    except IOError as e:
        print(f"⚠️  Error saving cache: {e}")


def sdf_to_dataframe(sdf_path: str, id_prefix: str, id_prop: str = "Catalog_ID") -> pd.DataFrame:
    """
    Read an SDF file and return a DataFrame with ID and SMILES columns.
    
    Extracts molecules from SDF, reads the specified ID property, and generates
    SMILES strings. Removes duplicates based on Catalog_ID.
    
    Parameters:
        sdf_path: Path to the SDF file
        id_prefix: Prefix for generating IDs (e.g., 'A' for aldehydes, 'C' for carboxylic)
        id_prop: Property name in SDF containing the catalog ID (default: "Catalog_ID")
    
    Returns:
        DataFrame with columns: ID, Catalog_ID, SMILES
    """
    if not re.fullmatch(r"[A-Za-z]+", id_prefix):
        raise ValueError("id_prefix must contain only letters (e.g. 'A', 'C', 'ALD').")
    
    rows = []
    seen_ids = set()
    
    suppl = Chem.SDMolSupplier(sdf_path)
    for mol in suppl:
        if mol is None:
            continue
        
        if not mol.HasProp(id_prop):
            continue
            
        catalog_id = mol.GetProp(id_prop)
        if catalog_id in seen_ids:
            continue
        
        # Try to generate SMILES
        try:
            smi = Chem.MolToSmiles(mol)
            if smi:
                seen_ids.add(catalog_id)
                rows.append({"Catalog_ID": catalog_id, "SMILES": smi})
        except Exception:
            continue
    
    if not rows:
        raise ValueError(f"No valid molecules found in {sdf_path}")
    
    df = pd.DataFrame(rows)
    df.insert(0, "ID", [f"{id_prefix}{i+1}" for i in range(len(df))])
    
    print(f"[SDF Reader] Loaded {len(df)} unique compounds from {sdf_path}")
    return df


def report_df_size(df: pd.DataFrame, label: str = "") -> None:
    """Print the number of rows in a DataFrame."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}{len(df)} rows")





def save_dataframe_as_csv(
    df: pd.DataFrame,
    output_base_path: str,
    columns: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
) -> None:
    """Save a DataFrame to CSV using format: <path>_<N>cmpds.csv

    Parameters:
        df: Input DataFrame to save.
        output_base_path: Path prefix (without extension); row count is appended automatically.
        columns: If provided, only these columns are written (in this order).
        rename_map: If provided, rename columns before writing (applied after column selection).
    """
    out = df[columns] if columns is not None else df
    if rename_map is not None:
        out = out.rename(columns=rename_map)
    n_rows = len(out)
    directory = os.path.dirname(output_base_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    output_path = f"{output_base_path}_{n_rows}cmpds.csv"
    out.to_csv(output_path, index=False)
    print(f"Saved {n_rows} compounds to: {output_path}")


def _process_descriptors_batch(
    batch_items: list[tuple[int, str]],
) -> list[tuple[int, dict]]:
    """Worker function for parallel descriptor calculation."""
    results = []
    for idx, smi in batch_items:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            desc = {
                "MW": Descriptors.MolWt(mol),
                "HBD": Lipinski.NumHDonors(mol),
                "HBA": Lipinski.NumHAcceptors(mol),
                "RotB": Lipinski.NumRotatableBonds(mol),
                "HvyAtm": mol.GetNumHeavyAtoms(),
                "Rings": rdMolDescriptors.CalcNumRings(mol),
                "HetAtm": rdMolDescriptors.CalcNumHeteroatoms(mol),
                "MR": Descriptors.MolMR(mol),
                "CAtm": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6),
                "Atoms": mol.GetNumAtoms(),
            }
            results.append((idx, desc))
        except Exception:
            pass
    return results


def add_rdkit_properties(
    df: pd.DataFrame,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Add RDKit descriptors (MW, HBD, HBA, RotB, HvyAtm, Rings, CAtm, HetAtm, MR, Atoms) to DataFrame.
    
    Parameters:
        df: Input DataFrame with SMILES column
        n_workers: Number of parallel workers (default: cpu_count - 1)
    
    Returns:
        DataFrame with added descriptor columns
    """
    if "SMILES" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'SMILES' column.")

    out = df.copy()
    
    # Create list of (index, smiles) tuples for valid molecules
    valid_items: list[tuple[int, str]] = []
    invalid_indices: list[int] = []
    
    for idx, smi in enumerate(out["SMILES"].astype(str)):
        if smi and smi != "nan":
            valid_items.append((idx, smi))
        else:
            invalid_indices.append(idx)
    
    if invalid_indices:
        removed = out.loc[invalid_indices, ["ID", "SMILES"]]
        print(f"⚠️  Removed {len(removed)} invalid SMILES rows")
    
    out = out.loc[~out.index.isin(invalid_indices)].reset_index(drop=True)
    
    # Prepare batches for multiprocessing
    n_workers = _get_n_workers(n_workers)
    batch_size = max(1, len(valid_items) // (n_workers * 10)) if len(valid_items) >= 1000 and n_workers > 1 else len(valid_items)
    batches = [
        valid_items[i:i + batch_size]
        for i in range(0, len(valid_items), batch_size)
    ]
    
    # Process in parallel or sequentially
    if len(valid_items) >= 1000 and n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            all_results = pool.map(_process_descriptors_batch, batches)
        
        # Flatten results
        results: list[tuple[int, dict]] = []
        for batch_result in all_results:
            results.extend(batch_result)
    else:
        results = _process_descriptors_batch(valid_items)
    
    # Add descriptors to DataFrame
    results_dict = {idx: desc for idx, desc in results}
    
    for col in ["MW", "HBD", "HBA", "RotB", "HvyAtm", "Rings", "HetAtm", "MR", "CAtm", "Atoms"]:
        out[col] = [results_dict.get(idx, {}).get(col, None) for idx in range(len(out))]
    
    return out


def filter_Veber(
    df: pd.DataFrame,
    max_tPSA: float | None = 90,
    max_RotB: int | None = 10,
    max_LogP: float | None = 3,
    min_tPSA: float | None = None,
    max_MWT: float | None = None,
    max_HBD: int | None = None,
    max_HBA: int | None = None,
    min_MR: float | None = None,
    max_MR: float | None = None,
    min_HvyAtm: int | None = None,
    max_HvyAtm: int | None = None,
    min_LogP: float | None = None,
    max_Rings: int | None = None,
    min_CAtm: int | None = None,
    min_HetAtm: int | None = None,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    print_report: bool = True,
    use_cache: bool = True,
    cache_file: Path | str | None = None,
    output_csv: Path | str | None = None,
    filter_chunk_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter DataFrame by Veber bioavailability criteria.
    
    Only calculates and checks each property when the corresponding limit is
    explicitly provided. Comparisons are inclusive (≤ / ≥) for all criteria
    except min_CAtm and min_HetAtm, which are exclusive (strictly >).
    LogP is calculated using Crippen's contribution-based method (Descriptors.MolLogP).

    For large datasets (>1M rows), use output_csv with filter_chunk_size to stream
    results to disk and avoid memory exhaustion.

    Parameters:
        df: Input DataFrame with SMILES column and RDKit properties already added.
        max_tPSA: Topological PSA upper limit (Å², includes S and P).
        max_RotB: Rotatable bonds upper limit.
        max_LogP: Crippen LogP upper limit.
        min_tPSA: Topological PSA lower limit (Å²).
        max_MWT: Molecular weight upper limit (Da).
        max_HBD: H-bond donors upper limit.
        max_HBA: H-bond acceptors upper limit.
        min_MR: Molecular refractivity lower limit.
        max_MR: Molecular refractivity upper limit.
        min_HvyAtm: Heavy atom count lower limit (inclusive).
        max_HvyAtm: Heavy atom count upper limit (inclusive).
        min_LogP: Crippen LogP lower limit.
        max_Rings: Ring count upper limit.
        min_CAtm: Carbon atom count lower limit (exclusive, strictly >).
        min_HetAtm: Heteroatom count lower limit (exclusive, strictly >).
        smiles_col: Column name for SMILES strings.
        id_col: Column name for compound IDs.
        print_report: Print acceptance/rejection summary.
        use_cache: Use persistent cache for calculated properties.
        cache_file: Cache file path (auto-generated from DataFrame and filters if None).
        output_csv: If provided, stream accepted rows to this CSV file instead of
            holding them in memory (recommended for >1M rows).
        filter_chunk_size: Chunk size for filtering when output_csv is used.
            If None, defaults to 100000. Set smaller for very large datasets.

    Returns:
        Tuple of (accepted, rejected) DataFrames. When output_csv is used,
        accepted DataFrame is empty (data is on disk).
    """
    df_work = df.copy()
    
    # --- Load cache if caching is enabled ---
    cache: dict[str, dict] = {}
    cache_hits = 0
    cache_misses = 0
    
    if use_cache:
        if cache_file is None:
            cache_file = Path("mol_files/3. Oxazolones/.cache/veber_filter_cache.json.gz")
        else:
            cache_file = Path(cache_file)
        
        cache = _load_cache(cache_file)
    else:
        cache_file = Path("/dev/null")  # Dummy value when not using cache

    # --- On-demand property calculation (only when the filter is active) ---
    # Use chunked processing to avoid OOM with large datasets (e.g., millions of oxazolones)
    need_tpsa = max_tPSA is not None or min_tPSA is not None
    need_logp = max_LogP is not None or min_LogP is not None
    need_mr = min_MR is not None or max_MR is not None
    need_catm = min_CAtm is not None

    if need_tpsa or need_logp or need_mr or need_catm:
        chunk_size = 10000
        n_rows = len(df_work)
        n_chunks = (n_rows + chunk_size - 1) // chunk_size
        
        if print_report:
            print(f"[Veber] Processing {n_rows:,} molecules in {n_chunks} chunks...")
        
        # Initialize columns
        if need_tpsa:
            df_work["tPSA"] = None
        if need_logp:
            df_work["LogP"] = None
        if need_mr:
            df_work["MR"] = None
        if need_catm:
            df_work["CAtm"] = 0
        
        # Process in chunks to avoid creating millions of mol objects at once
        for chunk_idx, chunk_start in enumerate(range(0, n_rows, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, n_rows)
            
            if print_report and (chunk_idx + 1) % 100 == 0:
                print(f"[Veber] Processing chunk {chunk_idx + 1}/{n_chunks}...")
            
            chunk_rows = df_work.iloc[chunk_start:chunk_end]
            chunk_smiles = chunk_rows[smiles_col].astype(str)
            chunk_mols = chunk_smiles.map(Chem.MolFromSmiles)
            
            # Calculate properties
            if need_tpsa:
                tpsa_values = []
                for i, (smi, m) in enumerate(zip(chunk_smiles, chunk_mols)):
                    cached_val = None
                    if use_cache and smi in cache and "tPSA" in cache[smi]:
                        cached_val = cache[smi]["tPSA"]
                    
                    if cached_val is not None:
                        tpsa_values.append(cached_val)
                        cache_hits += 1
                    elif m is not None:
                        val = Descriptors.TPSA(m, includeSandP=True)
                        tpsa_values.append(val)
                        if use_cache:
                            if smi not in cache:
                                cache[smi] = {}
                            cache[smi]["tPSA"] = val
                        cache_misses += 1
                    else:
                        tpsa_values.append(None)
                        cache_misses += 1
                df_work.loc[chunk_start:chunk_end-1, "tPSA"] = tpsa_values
            
            if need_logp:
                logp_values = []
                for i, (smi, m) in enumerate(zip(chunk_smiles, chunk_mols)):
                    cached_val = None
                    if use_cache and smi in cache and "LogP" in cache[smi]:
                        cached_val = cache[smi]["LogP"]
                    
                    if cached_val is not None:
                        logp_values.append(cached_val)
                        cache_hits += 1
                    elif m is not None:
                        val = Descriptors.MolLogP(m)
                        logp_values.append(val)
                        if use_cache:
                            if smi not in cache:
                                cache[smi] = {}
                            cache[smi]["LogP"] = val
                        cache_misses += 1
                    else:
                        logp_values.append(None)
                        cache_misses += 1
                df_work.loc[chunk_start:chunk_end-1, "LogP"] = logp_values
            
            if need_mr:
                mr_values = []
                for i, (smi, m) in enumerate(zip(chunk_smiles, chunk_mols)):
                    cached_val = None
                    if use_cache and smi in cache and "MR" in cache[smi]:
                        cached_val = cache[smi]["MR"]
                    
                    if cached_val is not None:
                        mr_values.append(cached_val)
                        cache_hits += 1
                    elif m is not None:
                        val = Descriptors.MolMR(m)
                        mr_values.append(val)
                        if use_cache:
                            if smi not in cache:
                                cache[smi] = {}
                            cache[smi]["MR"] = val
                        cache_misses += 1
                    else:
                        mr_values.append(None)
                        cache_misses += 1
                df_work.loc[chunk_start:chunk_end-1, "MR"] = mr_values
            
            if need_catm:
                catm_values = []
                for i, (smi, m) in enumerate(zip(chunk_smiles, chunk_mols)):
                    cached_val = None
                    if use_cache and smi in cache and "CAtm" in cache[smi]:
                        cached_val = cache[smi]["CAtm"]
                    
                    if cached_val is not None:
                        catm_values.append(cached_val)
                        cache_hits += 1
                    elif m is not None:
                        val = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6)
                        catm_values.append(val)
                        if use_cache:
                            if smi not in cache:
                                cache[smi] = {}
                            cache[smi]["CAtm"] = val
                        cache_misses += 1
                    else:
                        catm_values.append(0)
                        cache_misses += 1
                df_work.loc[chunk_start:chunk_end-1, "CAtm"] = catm_values
            
            # Save cache every 100 chunks (not every chunk) to reduce disk I/O
            if use_cache and cache and (chunk_idx + 1) % 100 == 0:
                _save_cache(cache_file, cache)
        
        # Save final cache after all chunks complete
        if use_cache and cache:
            _save_cache(cache_file, cache)

    # --- Filtering (all inclusive ≤/≥, except min_CAtm and min_HetAtm which are strict >) ---
    # Each criterion contributes a (bool_mask, label_series) pair.
    # Labels are computed only for the failing subset to avoid unnecessary string work.
    criteria: list[tuple[pd.Series, pd.Series]] = []

    if max_tPSA is not None:
        m = df_work["tPSA"] > max_tPSA
        criteria.append((m, df_work.loc[m, "tPSA"].map(lambda v: f"tPSA={v:.1f}>{max_tPSA}")))
    if min_tPSA is not None:
        m = df_work["tPSA"] < min_tPSA
        criteria.append((m, df_work.loc[m, "tPSA"].map(lambda v: f"tPSA={v:.1f}<{min_tPSA}")))
    if max_RotB is not None:
        m = df_work["RotB"] > max_RotB
        criteria.append((m, df_work.loc[m, "RotB"].map(lambda v: f"RotB={v}>{max_RotB}")))
    if max_LogP is not None:
        m = df_work["LogP"] > max_LogP
        criteria.append((m, df_work.loc[m, "LogP"].map(lambda v: f"LogP={v:.2f}>{max_LogP}")))
    if min_LogP is not None:
        m = df_work["LogP"] < min_LogP
        criteria.append((m, df_work.loc[m, "LogP"].map(lambda v: f"LogP={v:.2f}<{min_LogP}")))
    if max_MWT is not None:
        m = df_work["MW"] > max_MWT
        criteria.append((m, df_work.loc[m, "MW"].map(lambda v: f"MW={v:.1f}>{max_MWT}")))
    if max_HBD is not None:
        m = df_work["HBD"] > max_HBD
        criteria.append((m, df_work.loc[m, "HBD"].map(lambda v: f"HBD={v}>{max_HBD}")))
    if max_HBA is not None:
        m = df_work["HBA"] > max_HBA
        criteria.append((m, df_work.loc[m, "HBA"].map(lambda v: f"HBA={v}>{max_HBA}")))
    if min_MR is not None:
        m = df_work["MR"] < min_MR
        criteria.append((m, df_work.loc[m, "MR"].map(lambda v: f"MR={v:.2f}<{min_MR}")))
    if max_MR is not None:
        m = df_work["MR"] > max_MR
        criteria.append((m, df_work.loc[m, "MR"].map(lambda v: f"MR={v:.2f}>{max_MR}")))
    if min_HvyAtm is not None:
        m = df_work["HvyAtm"] < min_HvyAtm
        criteria.append((m, df_work.loc[m, "HvyAtm"].map(lambda v: f"HvyAtm={v}<{min_HvyAtm}")))
    if max_HvyAtm is not None:
        m = df_work["HvyAtm"] > max_HvyAtm
        criteria.append((m, df_work.loc[m, "HvyAtm"].map(lambda v: f"HvyAtm={v}>{max_HvyAtm}")))
    if max_Rings is not None:
        m = df_work["Rings"] > max_Rings
        criteria.append((m, df_work.loc[m, "Rings"].map(lambda v: f"Rings={v}>{max_Rings}")))
    if min_CAtm is not None:
        m = df_work["CAtm"] <= min_CAtm  # exclusive
        criteria.append((m, df_work.loc[m, "CAtm"].map(lambda v: f"CAtm={v}<={min_CAtm}")))
    if min_HetAtm is not None:
        m = df_work["HetAtm"] <= min_HetAtm  # exclusive
        criteria.append((m, df_work.loc[m, "HetAtm"].map(lambda v: f"HetAtm={v}<={min_HetAtm}")))

    if criteria:
        any_fail: pd.Series = criteria[0][0].copy()
        for mask, _ in criteria[1:]:
            any_fail |= mask

        rejection = pd.Series("", index=df_work.index)
        for mask, labels in criteria:
            existing = rejection[mask]
            rejection[mask] = existing.where(existing == "", existing + "; ") + labels
    else:
        any_fail = pd.Series(False, index=df_work.index)
        rejection = pd.Series("", index=df_work.index)

    df_work["Rejection"] = rejection
    
    # --- Streaming output mode for large datasets ---
    if output_csv is not None:
        if filter_chunk_size is None:
            filter_chunk_size = 100000
        
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        n_rows = len(df_work)
        n_chunks = (n_rows + filter_chunk_size - 1) // filter_chunk_size
        
        rejected_rows: list[dict] = []
        
        if print_report:
            print(f"[Veber] Streaming accepted rows to {output_path}...")
        
        for chunk_idx, chunk_start in enumerate(range(0, n_rows, filter_chunk_size)):
            chunk_end = min(chunk_start + filter_chunk_size, n_rows)
            
            if print_report and (chunk_idx + 1) % 10 == 0:
                print(f"[Veber] Processing filter chunk {chunk_idx + 1}/{n_chunks}...")
            
            chunk = df_work.iloc[chunk_start:chunk_end]
            
            # Separate accepted and rejected
            chunk_accepted = chunk[~chunk["Rejection"].astype(bool)].drop(columns=["Rejection"])
            chunk_rejected = chunk[chunk["Rejection"].astype(bool)]
            
            # Stream accepted to CSV
            if len(chunk_accepted) > 0:
                mode = "w" if chunk_idx == 0 else "a"
                header = chunk_idx == 0
                chunk_accepted.to_csv(output_path, mode=mode, index=False, header=header)
            
            # Collect rejected rows
            if len(chunk_rejected) > 0:
                rejected_rows.extend(chunk_rejected.to_dict("records"))
        
        df_rejected = pd.DataFrame(rejected_rows)
        df_accepted = pd.DataFrame()  # Empty - data is on disk
        
        # Write rejected to CSV in cache directory
        if len(df_rejected) > 0:
            rej_path = output_path.parent / f"{output_path.stem}_rejected{output_path.suffix}"
            df_rejected.to_csv(rej_path, index=False)
        
    else:
        # Original behavior - return DataFrames in memory
        df_rejected = df_work[any_fail].copy().reset_index(drop=True)
        df_accepted = df_work[~any_fail].drop(columns=["Rejection"]).reset_index(drop=True)

    if print_report:
        n_total = len(df)
        if output_csv is not None:
            n_accepted = len(df_work) - len(df_rejected)
            n_rejected = len(df_rejected)
        else:
            n_accepted = len(df_accepted)
            n_rejected = len(df_rejected)
        pct = (n_accepted / n_total * 100) if n_total > 0 else 0
        print(f"[filter_Veber] {n_accepted:,}/{n_total:,} accepted ({pct:.1f}%), {n_rejected:,} rejected")
        if use_cache and (cache_hits > 0 or cache_misses > 0):
            total_lookups = cache_hits + cache_misses
            hit_rate = (cache_hits / total_lookups * 100) if total_lookups > 0 else 0
            print(f"[filter_Veber] Cache: {cache_hits} hits, {cache_misses} misses ({hit_rate:.1f}% hit rate)")

    return df_accepted, df_rejected


# Module-level cache for compiled SMARTS patterns (thread-safe, computed once)
_COMPILED_BRENK_PATTERNS: list[tuple[str, Chem.Mol | None]] | None = None
_COMPILED_PAINS_PATTERNS: list[tuple[str, str, Chem.Mol | None]] | None = None


def _compile_brenk_pains_patterns() -> tuple[
    list[tuple[str, Chem.Mol | None]], 
    list[tuple[str, str, Chem.Mol | None]]
]:
    """Compile Brenk and PAINS SMARTS patterns once and cache at module level."""
    global _COMPILED_BRENK_PATTERNS, _COMPILED_PAINS_PATTERNS
    
    if _COMPILED_BRENK_PATTERNS is not None and _COMPILED_PAINS_PATTERNS is not None:
        return _COMPILED_BRENK_PATTERNS, _COMPILED_PAINS_PATTERNS
    
    from py_utils._smarts_catalog import BRENK_ALERTS, PAINS_ALERTS
    
    compiled_brenk: list[tuple[str, Chem.Mol | None]] = []
    for alert_name, smarts in BRENK_ALERTS.items():
        patt = Chem.MolFromSmarts(smarts)
        compiled_brenk.append((alert_name, patt))
    
    compiled_pains: list[tuple[str, str, Chem.Mol | None]] = []
    for pains_class, smarts_dict in PAINS_ALERTS.items():
        for filter_name, smarts in smarts_dict.items():
            patt = Chem.MolFromSmarts(smarts)
            compiled_pains.append((pains_class, filter_name, patt))
    
    _COMPILED_BRENK_PATTERNS = compiled_brenk
    _COMPILED_PAINS_PATTERNS = compiled_pains
    
    return compiled_brenk, compiled_pains


def _process_brenk_pains_batch(
    batch_items: list[tuple[int, str, bool]],
    n_workers: int = 1,
) -> tuple[dict[int, tuple[str, str]], dict[int, tuple[str, str]]]:
    """Worker function for parallel Brenk and PAINS filtering.
    
    batch_items: list of (index, smiles, process_pains) tuples
    """
    # Use cached compiled patterns (compiled once at module level)
    compiled_brenk, compiled_pains = _compile_brenk_pains_patterns()
    
    brenk_results: dict[int, tuple[str, str]] = {}
    pains_results: dict[int, tuple[str, str]] = {}
    
    for idx, smi, process_pains in batch_items:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            brenk_results[idx] = ("invalid_smiles", "")
            pains_results[idx] = ("invalid_smiles", "")
            continue
        
        # Brenk filtering
        matched_brenk = False
        for alert_name, patt in compiled_brenk:
            if patt and mol.HasSubstructMatch(patt):
                brenk_results[idx] = ("Brenk", alert_name)
                matched_brenk = True
                break
        if not matched_brenk:
            brenk_results[idx] = ("", "")
        
        # PAINS filtering (only if passed Brenk)
        if process_pains and brenk_results[idx][0] == "":
            matched_pains = False
            for pains_class, filter_name, patt in compiled_pains:
                if patt and mol.HasSubstructMatch(patt):
                    pains_results[idx] = (f"PAINS({pains_class})", filter_name)
                    matched_pains = True
                    break
            if not matched_pains:
                pains_results[idx] = ("", "")
        else:
            pains_results[idx] = ("", "")
    
    return brenk_results, pains_results


def filter_BrenkPAINS(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    print_report: bool = True,
    n_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out compounds containing Brenk structural alerts or PAINS substructures.

    Two-stage filter:
      1. **Brenk alerts** — functional groups associated with poor ADMET or
         mutagenicity (Brenk et al., J. Chem. Inf. Model. 2008).
      2. **PAINS** — pan-assay interference compounds (Baell & Holloway,
         J. Med. Chem. 2010). Organized by class (a, b, c).

    Rejected compounds get two extra columns in the rejected DataFrame:
      - MatchAlert: "Brenk", "PAINS(a)", "PAINS(b)", "PAINS(c)", or "invalid_smiles"
      - Substructure: the specific alert/filter name that matched

    Molecules that fail to parse are marked as "invalid_smiles".

    Parameters:
        df: Input DataFrame — must contain ``smiles_col`` and ``id_col``.
        smiles_col: Column containing SMILES strings (default: ``"SMILES"``).
        id_col: Column containing compound IDs (default: ``"ID"``).
        print_report: Print acceptance/rejection summary (default: True).
        n_workers: Number of parallel workers (default: cpu_count - 1).

    Returns:
        Tuple of (accepted, rejected) DataFrames.

    Raises:
        ValueError: If required columns are missing.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Missing column '{smiles_col}'.")
    if id_col not in df.columns:
        raise ValueError(f"Missing column '{id_col}'.")

    out_df = df.copy()
    
    # Prepare items for parallel processing
    smiles_list = out_df[smiles_col].astype(str).tolist()
    
    # Stage 1: Brenk filtering with multiprocessing
    n_workers = _get_n_workers(n_workers)
    
    # Create batch items - include flag for whether to process PAINS (will be determined later)
    items = [(i, smi, True) for i, smi in enumerate(smiles_list)]
    
    # Split into batches
    if len(items) >= 1000 and n_workers > 1:
        batch_size = max(1, len(items) // (n_workers * 10))
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_process_brenk_pains_batch, batches)
        
        # Merge results from all batches
        brenk_results: dict[int, tuple[str, str]] = {}
        pains_results: dict[int, tuple[str, str]] = {}
        
        for batch_brenk, batch_pains in results:
            brenk_results.update(batch_brenk)
            pains_results.update(batch_pains)
    else:
        # Sequential processing for small datasets
        brenk_results, pains_results = _process_brenk_pains_batch(items)
    
    # Apply results to DataFrame
    out_df["MatchAlert"] = [brenk_results.get(i, ("", ""))[0] for i in range(len(out_df))]
    out_df["Substructure"] = [brenk_results.get(i, ("", ""))[1] for i in range(len(out_df))]
    
    # Keep accepted Brenk compounds for PAINS filtering
    brenk_accepted = out_df[out_df["MatchAlert"] == ""].copy()
    brenk_rejected = out_df[out_df["MatchAlert"] != ""].copy()
    
    # Apply PAINS results to Brenk-accepted compounds
    pains_match_alerts: list[str] = []
    pains_substructures: list[str] = []
    
    for idx in brenk_accepted.index:
        pains_result = pains_results.get(idx, ("", ""))
        pains_match_alerts.append(pains_result[0])
        pains_substructures.append(pains_result[1])
    
    brenk_accepted["MatchAlert"] = pains_match_alerts
    brenk_accepted["Substructure"] = pains_substructures
    
    # Final results
    pains_rejected = brenk_accepted[brenk_accepted["MatchAlert"] != ""].copy()
    pains_accepted = brenk_accepted[brenk_accepted["MatchAlert"] == ""].copy()
    
    # Combine rejected from both stages
    df_rejected = pd.concat([brenk_rejected, pains_rejected], ignore_index=True)
    df_accepted = pains_accepted.drop(columns=["MatchAlert", "Substructure"]).reset_index(drop=True)
    df_rejected = df_rejected.reset_index(drop=True)

    if print_report:
        n_total = len(df)
        n_brenk_rejected = len(brenk_rejected)
        n_pains_rejected = len(pains_rejected)
        n_accepted = len(df_accepted)
        pct = (n_accepted / n_total * 100) if n_total > 0 else 0
        print(f"[filter_BrenkPAINS] {n_accepted}/{n_total} accepted ({pct:.1f}%)")
        print(f"[filter_BrenkPAINS] Rejected: Brenk={n_brenk_rejected}, PAINS={n_pains_rejected}")

    return df_accepted, df_rejected
