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





def save_dataframe_as_csv(df: pd.DataFrame, output_base_path: str) -> None:
    """Save a DataFrame to CSV using format: <path>_<N>cmpds.csv"""
    n_rows = len(df)
    directory = os.path.dirname(output_base_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    output_path = f"{output_base_path}_{n_rows}cmpds.csv"
    df.to_csv(output_path, index=False)
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
            }
            results.append((idx, desc))
        except Exception:
            pass
    return results


def add_rdkit_properties(
    df: pd.DataFrame,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Add RDKit descriptors (MW, HBD, HBA, RotB, HvyAtm, Rings, CAtm, HetAtm, MR) to DataFrame.
    
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
    
    for col in ["MW", "HBD", "HBA", "RotB", "HvyAtm", "Rings", "HetAtm", "MR", "CAtm"]:
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter DataFrame by Veber bioavailability criteria.

    Only calculates and checks each property when the corresponding limit is
    explicitly provided. Comparisons are inclusive (≤ / ≥) for all criteria
    except min_CAtm and min_HetAtm, which are exclusive (strictly >).
    LogP is calculated using Crippen's contribution-based method (Descriptors.MolLogP).

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

    Returns:
        Tuple of (accepted, rejected) DataFrames.
    """
    df_work = df.copy()

    # --- On-demand property calculation (only when the filter is active) ---
    need_tpsa = max_tPSA is not None or min_tPSA is not None
    need_logp = max_LogP is not None or min_LogP is not None
    need_mr = min_MR is not None or max_MR is not None
    need_catm = min_CAtm is not None

    if need_tpsa or need_logp or need_mr or need_catm:
        mols = df_work[smiles_col].astype(str).map(Chem.MolFromSmiles)
        if need_tpsa:
            df_work["tPSA"] = [
                Descriptors.TPSA(m, includeSandP=True) if m is not None else None
                for m in mols
            ]
        if need_logp:
            df_work["LogP"] = [
                Descriptors.MolLogP(m) if m is not None else None for m in mols
            ]
        if need_mr:
            df_work["MR"] = [
                Descriptors.MolMR(m) if m is not None else None for m in mols
            ]
        if need_catm:
            df_work["CAtm"] = [
                sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6)
                if m is not None else 0
                for m in mols
            ]

    # --- Filtering (all inclusive ≤/≥, except min_CAtm and min_HetAtm which are strict >) ---
    rejection_reasons = []
    for _, row in df_work.iterrows():
        reasons = []
        if max_tPSA is not None and row.get("tPSA", 0) > max_tPSA:
            reasons.append(f"tPSA={row['tPSA']:.1f}>{max_tPSA}")
        if max_RotB is not None and row.get("RotB", 0) > max_RotB:
            reasons.append(f"RotB={row['RotB']}>{max_RotB}")
        if max_LogP is not None and row.get("LogP", 0) > max_LogP:
            reasons.append(f"LogP={row['LogP']:.2f}>{max_LogP}")
        if min_tPSA is not None and row.get("tPSA", 0) < min_tPSA:
            reasons.append(f"tPSA={row['tPSA']:.1f}<{min_tPSA}")
        if max_MWT is not None and row.get("MW", 0) > max_MWT:
            reasons.append(f"MW={row['MW']:.1f}>{max_MWT}")
        if max_HBD is not None and row.get("HBD", 0) > max_HBD:
            reasons.append(f"HBD={row['HBD']}>{max_HBD}")
        if max_HBA is not None and row.get("HBA", 0) > max_HBA:
            reasons.append(f"HBA={row['HBA']}>{max_HBA}")
        if min_MR is not None and row.get("MR", 0) < min_MR:
            reasons.append(f"MR={row['MR']:.2f}<{min_MR}")
        if max_MR is not None and row.get("MR", 0) > max_MR:
            reasons.append(f"MR={row['MR']:.2f}>{max_MR}")
        if min_HvyAtm is not None and row.get("HvyAtm", 0) < min_HvyAtm:
            reasons.append(f"HvyAtm={row['HvyAtm']}<{min_HvyAtm}")
        if max_HvyAtm is not None and row.get("HvyAtm", 0) > max_HvyAtm:
            reasons.append(f"HvyAtm={row['HvyAtm']}>{max_HvyAtm}")
        if min_LogP is not None and row.get("LogP", 0) < min_LogP:
            reasons.append(f"LogP={row['LogP']:.2f}<{min_LogP}")
        if max_Rings is not None and row.get("Rings", 0) > max_Rings:
            reasons.append(f"Rings={row['Rings']}>{max_Rings}")
        if min_CAtm is not None and row.get("CAtm", 0) <= min_CAtm:  # exclusive
            reasons.append(f"CAtm={row['CAtm']}<={min_CAtm}")
        if min_HetAtm is not None and row.get("HetAtm", 0) <= min_HetAtm:  # exclusive
            reasons.append(f"HetAtm={row['HetAtm']}<={min_HetAtm}")
        rejection_reasons.append("; ".join(reasons) if reasons else "")

    df_work["Rejection_Reasons"] = rejection_reasons
    mask_rejected = df_work["Rejection_Reasons"] != ""
    df_rejected = df_work[mask_rejected].copy().reset_index(drop=True)
    df_accepted = df_work[~mask_rejected].copy().reset_index(drop=True)

    if "Rejection_Reasons" in df_accepted.columns:
        df_accepted = df_accepted.drop(columns=["Rejection_Reasons"])

    if print_report:
        n_total = len(df)
        n_accepted = len(df_accepted)
        n_rejected = len(df_rejected)
        pct = (n_accepted / n_total * 100) if n_total > 0 else 0
        print(f"[filter_Veber] {n_accepted}/{n_total} accepted ({pct:.1f}%), {n_rejected} rejected")

    return df_accepted, df_rejected




    df_work = df.copy()
    new_cache_entries: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Cache check (serial — fast dict lookups)
    # ------------------------------------------------------------------
    cached_results: dict[int, dict] = {}  # df_work positional idx -> result
    uncached_items: list[tuple[int, str, str]] = []  # (pos_idx, smiles, cache_key)

    smiles_series = df_work[smiles_col].astype(str)
    for pos_idx, smi in enumerate(smiles_series):
        cache_key = _get_cache_key(smi, max_halflife)
        if use_cache and cache_key in cache:
            cached_results[pos_idx] = cache[cache_key]
            cache_hits += 1
        else:
            uncached_items.append((pos_idx, smi, cache_key))
            cache_misses += 1

    if print_report:
        print(f"[Digest] {len(df_work):,} rows: {cache_hits:,} cache hits, "
              f"{cache_misses:,} misses")

    # ------------------------------------------------------------------
    # Parallel computation of cache misses
    # ------------------------------------------------------------------
    if uncached_items:
        n_workers = _get_n_workers(n_workers)

        if len(uncached_items) >= 1000 and n_workers > 1:
            batch_size = max(1, len(uncached_items) // (n_workers * 10))
            batches = [
                uncached_items[i:i + batch_size]
                for i in range(0, len(uncached_items), batch_size)
            ]

            if print_report:
                print(f"[Digest] Processing {len(uncached_items):,} misses "
                      f"with {n_workers} workers ({len(batches)} batches)...")

            worker_fn = partial(
                _process_digest_batch,
                unstable_groups=unstable_groups,
                max_halflife=max_halflife,
            )

            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(worker_fn, batches)

            for batch_results, batch_cache in results:
                cached_results.update(batch_results)
                new_cache_entries.update(batch_cache)
        else:
            # Small number of misses — process in main thread
            if print_report and uncached_items:
                print(f"[Digest] Processing {len(uncached_items):,} misses "
                      f"in main thread (< 1000)...")
            batch_results, batch_cache = _process_digest_batch(
                uncached_items,
                unstable_groups=unstable_groups,
                max_halflife=max_halflife,
            )
            cached_results.update(batch_results)
            new_cache_entries.update(batch_cache)

    # ------------------------------------------------------------------
    # Save updated cache
    # ------------------------------------------------------------------
    if use_cache and new_cache_entries:
        cache.update(new_cache_entries)
        _save_cache(cache_file, cache)

    # ------------------------------------------------------------------
    # Apply results to DataFrame
    # ------------------------------------------------------------------
    unstable_groups_col = []
    min_halflife_col = []
    for pos_idx in range(len(df_work)):
        result = cached_results.get(pos_idx, {"Unstable_Groups": "", "Min_HalfLife_Secs": None})
        unstable_groups_col.append(result["Unstable_Groups"])
        min_halflife_col.append(result["Min_HalfLife_Secs"])

    df_work["Unstable_Groups"] = unstable_groups_col
    df_work["Min_HalfLife_Secs"] = min_halflife_col

    mask_unstable = df_work["Unstable_Groups"] != ""
    df_unstable = df_work[mask_unstable].copy().reset_index(drop=True)
    df_stable = df_work[~mask_unstable].copy().reset_index(drop=True)

    if "Unstable_Groups" in df_stable.columns:
        df_stable = df_stable.drop(columns=["Unstable_Groups", "Min_HalfLife_Secs"])

    if print_report:
        n_total = len(df)
        n_stable = len(df_stable)
        n_unstable = len(df_unstable)
        pct = (n_stable / n_total * 100) if n_total > 0 else 0

        if use_cache:
            cache_pct = (
                (cache_hits / (cache_hits + cache_misses) * 100)
                if (cache_hits + cache_misses) > 0 else 0
            )
            print(f"[Digest] Cache: {cache_hits} hits, "
                  f"{cache_misses} misses ({cache_pct:.1f}% hit rate)")

        print(f"[Digest] {n_stable}/{n_total} stable ({pct:.1f}%), "
              f"{n_unstable} unstable")

        if n_unstable > 0:
            print("\nUnstable groups found:")
            all_groups: list[str] = []
            for groups_str in df_unstable["Unstable_Groups"]:
                all_groups.extend(groups_str.split("; "))
            group_counts = Counter(all_groups)
            for group, count in group_counts.most_common(10):
                print(f"  - {group}: {count} compounds")

    return df_stable, df_unstable


def _process_brenk_pains_batch(
    batch_items: list[tuple[int, str, bool]],
    n_workers: int = 1,
) -> tuple[dict[int, tuple[str, str]], dict[int, tuple[str, str]]]:
    """Worker function for parallel Brenk and PAINS filtering.
    
    batch_items: list of (index, smiles, process_pains) tuples
    """
    from py_utils._smarts_catalog import BRENK_ALERTS, PAINS_ALERTS
    
    # Compile patterns inside worker (not pickleable in main process)
    compiled_brenk: list[tuple[str, Chem.Mol | None]] = []
    for alert_name, smarts in BRENK_ALERTS.items():
        patt = Chem.MolFromSmarts(smarts)
        compiled_brenk.append((alert_name, patt))
    
    compiled_pains: list[tuple[str, str, Chem.Mol | None]] = []
    for pains_class, smarts_dict in PAINS_ALERTS.items():
        for filter_name, smarts in smarts_dict.items():
            patt = Chem.MolFromSmarts(smarts)
            compiled_pains.append((pains_class, filter_name, patt))
    
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
