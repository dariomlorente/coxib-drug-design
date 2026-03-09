from __future__ import annotations

# CHEMISTRY
# Chemical reactions and transformations using SMARTS/RDKit

import gzip
import hashlib
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, rdChemReactions

from py_utils._resources import _get_n_workers, _get_ram_budget_gb


# Cache directory for persistent storage
DEFAULT_CACHE_DIR = Path("mol_files/2. Intermediates/.cache")


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
        return {}


def _save_cache(cache_file: Path, cache: dict[str, Any]) -> None:
    """Save cache to gzip-compressed JSON file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(cache_file, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(cache, f, separators=(",", ":"))
    except IOError:
        pass


def rxn_SchottenBaumann(
    df_carboxyl: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    price_col: str = "PriceMol",
    priceg_col: str = "PriceG",
    keep_mol: bool = False,
    keep_non_carboxyl: bool = False,
    keep_no_product: bool = False,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Schotten–Baumann reaction (carboxylic acids / derivatives) with glycine-like partner (SeG surrogate).
    """
    if smiles_col not in df_carboxyl.columns:
        raise ValueError(f"Missing column '{smiles_col}'.")
    if id_col not in df_carboxyl.columns:
        raise ValueError(f"Missing column '{id_col}'.")
    if price_col not in df_carboxyl.columns:
        raise ValueError(f"Missing column '{price_col}' (required for deduplicate-by-cheapest).")

    patt_carboxyl = Chem.MolFromSmarts("[CX3](=O)[OX1H0-,OX2H1,Cl,Br,I]")

    smiles_SeG = "O=[C:5]([34SeH1:6])[CH2:4][NH2:8]"
    mol_SeG = Chem.MolFromSmiles(smiles_SeG)
    if mol_SeG is None:
        raise ValueError("Failed to build SeG reagent Mol (invalid SMILES).")

    rxn = rdChemReactions.ReactionFromSmarts(
        "([CX3:7](=[O:90])[OX1H0-,OX2H1,Cl,Br,I:91]).(O=[C:5]([O,S,N,Se:6])[CH2:4][NH2:8])>>"
        "(O=[C:5]([O,S,N,Se:6])[CH2:4][NH:8][CX3:7]=[O-0:91]).[O:90]"
    )
    if rxn is None:
        raise ValueError("Failed to build reaction from SMARTS.")

    out_rows = []
    removed_invalid = []
    problematic = []

    stats = {
        "input_rows": len(df_carboxyl),
        "invalid_input": 0,
        "not_carboxyl": 0,
        "no_product": 0,
        "problematic": 0,
        "output_rows_pre_dedup": 0,
        "output_rows_post_dedup": 0,
    }

    for _, row in df_carboxyl.iterrows():
        sid = row[id_col]
        smi_in = str(row[smiles_col])

        mol_in = Chem.MolFromSmiles(smi_in)
        if mol_in is None:
            stats["invalid_input"] += 1
            removed_invalid.append((sid, smi_in))
            continue

        if not mol_in.HasSubstructMatch(patt_carboxyl):
            stats["not_carboxyl"] += 1
            if keep_non_carboxyl:
                out_rows.append(row.to_dict())
            continue

        try:
            Chem.SanitizeMol(mol_in)
            products = rxn.RunReactants((mol_in, mol_SeG))
        except Exception as e:
            stats["problematic"] += 1
            problematic.append((sid, smi_in, str(e)))
            continue

        if not products:
            stats["no_product"] += 1
            if keep_no_product:
                out_rows.append(row.to_dict())
            continue

        seen = set()
        for prod_tuple in products:
            try:
                prod_mol = prod_tuple[0]
                Chem.SanitizeMol(prod_mol)
                psmi = Chem.MolToSmiles(prod_mol)
                if psmi in seen:
                    continue
                seen.add(psmi)

                # Create new row with only essential product columns
                new_row = {
                    id_col: sid,
                    smiles_col: psmi,
                    price_col: row[price_col],
                    "MW": Descriptors.MolWt(prod_mol),
                    "HAcp": Lipinski.NumHAcceptors(prod_mol),
                    "HDon": Lipinski.NumHDonors(prod_mol),
                    "RotBnd": Lipinski.NumRotatableBonds(prod_mol),
                    "HvyAtm": prod_mol.GetNumHeavyAtoms(),
                    "Rings": rdMolDescriptors.CalcNumRings(prod_mol),
                    "HetAtm": rdMolDescriptors.CalcNumHeteroatoms(prod_mol),
                }

                if keep_mol:
                    new_row["Mol"] = prod_mol
                out_rows.append(new_row)
            except Exception as e:
                stats["problematic"] += 1
                problematic.append((sid, smi_in, f"Product error: {e}"))

    out_df = pd.DataFrame(out_rows)
    stats["output_rows_pre_dedup"] = len(out_df)

    if len(out_df) > 0:
        out_df = _drop_priceg_and_keep_cheapest_smiles(
            out_df, smiles_col=smiles_col, price_col=price_col, priceg_col=priceg_col, print_report=print_report
        )

    stats["output_rows_post_dedup"] = len(out_df)

    if print_report:
        if removed_invalid:
            print("⚠️ Removed invalid input SMILES rows:")
            for sid, smi in removed_invalid[:200]:
                print(f"  - ID={sid}, SMILES='{smi}'")
        if problematic:
            print(f"⚠️ Problematic rows: {len(problematic)}")
        print("[SchottenBaumann] Stats:", stats)

    return out_df


def rxn_ErlenmeyerPlochl(
    df_aldehyde: pd.DataFrame,
    df_nacylglycine: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col_aldehyde: str = "ID",
    id_col_nacyl: str = "ID",
    price_col: str = "PriceMol",
    keep_mol: bool = False,
    print_report: bool = True,
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Erlenmeyer-Plochl reaction: combines aldehydes with N-acylglycines
    to form azlactones.

    Now with multiprocessing: the main thread checks the persistent cache
    (serial, fast) and only sends cache misses to parallel workers.
    Workers never receive the cache dict, keeping memory per-worker low.

    Parameters:
        df_aldehyde: DataFrame with aldehyde compounds.
        df_nacylglycine: DataFrame with N-acylglycine compounds.
        smiles_col: Column containing SMILES (default: ``"SMILES"``).
        id_col_aldehyde: ID column in aldehyde DataFrame (default: ``"ID"``).
        id_col_nacyl: ID column in nacylglycine DataFrame (default: ``"ID"``).
        price_col: Column with prices (default: ``"PriceMol"``).
        keep_mol: Keep RDKit mol objects (default: False).
        print_report: Print statistics (default: True).
        use_cache: Use persistent cache (default: True).
        cache_file: Cache file path (default: auto-generated).
        n_workers: Number of parallel workers (default: cpu_count - 1).

    Returns:
        DataFrame with azlactone products.
    """
    for df, name, id_col in [(df_aldehyde, "aldehyde", id_col_aldehyde),
                              (df_nacylglycine, "nacylglycine", id_col_nacyl)]:
        if smiles_col not in df.columns:
            raise ValueError(f"Missing '{smiles_col}' column in {name} DataFrame.")
        if id_col not in df.columns:
            raise ValueError(f"Missing '{id_col}' column in {name} DataFrame.")
        if price_col not in df.columns:
            raise ValueError(f"Missing '{price_col}' column in {name} DataFrame.")

    # Setup cache
    if cache_file is None:
        cache_file = DEFAULT_CACHE_DIR / "erlenmeyer_plochl_cache.json.gz"
    else:
        cache_file = Path(cache_file)

    cache = _load_cache(cache_file) if use_cache else {}
    cache_hits = 0
    cache_misses = 0
    new_cache_entries: dict[str, list] = {}

    patt_aldehyde = Chem.MolFromSmarts("[CX3H](=O)")

    stats = {
        "input_aldehydes": len(df_aldehyde),
        "input_nacylglycines": len(df_nacylglycine),
        "not_aldehyde": 0,
        "no_product": 0,
        "problematic": 0,
        "output_rows": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    # ------------------------------------------------------------------
    # Pre-process aldehydes (serial — only ~1665 rows)
    # ------------------------------------------------------------------
    valid_aldehydes: list[tuple[str, str, float]] = []  # (smi, id, price)
    for _, row in df_aldehyde.iterrows():
        smi = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.HasSubstructMatch(patt_aldehyde):
            valid_aldehydes.append((smi, str(row[id_col_aldehyde]), float(row[price_col])))
        else:
            stats["not_aldehyde"] += 1

    # Pre-process nacylglycines (serial — only ~4920 rows)
    nacylglycine_data: list[tuple[str, str, float]] = []  # (smi, id, price)
    for _, row in df_nacylglycine.iterrows():
        smi = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            nacylglycine_data.append((smi, str(row[id_col_nacyl]), float(row[price_col])))

    # ------------------------------------------------------------------
    # Cache check (serial — dict lookups, ~30 sec for 8M pairs)
    # ------------------------------------------------------------------
    out_rows: list[dict] = []
    work_items: list[tuple[int, int]] = []  # (ald_idx, nacyl_idx) needing computation

    for i, (ald_smi, ald_id, ald_price) in enumerate(valid_aldehydes):
        for j, (nacyl_smi, nacyl_id, nacyl_price) in enumerate(nacylglycine_data):
            cache_key = _get_cache_key(ald_smi, nacyl_smi)

            if use_cache and cache_key in cache:
                cached_result = cache[cache_key]
                if cached_result:
                    price_total = ald_price + nacyl_price
                    for prod_data in cached_result:
                        out_rows.append({
                            "ID": f"{ald_id}{nacyl_id}",
                            "SMILES": prod_data["smiles"],
                            "PriceMol_Total": price_total,
                            **{k: v for k, v in prod_data.items() if k != "smiles"},
                        })
                cache_hits += 1
            else:
                work_items.append((i, j))
                cache_misses += 1

    if print_report:
        total_pairs = len(valid_aldehydes) * len(nacylglycine_data)
        print(f"[ErlenmeyerPlochl] {total_pairs:,} total pairs: "
              f"{cache_hits:,} cache hits, {cache_misses:,} misses")

    # ------------------------------------------------------------------
    # Parallel computation of cache misses
    # ------------------------------------------------------------------
    if work_items:
        n_workers = _get_n_workers(n_workers)

        # Pack lightweight data for workers (no Mol objects, no cache)
        ald_tuples = [(smi, aid, price) for smi, aid, price in valid_aldehydes]
        nacyl_tuples = [(smi, nid, price) for smi, nid, price in nacylglycine_data]

        if len(work_items) >= 1000 and n_workers > 1:
            # Split work_items into batches for good load balancing
            batch_size = max(1, len(work_items) // (n_workers * 10))
            batches = [
                work_items[i:i + batch_size]
                for i in range(0, len(work_items), batch_size)
            ]

            if print_report:
                print(f"[ErlenmeyerPlochl] Processing {len(work_items):,} "
                      f"misses with {n_workers} workers "
                      f"({len(batches)} batches)...")

            worker_fn = partial(
                _process_ep_batch,
                ald_data=ald_tuples,
                nacyl_data=nacyl_tuples,
                keep_mol=keep_mol,
            )

            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(worker_fn, batches)

            for batch_rows, batch_cache, batch_stats in results:
                out_rows.extend(batch_rows)
                new_cache_entries.update(batch_cache)
                stats["no_product"] += batch_stats["no_product"]
                stats["problematic"] += batch_stats["problematic"]
        else:
            # Small number of misses — process in main thread
            if print_report and work_items:
                print(f"[ErlenmeyerPlochl] Processing {len(work_items):,} "
                      f"misses in main thread (< 1000)...")
            batch_rows, batch_cache, batch_stats = _process_ep_batch(
                work_items,
                ald_data=ald_tuples,
                nacyl_data=nacyl_tuples,
                keep_mol=keep_mol,
            )
            out_rows.extend(batch_rows)
            new_cache_entries.update(batch_cache)
            stats["no_product"] += batch_stats["no_product"]
            stats["problematic"] += batch_stats["problematic"]

    # ------------------------------------------------------------------
    # Save updated cache
    # ------------------------------------------------------------------
    if use_cache and new_cache_entries:
        cache.update(new_cache_entries)
        _save_cache(cache_file, cache)

    out_df = pd.DataFrame(out_rows)
    stats["output_rows"] = len(out_df)
    stats["cache_hits"] = cache_hits
    stats["cache_misses"] = cache_misses

    if len(out_df) > 0:
        out_df = out_df.sort_values(
            by=["SMILES", "PriceMol_Total"], ascending=[True, True],
        )
        out_df = out_df.drop_duplicates(
            subset=["SMILES"], keep="first",
        ).reset_index(drop=True)

    if print_report:
        if use_cache:
            total_lookups = cache_hits + cache_misses
            hit_rate = (cache_hits / total_lookups * 100) if total_lookups > 0 else 0
            print(f"[ErlenmeyerPlochl] Cache: {cache_hits} hits, "
                  f"{cache_misses} misses ({hit_rate:.1f}% hit rate)")
        print(f"[ErlenmeyerPlochl] Stats: {stats}")

    return out_df


def rxn_DivergenceSe34(
    df_azlactone: pd.DataFrame,
    smiles_substituents: list[str],
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    price_col: str = "PriceMol_Total",
    keep_mol: bool = False,
    print_report: bool = True,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Divergence of Se-34: replaces selenium in azlactones with O, S, NH, NF, or N-R substituents.
    
    Now with multiprocessing support for faster execution.
    
    Parameters:
        df_azlactone: DataFrame with azlactone compounds
        smiles_substituents: List of SMILES for N-R substituents
        smiles_col: Column name for SMILES (default: "SMILES")
        id_col: Column name for IDs (default: "ID")
        price_col: Column name for prices (default: "PriceMol_Total")
        keep_mol: Keep RDKit mol objects in output (default: False)
        print_report: Print progress statistics (default: True)
        n_workers: Number of parallel workers (default: cpu_count - 1).
            MBP M3 → 7, iMac M4 → 9. Pass explicitly to override.
    
    Returns:
        DataFrame with divergence products
    """
    if smiles_col not in df_azlactone.columns:
        raise ValueError(f"Missing '{smiles_col}' column in azlactone DataFrame.")
    if id_col not in df_azlactone.columns:
        raise ValueError(f"Missing '{id_col}' column in azlactone DataFrame.")
    if price_col not in df_azlactone.columns:
        raise ValueError(f"Missing '{price_col}' column in azlactone DataFrame.")
    
    # Use single-threaded version for small datasets (< 1000 rows)
    if len(df_azlactone) < 1000:
        return _rxn_DivergenceSe34_single(
            df_azlactone, smiles_substituents, smiles_col, id_col, price_col, keep_mol, print_report
        )
    
    n_workers = _get_n_workers(n_workers)
    
    # Convert DataFrame to list of dicts for pickling
    rows = df_azlactone.to_dict("records")
    n_rows = len(rows)
    
    # Create batches - aim for ~10-20 batches per worker for good load balancing
    batch_size = max(1, n_rows // (n_workers * 10))
    batch_indices = [list(range(i, min(i + batch_size, n_rows))) 
                     for i in range(0, n_rows, batch_size)]
    batch_data = [(rows, indices) for indices in batch_indices]
    
    if print_report:
        print(f"[DivergenceSe34] Processing {n_rows} azlactones using {n_workers} workers...")
        print(f"[DivergenceSe34] Split into {len(batch_data)} batches (~{batch_size} azlactones/batch)")
    
    # Process batches in parallel
    out_rows = []
    stats = {"input_azlactones": n_rows, "total_products": 0, "failed_basic": 0, "failed_nr": 0}
    
    # Use functools.partial to pass fixed arguments
    worker_func = partial(
        _process_azlactone_batch,
        smiles_substituents=smiles_substituents,
        smiles_col=smiles_col,
        id_col=id_col,
        price_col=price_col,
        keep_mol=keep_mol
    )
    
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(worker_func, batch_data)
    
    # Collect results
    for batch_rows, batch_stats in results:
        out_rows.extend(batch_rows)
        stats["failed_basic"] += batch_stats["failed_basic"]
        stats["failed_nr"] += batch_stats["failed_nr"]
    
    out_df = pd.DataFrame(out_rows)
    stats["total_products"] = len(out_df)
    
    if len(out_df) > 0:
        out_df = out_df.sort_values(by=["SMILES", "PriceMol_Total"], ascending=[True, True])
        out_df = out_df.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    
    if print_report:
        print(f"[DivergenceSe34] Stats: {stats}")
    
    return out_df


def _rxn_DivergenceSe34_single(
    df_azlactone: pd.DataFrame,
    smiles_substituents: list[str],
    smiles_col: str,
    id_col: str,
    price_col: str,
    keep_mol: bool,
    print_report: bool,
) -> pd.DataFrame:
    """Single-threaded version for small datasets."""
    rxn_div_o = rdChemReactions.ReactionFromSmarts("[34Se:6]>>[O:6]")
    rxn_div_s = rdChemReactions.ReactionFromSmarts("[34Se:6]>>[S:6]")
    rxn_div_nh = rdChemReactions.ReactionFromSmarts("[34Se:6]>>[NH:6]")
    rxn_div_nf = rdChemReactions.ReactionFromSmarts("[34Se:6]>>[N:6](F)")
    rxn_div_nr = rdChemReactions.ReactionFromSmarts(
        "([#6:5][34Se:93][#6:7]).([8O:94]=[N:6][#6:1])>>([#6:5][N:6]([#6:1])[#6:7]).([34Se:93]=[8O:94])"
    )

    if any(r is None for r in [rxn_div_o, rxn_div_s, rxn_div_nh, rxn_div_nf, rxn_div_nr]):
        raise ValueError("Failed to build divergence reactions from SMARTS.")

    out_rows = []
    stats = {"input_azlactones": len(df_azlactone), "total_products": 0, "failed_basic": 0, "failed_nr": 0}

    for _, azlactone_row in df_azlactone.iterrows():
        parent_id = str(azlactone_row[id_col])
        parent_smi = str(azlactone_row[smiles_col])
        parent_price = float(azlactone_row[price_col])

        azlactone_mol = Chem.MolFromSmiles(parent_smi)
        if azlactone_mol is None:
            continue

        try:
            Chem.SanitizeMol(azlactone_mol)

            for rxn_obj, div_type in [(rxn_div_o, "O"), (rxn_div_s, "S"), (rxn_div_nh, "NH"), (rxn_div_nf, "NF")]:
                products = rxn_obj.RunReactants((azlactone_mol,))
                if products and products[0]:
                    prod_mol = products[0][0]
                    Chem.SanitizeMol(prod_mol)
                    new_row = _create_divergence_row(parent_id, parent_price, prod_mol, div_type, parent_smi, keep_mol)
                    out_rows.append(new_row)

            for i, subs_smi in enumerate(smiles_substituents):
                subs_mol = Chem.MolFromSmiles(subs_smi)
                if subs_mol is None:
                    stats["failed_nr"] += 1
                    continue
                try:
                    Chem.SanitizeMol(subs_mol)
                    products = rxn_div_nr.RunReactants((azlactone_mol, subs_mol))
                    if products and products[0]:
                        prod_mol = products[0][0]
                        Chem.SanitizeMol(prod_mol)
                        div_type = f"N{i}"
                        new_row = _create_divergence_row(parent_id, parent_price, prod_mol, div_type, parent_smi, keep_mol)
                        out_rows.append(new_row)
                    else:
                        stats["failed_nr"] += 1
                except Exception:
                    stats["failed_nr"] += 1
        except Exception:
            stats["failed_basic"] += 1

    out_df = pd.DataFrame(out_rows)
    stats["total_products"] = len(out_df)

    if len(out_df) > 0:
        out_df = out_df.sort_values(by=["SMILES", "PriceMol_Total"], ascending=[True, True])
        out_df = out_df.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)

    if print_report:
        print(f"[DivergenceSe34] Stats: {stats}")

    return out_df


def rxn_DivergenceSe34_chunked(
    df_azlactone: pd.DataFrame,
    smiles_substituents: list,
    output_dir: str = "mol_files/2. Intermediates/Chunks",
    chunk_size: int = 750000,
    resume_from_chunk: int = 0,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    price_col: str = "PriceMol_Total",
    print_report: bool = True,
    n_workers: int | None = None,
    concat_result: bool = False,
) -> pd.DataFrame | list[str]:
    """
    Divergence of Se-34 with chunked processing for large datasets.

    Uses multiprocessing for faster execution. Default chunk_size of 750k
    targets ~30 minutes per chunk on MacBook Pro M3 (varies by data).

    By default returns a list of chunk file paths (memory-safe). Set
    concat_result=True to load and concatenate all chunks into a single
    DataFrame (only if the total result fits in RAM).

    Parameters:
        df_azlactone: DataFrame with azlactone compounds
        smiles_substituents: List of SMILES for N-R substituents
        output_dir: Directory to save chunk files
        chunk_size: Number of azlactones per chunk (default: 750000 for ~30min)
        resume_from_chunk: Start from this chunk index (for resuming)
        smiles_col: Column name for SMILES
        id_col: Column name for IDs
        price_col: Column name for prices
        print_report: Print progress messages
        n_workers: Number of parallel workers (default: cpu_count - 1).
            MBP M3 → 7, iMac M4 → 9. Pass explicitly to override.
        concat_result: If True, concatenate all chunks into one DataFrame
            and return it. WARNING: requires enough RAM for the full result.
            If False (default), return list of chunk file paths.

    Returns:
        list[str] of chunk file paths (default), or pd.DataFrame if
        concat_result=True.
    """
    import os
    import glob
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)

    n_azlactones = len(df_azlactone)
    n_chunks = (n_azlactones + chunk_size - 1) // chunk_size

    # Determine workers
    n_workers = _get_n_workers(n_workers)

    if print_report:
        print(f"[Chunked Divergence] Total azlactones: {n_azlactones}")
        print(f"[Chunked Divergence] Chunk size: {chunk_size}")
        print(f"[Chunked Divergence] Total chunks: {n_chunks}")
        print(f"[Chunked Divergence] Workers: {n_workers}")
        print(f"[Chunked Divergence] Estimated time: ~{n_chunks * 0.5:.1f} hours ({n_chunks * 30} minutes)")
        print(f"[Chunked Divergence] Tip: Adjust chunk_size if chunks take significantly more/less than 30min")

    existing_files = sorted(glob.glob(os.path.join(output_dir, "chunk_*_completed.csv")))
    completed_chunks = set()
    for f in existing_files:
        try:
            chunk_num = int(os.path.basename(f).split("_")[1])
            completed_chunks.add(chunk_num)
        except Exception:
            pass

    if print_report and completed_chunks:
        print(f"[Chunked Divergence] Resuming: {len(completed_chunks)} chunks already completed")

    chunk_file_paths: list[str] = []

    for chunk_idx in range(resume_from_chunk, n_chunks):
        chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:03d}_completed.csv")

        if chunk_idx in completed_chunks and os.path.exists(chunk_file):
            if print_report:
                n_rows = sum(1 for _ in open(chunk_file)) - 1  # header row
                print(f"[Chunk {chunk_idx+1}/{n_chunks}] Already on disk ({n_rows} products)")
            chunk_file_paths.append(chunk_file)
            continue

        start_time = datetime.now()
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_azlactones)
        chunk_azlactones = df_azlactone.iloc[start_idx:end_idx].copy()

        if print_report:
            print(f"\n[Chunk {chunk_idx+1}/{n_chunks}] Processing {len(chunk_azlactones)} azlactones with {n_workers} workers...")

        chunk_results = rxn_DivergenceSe34(
            chunk_azlactones, smiles_substituents,
            smiles_col=smiles_col, id_col=id_col, price_col=price_col,
            print_report=False, n_workers=n_workers,
        )

        chunk_results.to_csv(chunk_file, index=False)
        chunk_file_paths.append(chunk_file)

        if print_report:
            duration = (datetime.now() - start_time).total_seconds() / 60
            actual_chunk_size = len(chunk_azlactones)
            rate = actual_chunk_size / duration if duration > 0 else 0
            print(f"[Chunk {chunk_idx+1}/{n_chunks}] Completed: {len(chunk_results)} products in {duration:.1f}min ({rate:.0f} azlactones/min)")

            if duration > 40:
                suggested = int(chunk_size * 30 / duration)
                print(f"⚠️  Chunk took {duration:.0f}min (>30min). Consider reducing chunk_size to ~{suggested}")
            elif duration < 15:
                suggested = int(chunk_size * 30 / duration)
                print(f"⚠️  Chunk took only {duration:.0f}min (<15min). Consider increasing chunk_size to ~{suggested}")

    if print_report:
        print(f"\n[Chunked Divergence] All {len(chunk_file_paths)} chunks saved to: {output_dir}/")

    # --- Optionally concat everything in RAM ---
    if concat_result:
        if print_report:
            print("[Chunked Divergence] concat_result=True → loading all chunks into RAM...")
        all_dfs = [pd.read_csv(f) for f in chunk_file_paths]
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df.sort_values(
            by=["SMILES", "PriceMol_Total"], ascending=[True, True],
        )
        final_df = final_df.drop_duplicates(
            subset=["SMILES"], keep="first",
        ).reset_index(drop=True)
        if print_report:
            print(f"[Chunked Divergence] Total unique products: {len(final_df)}")
        return final_df

    return chunk_file_paths


def deduplicate_chunks(
    chunk_files: list[str],
    smiles_col: str = "SMILES",
    price_col: str = "PriceMol_Total",
    print_report: bool = True,
    reserve_gb: float = 4.0,
) -> list[str]:
    """
    Remove duplicate SMILES across chunk CSV files using a memory-efficient
    numpy hash-based index.

    For each duplicated SMILES (by hash), the entry with the lowest price
    is kept.  Chunks are rewritten in-place with duplicates removed.

    Algorithm (5 phases, peak RAM ~6.5 GB for 210M rows):
        1. Count rows per chunk and verify RAM budget.
        2. Build three pre-allocated numpy arrays (hash, price, chunk_id)
           by scanning each chunk for only the SMILES and price columns.
        3. ``np.lexsort`` by (price, hash) so the cheapest entry for each
           hash appears first.  Extract winner arrays.
        4. For each chunk, re-compute row hashes and use
           ``np.searchsorted`` against the sorted winner array to decide
           which rows to keep.
        5. Rewrite chunks in-place, removing loser rows.

    Hash collision risk: ~0.1 % for a single collision across 200M
    entries (one product lost out of 200M).  Negligible for research.

    Parameters:
        chunk_files: Paths to ``chunk_*_completed.csv`` files.
        smiles_col: Column name for SMILES (default: ``"SMILES"``).
        price_col: Column name for price (default: ``"PriceMol_Total"``).
        print_report: Print progress messages.
        reserve_gb: GB of RAM to reserve for OS + apps (default: 4.0).

    Returns:
        The same list of chunk file paths (some may have fewer rows).
    """
    n_chunks = len(chunk_files)
    ram_budget = _get_ram_budget_gb(reserve_gb)

    if print_report:
        print(f"[Deduplicate] RAM budget: {ram_budget:.1f} GB "
              f"(reserve: {reserve_gb:.1f} GB)")
        print(f"[Deduplicate] Scanning {n_chunks} chunks for duplicates...")

    # ------------------------------------------------------------------
    # Phase 1 — count rows per chunk (fast line count, no pandas)
    # ------------------------------------------------------------------
    chunk_sizes: list[int] = []
    for fpath in chunk_files:
        with open(fpath) as f:
            n = sum(1 for _ in f) - 1  # subtract header
        chunk_sizes.append(n)

    total_rows = sum(chunk_sizes)

    # Memory estimate: 13 bytes/row for arrays + 8 bytes/row for lexsort
    estimated_gb = total_rows * 21 / 1e9
    if estimated_gb > ram_budget * 0.85:
        raise MemoryError(
            f"[Deduplicate] Estimated peak memory ({estimated_gb:.1f} GB) "
            f"exceeds 85 % of RAM budget ({ram_budget:.1f} GB).  "
            f"Run on a machine with more RAM or reduce chunk count."
        )

    if print_report:
        print(f"[Deduplicate] Total rows: {total_rows:,}")
        print(f"[Deduplicate] Estimated peak memory: {estimated_gb:.1f} GB")

    # ------------------------------------------------------------------
    # Phase 2 — build pre-allocated numpy arrays (no concatenation peak)
    # ------------------------------------------------------------------
    hashes = np.empty(total_rows, dtype=np.int64)
    prices = np.empty(total_rows, dtype=np.float32)
    chunk_ids = np.empty(total_rows, dtype=np.uint16)

    offset = 0
    for cidx, fpath in enumerate(chunk_files):
        df_slim = pd.read_csv(fpath, usecols=[smiles_col, price_col])
        n = len(df_slim)
        hashes[offset:offset + n] = df_slim[smiles_col].map(hash).values.astype(np.int64)
        prices[offset:offset + n] = df_slim[price_col].values.astype(np.float32)
        chunk_ids[offset:offset + n] = cidx
        offset += n
        del df_slim
        if print_report:
            print(f"[Deduplicate] Scanned chunk {cidx}: {n:,} rows")

    # ------------------------------------------------------------------
    # Phase 3 — sort by (hash, price) and find winners
    # ------------------------------------------------------------------
    # lexsort sorts by last key first → primary=hashes, secondary=prices
    sort_order = np.lexsort((prices, hashes))

    # Apply sort in-place (reuse arrays to avoid double memory)
    hashes = hashes[sort_order]
    prices = prices[sort_order]
    chunk_ids = chunk_ids[sort_order]
    del sort_order

    # First occurrence of each unique hash = cheapest (sorted by price)
    unique_mask = np.empty(len(hashes), dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = hashes[1:] != hashes[:-1]

    n_unique = int(unique_mask.sum())
    n_duplicates = total_rows - n_unique

    if print_report:
        print(f"[Deduplicate] Unique hashes: {n_unique:,}")
        print(f"[Deduplicate] Duplicate rows to remove: {n_duplicates:,}")

    if n_duplicates == 0:
        if print_report:
            print("[Deduplicate] No duplicates found. Chunks unchanged.")
        del hashes, prices, chunk_ids, unique_mask
        return chunk_files

    # Extract winner arrays (sorted by hash — required for searchsorted)
    winner_hashes = hashes[unique_mask].copy()
    winner_chunks = chunk_ids[unique_mask].copy()

    # Free the big sorted arrays
    del hashes, prices, chunk_ids, unique_mask

    # ------------------------------------------------------------------
    # Phase 4+5 — rewrite each chunk, keeping only winner rows
    # ------------------------------------------------------------------
    total_kept = 0
    total_removed = 0

    for cidx, fpath in enumerate(chunk_files):
        df = pd.read_csv(fpath)
        before = len(df)

        # Recompute hashes for this chunk's rows
        row_hashes = df[smiles_col].map(hash).values.astype(np.int64)

        # Vectorized binary search into the sorted winner_hashes array
        search_idx = np.searchsorted(winner_hashes, row_hashes)

        # Clip to valid range (searchsorted can return len(winner_hashes))
        search_idx = np.clip(search_idx, 0, len(winner_hashes) - 1)

        # A row is kept iff its hash matches AND the winner chunk is this one
        mask = (
            (winner_hashes[search_idx] == row_hashes)
            & (winner_chunks[search_idx] == cidx)
        )

        df_clean = df[mask].reset_index(drop=True)
        removed = before - len(df_clean)

        if removed > 0:
            df_clean.to_csv(fpath, index=False)

        total_kept += len(df_clean)
        total_removed += removed

        if print_report:
            status = f"removed {removed:,}" if removed > 0 else "no changes"
            print(f"[Deduplicate] Chunk {cidx}: {before:,} -> {len(df_clean):,} ({status})")

        del df, df_clean, row_hashes, search_idx, mask

    del winner_hashes, winner_chunks

    if print_report:
        print(f"\n[Deduplicate] Done. Kept {total_kept:,} rows, "
              f"removed {total_removed:,} duplicates.")

    return chunk_files


def _process_ep_batch(
    work_items: list[tuple[int, int]],
    ald_data: list[tuple[str, str, float]],
    nacyl_data: list[tuple[str, str, float]],
    keep_mol: bool = False,
) -> tuple[list[dict], dict[str, list], dict[str, int]]:
    """
    Worker function for Erlenmeyer-Plochl parallel computation.

    Receives a batch of ``(ald_idx, nacyl_idx)`` pairs (cache misses only),
    computes the reaction for each pair, and returns results + new cache
    entries.  Must be at module level for multiprocessing pickling.

    Parameters:
        work_items: List of (aldehyde_index, nacylglycine_index) pairs.
        ald_data: List of (smiles, id, price) for all valid aldehydes.
        nacyl_data: List of (smiles, id, price) for all nacylglycines.
        keep_mol: Keep RDKit Mol objects in output rows.

    Returns:
        Tuple of (out_rows, new_cache_entries, stats).
    """
    # Recreate reaction (RDKit objects are not picklable)
    rxn_ep = rdChemReactions.ReactionFromSmarts(
        "([CX3H:3](=[O:92])).([34SeH:6][C:5](=O)[CH2:4][NH:8][CX3:7]=[O:91])>>"
        "([CH:3]=[C:4]1/[C:5](=O)[34Se:6][CX3:7]=[NH0:8]1).([OH:91][OH:92])"
    )

    out_rows: list[dict] = []
    new_cache: dict[str, list] = {}
    stats = {"no_product": 0, "problematic": 0}

    for ald_idx, nacyl_idx in work_items:
        ald_smi, ald_id, ald_price = ald_data[ald_idx]
        nacyl_smi, nacyl_id, nacyl_price = nacyl_data[nacyl_idx]

        cache_key = _get_cache_key(ald_smi, nacyl_smi)

        ald_mol = Chem.MolFromSmiles(ald_smi)
        nacyl_mol = Chem.MolFromSmiles(nacyl_smi)

        if ald_mol is None or nacyl_mol is None:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        try:
            Chem.SanitizeMol(ald_mol)
            Chem.SanitizeMol(nacyl_mol)
            products = rxn_ep.RunReactants((ald_mol, nacyl_mol))
        except Exception:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        if not products:
            new_cache[cache_key] = []
            stats["no_product"] += 1
            continue

        price_total = ald_price + nacyl_price
        seen: set[str] = set()
        product_data_list: list[dict] = []

        for prod_tuple in products:
            try:
                prod_mol = prod_tuple[0]
                Chem.SanitizeMol(prod_mol)
                psmi = Chem.MolToSmiles(prod_mol)
                if psmi in seen:
                    continue
                seen.add(psmi)

                prod_data = {
                    "smiles": psmi,
                    "MW": Descriptors.MolWt(prod_mol),
                    "HAcp": Lipinski.NumHAcceptors(prod_mol),
                    "HDon": Lipinski.NumHDonors(prod_mol),
                    "RotBnd": Lipinski.NumRotatableBonds(prod_mol),
                    "HvyAtm": prod_mol.GetNumHeavyAtoms(),
                    "Rings": rdMolDescriptors.CalcNumRings(prod_mol),
                    "HetAtm": rdMolDescriptors.CalcNumHeteroatoms(prod_mol),
                }
                product_data_list.append(prod_data)

                new_row: dict = {
                    "ID": f"{ald_id}{nacyl_id}",
                    "SMILES": psmi,
                    "PriceMol_Total": price_total,
                    **{k: v for k, v in prod_data.items() if k != "smiles"},
                }
                if keep_mol:
                    new_row["Mol"] = prod_mol
                out_rows.append(new_row)
            except Exception:
                stats["problematic"] += 1

        new_cache[cache_key] = product_data_list

    return out_rows, new_cache, stats


def _process_azlactone_batch(
    batch_data: tuple,
    smiles_substituents: list[str],
    smiles_col: str,
    id_col: str,
    price_col: str,
    keep_mol: bool
) -> tuple[list[dict], dict]:
    """
    Worker function to process a batch of azlactones in parallel.
    Must be at module level for multiprocessing pickling.
    """
    batch_rows, batch_indices = batch_data
    
    # Initialize reactions (RDKit objects are not picklable, recreate here)
    rxn_div_o = rdChemReactions.ReactionFromSmarts("[34Se:6]>>[O:6]")
    rxn_div_s = rdChemReactions.ReactionFromSmarts("[34Se:6]>>[S:6]")
    rxn_div_nh = rdChemReactions.ReactionFromSmarts("[34Se:6]>>[NH:6]")
    rxn_div_nf = rdChemReactions.ReactionFromSmarts("[34Se:6]>>[N:6](F)")
    rxn_div_nr = rdChemReactions.ReactionFromSmarts(
        "([#6:5][34Se:93][#6:7]).([8O:94]=[N:6][#6:1])>>([#6:5][N:6]([#6:1])[#6:7]).([34Se:93]=[8O:94])"
    )
    
    out_rows = []
    stats = {"failed_basic": 0, "failed_nr": 0}
    
    for idx in batch_indices:
        row = batch_rows[idx]
        parent_id = str(row[id_col])
        parent_smi = str(row[smiles_col])
        parent_price = float(row[price_col])
        
        azlactone_mol = Chem.MolFromSmiles(parent_smi)
        if azlactone_mol is None:
            continue
        
        try:
            Chem.SanitizeMol(azlactone_mol)
            
            # Basic divergences (O, S, NH, NF)
            for rxn_obj, div_type in [(rxn_div_o, "O"), (rxn_div_s, "S"), 
                                       (rxn_div_nh, "NH"), (rxn_div_nf, "NF")]:
                try:
                    products = rxn_obj.RunReactants((azlactone_mol,))
                    if products and products[0]:
                        prod_mol = products[0][0]
                        Chem.SanitizeMol(prod_mol)
                        new_row = _create_divergence_row(
                            parent_id, parent_price, prod_mol, div_type, parent_smi, keep_mol
                        )
                        out_rows.append(new_row)
                except Exception:
                    pass
            
            # N-R divergences
            for i, subs_smi in enumerate(smiles_substituents):
                subs_mol = Chem.MolFromSmiles(subs_smi)
                if subs_mol is None:
                    stats["failed_nr"] += 1
                    continue
                try:
                    Chem.SanitizeMol(subs_mol)
                    products = rxn_div_nr.RunReactants((azlactone_mol, subs_mol))
                    if products and products[0]:
                        prod_mol = products[0][0]
                        Chem.SanitizeMol(prod_mol)
                        div_type = f"N{i}"
                        new_row = _create_divergence_row(
                            parent_id, parent_price, prod_mol, div_type, parent_smi, keep_mol
                        )
                        out_rows.append(new_row)
                    else:
                        stats["failed_nr"] += 1
                except Exception:
                    stats["failed_nr"] += 1
                    
        except Exception:
            stats["failed_basic"] += 1
    
    return out_rows, stats


def _create_divergence_row(
    parent_id: str,
    parent_price: float,
    prod_mol: Chem.rdchem.Mol,
    divergence_type: str,
    parent_smi: str,
    keep_mol: bool = False,
) -> dict:
    """
    Build a result row dict for a single divergence product.

    Parameters:
        parent_id: Composite ID of the parent azlactone (e.g. ``"A12C40"``).
        parent_price: Total price of the parent reactants.
        prod_mol: RDKit Mol object of the product (already sanitized).
        divergence_type: Suffix indicating the variant (e.g. ``"O"``,
            ``"S"``, ``"NH"``, ``"N12"``).
        parent_smi: SMILES of the parent azlactone (unused in output,
            kept for potential debugging/logging).
        keep_mol: If True, include the Mol object in the row dict.

    Returns:
        Dict with keys ``ID``, ``SMILES``, ``PriceMol_Total``, and
        standard RDKit descriptors (``MW``, ``HAcp``, ``HDon``,
        ``RotBnd``, ``HvyAtm``, ``Rings``, ``HetAtm``).
    """
    psmi = Chem.MolToSmiles(prod_mol)
    composite_id = f"{parent_id}{divergence_type}"
    
    # Only essential product columns - ID already contains divergence_type, no need for parent details
    row = {
        "ID": composite_id,
        "SMILES": psmi,
        "PriceMol_Total": parent_price,
        "MW": Descriptors.MolWt(prod_mol),
        "HAcp": Lipinski.NumHAcceptors(prod_mol),
        "HDon": Lipinski.NumHDonors(prod_mol),
        "RotBnd": Lipinski.NumRotatableBonds(prod_mol),
        "HvyAtm": prod_mol.GetNumHeavyAtoms(),
        "Rings": rdMolDescriptors.CalcNumRings(prod_mol),
        "HetAtm": rdMolDescriptors.CalcNumHeteroatoms(prod_mol),
    }
    if keep_mol:
        row["Mol"] = prod_mol
    return row


def rxn_AminolysisGFPc(
    df_oxazolones: pd.DataFrame,
    df_amines: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col_oxazolone: str = "ID",
    id_col_amine: str = "ID",
    price_col: str = "PriceMol_Total",
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Aminolysis-GFPc reaction: opens oxazolone rings with amines to form imidazolones.

    Combines ``df_oxazolones`` (post-Veber filter) with ``df_amines`` (post-Brenk
    filter) pairwise. Products are imidazolones — the [A-G] step in the pipeline.

    Parameters:
        df_oxazolones: DataFrame with oxazolone compounds (``df_oxazolones_druglike``).
        df_amines: DataFrame with amine compounds (``df_amines_untoxic``).
        smiles_col: Column containing SMILES strings (default: ``"SMILES"``).
        id_col_oxazolone: ID column in oxazolone DataFrame (default: ``"ID"``).
        id_col_amine: ID column in amine DataFrame (default: ``"ID"``).
        price_col: Column with total reactant prices (default: ``"PriceMol_Total"``).
        use_cache: Use persistent cache (default: True).
        cache_file: Cache file path (default: auto-generated).
        print_report: Print statistics (default: True).

    Returns:
        DataFrame with imidazolone products (``df_imidazolones_raw``).

    Raises:
        ValueError: If required columns are missing.
        NotImplementedError: Always — reaction SMARTS not yet provided.
    """
    for df, name, id_col in [
        (df_oxazolones, "oxazolones", id_col_oxazolone),
        (df_amines, "amines", id_col_amine),
    ]:
        if smiles_col not in df.columns:
            raise ValueError(f"Missing '{smiles_col}' column in {name} DataFrame.")
        if id_col not in df.columns:
            raise ValueError(f"Missing '{id_col}' column in {name} DataFrame.")
        if price_col not in df.columns:
            raise ValueError(f"Missing '{price_col}' column in {name} DataFrame.")

    # TODO: reaction SMARTS — to be provided (Aminolysis-GFPc, oxazolone + amine → imidazolone)
    raise NotImplementedError(
        "rxn_AminolysisGFPc is not yet implemented. "
        "Provide the reaction SMARTS for the aminolysis step to proceed."
    )


def rxn_SulphurExchange(
    df_oxazolones: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    price_col: str = "PriceMol_Total",
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Sulphur-Exchange reaction: converts oxazolones to thiazolones by replacing oxygen with sulphur.

    Takes ``df_oxazolones_druglike`` (post-Veber, **not** ``df_oxazolones_raw``) as input.
    Product IDs receive an ``S`` suffix (e.g. ``OXA12C40`` → ``OXA12C40S``).

    Parameters:
        df_oxazolones: DataFrame with oxazolone compounds (must be ``df_oxazolones_druglike``).
        smiles_col: Column containing SMILES strings (default: ``"SMILES"``).
        id_col: Column containing compound IDs (default: ``"ID"``).
        price_col: Column with total reactant prices (default: ``"PriceMol_Total"``).
        use_cache: Use persistent cache (default: True).
        cache_file: Cache file path (default: auto-generated).
        print_report: Print statistics (default: True).

    Returns:
        DataFrame with thiazolone products (``df_thiazolones_raw``).
        Product IDs are suffixed with ``"S"``.

    Raises:
        ValueError: If required columns are missing.
        NotImplementedError: Always — reaction SMARTS not yet provided.

    Note:
        Input **must** be ``df_oxazolones_druglike`` (post-Veber, cell 6),
        not ``df_oxazolones_raw``. See Flowdiagram.md cell 8.
    """
    if smiles_col not in df_oxazolones.columns:
        raise ValueError(f"Missing '{smiles_col}' column in oxazolones DataFrame.")
    if id_col not in df_oxazolones.columns:
        raise ValueError(f"Missing '{id_col}' column in oxazolones DataFrame.")
    if price_col not in df_oxazolones.columns:
        raise ValueError(f"Missing '{price_col}' column in oxazolones DataFrame.")

    # TODO: reaction SMARTS — to be provided (Sulphur-Exchange, oxazolone O → S → thiazolone)
    raise NotImplementedError(
        "rxn_SulphurExchange is not yet implemented. "
        "Provide the reaction SMARTS for the sulphur-exchange step to proceed."
    )


def _drop_priceg_and_keep_cheapest_smiles(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    price_col: str = "PriceMol",
    priceg_col: str = "PriceG",
    print_report: bool = True,
) -> pd.DataFrame:
    """Drops PriceG column and removes duplicate SMILES keeping cheapest."""
    out = df.copy()
    
    if priceg_col in out.columns:
        out = out.drop(columns=[priceg_col])
    
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.sort_values(by=[smiles_col, price_col], ascending=[True, True], na_position="last")
    
    duplicated_mask = out.duplicated(subset=[smiles_col], keep="first")
    removed = out.loc[duplicated_mask, [smiles_col, price_col]]
    out = out.loc[~duplicated_mask].reset_index(drop=True)
    
    if print_report and len(removed) > 0:
        print(f"⚠️ Removed {len(removed)} duplicated SMILES (kept lowest PriceMol).")
    
    return out
