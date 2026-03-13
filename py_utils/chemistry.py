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
        return {}


def _save_cache(cache_file: Path, cache: dict[str, Any]) -> None:
    """Save cache to gzip-compressed JSON file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(cache_file, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(cache, f, separators=(",", ":"))
    except IOError:
        pass


def rxn_ErlenmeyerPlochl(
    df_aldehyde: pd.DataFrame,
    df_carboxylic: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col_aldehyde: str = "ID",
    id_col_carboxyl: str = "ID",
    price_col: str = "PriceMol",
    keep_mol: bool = False,
    print_report: bool = True,
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Erlenmeyer-Plochl reaction: combines aldehydes with carboxylic acids
    (or derivatives) and glycine to form oxazolones.

    Parameters:
        df_aldehyde: DataFrame with aldehyde compounds.
        df_carboxylic: DataFrame with carboxylic acid/derivative compounds.
        smiles_col: Column containing SMILES (default: ``"SMILES"``).
        id_col_aldehyde: ID column in aldehyde DataFrame (default: ``"ID"``).
        id_col_carboxyl: ID column in carboxylic DataFrame (default: ``"ID"``).
        price_col: Column with prices (default: ``"PriceMol"``).
        keep_mol: Keep RDKit mol objects (default: False).
        print_report: Print statistics (default: True).
        use_cache: Use persistent cache (default: True).
        cache_file: Cache file path (default: auto-generated).
        n_workers: Number of parallel workers (default: cpu_count - 1).

    Returns:
        DataFrame with oxazolone products.
    """
    # Validate inputs
    for df, name, id_col in [(df_aldehyde, "aldehyde", id_col_aldehyde),
                              (df_carboxylic, "carboxylic", id_col_carboxyl)]:
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

    # SMARTS patterns for reactants
    patt_aldehyde = Chem.MolFromSmarts("[#6:42][CX3H:41](=[O:91])")
    patt_carboxyl = Chem.MolFromSmarts("[#6:21][CX3:2](=[O:1])[OX1H0-,OX2H1,Cl,Br,I:92]")

    # Glycine reagent (single SMILES, not from DataFrame)
    smiles_gly = "O=[C:5]([OH1:6])[CH2:4][NH2:8]"
    mol_gly = Chem.MolFromSmiles(smiles_gly)
    if mol_gly is None:
        raise ValueError("Failed to build glycine reagent Mol (invalid SMILES).")

    # Reaction SMARTS: aldehyde + carboxylic acid + glycine → oxazolone + byproduct
    rxn_ep = rdChemReactions.ReactionFromSmarts(
        "([#6:42][CX3H:41](=[O:91])).([#6:21][CX3:2](=[O:1])[OX1H0-,OX2H1,Cl,Br,I:92])."
        "([O:51]=[C:5]([OH1:90])[CH2:4][NH2:3])>>"
        "([O:51]=[C:5]1[O:1][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
        "([OH:90]).([OH:91]).([O:92])"
    )

    if rxn_ep is None:
        raise ValueError("Failed to build reaction from SMARTS.")

    stats = {
        "input_aldehydes": len(df_aldehyde),
        "input_carboxylics": len(df_carboxylic),
        "invalid_aldehyde": 0,
        "invalid_carboxyl": 0,
        "not_aldehyde": 0,
        "not_carboxyl": 0,
        "no_product": 0,
        "problematic": 0,
        "output_rows": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    # Pre-process aldehydes
    valid_aldehydes: list[tuple[str, str, float]] = []  # (smi, id, price)
    for _, row in df_aldehyde.iterrows():
        smi = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["invalid_aldehyde"] += 1
        elif mol.HasSubstructMatch(patt_aldehyde):
            valid_aldehydes.append((smi, str(row[id_col_aldehyde]), float(row[price_col])))
        else:
            stats["not_aldehyde"] += 1

    # Pre-process carboxylic acids
    valid_carboxylics: list[tuple[str, str, float]] = []  # (smi, id, price)
    for _, row in df_carboxylic.iterrows():
        smi = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["invalid_carboxyl"] += 1
        elif mol.HasSubstructMatch(patt_carboxyl):
            valid_carboxylics.append((smi, str(row[id_col_carboxyl]), float(row[price_col])))
        else:
            stats["not_carboxyl"] += 1

    # Cache check
    out_rows: list[dict] = []
    work_items: list[tuple[int, int]] = []

    for i, (ald_smi, ald_id, ald_price) in enumerate(valid_aldehydes):
        for j, (carb_smi, carb_id, carb_price) in enumerate(valid_carboxylics):
            cache_key = _get_cache_key(ald_smi, carb_smi, smiles_gly)

            if use_cache and cache_key in cache:
                cached_result = cache[cache_key]
                if cached_result:
                    price_total = ald_price + carb_price
                    for prod_data in cached_result:
                        out_rows.append({
                            "ID": f"{ald_id}C{carb_id}",
                            "SMILES": prod_data["smiles"],
                            "PriceMol": price_total,
                            **{k: v for k, v in prod_data.items() if k != "smiles"},
                        })
                cache_hits += 1
            else:
                work_items.append((i, j))
                cache_misses += 1

    if print_report:
        total_pairs = len(valid_aldehydes) * len(valid_carboxylics)
        print(f"[ErlenmeyerPlochl] {total_pairs:,} total pairs: "
              f"{cache_hits:,} cache hits, {cache_misses:,} misses")

    # Parallel computation
    if work_items:
        n_workers = _get_n_workers(n_workers)
        ald_tuples = [(smi, aid, price) for smi, aid, price in valid_aldehydes]
        carb_tuples = [(smi, nid, price) for smi, nid, price in valid_carboxylics]

        if len(work_items) >= 1000 and n_workers > 1:
            batch_size = max(1, len(work_items) // (n_workers * 10))
            batches = [work_items[i:i + batch_size] for i in range(0, len(work_items), batch_size)]

            if print_report:
                print(f"[ErlenmeyerPlochl] Processing {len(work_items):,} "
                      f"misses with {n_workers} workers ({len(batches)} batches)...")

            worker_fn = partial(
                _process_ep_batch,
                ald_data=ald_tuples,
                carb_data=carb_tuples,
                smiles_gly=smiles_gly,
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
            if print_report and work_items:
                print(f"[ErlenmeyerPlochl] Processing {len(work_items):,} "
                      f"misses in main thread (< 1000)...")
            batch_rows, batch_cache, batch_stats = _process_ep_batch(
                work_items, ald_tuples, carb_tuples, smiles_gly, keep_mol
            )
            out_rows.extend(batch_rows)
            new_cache_entries.update(batch_cache)
            stats["no_product"] += batch_stats["no_product"]
            stats["problematic"] += batch_stats["problematic"]

    # Save cache
    if use_cache and new_cache_entries:
        cache.update(new_cache_entries)
        _save_cache(cache_file, cache)

    out_df = pd.DataFrame(out_rows)
    stats["output_rows"] = len(out_df)
    stats["cache_hits"] = cache_hits
    stats["cache_misses"] = cache_misses

    if len(out_df) > 0:
        out_df = out_df.sort_values(by=["SMILES", "PriceMol"], ascending=[True, True])
        out_df = out_df.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)

    if print_report:
        if use_cache:
            total_lookups = cache_hits + cache_misses
            hit_rate = (cache_hits / total_lookups * 100) if total_lookups > 0 else 0
            print(f"[ErlenmeyerPlochl] Cache: {cache_hits} hits, "
                  f"{cache_misses} misses ({hit_rate:.1f}% hit rate)")
        print(f"[ErlenmeyerPlochl] Stats: {stats}")

    return out_df


def _process_ep_batch(
    work_items: list[tuple[int, int]],
    ald_data: list[tuple[str, str, float]],
    carb_data: list[tuple[str, str, float]],
    smiles_gly: str,
    keep_mol: bool = False,
) -> tuple[list[dict], dict[str, list], dict[str, int]]:
    """Worker function for Erlenmeyer-Plochl parallel computation."""
    rxn_ep = rdChemReactions.ReactionFromSmarts(
        "([#6:42][CX3H:41](=[O:91])).([#6:21][CX3:2](=[O:1])[OX1H0-,OX2H1,Cl,Br,I:92])."
        "([O:51]=[C:5]([OH1:90])[CH2:4][NH2:3])>>"
        "([O:51]=[C:5]1[O:1][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
        "([OH:90]).([OH:91]).([O:92])"
    )
    mol_gly = Chem.MolFromSmiles(smiles_gly)

    out_rows: list[dict] = []
    new_cache: dict[str, list] = {}
    stats = {"no_product": 0, "problematic": 0}

    for ald_idx, carb_idx in work_items:
        ald_smi, ald_id, ald_price = ald_data[ald_idx]
        carb_smi, carb_id, carb_price = carb_data[carb_idx]
        cache_key = _get_cache_key(ald_smi, carb_smi, smiles_gly)

        ald_mol = Chem.MolFromSmiles(ald_smi)
        carb_mol = Chem.MolFromSmiles(carb_smi)

        if ald_mol is None or carb_mol is None or mol_gly is None:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        try:
            Chem.SanitizeMol(ald_mol)
            Chem.SanitizeMol(carb_mol)
            Chem.SanitizeMol(mol_gly)
            products = rxn_ep.RunReactants((ald_mol, carb_mol, mol_gly))
        except Exception:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        if not products:
            new_cache[cache_key] = []
            stats["no_product"] += 1
            continue

        price_total = ald_price + carb_price
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
                    "ID": f"{ald_id}C{carb_id}",
                    "SMILES": psmi,
                    "PriceMol": price_total,
                    **{k: v for k, v in prod_data.items() if k != "smiles"},
                }
                if keep_mol:
                    new_row["Mol"] = prod_mol
                out_rows.append(new_row)
            except Exception:
                stats["problematic"] += 1

        new_cache[cache_key] = product_data_list

    return out_rows, new_cache, stats


def rxn_AminolysisGFPc(
    df_oxazolones: pd.DataFrame,
    df_amines: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col_oxazolone: str = "ID",
    id_col_amine: str = "ID",
    price_col: str = "PriceMol",
    keep_mol: bool = False,
    print_report: bool = True,
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Aminolysis-GFPc reaction: opens oxazolone rings with amines to form imidazolones.

    Parameters:
        df_oxazolones: DataFrame with oxazolone compounds.
        df_amines: DataFrame with primary amine compounds.
        smiles_col: Column containing SMILES (default: ``"SMILES"``).
        id_col_oxazolone: ID column in oxazolone DataFrame (default: ``"ID"``).
        id_col_amine: ID column in amine DataFrame (default: ``"ID"``).
        price_col: Column with prices (default: ``"PriceMol"``).
        keep_mol: Keep RDKit mol objects (default: False).
        print_report: Print statistics (default: True).
        use_cache: Use persistent cache (default: True).
        cache_file: Cache file path (default: auto-generated).
        n_workers: Number of parallel workers (default: cpu_count - 1).

    Returns:
        DataFrame with imidazolone products.
    """
    # Validate inputs
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

    # Setup cache
    if cache_file is None:
        cache_file = DEFAULT_CACHE_DIR / "aminolysis_gfpc_cache.json.gz"
    else:
        cache_file = Path(cache_file)

    cache = _load_cache(cache_file) if use_cache else {}
    cache_hits = 0
    cache_misses = 0
    new_cache_entries: dict[str, list] = {}

    # SMARTS patterns for reactants
    patt_oxazolone = Chem.MolFromSmarts("[O:51]=[C:5]1[O:1][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42]")
    patt_amine = Chem.MolFromSmarts("[NX3H2:10][*:11]")

    if patt_oxazolone is None:
        raise ValueError("Failed to build oxazolone pattern from SMARTS.")
    if patt_amine is None:
        raise ValueError("Failed to build amine pattern from SMARTS.")

    # Reaction SMARTS: oxazolone + primary amine → imidazolone + water
    rxn_ag = rdChemReactions.ReactionFromSmarts(
        "([O:51]=[C:5]1[O:1][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
        "([NX3H2:10][*:11])>>"
        "([O:51]=[C:5]1[NX3:10]([*:11])[C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
        "([OX2H2:1])"
    )

    if rxn_ag is None:
        raise ValueError("Failed to build reaction from SMARTS.")

    stats = {
        "input_oxazolones": len(df_oxazolones),
        "input_amines": len(df_amines),
        "invalid_oxazolone": 0,
        "invalid_amine": 0,
        "not_oxazolone": 0,
        "not_amine": 0,
        "no_product": 0,
        "problematic": 0,
        "output_rows": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    # Pre-process oxazolones
    valid_oxazolones: list[tuple[str, str, float]] = []  # (smi, id, price)
    for _, row in df_oxazolones.iterrows():
        smi = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["invalid_oxazolone"] += 1
        elif mol.HasSubstructMatch(patt_oxazolone):
            valid_oxazolones.append((smi, str(row[id_col_oxazolone]), float(row[price_col])))
        else:
            stats["not_oxazolone"] += 1

    # Pre-process amines
    valid_amines: list[tuple[str, str, float]] = []  # (smi, id, price)
    for _, row in df_amines.iterrows():
        smi = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["invalid_amine"] += 1
        elif mol.HasSubstructMatch(patt_amine):
            valid_amines.append((smi, str(row[id_col_amine]), float(row[price_col])))
        else:
            stats["not_amine"] += 1

    # Cache check
    out_rows: list[dict] = []
    work_items: list[tuple[int, int]] = []

    for i, (ox_smi, ox_id, ox_price) in enumerate(valid_oxazolones):
        for j, (am_smi, am_id, am_price) in enumerate(valid_amines):
            cache_key = _get_cache_key(ox_smi, am_smi)

            if use_cache and cache_key in cache:
                cached_result = cache[cache_key]
                if cached_result:
                    price_total = ox_price + am_price
                    for prod_data in cached_result:
                        out_rows.append({
                            "ID": f"{ox_id}A{am_id}",
                            "SMILES": prod_data["smiles"],
                            "PriceMol": price_total,
                            **{k: v for k, v in prod_data.items() if k != "smiles"},
                        })
                cache_hits += 1
            else:
                work_items.append((i, j))
                cache_misses += 1

    if print_report:
        total_pairs = len(valid_oxazolones) * len(valid_amines)
        print(f"[AminolysisGFPc] {total_pairs:,} total pairs: "
              f"{cache_hits:,} cache hits, {cache_misses:,} misses")

    # Parallel computation
    if work_items:
        n_workers = _get_n_workers(n_workers)
        ox_tuples = [(smi, oid, price) for smi, oid, price in valid_oxazolones]
        am_tuples = [(smi, aid, price) for smi, aid, price in valid_amines]

        if len(work_items) >= 1000 and n_workers > 1:
            batch_size = max(1, len(work_items) // (n_workers * 10))
            batches = [work_items[i:i + batch_size] for i in range(0, len(work_items), batch_size)]

            if print_report:
                print(f"[AminolysisGFPc] Processing {len(work_items):,} "
                      f"misses with {n_workers} workers ({len(batches)} batches)...")

            worker_fn = partial(
                _process_ag_batch,
                ox_data=ox_tuples,
                am_data=am_tuples,
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
            if print_report and work_items:
                print(f"[AminolysisGFPc] Processing {len(work_items):,} "
                      f"misses in main thread (< 1000)...")
            batch_rows, batch_cache, batch_stats = _process_ag_batch(
                work_items, ox_tuples, am_tuples, keep_mol
            )
            out_rows.extend(batch_rows)
            new_cache_entries.update(batch_cache)
            stats["no_product"] += batch_stats["no_product"]
            stats["problematic"] += batch_stats["problematic"]

    # Save cache
    if use_cache and new_cache_entries:
        cache.update(new_cache_entries)
        _save_cache(cache_file, cache)

    out_df = pd.DataFrame(out_rows)
    stats["output_rows"] = len(out_df)
    stats["cache_hits"] = cache_hits
    stats["cache_misses"] = cache_misses

    if len(out_df) > 0:
        out_df = out_df.sort_values(by=["SMILES", "PriceMol"], ascending=[True, True])
        out_df = out_df.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)

    if print_report:
        if use_cache:
            total_lookups = cache_hits + cache_misses
            hit_rate = (cache_hits / total_lookups * 100) if total_lookups > 0 else 0
            print(f"[AminolysisGFPc] Cache: {cache_hits} hits, "
                  f"{cache_misses} misses ({hit_rate:.1f}% hit rate)")
        print(f"[AminolysisGFPc] Stats: {stats}")

    return out_df


def _process_ag_batch(
    work_items: list[tuple[int, int]],
    ox_data: list[tuple[str, str, float]],
    am_data: list[tuple[str, str, float]],
    keep_mol: bool = False,
) -> tuple[list[dict], dict[str, list], dict[str, int]]:
    """Worker function for Aminolysis-GFPc parallel computation."""
    rxn_ag = rdChemReactions.ReactionFromSmarts(
        "([O:51]=[C:5]1[O:1][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
        "([NX3H2:10][*:11])>>"
        "([O:51]=[C:5]1[NX3:10]([*:11])[C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
        "([OX2H2:1])"
    )

    out_rows: list[dict] = []
    new_cache: dict[str, list] = {}
    stats = {"no_product": 0, "problematic": 0}

    for ox_idx, am_idx in work_items:
        ox_smi, ox_id, ox_price = ox_data[ox_idx]
        am_smi, am_id, am_price = am_data[am_idx]
        cache_key = _get_cache_key(ox_smi, am_smi)

        ox_mol = Chem.MolFromSmiles(ox_smi)
        am_mol = Chem.MolFromSmiles(am_smi)

        if ox_mol is None or am_mol is None:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        try:
            Chem.SanitizeMol(ox_mol)
            Chem.SanitizeMol(am_mol)
            products = rxn_ag.RunReactants((ox_mol, am_mol))
        except Exception:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        if not products:
            new_cache[cache_key] = []
            stats["no_product"] += 1
            continue

        price_total = ox_price + am_price
        seen: set[str] = set()
        product_data_list: list[dict] = []

        for prod_tuple in products:
            # prod_tuple contains: (imidazolone_product, water_subproduct)
            # We want the imidazolone (first element)
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
                    "ID": f"{ox_id}A{am_id}",
                    "SMILES": psmi,
                    "PriceMol": price_total,
                    **{k: v for k, v in prod_data.items() if k != "smiles"},
                }
                if keep_mol:
                    new_row["Mol"] = prod_mol
                out_rows.append(new_row)
            except Exception:
                stats["problematic"] += 1

        new_cache[cache_key] = product_data_list

    return out_rows, new_cache, stats


def rxn_SulphurExchange(
    df_oxazolones: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    price_col: str = "PriceMol",
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Sulphur-Exchange reaction: converts oxazolones to thiazolones by replacing oxygen with sulphur.

    Not yet implemented. TODO: Provide reaction SMARTS.
    """
    if smiles_col not in df_oxazolones.columns:
        raise ValueError(f"Missing '{smiles_col}' column in oxazolones DataFrame.")
    if id_col not in df_oxazolones.columns:
        raise ValueError(f"Missing '{id_col}' column in oxazolones DataFrame.")
    if price_col not in df_oxazolones.columns:
        raise ValueError(f"Missing '{price_col}' column in oxazolones DataFrame.")

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
