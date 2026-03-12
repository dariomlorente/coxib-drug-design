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


def read_smiles_csv(csv_path: str, id_prefix: str) -> pd.DataFrame:
    """
    Read a CSV file containing a 'SMILES' column and return a DataFrame.
    """
    if not re.fullmatch(r"[A-Za-z]+", id_prefix):
        raise ValueError("id_prefix must contain only letters (e.g. 'A', 'C', 'ALD').")

    df = pd.read_csv(csv_path)

    if "SMILES" not in df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column.")

    df["SMILES"] = df["SMILES"].astype(str).str.strip()
    df = df[df["SMILES"].notna() & (df["SMILES"] != "")]
    df = df.reset_index(drop=True)

    if "ID" not in df.columns:
        df.insert(0, "ID", [f"{id_prefix}{i+1}" for i in range(len(df))])

    return df


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


def replace_price_3g_eur(
    df: pd.DataFrame,
    max_price: float = 100,
    smiles_col: str = "SMILES",
    price_col: str = "PRICE_3G_EUR",
) -> pd.DataFrame:
    """
    Replace column PRICE_3G_EUR with PriceG and PriceMol.
    Also drops rows with PriceG > max_price.
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame is missing required column: {price_col}")
    if smiles_col not in df.columns:
        raise ValueError(f"DataFrame is missing required column: {smiles_col}")
    if max_price <= 0:
        raise ValueError("max_price must be > 0")

    out_df = df.copy()
    out_df[price_col] = pd.to_numeric(out_df[price_col], errors="coerce")
    out_df["PriceG"] = out_df[price_col] / 3.0

    def _mw_from_smiles(smi: Any) -> float:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            return float("nan")
        return float(Descriptors.MolWt(mol))

    mw = out_df[smiles_col].map(_mw_from_smiles)
    out_df["PriceMol"] = out_df["PriceG"] / mw
    out_df = out_df.drop(columns=[price_col])
    
    # Filter: keep only rows with valid, positive prices within limit
    # PriceG must be > 0 (not NaN, not 0, not negative) and <= max_price
    valid_prices = (
        out_df["PriceG"].notna() & 
        (out_df["PriceG"] > 0) & 
        (out_df["PriceG"] <= max_price)
    )
    removed_count = len(out_df) - valid_prices.sum()
    if removed_count > 0:
        print(f"⚠️  Removed {removed_count} compounds with invalid prices (<= 0 or > {max_price})")
    out_df = out_df[valid_prices].reset_index(drop=True)

    return out_df


def save_dataframe_as_csv(df: pd.DataFrame, output_base_path: str) -> None:
    """Save a DataFrame to CSV using format: <path>_<N>cmpds.csv"""
    n_rows = len(df)
    directory = os.path.dirname(output_base_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    output_path = f"{output_base_path}_{n_rows}cmpds.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {n_rows} compounds to: {output_path}")


def save_dataframe_as_sdf(df: pd.DataFrame, output_path: str, smiles_col: str = "SMILES", id_col: str = "ID") -> None:
    """Save a DataFrame to SDF file with 2D structures."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    writer = SDWriter(output_path)
    n_written = 0

    for _, row in df.iterrows():
        smi = str(row.get(smiles_col, ""))
        mol_id = str(row.get(id_col, f"mol_{n_written}"))
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol.SetProp("_Name", mol_id)
            for col in df.columns:
                if col not in [smiles_col, id_col, "Mol"]:
                    value = row.get(col)
                    if pd.notna(value):
                        mol.SetProp(str(col), str(value))
            writer.write(mol)
            n_written += 1

    writer.close()
    print(f"Saved {n_written} molecules to: {output_path}")


def save_dataframe_as_smi(df: pd.DataFrame, output_path: str, smiles_col: str = "SMILES") -> None:
    """Save SMILES column from DataFrame to .smi file."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    n_written = 0
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            smi = str(row.get(smiles_col, ""))
            if smi and smi != "nan":
                f.write(smi + "\n")
                n_written += 1

    print(f"Saved {n_written} SMILES to: {output_path}")


def add_rdkit_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Add RDKit descriptors (MW, HAcp, HDon, RotBnd, HvyAtm, Rings, HetAtm) to DataFrame."""
    if "SMILES" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'SMILES' column.")

    out = df.copy()
    mols = out["SMILES"].astype(str).map(Chem.MolFromSmiles)
    invalid_mask = mols.isna()

    if invalid_mask.any():
        removed = out.loc[invalid_mask, ["ID", "SMILES"]]
        print(f"⚠️  Removed {len(removed)} invalid SMILES rows")

    out = out.loc[~invalid_mask].reset_index(drop=True)
    mols = mols.loc[~invalid_mask]

    out["MW"] = [Descriptors.MolWt(m) for m in mols]
    out["HAcp"] = [Lipinski.NumHAcceptors(m) for m in mols]
    out["HDon"] = [Lipinski.NumHDonors(m) for m in mols]
    out["RotBnd"] = [Lipinski.NumRotatableBonds(m) for m in mols]
    out["HvyAtm"] = [m.GetNumHeavyAtoms() for m in mols]
    out["Rings"] = [rdMolDescriptors.CalcNumRings(m) for m in mols]
    out["HetAtm"] = [rdMolDescriptors.CalcNumHeteroatoms(m) for m in mols]

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
        if max_RotB is not None and row.get("RotBnd", 0) > max_RotB:
            reasons.append(f"RotB={row['RotBnd']}>{max_RotB}")
        if max_LogP is not None and row.get("LogP", 0) > max_LogP:
            reasons.append(f"LogP={row['LogP']:.2f}>{max_LogP}")
        if min_tPSA is not None and row.get("tPSA", 0) < min_tPSA:
            reasons.append(f"tPSA={row['tPSA']:.1f}<{min_tPSA}")
        if max_MWT is not None and row.get("MW", 0) > max_MWT:
            reasons.append(f"MW={row['MW']:.1f}>{max_MWT}")
        if max_HBD is not None and row.get("HDon", 0) > max_HBD:
            reasons.append(f"HBD={row['HDon']}>{max_HBD}")
        if max_HBA is not None and row.get("HAcp", 0) > max_HBA:
            reasons.append(f"HBA={row['HAcp']}>{max_HBA}")
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


def filter_by_properties(
    df: pd.DataFrame,
    max_MW: float | None = None,
    max_HA: int | None = None,
    max_HD: int | None = None,
    max_RotB: int | None = None,
    max_HeavyAtoms: int | None = None,
    max_Rings: int | None = None,
    max_Heteroatoms: int | None = None,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    print_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter DataFrame by molecular properties (Lipinski/Veber criteria)."""
    df_work = df.copy()
    rejection_reasons = []

    for _, row in df_work.iterrows():
        reasons = []
        if max_MW is not None and row.get("MW", 0) > max_MW:
            reasons.append(f"MW={row['MW']:.1f}>{max_MW}")
        if max_HA is not None and row.get("HAcp", 0) > max_HA:
            reasons.append(f"HA={row['HAcp']}>{max_HA}")
        if max_HD is not None and row.get("HDon", 0) > max_HD:
            reasons.append(f"HD={row['HDon']}>{max_HD}")
        if max_RotB is not None and row.get("RotBnd", 0) > max_RotB:
            reasons.append(f"RotB={row['RotBnd']}>{max_RotB}")
        if max_HeavyAtoms is not None and row.get("HvyAtm", 0) > max_HeavyAtoms:
            reasons.append(f"HvyAtm={row['HvyAtm']}>{max_HeavyAtoms}")
        if max_Rings is not None and row.get("Rings", 0) > max_Rings:
            reasons.append(f"Rings={row['Rings']}>{max_Rings}")
        if max_Heteroatoms is not None and row.get("HetAtm", 0) > max_Heteroatoms:
            reasons.append(f"HetAtm={row['HetAtm']}>{max_Heteroatoms}")
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
        print(f"[Filter] {n_accepted}/{n_total} accepted ({pct:.1f}%), {n_rejected} rejected")

    return df_accepted, df_rejected


def add_dicarboxylic_flag(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    out_col: str = "Dicarb",
) -> pd.DataFrame:
    """Add boolean column indicating whether compound has exactly two carboxyl groups."""
    smarts = "[CX3](=O)[OX1H0-,OX2H1,Cl,Br,I]"
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        raise ValueError("Invalid SMARTS pattern for carboxyl group.")

    out_df = df.copy()

    def _has_two_carboxyls(smi: str) -> bool:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            return False
        matches = mol.GetSubstructMatches(patt)
        return len(matches) == 2

    out_df[out_col] = out_df[smiles_col].map(_has_two_carboxyls)
    return out_df


def _process_digest_batch(
    items: list[tuple[int, str, str]],
    *,
    unstable_groups: list[tuple[str, int, str]],
    max_halflife: float,
) -> tuple[dict[int, dict], dict[str, dict]]:
    """
    Worker function for parallel digest processing.

    Checks each molecule for unstable functional groups by matching
    SMARTS patterns.  Designed for ``multiprocessing.Pool.map`` via
    ``functools.partial``.

    Parameters:
        items: Batch of ``(pos_idx, smiles, cache_key)`` tuples
            (only cache misses).
        unstable_groups: List of ``(group_name, halflife_secs, smarts)``
            tuples defining the labile functional groups.
        max_halflife: Maximum half-life threshold in seconds.

    Returns:
        Tuple of ``(results, new_cache)`` where *results* maps
        positional indices to result dicts and *new_cache* maps cache
        keys to the same result dicts for persistence.
    """
    # Compile SMARTS patterns inside the worker (RDKit pattern objects
    # are not pickle-safe, so they cannot be sent from the main process).
    compiled_patterns: list[tuple[str, int, Chem.rdchem.Mol | None]] = []
    for group_name, halflife, smarts in unstable_groups:
        if halflife < max_halflife:
            patt = Chem.MolFromSmarts(smarts)
            compiled_patterns.append((group_name, halflife, patt))

    results: dict[int, dict] = {}
    new_cache: dict[str, dict] = {}

    for pos_idx, smi, cache_key in items:
        mol = Chem.MolFromSmiles(smi)
        found_groups: list[str] = []
        min_halflife = float("inf")

        if mol is not None:
            try:
                for group_name, halflife, patt in compiled_patterns:
                    if patt is not None and mol.HasSubstructMatch(patt):
                        found_groups.append(f"{group_name}({halflife}s)")
                        min_halflife = min(min_halflife, halflife)
            except Exception:
                pass  # Malformed molecule — treat as stable

        if found_groups:
            result = {
                "Unstable_Groups": "; ".join(found_groups),
                "Min_HalfLife_Secs": (
                    min_halflife if min_halflife != float("inf") else None
                ),
            }
        else:
            result = {"Unstable_Groups": "", "Min_HalfLife_Secs": None}

        results[pos_idx] = result
        new_cache[cache_key] = result

    return results, new_cache


def digest(
    df: pd.DataFrame,
    max_halflife: float = 7200,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    print_report: bool = True,
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    n_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out compounds with toxic or labile functional groups.

    Groups are metabolized by glutathione, blood pH, or body homeostasis.
    Now with multiprocessing: the main thread checks the persistent cache
    (serial, fast) and only sends cache misses to parallel workers.

    Parameters:
        df: DataFrame with SMILES column.
        max_halflife: Maximum half-life threshold in seconds
            (default: 7200 = 2 hours).
        smiles_col: Column containing SMILES strings (default: ``"SMILES"``).
        id_col: Column for IDs (default: ``"ID"``).
        print_report: Print summary statistics (default: True).
        use_cache: Use persistent cache (default: True).
        cache_file: Cache file path (default: auto-generated based on
            *max_halflife*).
        n_workers: Number of parallel workers (default: cpu_count - 1).

    Returns:
        Tuple of (stable DataFrame, unstable DataFrame).
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Missing column: {smiles_col}")

    # Setup cache
    if cache_file is None:
        cache_file = DEFAULT_CACHE_DIR / f"digest_cache_{int(max_halflife)}s.json.gz"
    else:
        cache_file = Path(cache_file)

    cache = _load_cache(cache_file) if use_cache else {}
    cache_hits = 0
    cache_misses = 0

    unstable_groups = [
        ("Organic peroxide", 1, "[*:1][OX2][OX2][*:2]"),
        ("Acyl halide", 2, "[CX3:1](=[O:2])[Cl,Br,I]"),
        ("Isocyanate", 5, "[*:1][NX2:2]=[CX2]=[OX1]"),
        ("Carboxylic anhydride", 10, "[CX3:1](=[OX1:11])[OX2:3][CX3:2](=[OX1:22])"),
        ("Hemiaminal", 30, "[*:11][NX3:1]([*:22])[CX4:2]([*:21])([#6:22])[OX2H:3]"),
        ("Aldehyde", 60, "[O:1]=[CX3H:2][#6:3]"),
        ("Azo compound", 90, "[CX4:1][NX2:2]=[NX2:3][CX4:4]"),
        ("Isocyanide", 180, "[#6:1][NX2H0:2]~[CX1&H0,CX2&H1:3]"),
        ("Aliphatic azide", 300, "[#6:1][NX2]~[NX2]~[NX1]"),
        ("Epoxide", 600, "[*:12][C@@X4:1]1([*:11])[OX2:3][C@@X4:2]1([*:21])[*:22]"),
        ("Aziridine", 1200, "[*:12][C@@X4:1]1([*:11])[NX3H:3][C@@X4:2]1([*:21])[*:22]"),
        ("Hydrazine", 1500, "[*:1][NX3:2]([*:3])[NX3:4]([*:5])[*:6]"),
        ("Thiol", 1800, "[cX3,CX4:1][SX2H:2]"),
        ("Thioester", 3600, "[CX3:1](=[OX1:2])[SX2:3][cX3,CX4:4]"),
        ("Thiosulfonate", 4800, "[SX4:1](=[OX1:2])(=[OX1:22])[SX2:3][cX3,CX4:4]"),
        ("Ester", 7200, "[CX3:1](=[OX1:2])[OX2:3][#6:4]"),
    ]

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


def cleanup_generated_files(base_path: str = ".", verbose: bool = True) -> None:
    """
    Remove all generated files that are not tracked by git.
    
    Cleans CSV files, JSON cache files, Python cache directories,
    and Jupyter checkpoints from gitignored directories.
    
    Parameters:
        base_path: Base directory path (default: current directory)
        verbose: Print cleanup progress (default: True)
    """
    import shutil
    
    paths_to_clean = [
        "mol_files/1. Building Blocks",
        "mol_files/2. Intermediates",
        "mol_files/3. Products",
        "mol_files/0. EnamineSDFs/price_cache",
    ]
    
    removed_count = 0
    
    for relative_path in paths_to_clean:
        full_path = os.path.join(base_path, relative_path)
        if os.path.exists(full_path):
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if item.endswith(('.csv', '.json', '.smi', '.sdf')):
                    os.remove(item_path)
                    removed_count += 1
                    if verbose:
                        print(f"Removed: {item_path}")
    
    # Clean __pycache__ directories
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                shutil.rmtree(pycache_path)
                removed_count += 1
                if verbose:
                    print(f"Removed: {pycache_path}")
    
    # Clean .ipynb_checkpoints
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == ".ipynb_checkpoints":
                checkpoint_path = os.path.join(root, dir_name)
                shutil.rmtree(checkpoint_path)
                removed_count += 1
                if verbose:
                    print(f"Removed: {checkpoint_path}")
    
    if verbose:
        print(f"[Cleanup] Removed {removed_count} generated files/directories")


def filter_BrenkPAINS(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    print_report: bool = True,
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

    Returns:
        Tuple of (accepted, rejected) DataFrames.

    Raises:
        ValueError: If required columns are missing.
    """
    from py_utils._smarts_catalog import BRENK_ALERTS, PAINS_ALERTS

    if smiles_col not in df.columns:
        raise ValueError(f"Missing column '{smiles_col}'.")
    if id_col not in df.columns:
        raise ValueError(f"Missing column '{id_col}'.")

    out_df = df.copy()

    # First stage: Brenk filtering
    brenk_match_alerts: list[str] = []
    brenk_substructures: list[str] = []
    compiled_brenk: list[tuple[str, Chem.Mol | None]] = []
    
    for alert_name, smarts in BRENK_ALERTS.items():
        patt = Chem.MolFromSmarts(smarts)
        compiled_brenk.append((alert_name, patt))

    for smi in out_df[smiles_col].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            brenk_match_alerts.append("invalid_smiles")
            brenk_substructures.append("")
            continue
        
        matched = False
        for alert_name, patt in compiled_brenk:
            if patt and mol.HasSubstructMatch(patt):
                brenk_match_alerts.append("Brenk")
                brenk_substructures.append(alert_name)
                matched = True
                break
        if not matched:
            brenk_match_alerts.append("")
            brenk_substructures.append("")

    out_df["MatchAlert"] = brenk_match_alerts
    out_df["Substructure"] = brenk_substructures
    
    # Keep accepted Brenk compounds for PAINS filtering
    brenk_accepted = out_df[out_df["MatchAlert"] == ""].copy()
    brenk_rejected = out_df[out_df["MatchAlert"] != ""].copy()
    
    # Second stage: PAINS filtering (on Brenk-accepted compounds only)
    pains_match_alerts: list[str] = []
    pains_substructures: list[str] = []
    compiled_pains: list[tuple[str, str, Chem.Mol | None]] = []
    
    for pains_class, smarts_dict in PAINS_ALERTS.items():
        for filter_name, smarts in smarts_dict.items():
            patt = Chem.MolFromSmarts(smarts)
            compiled_pains.append((pains_class, filter_name, patt))

    for smi in brenk_accepted[smiles_col].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            pains_match_alerts.append("invalid_smiles")
            pains_substructures.append("")
            continue
        
        matched = False
        for pains_class, filter_name, patt in compiled_pains:
            if patt and mol.HasSubstructMatch(patt):
                pains_match_alerts.append(f"PAINS({pains_class})")
                pains_substructures.append(filter_name)
                matched = True
                break
        if not matched:
            pains_match_alerts.append("")
            pains_substructures.append("")
    
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
