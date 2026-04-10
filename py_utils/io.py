from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolToSmiles, rdMolDescriptors

from ._utils import _get_n_workers


def sdf_to_dataframe(sdf_path: str, id_prefix: str, id_prop: str = "Catalog_ID") -> pd.DataFrame:
    """
    Read an SDF file and return a DataFrame with ID and SMILES columns.

    Extracts molecules from SDF, reads the specified ID property, and generates
    SMILES strings. Removes duplicates based on Catalog_ID.

    Parameters:
        sdf_path: Path to the SDF file.
        id_prefix: Prefix for generating IDs (e.g., 'A' for aldehydes, 'C' for carboxylic).
        id_prop: Property name in SDF containing the catalog ID (default: "Catalog_ID").

    Returns:
        DataFrame with columns: ID, Catalog_ID, SMILES.
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
    """
    Save a DataFrame to CSV with row count in filename.

    Parameters:
        df: Input DataFrame to save.
        output_base_path: Path prefix (without extension); row count is appended automatically.
        columns: If provided, only these columns are written (in this order).
        rename_map: If provided, rename columns before writing.
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


def _process_descriptors_batch(batch_items: list[tuple[int, str]]) -> list[tuple[int, dict]]:
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
                "tPSA": Descriptors.TPSA(mol),
                "LogP": Descriptors.MolLogP(mol),
            }
            results.append((idx, desc))
        except Exception:
            pass
    return results


def add_rdkit_properties(
    df: pd.DataFrame,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Add RDKit descriptors to DataFrame.

    Calculates MW, HBD, HBA, RotB, HvyAtm, Rings, HetAtm, MR, CAtm, Atoms, tPSA,
    LogP for each molecule. Runs in parallel when >= 1000 molecules.

    Parameters:
        df: Input DataFrame with SMILES column.
        n_workers: Number of parallel workers (default: cpu_count - 1).

    Returns:
        DataFrame with added descriptor columns.
    """
    if "SMILES" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'SMILES' column.")

    out = df.copy()

    valid_items: list[tuple[int, str]] = []
    invalid_indices: list[int] = []

    for idx, smi in enumerate(out["SMILES"].astype(str)):
        if smi and smi != "nan":
            valid_items.append((idx, smi))
        else:
            invalid_indices.append(idx)

    if invalid_indices:
        print(f"Removed {len(invalid_indices)} invalid SMILES rows")

    out = out.loc[~out.index.isin(invalid_indices)].reset_index(drop=True)

    n_workers = _get_n_workers(n_workers)
    batch_size = (
        max(1, len(valid_items) // (n_workers * 10))
        if len(valid_items) >= 1000 and n_workers > 1
        else len(valid_items)
    )
    batches = [
        valid_items[i : i + batch_size]
        for i in range(0, len(valid_items), batch_size)
    ]

    if len(valid_items) >= 1000 and n_workers > 1:
        import multiprocessing as mp

        with mp.Pool(processes=n_workers) as pool:
            all_results = pool.map(_process_descriptors_batch, batches)

        results: list[tuple[int, dict]] = []
        for batch_result in all_results:
            results.extend(batch_result)
    else:
        results = _process_descriptors_batch(valid_items)

    results_dict = {idx: desc for idx, desc in results}

    for col in [
        "MW", "HBD", "HBA", "RotB", "HvyAtm", "Rings", "HetAtm", "MR",
        "CAtm", "Atoms", "tPSA", "LogP",
    ]:
        out[col] = [results_dict.get(idx, {}).get(col, None) for idx in range(len(out))]

    return out
