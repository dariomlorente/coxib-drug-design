from __future__ import annotations

import multiprocessing as mp
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from ._utils import _get_n_workers


def _process_descriptors_batch(batch_items: list[tuple[int, str]]) -> list[tuple[int, dict]]:
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


REQUIRED_BIOAVAILABILITY_COLS = [
    "MW",
    "LogP",
    "HBD",
    "HBA",
    "MR",
    "Atoms",
    "tPSA",
    "RotB",
    "CAtm",
    "HetAtm",
    "Rings",
]

DESCRIPTOR_COLUMNS = [
    "MW",
    "HBD",
    "HBA",
    "RotB",
    "HvyAtm",
    "Rings",
    "HetAtm",
    "MR",
    "CAtm",
    "Atoms",
    "tPSA",
    "LogP",
]


def ensure_required_bioavailability_columns(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    print_report: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    missing = [col for col in REQUIRED_BIOAVAILABILITY_COLS if col not in out.columns]
    if not missing:
        return out

    if "Atoms" in missing and {"CAtm", "HetAtm"}.issubset(out.columns):
        c_atoms = pd.to_numeric(out["CAtm"], errors="coerce")
        hetero_atoms = pd.to_numeric(out["HetAtm"], errors="coerce")
        out["Atoms"] = c_atoms + hetero_atoms
        missing.remove("Atoms")
        if print_report:
            print("[Descriptors] Filled missing 'Atoms' as CAtm + HetAtm")

    if not missing:
        return out

    if smiles_col not in out.columns:
        raise ValueError(
            f"Missing column '{smiles_col}' required to compute descriptors: {', '.join(missing)}"
        )

    calculators = {
        "MW": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "HBD": Lipinski.NumHDonors,
        "HBA": Lipinski.NumHAcceptors,
        "MR": Descriptors.MolMR,
        "Atoms": lambda mol: mol.GetNumAtoms(),
        "tPSA": lambda mol: Descriptors.TPSA(mol, includeSandP=True),
        "RotB": Lipinski.NumRotatableBonds,
        "CAtm": lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6),
        "HetAtm": lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)),
        "Rings": Lipinski.RingCount,
    }

    mols = [
        Chem.MolFromSmiles(smiles) if smiles and smiles != "nan" else None
        for smiles in out[smiles_col].astype(str)
    ]

    for col in missing:
        fn = calculators[col]
        values: list[float | int | None] = []
        for mol in mols:
            if mol is None:
                values.append(None)
                continue
            try:
                values.append(fn(mol))
            except Exception:
                values.append(None)
        out[col] = values

    if print_report:
        print(f"[Descriptors] Added missing columns: {', '.join(missing)}")

    return out
