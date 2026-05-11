from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

from ._paths import QED_CACHE
from ._descriptors import ensure_required_bioavailability_columns


def _qed_from_smiles(smiles: str, precision: int) -> float | None:
    if not smiles or smiles == "nan":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        return round(float(QED.default(mol)), precision)
    except Exception:
        return None


def _compute_qed_batch(payload: tuple[list[str], int]) -> list[float | None]:
    smiles_batch, precision = payload
    return [_qed_from_smiles(smiles, precision=precision) for smiles in smiles_batch]


def add_qed_column(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    price_col: str = "PriceMol",
    qed_col: str = "QED",
    precision: int = 4,
    n_workers: int | None = None,
    mp_min_rows: int = 50000,
    print_report: bool = True,
) -> pd.DataFrame:
    if smiles_col not in df.columns:
        raise ValueError(f"Missing column '{smiles_col}'.")
    if price_col not in df.columns:
        raise ValueError(f"Missing column '{price_col}'.")

    out = df.copy()
    smiles_list = out[smiles_col].astype(str).tolist()

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    if len(smiles_list) >= mp_min_rows and n_workers > 1:
        batch_size = max(5000, len(smiles_list) // (n_workers * 8))
        batches = [
            smiles_list[i : i + batch_size]
            for i in range(0, len(smiles_list), batch_size)
        ]
        payloads = [(batch, precision) for batch in batches]
        with mp.Pool(processes=n_workers) as pool:
            batch_results = pool.map(_compute_qed_batch, payloads)
        qed_values = [val for batch in batch_results for val in batch]
    else:
        qed_values = [_qed_from_smiles(smiles, precision=precision) for smiles in smiles_list]

    if qed_col in out.columns:
        out = out.drop(columns=[qed_col])

    insert_at = out.columns.get_loc(price_col) + 1
    out.insert(insert_at, qed_col, qed_values)

    if print_report:
        valid_qed = pd.Series(qed_values, dtype="float64").notna().sum()
        missing_qed = len(out) - int(valid_qed)
        print(
            f"[QED] Computed {int(valid_qed):,}/{len(out):,} values "
            f"({missing_qed:,} missing)"
        )

    return out


def load_or_compute_qed(
    df: pd.DataFrame,
    stage_name: str,
    cache_dir: str | Path = QED_CACHE,
    force_recompute: bool = False,
    smiles_col: str = "SMILES",
    price_col: str = "PriceMol",
    print_report: bool = True,
) -> tuple[pd.DataFrame, Path]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{stage_name}_qed_{len(df)}cmpds.csv"

    if cache_path.exists() and not force_recompute:
        if print_report:
            print(f"[load_or_compute_qed] Loading {cache_path.name} \u2713")
        return pd.read_csv(cache_path), cache_path

    if print_report:
        print(f"[load_or_compute_qed] Computing QED for {stage_name}...")

    prepared = ensure_required_bioavailability_columns(
        df,
        smiles_col=smiles_col,
        print_report=print_report,
    )
    with_qed = add_qed_column(
        prepared,
        smiles_col=smiles_col,
        price_col=price_col,
        print_report=print_report,
    )

    with_qed.to_csv(cache_path, index=False)

    if print_report:
        print(f"[load_or_compute_qed] Saved {cache_path.name} ({len(with_qed):,} rows)")

    return with_qed, cache_path
