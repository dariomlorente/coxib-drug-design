from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_chembl_ic50_summary(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing IC50 summary file: {path}")

    df = pd.read_csv(path, sep=";", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def add_qsar_targets(
    df: pd.DataFrame,
    cox1_col: str = "IC50_COX1_median",
    cox2_col: str = "IC50_COX2_median",
) -> pd.DataFrame:
    if cox1_col not in df.columns:
        raise ValueError(f"Missing column '{cox1_col}'.")
    if cox2_col not in df.columns:
        raise ValueError(f"Missing column '{cox2_col}'.")

    out = df.copy()
    out[cox1_col] = pd.to_numeric(out[cox1_col], errors="coerce")
    out[cox2_col] = pd.to_numeric(out[cox2_col], errors="coerce")

    cox2_values = out[cox2_col]
    cox1_values = out[cox1_col]

    out["pIC50_COX2"] = np.nan
    out["pIC50_COX1"] = np.nan

    valid_cox2 = cox2_values > 0
    valid_cox1 = cox1_values > 0
    out.loc[valid_cox2, "pIC50_COX2"] = -np.log10(cox2_values.loc[valid_cox2] * 1e-9)
    out.loc[valid_cox1, "pIC50_COX1"] = -np.log10(cox1_values.loc[valid_cox1] * 1e-9)

    active = pd.Series(pd.NA, index=out.index, dtype="boolean")
    mask = cox2_values.notna()
    active.loc[mask] = cox2_values.loc[mask] <= 1000
    out["active_COX2"] = active

    return out


def make_stratification_bins(
    values: pd.Series,
    n_bins: int = 10,
    min_per_bin: int = 2,
) -> pd.Series:
    clean = values.dropna()
    if clean.empty:
        raise ValueError("No values available for stratification.")

    max_bins = min(n_bins, int(clean.nunique()))
    for bins in range(max_bins, 1, -1):
        edges = np.linspace(clean.min(), clean.max(), bins + 1)
        strata = pd.cut(values, bins=edges, include_lowest=True)
        counts = strata.value_counts(dropna=True)
        if not counts.empty and counts.min() >= min_per_bin:
            return strata

    raise ValueError("Unable to create stratification bins with adequate counts.")


def compute_centroid_distances(
    features: np.ndarray,
    centroid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one row.")

    if centroid is None:
        centroid_vec = features.mean(axis=0)
    else:
        centroid_vec = np.asarray(centroid, dtype=float)
        if centroid_vec.ndim != 1:
            raise ValueError("centroid must be a 1D array.")
        if centroid_vec.shape[0] != features.shape[1]:
            raise ValueError(
                "centroid length does not match feature dimension: "
                f"expected {features.shape[1]}, got {centroid_vec.shape[0]}"
            )

    distances = np.linalg.norm(features - centroid_vec, axis=1)
    return distances, centroid_vec


def pic50_to_ic50_nm(pic50_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(pic50_values, dtype=float)
    return np.power(10.0, 9.0 - values)


def compute_selectivity_index(
    ic50_cox1_nm: Iterable[float],
    ic50_cox2_nm: Iterable[float],
) -> np.ndarray:
    cox1 = np.asarray(ic50_cox1_nm, dtype=float)
    cox2 = np.asarray(ic50_cox2_nm, dtype=float)
    return np.where(cox2 > 0, cox1 / cox2, np.nan)


def compute_qsar_score(
    pic50_cox2: Iterable[float],
    pic50_cox1: Iterable[float],
) -> np.ndarray:
    co2 = np.asarray(pic50_cox2, dtype=float)
    co1 = np.asarray(pic50_cox1, dtype=float)
    return 2.0 * co2 - co1
