from __future__ import annotations

import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
from .io import add_rdkit_properties
from ._utils import _get_cache_key, _load_cache, _save_cache


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


def report_df_size(df: pd.DataFrame, label: str) -> None:
    """Print DataFrame size with a bracketed label."""
    print(f"[{label}] {len(df):,} rows")


def load_chembl_ic50_summary(path: str | Path) -> pd.DataFrame:
    """
    Load the merged ChEMBL IC50 summary (semicolon-delimited).

    Parameters
    ----------
    path
        Path to ``ChEMBL_IC50nSI.csv``.

    Returns
    -------
    pd.DataFrame
        Loaded summary table with stripped column names.
    """
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
    """
    Add pIC50 targets and active label for COX2.

    Parameters
    ----------
    df
        Input DataFrame with IC50 columns.
    cox1_col
        Column name for COX1 median IC50 (nM).
    cox2_col
        Column name for COX2 median IC50 (nM).

    Returns
    -------
    pd.DataFrame
        Copy of the input with pIC50 columns and active_COX2 label.
    """
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
    """
    Create range-based bins for stratified splits on continuous targets.

    Parameters
    ----------
    values
        Series to bin (e.g. pIC50 values).
    n_bins
        Initial number of equal-width bins.
    min_per_bin
        Minimum number of samples required per bin.

    Returns
    -------
    pd.Series
        Categorical bin labels for stratification.
    """
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
    """Compute Euclidean distances to a centroid.

    Parameters
    ----------
    features
        2D array of (optionally scaled) feature values.
    centroid
        Optional centroid vector. When None, the centroid is computed as the
        mean of ``features``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Distances and the centroid vector used.
    """
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
    """
    Convert pIC50 values (M) to IC50 in nM.

    Parameters
    ----------
    pic50_values
        Iterable of pIC50 values.

    Returns
    -------
    np.ndarray
        IC50 values in nM.
    """
    values = np.asarray(pic50_values, dtype=float)
    return np.power(10.0, 9.0 - values)


def compute_selectivity_index(
    ic50_cox1_nm: Iterable[float],
    ic50_cox2_nm: Iterable[float],
) -> np.ndarray:
    """
    Compute selectivity index as IC50_COX1 / IC50_COX2.

    Parameters
    ----------
    ic50_cox1_nm
        Predicted COX1 IC50 values (nM).
    ic50_cox2_nm
        Predicted COX2 IC50 values (nM).

    Returns
    -------
    np.ndarray
        Selectivity index values.
    """
    cox1 = np.asarray(ic50_cox1_nm, dtype=float)
    cox2 = np.asarray(ic50_cox2_nm, dtype=float)
    return np.where(cox2 > 0, cox1 / cox2, np.nan)


def compute_qsar_score(
    pic50_cox2: Iterable[float],
    pic50_cox1: Iterable[float],
) -> np.ndarray:
    """
    Compute QSAR score as 2 * pIC50_COX2 - pIC50_COX1.

    Parameters
    ----------
    pic50_cox2
        Predicted pIC50 values for COX2.
    pic50_cox1
        Predicted pIC50 values for COX1.

    Returns
    -------
    np.ndarray
        QSAR scores.
    """
    co2 = np.asarray(pic50_cox2, dtype=float)
    co1 = np.asarray(pic50_cox1, dtype=float)
    return 2.0 * co2 - co1


def _clean_smiles_column(
    df: pd.DataFrame,
    label: str,
    smiles_candidates: list[str] | None = None,
) -> pd.DataFrame:
    candidates = smiles_candidates or ["Smiles", "SMILES", "Canonical SMILES", "canonical_smiles"]
    smiles_col = next((col for col in candidates if col in df.columns), None)
    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Checked: {candidates}")

    out = df.copy()
    start_rows = len(out)
    out = out.dropna(subset=[smiles_col]).copy()
    print(f"[{label}] Removed {start_rows - len(out):,} rows with NaN SMILES")

    out[smiles_col] = out[smiles_col].astype(str).str.strip()
    empty_mask = out[smiles_col].eq("")
    print(f"[{label}] Removed {empty_mask.sum():,} empty/blank SMILES")
    out = out.loc[~empty_mask].copy()

    valid_mask = out[smiles_col].apply(lambda smi: Chem.MolFromSmiles(smi) is not None)
    print(f"[{label}] Removed {len(out) - valid_mask.sum():,} invalid SMILES")
    out = out.loc[valid_mask].copy()

    if smiles_col != "SMILES":
        out = out.rename(columns={smiles_col: "SMILES"})

    return out


def _prepare_qsar_training_data(
    chembl_path: str | Path,
) -> pd.DataFrame:
    df_chembl = load_chembl_ic50_summary(chembl_path)
    df_chembl = _clean_smiles_column(df_chembl, label="QSAR")
    df_chembl = add_rdkit_properties(df_chembl)
    df_chembl = add_qsar_targets(df_chembl)
    df_chembl = df_chembl.dropna(subset=DESCRIPTOR_COLUMNS)
    print(f"[QSAR] ChEMBL rows with descriptors: {len(df_chembl):,}")
    return df_chembl


def _train_qsar_models(
    df_chembl: pd.DataFrame,
    n_estimators: int = 500,
    random_state: int = 42,
) -> dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df_cox2 = df_chembl[df_chembl["pIC50_COX2"].notna()].copy()
    df_cox1 = df_chembl[df_chembl["pIC50_COX1"].notna()].copy()

    X_cox2 = df_cox2[DESCRIPTOR_COLUMNS].to_numpy()
    y_cox2 = df_cox2["pIC50_COX2"].to_numpy()
    y_active = df_cox2["active_COX2"].astype(int).to_numpy()

    strata = make_stratification_bins(df_cox2["pIC50_COX2"], n_bins=10)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_active_train,
        y_active_test,
    ) = train_test_split(
        X_cox2,
        y_cox2,
        y_active,
        test_size=0.2,
        random_state=random_state,
        stratify=strata,
    )

    rf_reg = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_reg.fit(X_train, y_train)

    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_active_train)

    y_pred = rf_reg.predict(X_test)
    y_class = rf_clf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    acc = accuracy_score(y_active_test, y_class)

    print(f"[QSAR] COX2 RF regressor: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
    print(f"[QSAR] COX2 RF classifier: Accuracy={acc:.3f}")

    X_cox1 = df_cox1[DESCRIPTOR_COLUMNS].to_numpy()
    y_cox1 = df_cox1["pIC50_COX1"].to_numpy()

    rf_cox1 = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_cox1.fit(X_cox1, y_cox1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    train_distances, centroid = compute_centroid_distances(X_train_scaled)
    ad_threshold = np.quantile(train_distances, 0.95)
    print(f"[QSAR] AD cutoff (95th percentile): {ad_threshold:.3f}")

    return {
        "rf_reg_cox2": rf_reg,
        "rf_clf_cox2": rf_clf,
        "rf_reg_cox1": rf_cox1,
        "scaler": scaler,
        "ad_threshold": ad_threshold,
        "centroid": centroid,
    }


def _load_qsar_model_cache(cache_file: Path) -> dict[str, object] | None:
    cache = _load_cache(cache_file)
    if not cache:
        return None

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    try:
        if cache.get("version") != 1:
            return None
        if cache.get("type") != "qsar_models":
            return None

        df_cache = cache.get("models")
        if df_cache is None:
            return None

        df_cox2 = pd.DataFrame(df_cache.get("cox2", []))
        df_cox1 = pd.DataFrame(df_cache.get("cox1", []))
        if df_cox2.empty or df_cox1.empty:
            return None

        rf_reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        rf_reg.fit(df_cox2[DESCRIPTOR_COLUMNS].to_numpy(), df_cox2["pIC50_COX2"].to_numpy())

        rf_clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        rf_clf.fit(
            df_cox2[DESCRIPTOR_COLUMNS].to_numpy(),
            df_cox2["active_COX2"].astype(int).to_numpy(),
        )

        rf_cox1 = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        rf_cox1.fit(df_cox1[DESCRIPTOR_COLUMNS].to_numpy(), df_cox1["pIC50_COX1"].to_numpy())

        scaler = StandardScaler()
        X_train = df_cox2[DESCRIPTOR_COLUMNS].to_numpy()
        X_train_scaled = scaler.fit_transform(X_train)
        train_distances, centroid = compute_centroid_distances(X_train_scaled)
        ad_threshold = np.quantile(train_distances, 0.95)

        return {
            "rf_reg_cox2": rf_reg,
            "rf_clf_cox2": rf_clf,
            "rf_reg_cox1": rf_cox1,
            "scaler": scaler,
            "ad_threshold": float(ad_threshold),
            "centroid": centroid,
        }
    except Exception:
        return None


def _save_qsar_model_cache(cache_file: Path, df_chembl: pd.DataFrame) -> None:
    df_cox2 = df_chembl[df_chembl["pIC50_COX2"].notna()].copy()
    df_cox1 = df_chembl[df_chembl["pIC50_COX1"].notna()].copy()

    cache = {
        "version": 1,
        "type": "qsar_models",
        "models": {
            "cox2": df_cox2[[*DESCRIPTOR_COLUMNS, "pIC50_COX2", "active_COX2"]].to_dict("records"),
            "cox1": df_cox1[[*DESCRIPTOR_COLUMNS, "pIC50_COX1"]].to_dict("records"),
        },
    }
    _save_cache(cache_file, cache)


def _predict_qsar_for_series(
    df: pd.DataFrame,
    label: str,
    models: dict[str, object],
) -> pd.DataFrame:
    out = _clean_smiles_column(df, label=f"QSAR {label}")
    out = add_rdkit_properties(out)
    out = out.dropna(subset=DESCRIPTOR_COLUMNS)
    print(f"[QSAR {label}] Rows with descriptors: {len(out):,}")

    if out.empty:
        return out

    X = out[DESCRIPTOR_COLUMNS].to_numpy()
    rf_reg = models["rf_reg_cox2"]
    rf_cox1 = models["rf_reg_cox1"]
    scaler = models["scaler"]
    ad_threshold = float(models["ad_threshold"])
    centroid = models.get("centroid")

    pic50_cox2_pred = rf_reg.predict(X)
    pic50_cox1_pred = rf_cox1.predict(X)

    ic50_cox2_pred = pic50_to_ic50_nm(pic50_cox2_pred)
    ic50_cox1_pred = pic50_to_ic50_nm(pic50_cox1_pred)
    si_pred = compute_selectivity_index(ic50_cox1_pred, ic50_cox2_pred)
    qsar_score = compute_qsar_score(pic50_cox2_pred, pic50_cox1_pred)

    X_scaled = scaler.transform(X)
    if centroid is None:
        rep_distances, _ = compute_centroid_distances(X_scaled)
    else:
        rep_distances, _ = compute_centroid_distances(X_scaled, centroid=np.asarray(centroid))

    out["AD_Distance"] = rep_distances
    out["In_AD"] = out["AD_Distance"] <= ad_threshold
    out["pIC50_COX2_pred"] = pic50_cox2_pred
    out["pIC50_COX1_pred"] = pic50_cox1_pred
    out["IC50_COX2_pred_nM"] = ic50_cox2_pred
    out["IC50_COX1_pred_nM"] = ic50_cox1_pred
    out["SI_pred"] = si_pred
    out["QSAR_score"] = qsar_score

    return out


def _select_top_by_qsar_score(
    df: pd.DataFrame,
    acceptance_rate: float,
    minimum: int,
    score_col: str = "QSAR_score",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = len(df)
    if total == 0:
        return df.copy(), df.copy()

    keep_pct = int(total * acceptance_rate)
    if acceptance_rate > 0 and keep_pct == 0:
        keep_pct = 1

    keep_n = min(total, max(keep_pct, minimum))
    ordered = df.sort_values(score_col, ascending=False, kind="stable").reset_index(drop=True)
    accepted = ordered.head(keep_n).copy()
    rejected = ordered.tail(total - keep_n).copy()
    return accepted, rejected


def run_qsar_winnow(
    df_imidazolones_druglike: pd.DataFrame,
    df_thiazolones_druglike: pd.DataFrame,
    chembl_path: str | Path = "protein_files/IC50s/ChEMBL_IC50nSI.csv",
    acceptance_rate: float = 0.01,
    minimum: int = 1000,
    output_dir: str | Path = "mol_files/7. QSAR",
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    print_report: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """
    Run ML-QSAR winnowing for imidazolones and thiazolones.

    Parameters
    ----------
    df_imidazolones_druglike
        Imidazolones after bioavailability filtering.
    df_thiazolones_druglike
        Thiazolones after bioavailability filtering.
    chembl_path
        Path to ChEMBL IC50 summary file.
    acceptance_rate
        Fraction for top percentile selection (e.g. 0.01 for top 1%).
    minimum
        Minimum count of top compounds to keep.
    output_dir
        Output directory for QSAR results.
    use_cache
        Use a persistent cache to avoid re-training on every run.
    cache_file
        Optional cache file path. When None, defaults to
        ``{output_dir}/.cache/qsar_models.json.gz``.
    print_report
        Print progress and output paths.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], dict[str, Path]]
        Accepted/rejected DataFrames and output paths.
    """
    if acceptance_rate < 0 or acceptance_rate > 1:
        raise ValueError("acceptance_rate must be between 0 and 1.")
    if minimum <= 0:
        raise ValueError("minimum must be greater than 0.")

    output_dir = Path(output_dir)

    cache_path = None
    if use_cache:
        if cache_file is None:
            cache_path = output_dir / ".cache" / "qsar_models.json.gz"
        else:
            cache_path = Path(cache_file)

    models = None
    if cache_path is not None:
        models = _load_qsar_model_cache(cache_path)
        if models is not None and print_report:
            try:
                cache_display = cache_path.resolve().relative_to(Path.cwd())
            except Exception:
                cache_display = Path(cache_path.name)
            print(f"[QSAR] Loaded model cache: {cache_display}")

    if models is None:
        df_chembl = _prepare_qsar_training_data(chembl_path)
        models = _train_qsar_models(df_chembl)
        if cache_path is not None:
            _save_qsar_model_cache(cache_path, df_chembl)
            if print_report:
                try:
                    cache_display = cache_path.resolve().relative_to(Path.cwd())
                except Exception:
                    cache_display = Path(cache_path.name)
                print(f"[QSAR] Saved model cache: {cache_display}")

    df_imi_pred = _predict_qsar_for_series(df_imidazolones_druglike, "Imidazolones", models)
    df_thi_pred = _predict_qsar_for_series(df_thiazolones_druglike, "Thiazolones", models)

    df_imi_acc, df_imi_rej = _select_top_by_qsar_score(df_imi_pred, acceptance_rate, minimum)
    df_thi_acc, df_thi_rej = _select_top_by_qsar_score(df_thi_pred, acceptance_rate, minimum)

    if print_report:
        keep_imi = len(df_imi_acc)
        keep_thi = len(df_thi_acc)
        pct_imi = max(1, int(len(df_imi_pred) * acceptance_rate)) if len(df_imi_pred) else 0
        pct_thi = max(1, int(len(df_thi_pred) * acceptance_rate)) if len(df_thi_pred) else 0
        print(
            f"[QSAR] Selection counts (max of {acceptance_rate:.2%} vs {minimum}): "
            f"Imidazolones keep_n={keep_imi} ({acceptance_rate:.2%}={pct_imi}), "
            f"Thiazolones keep_n={keep_thi} ({acceptance_rate:.2%}={pct_thi})"
        )
        print(f"[QSAR] Imidazolones: {len(df_imi_acc):,} accepted, {len(df_imi_rej):,} rejected")
        print(f"[QSAR] Thiazolones: {len(df_thi_acc):,} accepted, {len(df_thi_rej):,} rejected")

    rejected_dir = output_dir / ".rejected"
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "imidazolones_accepted": output_dir
        / f"Imidazolones_qsar_{len(df_imi_acc)}cmpds.csv",
        "thiazolones_accepted": output_dir
        / f"Thiazolones_qsar_{len(df_thi_acc)}cmpds.csv",
        "imidazolones_rejected": rejected_dir
        / f"Imidazolones_rejected_qsar_{len(df_imi_rej)}cmpds.csv",
        "thiazolones_rejected": rejected_dir
        / f"Thiazolones_rejected_qsar_{len(df_thi_rej)}cmpds.csv",
    }

    df_imi_acc.to_csv(paths["imidazolones_accepted"], index=False)
    df_thi_acc.to_csv(paths["thiazolones_accepted"], index=False)
    df_imi_rej.to_csv(paths["imidazolones_rejected"], index=False)
    df_thi_rej.to_csv(paths["thiazolones_rejected"], index=False)

    if print_report:
        print(f"[Save] {paths['imidazolones_accepted']}")
        print(f"[Save] {paths['thiazolones_accepted']}")
        print(f"[Save] {paths['imidazolones_rejected']}")
        print(f"[Save] {paths['thiazolones_rejected']}")

    outputs = {
        "imidazolones_accepted": df_imi_acc,
        "thiazolones_accepted": df_thi_acc,
        "imidazolones_rejected": df_imi_rej,
        "thiazolones_rejected": df_thi_rej,
    }

    return outputs, paths


def _extract_counted_suffix(path: Path, prefix: str) -> int | None:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)cmpds$")
    match = pattern.match(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def find_latest_stage_csv(
    stage_dir: str | Path,
    stage_name: str,
    filter_mode: str = "brenkpains",
) -> Path:
    """
    Find latest counted CSV for a stage/filter pair.

    Parameters
    ----------
    stage_dir
        Directory containing stage CSV files.
    stage_name
        Stage name, e.g. ``Imidazolones``.
    filter_mode
        Suffix mode in filenames, e.g. ``brenkpains``.

    Returns
    -------
    Path
        Path to selected CSV file.
    """
    stage_dir = Path(stage_dir)
    prefix = f"{stage_name}_{filter_mode}"

    candidates: list[tuple[Path, int]] = []
    for path in stage_dir.glob(f"{prefix}_*cmpds.csv"):
        count = _extract_counted_suffix(path, prefix)
        if count is not None:
            candidates.append((path, count))

    if not candidates:
        raise ValueError(f"No file found for pattern '{prefix}_*cmpds.csv' in {stage_dir}.")

    return max(candidates, key=lambda item: (item[1], item[0].stat().st_mtime))[0]


def load_generated_product_sets(
    imidazolones_dir: str | Path = "mol_files/4. Imidazolones",
    thiazolones_dir: str | Path = "mol_files/5. Thiazolones",
    filter_mode: str = "brenkpains",
    print_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """
    Load generated product sets for phase 2.

    Parameters
    ----------
    imidazolones_dir
        Directory containing imidazolone CSV files.
    thiazolones_dir
        Directory containing thiazolone CSV files.
    filter_mode
        Input mode used in file naming.
    print_report
        Print loading report.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, Path, Path]
        Imidazolones DataFrame, thiazolones DataFrame, and both file paths.
    """
    imidazolones_path = find_latest_stage_csv(imidazolones_dir, "Imidazolones", filter_mode)
    thiazolones_path = find_latest_stage_csv(thiazolones_dir, "Thiazolones", filter_mode)

    df_imidazolones = pd.read_csv(imidazolones_path)
    df_thiazolones = pd.read_csv(thiazolones_path)

    expected_imi = _extract_counted_suffix(imidazolones_path, f"Imidazolones_{filter_mode}")
    expected_thi = _extract_counted_suffix(thiazolones_path, f"Thiazolones_{filter_mode}")

    if print_report:
        print(
            f"[LoadProducts] Imidazolones: {imidazolones_path.name} "
            f"({len(df_imidazolones):,} rows)"
        )
        print(
            f"[LoadProducts] Thiazolones:  {thiazolones_path.name} "
            f"({len(df_thiazolones):,} rows)"
        )
        if expected_imi is not None and len(df_imidazolones) != expected_imi:
            print(
                f"⚠️ [LoadProducts] Imidazolones filename count ({expected_imi:,}) "
                f"does not match loaded rows ({len(df_imidazolones):,})"
            )
        if expected_thi is not None and len(df_thiazolones) != expected_thi:
            print(
                f"⚠️ [LoadProducts] Thiazolones filename count ({expected_thi:,}) "
                f"does not match loaded rows ({len(df_thiazolones):,})"
            )

    return df_imidazolones, df_thiazolones, imidazolones_path, thiazolones_path


def ensure_required_bioavailability_columns(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Ensure all descriptor columns required by the bioavailability filter exist.

    Missing ``Atoms`` is filled from ``CAtm + HetAtm`` when possible; remaining
    missing descriptors are computed from SMILES.

    Parameters
    ----------
    df
        Input DataFrame.
    smiles_col
        SMILES column name.
    print_report
        Print descriptor completion report.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all required columns.
    """
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
    """
    Add QED column after ``PriceMol``.

    Parameters
    ----------
    df
        Input DataFrame.
    smiles_col
        SMILES column name.
    price_col
        Price column used as insertion anchor.
    qed_col
        Output QED column name.
    precision
        Decimal rounding for QED values.
    n_workers
        Number of worker processes for large datasets.
    mp_min_rows
        Minimum dataset size to enable multiprocessing.
    print_report
        Print QED computation report.

    Returns
    -------
    pd.DataFrame
        DataFrame with QED column inserted after ``PriceMol``.
    """
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
    cache_dir: str | Path = "mol_files/6. QED/.cache",
    force_recompute: bool = False,
    smiles_col: str = "SMILES",
    price_col: str = "PriceMol",
    print_report: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """
    Load QED-annotated CSV from cache when available, otherwise compute and cache.

    Parameters
    ----------
    df
        Input DataFrame.
    stage_name
        Stage name, e.g. ``Imidazolones``.
    cache_dir
        Cache directory for QED CSV files.
    force_recompute
        Ignore cache and recompute when True.
    smiles_col
        SMILES column name.
    price_col
        Price column name.
    print_report
        Print caching report.

    Returns
    -------
    tuple[pd.DataFrame, Path]
        QED DataFrame and cache path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{stage_name}_qed_{len(df)}cmpds.csv"

    if cache_path.exists() and not force_recompute:
        if print_report:
            print(f"[load_or_compute_qed] Loading {cache_path.name} ✓")
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


def filter_bioavailability(
    df: pd.DataFrame,
    qed_col: str = "QED",
    violation_col: str = "Violation",
    print_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Lipinski, Ghose, Egan, Muegge, and Veber rules.

    A compound is accepted when it violates at most one rule.

    Parameters
    ----------
    df
        Input DataFrame containing descriptor columns.
    qed_col
        QED column name.
    violation_col
        Output column with violated rule names.
    print_report
        Print acceptance/rejection summary.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Accepted and rejected DataFrames.
    """
    required_cols = [qed_col] + REQUIRED_BIOAVAILABILITY_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for bioavailability filter: {', '.join(missing_cols)}"
        )

    num = df[REQUIRED_BIOAVAILABILITY_COLS].apply(pd.to_numeric, errors="coerce")

    pass_lipinski = (
        (num["MW"] <= 500)
        & (num["LogP"] <= 4.15)
        & (num["HBD"] <= 5)
        & (num["HBA"] <= 10)
    )
    pass_ghose = (
        (num["MW"] >= 160)
        & (num["MW"] <= 480)
        & (num["LogP"] >= -0.4)
        & (num["LogP"] <= 5.6)
        & (num["MR"] >= 40)
        & (num["MR"] <= 130)
        & (num["Atoms"] >= 20)
        & (num["Atoms"] <= 70)
    )
    pass_egan = (num["LogP"] <= 5.88) & (num["tPSA"] <= 131.6)
    pass_muegge = (
        (num["MW"] >= 200)
        & (num["MW"] <= 600)
        & (num["LogP"] >= -2)
        & (num["LogP"] <= 5)
        & (num["HBD"] <= 5)
        & (num["HBA"] <= 10)
        & (num["CAtm"] >= 5)
        & (num["HetAtm"] >= 2)
        & (num["Rings"] <= 7)
        & (num["RotB"] <= 15)
        & (num["tPSA"] <= 150)
    )
    pass_veber = (num["RotB"] <= 10) & (num["tPSA"] <= 140)

    rule_pass = pd.DataFrame(
        {
            "Lipinski": pass_lipinski,
            "Ghose": pass_ghose,
            "Egan": pass_egan,
            "Muegge": pass_muegge,
            "Veber": pass_veber,
        },
        index=df.index,
    )

    rule_fail = ~rule_pass.fillna(False)
    fail_count = rule_fail.sum(axis=1)

    violations = pd.Series("none", index=df.index, dtype="object")

    single_fail_mask = fail_count == 1
    if single_fail_mask.any():
        violations.loc[single_fail_mask] = rule_fail.loc[single_fail_mask].idxmax(axis=1)

    multi_fail_mask = fail_count > 1
    if multi_fail_mask.any():
        fail_rows = rule_fail.loc[multi_fail_mask, rule_pass.columns].to_numpy(dtype=bool)
        names = list(rule_pass.columns)
        joined = [
            ", ".join(name for name, is_failed in zip(names, row) if is_failed)
            for row in fail_rows
        ]
        violations.loc[multi_fail_mask] = joined

    out = df.copy()
    if violation_col in out.columns:
        out = out.drop(columns=[violation_col])

    insert_at = out.columns.get_loc(qed_col) + 1
    out.insert(insert_at, violation_col, violations.values)

    accepted_mask = fail_count <= 1
    accepted = out.loc[accepted_mask].reset_index(drop=True)
    rejected = out.loc[~accepted_mask].reset_index(drop=True)

    if print_report:
        total = len(out)
        n_accepted = len(accepted)
        n_rejected = len(rejected)
        accepted_pct = (n_accepted / total * 100) if total else 0.0
        print(
            f"[filter_bioavailability] {n_accepted:,}/{total:,} accepted "
            f"({accepted_pct:.1f}%), {n_rejected:,} rejected"
        )

    return accepted, rejected


def save_bioavailability_outputs(
    df_imidazolones_druglike: pd.DataFrame,
    df_thiazolones_druglike: pd.DataFrame,
    df_imidazolones_nondruglike: pd.DataFrame,
    df_thiazolones_nondruglike: pd.DataFrame,
    output_dir: str | Path = "mol_files/6. QED",
    print_report: bool = True,
) -> dict[str, Path]:
    """
    Save accepted and rejected phase-2 datasets using row-count suffix naming.

    Parameters
    ----------
    df_imidazolones_druglike
        Accepted imidazolones.
    df_thiazolones_druglike
        Accepted thiazolones.
    df_imidazolones_nondruglike
        Rejected imidazolones.
    df_thiazolones_nondruglike
        Rejected thiazolones.
    output_dir
        Base output directory.
    print_report
        Print saved paths.

    Returns
    -------
    dict[str, Path]
        Output paths by dataset role.
    """
    output_dir = Path(output_dir)
    rejected_dir = output_dir / ".rejected"
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "imidazolones_accepted": output_dir
        / f"Imidazolones_{len(df_imidazolones_druglike)}cmpds.csv",
        "thiazolones_accepted": output_dir
        / f"Thiazolones_{len(df_thiazolones_druglike)}cmpds.csv",
        "imidazolones_rejected": rejected_dir
        / f"Imidazolones_rejected_bioavailability_{len(df_imidazolones_nondruglike)}cmpds.csv",
        "thiazolones_rejected": rejected_dir
        / f"Thiazolones_rejected_bioavailability_{len(df_thiazolones_nondruglike)}cmpds.csv",
    }

    df_imidazolones_druglike.to_csv(paths["imidazolones_accepted"], index=False)
    df_thiazolones_druglike.to_csv(paths["thiazolones_accepted"], index=False)
    df_imidazolones_nondruglike.to_csv(paths["imidazolones_rejected"], index=False)
    df_thiazolones_nondruglike.to_csv(paths["thiazolones_rejected"], index=False)

    if print_report:
        print(f"[Save] {paths['imidazolones_accepted']}")
        print(f"[Save] {paths['thiazolones_accepted']}")
        print(f"[Save] {paths['imidazolones_rejected']}")
        print(f"[Save] {paths['thiazolones_rejected']}")

    return paths


def _normalize_acceptance_rate(rate: float | None) -> float | None:
    if rate is None:
        return None
    if rate < 0 or rate > 1:
        raise ValueError("acceptance_rate must be between 0 and 1.")
    return float(rate)


def _validate_max_sample_size(max_sample_size: int | None) -> int | None:
    if max_sample_size is None:
        return None
    if max_sample_size <= 0:
        raise ValueError("max_sample_size must be greater than 0 when provided.")
    return int(max_sample_size)


def apply_price_controls(
    df: pd.DataFrame,
    price_col: str = "PriceMol",
    max_price: float | None = None,
    acceptance_rate: float | None = None,
    max_sample_size: int | None = None,
    rejection_col: str = "PriceCtrlRejection",
    print_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply sequential price-based controls: max price -> acceptance rate -> max sample size.

    Parameters
    ----------
    df
        Input DataFrame.
    price_col
        Price column used for sorting and filtering.
    max_price
        Optional hard upper bound for price (inclusive).
    acceptance_rate
        Optional fraction in [0, 1]. Keeps floor(rate * n) rows after max-price
        filtering, with minimum 1 row when rate > 0 and n > 0.
    max_sample_size
        Optional cap on final kept rows (top-N cheapest after previous controls).
    rejection_col
        Column name storing rejection reason(s).
    print_report
        Print filtering summary.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Accepted and rejected DataFrames.
    """
    if price_col not in df.columns:
        raise ValueError(f"Missing column '{price_col}'.")

    acceptance_rate = _normalize_acceptance_rate(acceptance_rate)
    max_sample_size = _validate_max_sample_size(max_sample_size)

    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")

    if out[price_col].isna().any() and print_report:
        print(
            f"⚠️ [apply_price_controls] Found {int(out[price_col].isna().sum()):,} rows "
            f"with invalid {price_col}; they are treated as most expensive"
        )

    out = out.sort_values(by=[price_col], kind="stable", na_position="last").reset_index(drop=True)

    reasons = pd.Series("", index=out.index, dtype="object")

    def _append_reason(mask: pd.Series, reason: str) -> None:
        if not bool(mask.any()):
            return
        current = reasons.loc[mask]
        reasons.loc[mask] = np.where(current.eq(""), reason, current + f", {reason}")

    accepted_mask = pd.Series(True, index=out.index)

    if max_price is not None:
        if max_price < 0:
            raise ValueError("max_price must be >= 0 when provided.")
        over_price = out[price_col] > max_price
        over_price = over_price.fillna(True)
        _append_reason(over_price, "max_price")
        accepted_mask &= ~over_price

    if acceptance_rate is not None:
        current_idx = out.index[accepted_mask]
        current_n = len(current_idx)
        if current_n > 0:
            keep_n = int(current_n * acceptance_rate)
            if acceptance_rate > 0 and keep_n == 0:
                keep_n = 1
            if keep_n < current_n:
                drop_idx = current_idx[keep_n:]
                drop_mask = pd.Series(False, index=out.index)
                drop_mask.loc[drop_idx] = True
                _append_reason(drop_mask, "acceptance_rate")
                accepted_mask.loc[drop_idx] = False

    if max_sample_size is not None:
        current_idx = out.index[accepted_mask]
        current_n = len(current_idx)
        if current_n > max_sample_size:
            drop_idx = current_idx[max_sample_size:]
            drop_mask = pd.Series(False, index=out.index)
            drop_mask.loc[drop_idx] = True
            _append_reason(drop_mask, "max_sample_size")
            accepted_mask.loc[drop_idx] = False

    accepted = out.loc[accepted_mask].copy().reset_index(drop=True)
    rejected = out.loc[~accepted_mask].copy().reset_index(drop=True)

    if rejection_col in accepted.columns:
        accepted = accepted.drop(columns=[rejection_col])
    if rejection_col in rejected.columns:
        rejected = rejected.drop(columns=[rejection_col])

    rejected.insert(len(rejected.columns), rejection_col, reasons.loc[~accepted_mask].values)

    if print_report:
        total = len(out)
        n_acc = len(accepted)
        n_rej = len(rejected)
        acc_pct = (n_acc / total * 100) if total else 0.0
        print(
            f"[apply_price_controls] {n_acc:,}/{total:,} accepted "
            f"({acc_pct:.1f}%), {n_rej:,} rejected"
        )

    return accepted, rejected


def save_price_control_outputs(
    df_imidazolones_input: pd.DataFrame,
    df_thiazolones_input: pd.DataFrame,
    df_imidazolones_rejected_pricectrl: pd.DataFrame,
    df_thiazolones_rejected_pricectrl: pd.DataFrame,
    input_dir: str | Path = "mol_files/8. Clustering/.inputs",
    rejected_dir: str | Path = "mol_files/8. Clustering/.rejected",
    print_report: bool = True,
) -> dict[str, Path]:
    """
    Save price-controlled clustering inputs and rejected sets.

    Parameters
    ----------
    df_imidazolones_input
        Accepted imidazolones for clustering input.
    df_thiazolones_input
        Accepted thiazolones for clustering input.
    df_imidazolones_rejected_pricectrl
        Rejected imidazolones from price controls.
    df_thiazolones_rejected_pricectrl
        Rejected thiazolones from price controls.
    input_dir
        Directory for accepted clustering inputs.
    rejected_dir
        Directory for rejected compounds.
    print_report
        Print saved paths.

    Returns
    -------
    dict[str, Path]
        Output paths by dataset role.
    """
    input_dir = Path(input_dir)
    rejected_dir = Path(rejected_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "imidazolones_input": input_dir
        / f"Imidazolones_input_{len(df_imidazolones_input)}cmpds.csv",
        "thiazolones_input": input_dir
        / f"Thiazolones_input_{len(df_thiazolones_input)}cmpds.csv",
        "imidazolones_rejected_pricectrl": rejected_dir
        / f"Imidazolones_rejected_pricectrl_{len(df_imidazolones_rejected_pricectrl)}cmpds.csv",
        "thiazolones_rejected_pricectrl": rejected_dir
        / f"Thiazolones_rejected_pricectrl_{len(df_thiazolones_rejected_pricectrl)}cmpds.csv",
    }

    df_imidazolones_input.to_csv(paths["imidazolones_input"], index=False)
    df_thiazolones_input.to_csv(paths["thiazolones_input"], index=False)
    df_imidazolones_rejected_pricectrl.to_csv(paths["imidazolones_rejected_pricectrl"], index=False)
    df_thiazolones_rejected_pricectrl.to_csv(paths["thiazolones_rejected_pricectrl"], index=False)

    if print_report:
        print(f"[Save] {paths['imidazolones_input']}")
        print(f"[Save] {paths['thiazolones_input']}")
        print(f"[Save] {paths['imidazolones_rejected_pricectrl']}")
        print(f"[Save] {paths['thiazolones_rejected_pricectrl']}")

    return paths


def run_clustering_input_export(
    df_imidazolones_druglike: pd.DataFrame,
    df_thiazolones_druglike: pd.DataFrame,
    max_price_imi: float | None = None,
    max_price_thi: float | None = None,
    accept_rate_imi: float = 0.67,
    accept_rate_thi: float = 0.67,
    max_sample_imi: int = 30000,
    max_sample_thi: int = 30000,
) -> dict[str, Path]:
    """
    Run clustering input export pipeline for both series.

    Parameters
    ----------
    df_imidazolones_druglike
        Accepted imidazolones from the prior stage.
    df_thiazolones_druglike
        Accepted thiazolones from the prior stage.
    max_price_imi
        Max price for imidazolones.
    max_price_thi
        Max price for thiazolones.
    accept_rate_imi
        Acceptance rate for imidazolones.
    accept_rate_thi
        Acceptance rate for thiazolones.
    max_sample_imi
        Max sample size for imidazolones.
    max_sample_thi
        Max sample size for thiazolones.

    Returns
    -------
    dict[str, Path]
        Paths with keys: imidazolones_input, thiazolones_input,
        imidazolones_rejected_pricectrl, thiazolones_rejected_pricectrl.
    """
    df_imi_in, df_imi_rej = apply_price_controls(
        df_imidazolones_druglike,
        max_price=max_price_imi,
        acceptance_rate=accept_rate_imi,
        max_sample_size=max_sample_imi,
    )
    df_thi_in, df_thi_rej = apply_price_controls(
        df_thiazolones_druglike,
        max_price=max_price_thi,
        acceptance_rate=accept_rate_thi,
        max_sample_size=max_sample_thi,
    )
    return save_price_control_outputs(
        df_imidazolones_input=df_imi_in,
        df_thiazolones_input=df_thi_in,
        df_imidazolones_rejected_pricectrl=df_imi_rej,
        df_thiazolones_rejected_pricectrl=df_thi_rej,
        input_dir="mol_files/8. Clustering/.inputs",
        rejected_dir="mol_files/8. Clustering/.rejected",
    )


def plot_qed_histograms(
    df_imidazolones_druglike: pd.DataFrame,
    df_thiazolones_druglike: pd.DataFrame,
    df_imidazolones_nondruglike: pd.DataFrame,
    df_thiazolones_nondruglike: pd.DataFrame,
    qed_col: str = "QED",
    bins: int = 50,
    figsize: tuple[int, int] = (12, 4),
) -> tuple[object, tuple[object, object]]:
    """
    Plot QED histograms for rejected and accepted compounds.

    Parameters
    ----------
    df_imidazolones_druglike
        Accepted imidazolones.
    df_thiazolones_druglike
        Accepted thiazolones.
    df_imidazolones_nondruglike
        Rejected imidazolones.
    df_thiazolones_nondruglike
        Rejected thiazolones.
    qed_col
        QED column name.
    bins
        Histogram bin count.
    figsize
        Figure size.

    Returns
    -------
    tuple[object, tuple[object, object]]
        Figure and axes tuple.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to plot QED histograms. Install it in the 'coxibs' environment."
        ) from exc

    qed_rejected = pd.concat(
        [
            df_imidazolones_nondruglike[qed_col],
            df_thiazolones_nondruglike[qed_col],
        ],
        ignore_index=True,
    ).dropna()

    qed_accepted = pd.concat(
        [
            df_imidazolones_druglike[qed_col],
            df_thiazolones_druglike[qed_col],
        ],
        ignore_index=True,
    ).dropna()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(
        qed_rejected,
        bins=bins,
        color="#d95f02",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].set_title(f"Rejected compounds (n={len(qed_rejected):,})")
    axes[0].set_xlabel("QED")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, 1)

    axes[1].hist(
        qed_accepted,
        bins=bins,
        color="#1b9e77",
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_title(f"Accepted compounds (n={len(qed_accepted):,})")
    axes[1].set_xlabel("QED")
    axes[1].set_xlim(0, 1)

    fig.suptitle("QED distribution after bioavailability filter")
    plt.tight_layout()
    return fig, (axes[0], axes[1])
