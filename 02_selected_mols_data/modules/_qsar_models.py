from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._paths import INPUT_IC50, QSAR_DIR
from ._utils import _load_cache, _save_cache
from ._descriptors import add_rdkit_properties, DESCRIPTOR_COLUMNS
from ._io import _clean_smiles_column
from ._qsar_scoring import (
    load_chembl_ic50_summary,
    add_qsar_targets,
    make_stratification_bins,
    compute_centroid_distances,
    pic50_to_ic50_nm,
    compute_selectivity_index,
    compute_qsar_score,
)


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
    chembl_path: str | Path = INPUT_IC50,
    acceptance_rate: float = 0.01,
    minimum: int = 1000,
    output_dir: str | Path = QSAR_DIR,
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    print_report: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
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


# =============================================================================
# QSARPipeline
# =============================================================================


class QSARPipeline:
    """
    Trains Random Forest QSAR models on ChEMBL IC50 data and scores new
    compounds.

    Wraps _prepare_qsar_training_data(), _train_qsar_models(),
    _predict_qsar_for_series(), _select_top_by_qsar_score(), and
    run_qsar_winnow(). Models are loaded from cache if available.

    Parameters
    ----------
    chembl_path : str or Path, default=INPUT_IC50
        Path to ChEMBL IC50 summary CSV.
    acceptance_rate : float, default=0.01
        Fraction of top-scoring compounds to keep per series.
    minimum : int, default=1000
        Minimum number of compounds to keep per series.
    output_dir : str or Path, default=QSAR_DIR
        Directory for QSAR output files.
    use_cache : bool, default=True
        Load/save trained models from disk cache.
    cache_file : str or Path or None, default=None
        Explicit model cache path. Auto-generated if None.
    """

    def __init__(
        self,
        chembl_path: str | Path = INPUT_IC50,
        acceptance_rate: float = 0.01,
        minimum: int = 1000,
        output_dir: str | Path = QSAR_DIR,
        use_cache: bool = True,
        cache_file: str | Path | None = None,
    ) -> None:
        self.chembl_path = chembl_path
        self.acceptance_rate = acceptance_rate
        self.minimum = minimum
        self.output_dir = output_dir
        self.use_cache = use_cache
        self.cache_file = cache_file
        self._fitted: bool = False
        self._models: dict[str, object] | None = None

    def fit(self, force_retrain: bool = False) -> QSARPipeline:
        """
        Train or load cached QSAR models.

        If use_cache is True and a cached model file exists, loads from
        cache. Otherwise trains new models and optionally saves them.

        Parameters
        ----------
        force_retrain : bool, default=False
            Ignore cache and retrain models from scratch.

        Returns
        -------
        QSARPipeline
            Self, with self._models populated and self._fitted set to True.
        """
        output_dir = Path(self.output_dir)

        cache_path: Path | None = None
        if self.use_cache:
            if self.cache_file is None:
                cache_path = output_dir / ".cache" / "qsar_models.json.gz"
            else:
                cache_path = Path(self.cache_file)

        models: dict[str, object] | None = None
        if cache_path is not None and not force_retrain:
            models = _load_qsar_model_cache(cache_path)

        if models is None:
            df_chembl = _prepare_qsar_training_data(self.chembl_path)
            models = _train_qsar_models(df_chembl)
            if cache_path is not None:
                _save_qsar_model_cache(cache_path, df_chembl)

        self._models = models
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Predict COX-1/2 activity and selectivity for a compound series.

        Parameters
        ----------
        df : pd.DataFrame
            Compound DataFrame with SMILES column.
        label : str
            Series label for logging (e.g. 'Imidazolones', 'Thiazolones').

        Returns
        -------
        pd.DataFrame
            DataFrame with prediction columns appended.

        Raises
        ------
        RuntimeError
            If .fit() has not been called before .predict().
        """
        if not self._fitted or self._models is None:
            raise RuntimeError("Call .fit() before .predict()")
        return _predict_qsar_for_series(df, label, self._models)

    def winnow(
        self,
        df_imidazolones: pd.DataFrame,
        df_thiazolones: pd.DataFrame,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
        """
        Run the full QSAR winnow pipeline on both compound series.

        Convenience wrapper around run_qsar_winnow() using the configured
        parameters.

        Parameters
        ----------
        df_imidazolones : pd.DataFrame
            Drug-like imidazolone compounds.
        df_thiazolones : pd.DataFrame
            Drug-like thiazolone compounds.

        Returns
        -------
        tuple[dict[str, pd.DataFrame], dict[str, Path]]
            (outputs_dict, paths_dict) as returned by run_qsar_winnow().
        """
        return run_qsar_winnow(
            df_imidazolones,
            df_thiazolones,
            chembl_path=self.chembl_path,
            acceptance_rate=self.acceptance_rate,
            minimum=self.minimum,
            output_dir=self.output_dir,
            use_cache=self.use_cache,
            cache_file=self.cache_file,
        )
