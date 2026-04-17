from __future__ import annotations

import multiprocessing as mp
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED


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


def report_df_size(df: pd.DataFrame, label: str) -> None:
    """Print DataFrame size with a bracketed label."""
    print(f"[{label}] {len(df):,} rows")


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
    output_dir: str | Path = "mol_files/7. Clustering",
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
    output_dir
        Base directory for clustering stage outputs.
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
        "imidazolones_input": output_dir
        / f"Imidazolones_input_{len(df_imidazolones_input)}cmpds.csv",
        "thiazolones_input": output_dir
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

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

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
