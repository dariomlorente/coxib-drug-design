from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._paths import QED_DIR
from ._descriptors import REQUIRED_BIOAVAILABILITY_COLS


def filter_bioavailability(
    df: pd.DataFrame,
    qed_col: str = "QED",
    violation_col: str = "Violation",
    print_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    output_dir: str | Path = QED_DIR,
    print_report: bool = True,
) -> dict[str, Path]:
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


def plot_qed_histograms(
    df_imidazolones_druglike: pd.DataFrame,
    df_thiazolones_druglike: pd.DataFrame,
    df_imidazolones_nondruglike: pd.DataFrame,
    df_thiazolones_nondruglike: pd.DataFrame,
    qed_col: str = "QED",
    bins: int = 50,
    figsize: tuple[int, int] = (12, 4),
) -> tuple[object, tuple[object, object]]:
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


# =============================================================================
# BioavailabilityFilter
# =============================================================================


class BioavailabilityFilter:
    """
    Applies the 4-of-5 drug-likeness rules filter (Lipinski, Ghose, Egan,
    Muegge, Veber).

    Wraps filter_bioavailability() and plot_qed_histograms().

    Parameters
    ----------
    qed_col : str, default="QED"
        Column name for QED scores.
    violation_col : str, default="Violation"
        Column name for violation details.
    """

    def __init__(
        self,
        qed_col: str = "QED",
        violation_col: str = "Violation",
    ) -> None:
        self.qed_col = qed_col
        self.violation_col = violation_col

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply the 4-of-5 bioavailability rules filter.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with QED and bioavailability descriptor columns.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (accepted, rejected) DataFrames.
        """
        return filter_bioavailability(
            df,
            qed_col=self.qed_col,
            violation_col=self.violation_col,
        )

    def plot_qed(
        self,
        df_accepted: pd.DataFrame,
        df_rejected: pd.DataFrame,
    ) -> tuple[object, tuple[object, object]]:
        """
        Plot QED histograms for accepted vs rejected compounds.

        Delegates to plot_qed_histograms(), passing the same DataFrame to
        both the imidazolone and thiazolone slots.

        Parameters
        ----------
        df_accepted : pd.DataFrame
            Accepted compounds DataFrame.
        df_rejected : pd.DataFrame
            Rejected compounds DataFrame.

        Returns
        -------
        tuple[Figure, tuple[Axes, Axes]]
            Matplotlib figure and axes tuple.
        """
        return plot_qed_histograms(
            df_accepted,
            df_accepted,
            df_rejected,
            df_rejected,
            qed_col=self.qed_col,
        )
