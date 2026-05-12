from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._paths import CLUSTERING_INPUTS, CLUSTERING_REJECTED


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
    if price_col not in df.columns:
        raise ValueError(f"Missing column '{price_col}'.")

    acceptance_rate = _normalize_acceptance_rate(acceptance_rate)
    max_sample_size = _validate_max_sample_size(max_sample_size)

    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")

    if out[price_col].isna().any() and print_report:
        print(
            f"\u26a0\ufe0f [apply_price_controls] Found {int(out[price_col].isna().sum()):,} rows "
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
    input_dir: str | Path = CLUSTERING_INPUTS,
    rejected_dir: str | Path = CLUSTERING_REJECTED,
    print_report: bool = True,
) -> dict[str, Path]:
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


# =============================================================================
# PriceController
# =============================================================================


class PriceController:
    """
    Applies price-based filtering and acceptance-rate controls.

    Wraps apply_price_controls(). Stores max_price, acceptance_rate,
    max_sample_size, and price_col at construction time so the same
    controller can be applied consistently to both Imidazolones and
    Thiazolones.

    Parameters
    ----------
    max_price : float or None, optional
        Maximum price per molecule. Default: None.
    acceptance_rate : float or None, optional
        Fraction of compounds to keep (0–1). Default: None.
    max_sample_size : int or None, optional
        Hard cap on accepted compounds. Default: None.
    price_col : str, optional
        Column name for prices. Default: "PriceMol".
    """

    def __init__(
        self,
        max_price: float | None = None,
        acceptance_rate: float | None = None,
        max_sample_size: int | None = None,
        price_col: str = "PriceMol",
    ) -> None:
        self.max_price = max_price
        self.acceptance_rate = acceptance_rate
        self.max_sample_size = max_sample_size
        self.price_col = price_col

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply price-based filtering and acceptance-rate controls.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with a ``price_col`` column.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (accepted, rejected) DataFrames.
        """
        return apply_price_controls(
            df,
            price_col=self.price_col,
            max_price=self.max_price,
            acceptance_rate=self.acceptance_rate,
            max_sample_size=self.max_sample_size,
        )


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
        input_dir=CLUSTERING_INPUTS,
        rejected_dir=CLUSTERING_REJECTED,
    )
