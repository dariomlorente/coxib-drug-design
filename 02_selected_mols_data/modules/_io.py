from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rdkit import Chem

from ._paths import P1_OUTPUTS


def report_df_size(df: pd.DataFrame, label: str = "") -> None:
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
    filter_mode: str = "",
) -> Path:
    stage_dir = Path(stage_dir)
    prefix = stage_name if not filter_mode else f"{stage_name}_{filter_mode}"

    candidates: list[tuple[Path, int]] = []
    for path in stage_dir.glob(f"{prefix}_*cmpds.csv"):
        count = _extract_counted_suffix(path, prefix)
        if count is not None:
            candidates.append((path, count))

    if not candidates:
        raise ValueError(f"No file found for pattern '{prefix}_*cmpds.csv' in {stage_dir}.")

    return max(candidates, key=lambda item: (item[1], item[0].stat().st_mtime))[0]


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


def load_generated_product_sets(
    imidazolones_dir: str | Path = P1_OUTPUTS,
    thiazolones_dir: str | Path = P1_OUTPUTS,
    filter_mode: str = "",
    print_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    imidazolones_path = find_latest_stage_csv(imidazolones_dir, "Imidazolones", filter_mode)
    thiazolones_path = find_latest_stage_csv(thiazolones_dir, "Thiazolones", filter_mode)

    df_imidazolones = pd.read_csv(imidazolones_path)
    df_thiazolones = pd.read_csv(thiazolones_path)

    mode_suffix = f"_{filter_mode}" if filter_mode else ""
    expected_imi = _extract_counted_suffix(imidazolones_path, f"Imidazolones{mode_suffix}")
    expected_thi = _extract_counted_suffix(thiazolones_path, f"Thiazolones{mode_suffix}")

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
                f"\u26a0\ufe0f [LoadProducts] Imidazolones filename count ({expected_imi:,}) "
                f"does not match loaded rows ({len(df_imidazolones):,})"
            )
        if expected_thi is not None and len(df_thiazolones) != expected_thi:
            print(
                f"\u26a0\ufe0f [LoadProducts] Thiazolones filename count ({expected_thi:,}) "
                f"does not match loaded rows ({len(df_thiazolones):,})"
            )

    return df_imidazolones, df_thiazolones, imidazolones_path, thiazolones_path
