from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from rdkit import Chem

DEFAULT_TARGET_IDS = ("CHEMBL221", "CHEMBL230")
DEFAULT_CHUNK_SIZE = 200_000
DEFAULT_BAO_LABELS = (
    "single protein format",
    "cell-free format",
    "cell-based format",
)
DEFAULT_IC50_SUMMARY = "protein_files/IC50s/ChEMBL_IC50nSI.csv"
DEFAULT_IC50_OUTPUT_DIR = "mol_files/8. From DrugBank"
DEFAULT_COXIB_SMARTS = "[O:51]=[C:5]1[SX2,NX3:10][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]"


def _has_header(path: Path) -> bool:
    """Return True if a ChEMBL export file includes the header row."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first_line = handle.readline()
    except OSError as exc:
        raise ValueError(f"Could not read {path}.") from exc
    return "Target ChEMBL ID" in first_line and "Standard Type" in first_line


def _get_reference_columns(csv_files: Sequence[Path]) -> list[str]:
    """Read the column names from the first file that includes headers."""
    header_source = next((path for path in csv_files if _has_header(path)), None)
    if header_source is None:
        raise ValueError("No header row found in the ChEMBL CSV files.")

    columns = pd.read_csv(
        header_source,
        sep=";",
        nrows=0,
        low_memory=False,
    ).columns.str.strip().tolist()
    if not columns:
        raise ValueError(f"No columns detected in {header_source}.")
    return columns


def _filter_ic50_rows(
    df: pd.DataFrame,
    allowed_bao_labels: Sequence[str],
    units_substring: str = "nM",
    relation_substring: str = "=",
) -> pd.DataFrame:
    """Filter IC50 rows by units, BAO label, and relation criteria."""
    required_cols = {
        "Molecule ChEMBL ID",
        "Smiles",
        "Standard Value",
        "Standard Units",
        "Standard Relation",
        "BAO Label",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    bao_set = {label.strip().lower() for label in allowed_bao_labels}
    units_mask = df["Standard Units"].fillna("").str.contains(
        units_substring,
        case=False,
        na=False,
    )
    relation_mask = df["Standard Relation"].fillna("").str.contains(
        relation_substring,
        regex=False,
        na=False,
    )
    bao_mask = (
        df["BAO Label"]
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(bao_set)
    )

    filtered = df.loc[units_mask & relation_mask & bao_mask].copy()
    filtered["Standard Value"] = pd.to_numeric(
        filtered["Standard Value"],
        errors="coerce",
    )
    filtered = filtered.dropna(subset=["Standard Value"])
    return filtered


def _select_smiles(df: pd.DataFrame) -> pd.Series:
    """Select the first non-empty SMILES per ChEMBL ID."""
    smiles = df["Smiles"].replace("", pd.NA)
    return (
        df.assign(_smiles=smiles)
        .dropna(subset=["_smiles"])
        .groupby("Molecule ChEMBL ID", dropna=True)["_smiles"]
        .first()
    )


def merge_ic50_summary(
    cox1_csv: str | Path = "protein_files/IC50s/CHEMBL221_ic50.csv",
    cox2_csv: str | Path = "protein_files/IC50s/CHEMBL230_ic50.csv",
    output_csv: str | Path = "protein_files/IC50s/ChEMBL_IC50nSI.csv",
    allowed_bao_labels: Sequence[str] = DEFAULT_BAO_LABELS,
    units_substring: str = "nM",
    relation_substring: str = "=",
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Merge IC50 summaries for COX1 and COX2 into a single table.

    Parameters:
        cox1_csv: Path to CHEMBL221 IC50 CSV.
        cox2_csv: Path to CHEMBL230 IC50 CSV.
        output_csv: Path to output CSV with merged statistics.
        allowed_bao_labels: Allowed BAO Label values for filtering.
        units_substring: Substring that must appear in Standard Units.
        relation_substring: Substring that must appear in Standard Relation.
        print_report: Print progress and summary information.

    Returns:
        DataFrame with merged IC50 median/mean columns.
    """
    cox1_path = Path(cox1_csv)
    cox2_path = Path(cox2_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cox1_path.exists():
        raise FileNotFoundError(f"Missing COX1 IC50 file: {cox1_path}")
    if not cox2_path.exists():
        raise FileNotFoundError(f"Missing COX2 IC50 file: {cox2_path}")

    cox1_df = pd.read_csv(cox1_path, sep=";", dtype=str, low_memory=False)
    cox2_df = pd.read_csv(cox2_path, sep=";", dtype=str, low_memory=False)
    cox1_df.columns = cox1_df.columns.str.strip()
    cox2_df.columns = cox2_df.columns.str.strip()

    cox1_filtered = _filter_ic50_rows(
        cox1_df,
        allowed_bao_labels,
        units_substring=units_substring,
        relation_substring=relation_substring,
    )
    cox2_filtered = _filter_ic50_rows(
        cox2_df,
        allowed_bao_labels,
        units_substring=units_substring,
        relation_substring=relation_substring,
    )

    cox1_stats = (
        cox1_filtered.groupby("Molecule ChEMBL ID", dropna=True)["Standard Value"]
        .agg(["median", "mean"])
        .rename(
            columns={
                "median": "IC50_COX1_median",
                "mean": "IC50_COX1_mean",
            }
        )
    )
    cox2_stats = (
        cox2_filtered.groupby("Molecule ChEMBL ID", dropna=True)["Standard Value"]
        .agg(["median", "mean"])
        .rename(
            columns={
                "median": "IC50_COX2_median",
                "mean": "IC50_COX2_mean",
            }
        )
    )

    merged = (
        cox1_stats.reset_index()
        .rename(columns={"Molecule ChEMBL ID": "ChEMBL_ID"})
        .merge(
            cox2_stats.reset_index().rename(
                columns={"Molecule ChEMBL ID": "ChEMBL_ID"}
            ),
            on="ChEMBL_ID",
            how="outer",
        )
    )

    smiles_cox1 = _select_smiles(cox1_filtered)
    smiles_cox2 = _select_smiles(cox2_filtered)
    smiles = smiles_cox2.to_frame("SMILES").combine_first(
        smiles_cox1.to_frame("SMILES")
    )
    merged = merged.merge(smiles, left_on="ChEMBL_ID", right_index=True, how="left")

    ordered_cols = [
        "ChEMBL_ID",
        "SMILES",
        "IC50_COX1_median",
        "IC50_COX2_median",
        "IC50_COX1_mean",
        "IC50_COX2_mean",
    ]
    for col in ordered_cols:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[ordered_cols]

    merged.to_csv(output_path, index=False, sep=";")

    if print_report:
        print("[IC50] Summary saved.")
        print(f"[IC50] COX1 rows after filters: {len(cox1_filtered)}")
        print(f"[IC50] COX2 rows after filters: {len(cox2_filtered)}")
        print(f"[IC50] Unique IDs: {merged['ChEMBL_ID'].nunique()} -> {output_path}")

    return merged


def merge_ic50_into_csv(
    input_csv: str | Path,
    ic50_summary_csv: str | Path = DEFAULT_IC50_SUMMARY,
    output_dir: str | Path = DEFAULT_IC50_OUTPUT_DIR,
    smiles_col: str = "SMILES",
    price_col: str = "PriceMol",
    qed_col: str = "QED",
    violation_col: str = "Violation",
    print_report: bool = True,
) -> Path:
    """
    Merge IC50 summary metrics into a compound CSV by matching SMILES.

    Parameters:
        input_csv: Input CSV path (comma-delimited).
        ic50_summary_csv: Path to ChEMBL_IC50nSI.csv (semicolon-delimited).
        output_dir: Directory to write IC50s_<input>.csv.
        smiles_col: SMILES column name in the input CSV.
        price_col: Price column used to deduplicate by cheapest SMILES.
        qed_col: Column after which IC50 columns are inserted if present.
        violation_col: Column before which IC50 columns are inserted if present.
        print_report: Print progress and summary information.

    Returns:
        Path to the merged output CSV.
    """
    input_path = Path(input_csv)
    ic50_path = Path(ic50_summary_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not ic50_path.exists():
        raise FileNotFoundError(f"IC50 summary not found: {ic50_path}")

    df = pd.read_csv(input_path, sep=",", low_memory=False)
    df.columns = df.columns.str.strip()
    if smiles_col not in df.columns:
        raise ValueError(f"Missing column '{smiles_col}' in {input_path}.")
    if price_col not in df.columns:
        raise ValueError(f"Missing column '{price_col}' in {input_path}.")

    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    price_values = pd.to_numeric(df[price_col], errors="coerce")
    df["_price_sort"] = price_values.fillna(float("inf"))
    df = (
        df.sort_values([smiles_col, "_price_sort"], ascending=[True, True])
        .drop_duplicates(subset=[smiles_col], keep="first")
        .drop(columns=["_price_sort"])
    )

    ic50_df = pd.read_csv(ic50_path, sep=";", dtype=str, low_memory=False)
    ic50_df.columns = ic50_df.columns.str.strip()
    ic50_df["SMILES"] = ic50_df["SMILES"].fillna("").str.strip()
    ic50_df = ic50_df[ic50_df["SMILES"].ne("")].copy()

    ic50_cols = [
        "IC50_COX1_median",
        "IC50_COX2_median",
        "IC50_COX1_mean",
        "IC50_COX2_mean",
    ]
    for col in ic50_cols:
        ic50_df[col] = pd.to_numeric(ic50_df[col], errors="coerce")

    ic50_df["_non_null"] = ic50_df[ic50_cols].notna().sum(axis=1)
    ic50_df = (
        ic50_df.sort_values(["SMILES", "_non_null"], ascending=[True, False])
        .drop_duplicates(subset=["SMILES"], keep="first")
        .drop(columns=["_non_null"])
    )

    merged = df.merge(ic50_df[["SMILES"] + ic50_cols], on="SMILES", how="inner")

    for col in ic50_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["SI_median"] = pd.NA
    merged["SI_mean"] = pd.NA
    median_mask = (
        merged["IC50_COX1_median"].notna()
        & merged["IC50_COX2_median"].notna()
        & merged["IC50_COX2_median"].ne(0)
    )
    mean_mask = (
        merged["IC50_COX1_mean"].notna()
        & merged["IC50_COX2_mean"].notna()
        & merged["IC50_COX2_mean"].ne(0)
    )
    merged.loc[median_mask, "SI_median"] = (
        merged.loc[median_mask, "IC50_COX1_median"]
        / merged.loc[median_mask, "IC50_COX2_median"]
    )
    merged.loc[mean_mask, "SI_mean"] = (
        merged.loc[mean_mask, "IC50_COX1_mean"]
        / merged.loc[mean_mask, "IC50_COX2_mean"]
    )

    insert_cols = [
        "IC50_COX1_median",
        "IC50_COX2_median",
        "SI_median",
        "IC50_COX1_mean",
        "IC50_COX2_mean",
        "SI_mean",
    ]
    for col in insert_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    ordered_cols = [col for col in merged.columns if col not in insert_cols]
    if qed_col in ordered_cols and violation_col in ordered_cols:
        qed_idx = ordered_cols.index(qed_col)
        violation_idx = ordered_cols.index(violation_col)
        if qed_idx < violation_idx:
            ordered_cols = (
                ordered_cols[: qed_idx + 1]
                + insert_cols
                + ordered_cols[qed_idx + 1 :]
            )
        else:
            ordered_cols = ordered_cols + insert_cols
    else:
        ordered_cols = ordered_cols + insert_cols

    merged = merged[ordered_cols]

    output_file = output_path / f"IC50s_{input_path.name}"
    merged.to_csv(output_file, index=False)

    if print_report:
        print(f"[IC50] Merged {input_path.name} -> {output_file}")
        print(f"[IC50] Rows kept after SMILES match: {len(merged)}")

    return output_file


def find_chembl_ids_by_smarts(
    input_csv: str | Path = DEFAULT_IC50_SUMMARY,
    output_csv: str | Path = "protein_files/IC50s/ChEMBL_IC50nSI_smarts.csv",
    smarts: str = DEFAULT_COXIB_SMARTS,
    smiles_col: str = "SMILES",
    id_col: str = "ChEMBL_ID",
    print_report: bool = True,
) -> Path:
    """
    Export ChEMBL entries whose SMILES match a SMARTS substructure.

    Parameters:
        input_csv: Path to the ChEMBL IC50 summary CSV.
        output_csv: Path to write the matching rows.
        smarts: SMARTS pattern to match.
        smiles_col: Column name containing SMILES strings.
        id_col: Column name containing ChEMBL IDs.
        print_report: Print summary information.

    Returns:
        Path to the output CSV.
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing IC50 summary file: {input_path}")

    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ValueError("Invalid SMARTS pattern.")

    df = pd.read_csv(input_path, sep=";", dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    if smiles_col not in df.columns:
        raise ValueError(f"Missing column '{smiles_col}' in {input_path}.")
    if id_col not in df.columns:
        raise ValueError(f"Missing column '{id_col}' in {input_path}.")

    matches: list[dict[str, str]] = []
    invalid_smiles = 0
    checked = 0

    for _, row in df.iterrows():
        chembl_id = "" if pd.isna(row[id_col]) else str(row[id_col]).strip()
        smiles = "" if pd.isna(row[smiles_col]) else str(row[smiles_col]).strip()
        if not chembl_id or not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles += 1
            continue
        checked += 1
        if mol.HasSubstructMatch(pattern):
            matches.append(
                {
                    "ChEMBL_ID": chembl_id,
                    "SMILES": smiles,
                    "IC50_COX1_median": row.get("IC50_COX1_median", ""),
                    "IC50_COX2_median": row.get("IC50_COX2_median", ""),
                    "IC50_COX1_mean": row.get("IC50_COX1_mean", ""),
                    "IC50_COX2_mean": row.get("IC50_COX2_mean", ""),
                }
            )

    out_df = pd.DataFrame(matches)
    if out_df.empty:
        out_df = pd.DataFrame(
            columns=[
                "ChEMBL_ID",
                "SMILES",
                "IC50_COX1_median",
                "IC50_COX2_median",
                "SI_median",
                "IC50_COX1_mean",
                "IC50_COX2_mean",
                "SI_mean",
            ]
        )
        out_df.to_csv(output_path, index=False, sep=";")
        if print_report:
            print(f"[SMARTS] Checked molecules: {checked}")
            print(f"[SMARTS] Invalid SMILES: {invalid_smiles}")
            print("[SMARTS] Matches: 0")
            print(f"[SMARTS] Output saved to {output_path}")
        return output_path

    for col in (
        "IC50_COX1_median",
        "IC50_COX2_median",
        "IC50_COX1_mean",
        "IC50_COX2_mean",
    ):
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce")

    out_df["SI_median"] = pd.NA
    out_df["SI_mean"] = pd.NA
    median_mask = (
        out_df["IC50_COX1_median"].notna()
        & out_df["IC50_COX2_median"].notna()
        & out_df["IC50_COX2_median"].ne(0)
    )
    mean_mask = (
        out_df["IC50_COX1_mean"].notna()
        & out_df["IC50_COX2_mean"].notna()
        & out_df["IC50_COX2_mean"].ne(0)
    )
    out_df.loc[median_mask, "SI_median"] = (
        out_df.loc[median_mask, "IC50_COX1_median"]
        / out_df.loc[median_mask, "IC50_COX2_median"]
    )
    out_df.loc[mean_mask, "SI_mean"] = (
        out_df.loc[mean_mask, "IC50_COX1_mean"]
        / out_df.loc[mean_mask, "IC50_COX2_mean"]
    )

    out_df = out_df[
        [
            "ChEMBL_ID",
            "SMILES",
            "IC50_COX1_median",
            "IC50_COX2_median",
            "SI_median",
            "IC50_COX1_mean",
            "IC50_COX2_mean",
            "SI_mean",
        ]
    ]
    out_df.to_csv(output_path, index=False, sep=";")

    if print_report:
        print(f"[SMARTS] Checked molecules: {checked}")
        print(f"[SMARTS] Invalid SMILES: {invalid_smiles}")
        print(f"[SMARTS] Matches: {len(out_df)}")
        print(f"[SMARTS] Output saved to {output_path}")

    return output_path


def extract_ic50_by_target(
    input_dir: str | Path = "protein_files/ChEMBL",
    output_dir: str | Path = "protein_files/IC50s",
    target_ids: Sequence[str] = DEFAULT_TARGET_IDS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = True,
    print_report: bool = True,
) -> tuple[dict[str, int], dict[str, Path]]:
    """
    Extract IC50 rows for selected ChEMBL targets into per-target CSV files.

    Parameters:
        input_dir: Directory containing the ChEMBL CSV exports.
        output_dir: Directory to write per-target IC50 CSVs.
        target_ids: Target ChEMBL IDs to extract.
        chunk_size: Number of rows per chunk to stream.
        overwrite: Remove existing output files before writing.
        print_report: Print progress and summary information.

    Returns:
        Tuple of (row_counts, output_paths) where row_counts maps target ID to
        number of IC50 rows written, and output_paths maps target ID to output path.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_path}.")

    target_ids = tuple(target_ids)
    output_paths = {
        target_id: output_path / f"{target_id}_ic50.csv" for target_id in target_ids
    }

    if overwrite:
        for path in output_paths.values():
            if path.exists():
                path.unlink()

    reference_columns = _get_reference_columns(csv_files)
    required_cols = {"Target ChEMBL ID", "Standard Type", "Standard Value"}
    row_counts = {target_id: 0 for target_id in target_ids}

    for csv_path in csv_files:
        if print_report:
            print(f"[IC50] Reading {csv_path.name}")

        read_kwargs = dict(
            sep=";",
            dtype=str,
            chunksize=chunk_size,
            low_memory=False,
        )
        if not _has_header(csv_path):
            read_kwargs.update({"header": None, "names": reference_columns})

        for chunk in pd.read_csv(csv_path, **read_kwargs):
            chunk.columns = chunk.columns.str.strip()
            missing = required_cols - set(chunk.columns)
            if missing:
                raise ValueError(
                    f"Missing columns in {csv_path.name}: {sorted(missing)}"
                )

            target_col = chunk["Target ChEMBL ID"].fillna("").str.strip()
            type_col = chunk["Standard Type"].fillna("").str.strip().str.upper()
            value_col = chunk["Standard Value"].fillna("").str.strip()

            base_mask = type_col.eq("IC50") & value_col.ne("")
            if not base_mask.any():
                continue

            for target_id, out_file in output_paths.items():
                mask = base_mask & target_col.eq(target_id)
                if not mask.any():
                    continue

                filtered = chunk.loc[mask]
                filtered.to_csv(
                    out_file,
                    mode="a",
                    index=False,
                    sep=";",
                    header=not out_file.exists(),
                )
                row_counts[target_id] += len(filtered)

    if print_report:
        print("[IC50] Done.")
        for target_id in target_ids:
            print(
                f"[IC50] {target_id}: {row_counts[target_id]} rows -> {output_paths[target_id]}"
            )

    return row_counts, output_paths
