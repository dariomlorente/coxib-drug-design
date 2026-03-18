from __future__ import annotations

import shutil
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import pandas as pd


_STAGE_REGISTRY: dict[str, str] = {
    "Aldehydes": "mol_files/2. Building Blocks",
    "Carboxylics": "mol_files/2. Building Blocks",
    "Amines": "mol_files/2. Building Blocks",
    "Oxazolones": "mol_files/3. Oxazolones",
    "Imidazolones": "mol_files/4. Imidazolones",
    "Thiazolones": "mol_files/5. Thiazolones",
}


def stage_path(stage_name: str, suffix: str = "") -> Path:
    """
    Resolve path for a pipeline stage file.
    
    Args:
        stage_name: Stage name (e.g., "Oxazolones", "Oxazolones_rejected")
        suffix: Optional suffix (e.g., "_checkpoint", "_rejected")
    
    Returns:
        Path to the stage file in the appropriate mol_files subdirectory.
    
    Examples:
        stage_path("Oxazolones") 
        # → mol_files/3. Oxazolones/Oxazolones.csv
        
        stage_path("Oxazolones_rejected")
        # → mol_files/3. Oxazolones/Oxazolones_rejected.csv
        
        stage_path("Oxazolones", "_checkpoint")
        # → mol_files/3. Oxazolones/Oxazolones_checkpoint.csv
    """
    is_rejected = stage_name.endswith("_rejected")
    base_name = stage_name.replace("_rejected", "")
    
    if base_name not in _STAGE_REGISTRY:
        raise ValueError(f"Unknown stage name: {base_name}. "
                        f"Known stages: {list(_STAGE_REGISTRY.keys())}")
    
    base_dir = Path(_STAGE_REGISTRY[base_name])
    
    if suffix:
        file_name = f"{base_name}{suffix}.csv"
    elif is_rejected:
        file_name = f"{stage_name}.csv"
    else:
        file_name = f"{stage_name}.csv"
    
    return base_dir / file_name


def checkpoint_path(stage_name: str) -> Path:
    """
    Get path for a checkpoint file.
    
    Args:
        stage_name: Stage name (e.g., "Oxazolones")
    
    Returns:
        Path like mol_files/3. Oxazolones/Oxazolones_checkpoint.csv
    """
    return stage_path(stage_name, "_checkpoint")


def rejected_path(stage_name: str, suffix: str = "") -> Path:
    """
    Path to rejected compounds in .rejected/ subfolder.
    
    Args:
        stage_name: Stage name (e.g., "Oxazolones", "Imidazolones")
        suffix: Optional suffix (e.g., "_brenkpains")
    
    Returns:
        Path like mol_files/3. Oxazolones/.rejected/Oxazolones_rejected.csv
    
    Examples:
        rejected_path("Oxazolones")
        # → mol_files/3. Oxazolones/.rejected/Oxazolones_rejected.csv
        
        rejected_path("Imidazolones", "_brenkpains")
        # → mol_files/4. Imidazolones/.rejected/Imidazolones_brenkpains_rejected.csv
    """
    if stage_name not in _STAGE_REGISTRY:
        raise ValueError(f"Unknown stage name: {stage_name}. "
                        f"Known stages: {list(_STAGE_REGISTRY.keys())}")
    
    base_dir = Path(_STAGE_REGISTRY[stage_name])
    suffix_str = f"_{suffix}" if suffix else ""
    return base_dir / ".rejected" / f"{stage_name}{suffix_str}_rejected.csv"


def init_stage_dirs() -> None:
    """Create all stage directories, .cache/ and .rejected/ subdirectories."""
    # Get unique stage directories from the registry
    stage_dirs = set(Path(p) for p in _STAGE_REGISTRY.values())
    for d in stage_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / ".cache").mkdir(exist_ok=True)
        (d / ".rejected").mkdir(exist_ok=True)


def _get_csv_row_count(csv_path: Path) -> int:
    """Get row count from CSV without loading full file."""
    try:
        with open(csv_path, 'r') as f:
            return sum(1 for _ in f) - 1
    except (OSError, IOError):
        return -1


def load_or_run(
    compute_fn: Callable[[], pd.DataFrame],
    output_csv: str | Path,
    checkpoint_csv: str | Path | None = None,
    force_recompute: bool = False,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Load DataFrame from CSV if it exists, otherwise compute it.
    
    Use this for reactions that produce a single output DataFrame.
    Supports checkpoint-based resume: if compute_fn crashes mid-execution,
    the checkpoint file survives and can be resumed on next run.
    
    The compute_fn should accept **kwargs with checkpoint_csv parameter.
    
    Args:
        compute_fn: Function that returns the DataFrame to cache.
            Can accept checkpoint_csv as a keyword argument.
        output_csv: Path to save/load the final CSV.
        checkpoint_csv: Path for temporary checkpoint file during computation.
            If None, derived from output_csv path.
        force_recompute: If True, ignore existing CSV and recompute.
        print_report: Print loading/computing messages.
    
    Returns:
        DataFrame from cache or compute_fn result.
    
    Checkpoint flow:
        1. If output_csv exists and not force_recompute → load and return
        2. If checkpoint_csv exists → load checkpoint, compute from resume point
        3. Compute normally, writing to checkpoint after each chunk
        4. On success → move checkpoint to output_csv, delete checkpoint
    """
    output_path = Path(output_csv)
    
    if checkpoint_csv is None:
        checkpoint_path_obj = output_path.parent / f"{output_path.stem}_checkpoint.csv"
    else:
        checkpoint_path_obj = Path(checkpoint_csv)
    
    if not force_recompute and output_path.exists():
        rows = _get_csv_row_count(output_path)
        if print_report:
            print(f"[load_or_run] Loading {output_path.name} ({rows:,} rows) ✓")
        return pd.read_csv(output_path)
    
    if checkpoint_path_obj.exists() and not force_recompute:
        rows = _get_csv_row_count(checkpoint_path_obj)
        if print_report:
            print(f"[load_or_run] Resuming from checkpoint {checkpoint_path_obj.name} ({rows:,} rows)")
        return pd.read_csv(checkpoint_path_obj)
    
    if print_report:
        print(f"[load_or_run] Computing {output_path.name}...")
    
    df = compute_fn(checkpoint_csv=str(checkpoint_path_obj))
    
    if print_report:
        print(f"[load_or_run] Saving {output_path.name} ({len(df):,} rows)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    if checkpoint_path_obj.exists() and checkpoint_path_obj != output_path:
        checkpoint_path_obj.unlink()
    
    return df


def load_or_filter(
    compute_fn: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
    accepted_csv: str | Path,
    rejected_csv: str | Path,
    force_recompute: bool = False,
    print_report: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load accepted/rejected DataFrames from CSVs if they exist, otherwise compute.
    
    Use this for filters that produce accepted and rejected DataFrames.
    
    Args:
        compute_fn: Function that returns (accepted_df, rejected_df) tuple.
        accepted_csv: Path to save/load the accepted compounds CSV.
        rejected_csv: Path to save/load the rejected compounds CSV.
        force_recompute: If True, ignore existing CSVs and recompute.
        print_report: Print loading/computing messages.
    
    Returns:
        Tuple of (accepted_df, rejected_df).
    """
    accepted_path = Path(accepted_csv)
    rejected_path = Path(rejected_csv)
    
    if (not force_recompute and accepted_path.exists() and rejected_path.exists()):
        acc_rows = _get_csv_row_count(accepted_path)
        rej_rows = _get_csv_row_count(rejected_path)
        if print_report:
            print(f"[load_or_filter] Loading {accepted_path.name} ({acc_rows:,} rows) "
                  f"+ {rejected_path.name} ({rej_rows:,} rows) ✓")
        return pd.read_csv(accepted_path), pd.read_csv(rejected_path)
    
    if print_report:
        print(f"[load_or_filter] Computing {accepted_path.name}...")
    
    df_accepted, df_rejected = compute_fn()
    
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    df_accepted.to_csv(accepted_path, index=False)
    df_rejected.to_csv(rejected_path, index=False)
    
    if print_report:
        print(f"[load_or_filter] Saved {accepted_path.name} ({len(df_accepted):,} accepted) "
              f"+ {rejected_path.name} ({len(df_rejected):,} rejected)")
    
    return df_accepted, df_rejected


def save_dataframe(
    df: pd.DataFrame,
    output_csv: str | Path,
    print_report: bool = True,
) -> None:
    """
    Save DataFrame to CSV with consistent reporting.
    
    Args:
        df: DataFrame to save.
        output_csv: Path to save the CSV.
        print_report: Print save message.
    """
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if print_report:
        print(f"[save_dataframe] Saved {path.name} ({len(df):,} rows)")
