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

def stage_path(
    stage_name: str,
    row_count: int | None = None,
    filter_mode: str | None = None,
) -> Path:
    """
    Generate path for a pipeline stage file with optional row count and filter suffix.
    
    File naming convention:
      - Raw reaction output: {Stage}_raw_{N}cmpds.csv
      - Filtered output: {Stage}_{filter}_{N}cmpds.csv
    
    Args:
        stage_name: Stage name (e.g., "Oxazolones", "Imidazolones")
        row_count: Optional row count to include in filename (e.g., 5655971)
        filter_mode: Optional filter suffix ("veber", "brenkpains", "druglike", etc.)
    
    Returns:
        Path to the stage file in the appropriate mol_files subdirectory.
    
    Examples:
        stage_path("Oxazolones", row_count=5220, filter_mode="raw")
        # → mol_files/3. Oxazolones/Oxazolones_raw_5220cmpds.csv
        
        stage_path("Oxazolones", row_count=4087, filter_mode="veber")
        # → mol_files/3. Oxazolones/Oxazolones_veber_4087cmpds.csv
        
        stage_path("Imidazolones", row_count=118151, filter_mode="brenkpains")
        # → mol_files/4. Imidazolones/Imidazolones_brenkpains_118151cmpds.csv
    """
    if stage_name not in _STAGE_REGISTRY:
        raise ValueError(f"Unknown stage name: {stage_name}. "
                        f"Known stages: {list(_STAGE_REGISTRY.keys())}")
    
    if filter_mode is not None and filter_mode not in ("raw", "veber", "brenkpains", "druglike"):
        raise ValueError(f"Unknown filter_mode: {filter_mode}. "
                        f"Known modes: raw, veber, brenkpains")
    
    base_dir = Path(_STAGE_REGISTRY[stage_name])
    
    if filter_mode == "raw":
        file_name = f"{stage_name}_raw"
    else:
        file_name = f"{stage_name}_{filter_mode}" if filter_mode else stage_name
    
    if row_count is not None:
        file_name = f"{file_name}_{row_count}cmpds.csv"
    else:
        file_name = f"{file_name}.csv"
    
    return base_dir / file_name


def checkpoint_path(stage_name: str) -> Path:
    """
    Get path for a checkpoint metadata file (JSON format).
    
    Args:
        stage_name: Stage name (e.g., "Oxazolones")
    
    Returns:
        Path like mol_files/3. Oxazolones/.cache/Oxazolones_checkpoint.json
    """
    base_dir = Path(_STAGE_REGISTRY[stage_name])
    return base_dir / ".cache" / f"{stage_name}_checkpoint.json"


def rejected_path(stage_name: str, filter_type: str = "", row_count: int | None = None) -> Path:
    """
    Generate path for rejected compounds in .rejected/ subfolder.
    
    Args:
        stage_name: Stage name (e.g., "Oxazolones", "Imidazolones")
        filter_type: Optional filter type suffix (e.g., "veber", "brenkpains")
        row_count: Optional row count to include in filename
    
    Returns:
        Path to rejected compounds in .rejected/ folder with row count
    
    Examples:
        rejected_path("Oxazolones", row_count=5655971)
        # → mol_files/3. Oxazolones/.rejected/Oxazolones_rejected_veber_5655971cmpds.csv
        
        rejected_path("Imidazolones", "brenkpains", row_count=100)
        # → mol_files/4. Imidazolones/.rejected/Imidazolones_rejected_brenkpains_100cmpds.csv
    """
    if stage_name not in _STAGE_REGISTRY:
        raise ValueError(f"Unknown stage name: {stage_name}. "
                        f"Known stages: {list(_STAGE_REGISTRY.keys())}")
    
    base_dir = Path(_STAGE_REGISTRY[stage_name])
    
    # Build the suffix: _rejected, _rejected_veber, _rejected_brenkpains
    if filter_type:
        suffix = f"_rejected_{filter_type}"
    else:
        suffix = "_rejected"
    
    if row_count is not None:
        file_name = f"{stage_name}{suffix}_{row_count}cmpds.csv"
    else:
        file_name = f"{stage_name}{suffix}.csv"
    
    return base_dir / ".rejected" / file_name


def init_stage_dirs() -> None:
    """Create all stage directories, .cache/ and .rejected/ subdirectories."""
    stage_dirs = set(Path(p) for p in _STAGE_REGISTRY.values())
    for d in stage_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / ".cache").mkdir(exist_ok=True)
        (d / ".rejected").mkdir(exist_ok=True)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Write DataFrame to CSV atomically using temp file + rename.
    
    This prevents corruption if the write is interrupted.
    """
    temp_path = path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    # Atomic rename on POSIX systems
    temp_path.rename(path)


def _append_to_csv_atomic(df: pd.DataFrame, path: Path, mode: str = "a") -> None:
    """
    Append DataFrame to CSV atomically.
    
    For 'w' mode, creates new file. For 'a' mode, loads existing, appends, 
    and writes atomically.
    """
    if mode == "w" or not path.exists():
        _atomic_write_csv(df, path)
    else:
        # Load existing, append, then atomically rewrite
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
        _atomic_write_csv(combined, path)


def _get_csv_row_count(csv_path: Path) -> int:
    """Get row count from CSV without loading full file."""
    try:
        with open(csv_path, 'r') as f:
            return sum(1 for _ in f) - 1
    except (OSError, IOError):
        return -1


def load_or_run(
    compute_fn: Callable[..., pd.DataFrame],
    output_csv: str | Path,
    stage_name: str | None = None,
    force_recompute: bool = False,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Load DataFrame from CSV if it exists, otherwise compute it.
    
    Use this for reactions that produce a single output DataFrame.
    Supports robust checkpoint-based resume using JSON metadata.
    
    Args:
        compute_fn: Function that returns the DataFrame to cache.
            Should accept `checkpoint_manager` keyword argument.
        output_csv: Path to save/load the final CSV.
        stage_name: Stage name for checkpoint management.
            If None, inferred from output_csv stem.
        force_recompute: If True, ignore existing CSV and recompute.
        print_report: Print loading/computing messages.
    
    Returns:
        DataFrame from cache or compute_fn result.
    
    Checkpoint flow:
        1. If output_csv exists and not force_recompute → load and return
        2. If checkpoint.json exists with status=complete → load from cache
        3. If checkpoint.json exists with status=in_progress → resume computation
        4. Otherwise → compute from scratch
    """
    from ._checkpoint import CheckpointManager
    
    output_path = Path(output_csv)
    base_dir = output_path.parent
    
    # Determine stage name
    if stage_name is None:
        import re
        filename = output_path.stem
        stage_name = re.sub(r"_(raw|veber|brenkpains|druglike)(_\d+cpmds)?$", "", filename)
    
    # Initialize checkpoint manager
    checkpoint = CheckpointManager(stage_name, base_dir)
    
    # Helper to find actual output file (with or without row count suffix)
    def find_actual_output(base_dir: Path, stage_name: str, stage_type: str) -> Path:
        """
        Find the actual output file, checking for variants with row counts.
        
        For raw outputs: {Stage}_raw_{N}cmpds.csv or {Stage}_raw.csv
        For filtered outputs: {Stage}_{filter}_{N}cmpds.csv
        """
        import re
        
        # Try to find file by pattern based on stage_type
        if stage_type == "raw":
            # Look for raw files with row count: {Stage}_raw_*cmpds.csv
            pattern = f"{stage_name}_raw_*cmpds.csv"
            for f in base_dir.glob(pattern):
                return f
            # Fallback to raw without count
            return base_dir / f"{stage_name}_raw.csv"
        else:
            # For filtered outputs, pattern includes filter name
            pattern = f"{stage_name}_{stage_type}_*cmpds.csv"
            for f in base_dir.glob(pattern):
                return f
            return base_dir / f"{stage_name}_{stage_type}.csv"
    
    # Determine what type of output this is
    stage_type = "raw" if "_raw" in output_path.stem else ""
    if not stage_type:
        # Extract filter type from path
        import re
        match = re.search(r"_(veber|brenkpains|druglike)", output_path.stem)
        stage_type = match.group(1) if match else ""
    
    # Check if actual output file exists (with or without row count)
    actual_output = find_actual_output(base_dir, stage_name, stage_type)
    
    if not force_recompute and actual_output.exists():
        rows = _get_csv_row_count(actual_output)
        if rows > 0:
            if print_report:
                print(f"[load_or_run] Loading {actual_output.name} ({rows:,} rows) ✓")
            # Mark as complete in checkpoint if not already
            if not checkpoint.is_complete():
                checkpoint.set_complete(row_count=rows)
            return pd.read_csv(actual_output)
    
    # Check checkpoint status
    if checkpoint.has_error():
        if print_report:
            print(f"[load_or_run] ⚠️  Previous attempt failed: {checkpoint.data.get('error')}")
        if not force_recompute:
            # Ask user or automatically recompute
            print(f"[load_or_run] Force recomputing due to previous failure")
            force_recompute = True
    
    if not force_recompute and checkpoint.is_complete():
        # Load from cache (checkpoint has complete status)
        # Find actual output file (with or without row count)
        actual_output = find_actual_output(base_dir, stage_name, stage_type)
        if actual_output.exists():
            if print_report:
                print(f"[load_or_run] Loading {actual_output.name} from checkpoint ✓")
            return pd.read_csv(actual_output)
        else:
            if print_report:
                print(f"[load_or_run] Checkpoint marked complete but file missing, recomputing")
            checkpoint.reset()
    
    if checkpoint.is_in_progress() and not force_recompute:
        if print_report:
            progress = checkpoint.get_progress()
            completed = progress.get("completed_chunks", 0)
            total = progress.get("total_chunks", 0)
            print(f"[load_or_run] Resuming from checkpoint: {completed}/{total} chunks completed")
            ids = checkpoint.get_completed_ids("aldehyde")
            if ids:
                print(f"[load_or_run]   Completed aldehydes: {len(ids)}")
    else:
        if print_report:
            print(f"[load_or_run] Computing {stage_name}...")
        checkpoint.reset()
    
    # Call compute function with checkpoint manager
    try:
        df = compute_fn(checkpoint_manager=checkpoint)
        
        # Save output CSV (atomic) with row count in filename
        if len(df) > 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Detect if this is a raw output and add row count to filename
            is_raw = "_raw" in output_path.stem
            if is_raw and not output_path.stem.endswith(f"_raw_{len(df)}cmpds"):
                import re
                base_name = re.sub(r"_raw(_\d+cpmds)?$", "", output_path.stem)
                new_name = f"{base_name}_raw_{len(df)}cmpds.csv"
                output_path = output_path.parent / new_name
            
            _atomic_write_csv(df, output_path)
            
            if print_report:
                print(f"[load_or_run] Saved {output_path.name} ({len(df):,} rows)")
            
            # Mark as complete
            checkpoint.set_complete(row_count=len(df))
        else:
            if print_report:
                print(f"[load_or_run] Warning: Empty result for {stage_name}")
            checkpoint.set_failed("Empty result from compute_fn")
            
        return df
        
    except Exception as e:
        error_msg = str(e)
        checkpoint.set_failed(error_msg)
        if print_report:
            print(f"[load_or_run] ✗ Failed: {error_msg}")
        raise


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
    
    # Update paths with actual row counts
    n_acc = len(df_accepted)
    n_rej = len(df_rejected)
    
    # Insert row count into filename (replace .csv with _Ncmpds.csv)
    acc_base = accepted_path.stem
    rej_base = rejected_path.stem
    
    if not acc_base.endswith(f"_{n_acc}cmpds"):
        # Remove old row count if present
        import re
        acc_base = re.sub(r"_\d+cmpds$", "", acc_base)
        rej_base = re.sub(r"_\d+cmpds$", "", rej_base)
        
        new_acc_name = f"{acc_base}_{n_acc}cmpds.csv"
        new_rej_name = f"{rej_base}_{n_rej}cmpds.csv"
        
        accepted_path = accepted_path.parent / new_acc_name
        rejected_path = rejected_path.parent / new_rej_name
    
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    Uses atomic writes to prevent corruption.
    
    Args:
        df: DataFrame to save.
        output_csv: Path to save the CSV.
        print_report: Print save message.
    """
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(df, path)
    if print_report:
        print(f"[save_dataframe] Saved {path.name} ({len(df):,} rows)")
