from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

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


def _strip_rowcount_suffix(stem: str) -> str:
    """Remove trailing _{N}cmpds suffix from a filename stem."""
    match = re.match(r"^(?P<prefix>.+)_\d+cmpds$", stem)
    if match is None:
        return stem
    return match.group("prefix")


def _with_rowcount_suffix(stem: str, row_count: int) -> str:
    """Return stem with canonical trailing _{N}cmpds suffix."""
    base = _strip_rowcount_suffix(stem)
    return f"{base}_{row_count}cmpds"


def _find_latest_counted_csv(base_dir: Path, base_stem: str) -> Path | None:
    """Find latest CSV matching <base_stem>_<N>cmpds.csv exactly."""
    pattern = re.compile(rf"^{re.escape(base_stem)}_(\d+)cmpds$")
    candidates: list[Path] = []

    for path in base_dir.glob(f"{base_stem}_*cmpds.csv"):
        if pattern.match(path.stem):
            candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_existing_output(requested_path: Path) -> Path | None:
    """Resolve existing CSV for a requested output path with row-count awareness."""
    base_stem = _strip_rowcount_suffix(requested_path.stem)

    counted = _find_latest_counted_csv(requested_path.parent, base_stem)
    if counted is not None:
        return counted

    exact = requested_path.parent / f"{base_stem}.csv"
    if exact.exists():
        return exact

    return None


def _checkpoint_has_progress(checkpoint: object) -> bool:
    """Return True if checkpoint contains non-empty progress/IDs."""
    progress = checkpoint.get_progress()
    if progress.get("completed_chunks", 0) > 0:
        return True

    completed_ids = checkpoint.data.get("completed_ids", {})
    return any(bool(ids) for ids in completed_ids.values())


def _cleanup_stage_temp_csvs(cache_dir: Path) -> None:
    """Delete transient stage temp CSV files used for chunked reaction resume."""
    for temp_path in cache_dir.glob(".tmp_*_results.csv"):
        try:
            temp_path.unlink()
        except OSError:
            continue


def load_or_run(
    compute_fn: Callable[..., pd.DataFrame],
    output_csv: str | Path,
    stage_name: str | None = None,
    force_recompute: bool = False,
    params: dict[str, object] | None = None,
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
        params: Optional parameter dict used to validate cache/checkpoint compatibility.
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
        filename = _strip_rowcount_suffix(output_path.stem)
        stage_name = re.sub(r"_(raw|veber|brenkpains|druglike)$", "", filename)
    
    # Initialize checkpoint manager
    checkpoint = CheckpointManager(stage_name, base_dir)

    if params is not None and not force_recompute and not checkpoint.validate_params(params):
        if print_report:
            print("[load_or_run] Checkpoint params changed, recomputing")
        force_recompute = True
    
    requested_output = output_path.parent / f"{_strip_rowcount_suffix(output_path.stem)}.csv"

    # Check if actual output file exists (with or without row count)
    actual_output = _resolve_existing_output(requested_output)
    
    # If force_recompute, reset checkpoint at the start to avoid stale state
    if force_recompute:
        checkpoint.reset()
        _cleanup_stage_temp_csvs(checkpoint.path.parent)

    if params is not None:
        checkpoint.set_input_params(params)
    
    if not force_recompute and actual_output is not None and actual_output.exists():
        rows = _get_csv_row_count(actual_output)
        if rows > 0:
            if _checkpoint_has_progress(checkpoint) and not checkpoint.is_complete():
                if print_report:
                    print("[load_or_run] Checkpoint in progress; resuming computation")
            else:
                if print_report:
                    print(f"[load_or_run] Loading {actual_output.name} ({rows:,} rows) ✓")
                # Mark as complete in checkpoint if not already
                if not checkpoint.is_complete():
                    checkpoint.set_complete(row_count=rows)
                return pd.read_csv(actual_output)

    if checkpoint.is_complete() and not force_recompute:
        actual_output = _resolve_existing_output(requested_output)
        if actual_output is not None and actual_output.exists():
            if print_report:
                print(f"[load_or_run] Loading {actual_output.name} from checkpoint ✓")
            return pd.read_csv(actual_output)
        if print_report:
            print("[load_or_run] Checkpoint marked complete but output missing, recomputing")
        checkpoint.reset()
    
    # Check checkpoint status
    if checkpoint.has_error():
        if print_report:
            print(f"[load_or_run] ⚠️  Previous attempt failed: {checkpoint.data.get('error')}")
        if not force_recompute:
            # Ask user or automatically recompute
            print(f"[load_or_run] Force recomputing due to previous failure")
            force_recompute = True
    
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
            
            # Canonical counted filename for all stage outputs
            counted_stem = _with_rowcount_suffix(output_path.stem, len(df))
            output_path = output_path.parent / f"{counted_stem}.csv"
            
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


def _find_existing_filter_csv(
    accepted_path: Path,
    rejected_path: Path,
    input_row_count: int | None = None,
) -> tuple[Path | None, Path | None]:
    """
    Find existing CSV files for filters, accounting for row count in filename.
    
    Search pattern: {stage}_{filter_type}_{N}cmpds.csv
    
    Args:
        accepted_path: Expected path to accepted compounds (without row count)
        rejected_path: Expected path to rejected compounds (without row count)
    
    Returns:
        Tuple of (found_accepted_path, found_rejected_path) or (None, None) if not found
    """
    accepted_stem = _strip_rowcount_suffix(accepted_path.stem)
    pattern = r"^([A-Za-z]+)_(\w+)$"
    match = re.match(pattern, accepted_stem)
    
    if not match:
        return None, None
    
    stage_name = match.group(1)
    filter_type = match.group(2)
    
    accepted_candidates: list[Path] = []
    accepted_exact = accepted_path.parent / f"{stage_name}_{filter_type}.csv"
    if accepted_exact.exists():
        accepted_candidates.append(accepted_exact)
    accepted_candidates.extend(sorted(
        accepted_path.parent.glob(f"{stage_name}_{filter_type}_*cmpds.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ))
    accepted_candidates.extend(sorted(
        accepted_path.parent.glob(f"test_{stage_name}_{filter_type}_*cmpds.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ))

    accepted_candidates = list(dict.fromkeys(accepted_candidates))

    if not accepted_candidates:
        return None, None

    # Now find corresponding rejected candidates
    rejected_parent = rejected_path.parent
    rejected_parent.mkdir(parents=True, exist_ok=True)

    rejected_candidates: list[Path] = []
    rejected_exact = rejected_parent / f"{stage_name}_rejected_{filter_type}.csv"
    if rejected_exact.exists():
        rejected_candidates.append(rejected_exact)
    rejected_candidates.extend(sorted(
        rejected_parent.glob(f"{stage_name}_rejected_{filter_type}_*cmpds.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ))
    rejected_candidates.extend(sorted(
        rejected_parent.glob(f"test_{stage_name}_rejected_{filter_type}_*cmpds.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ))

    rejected_candidates = list(dict.fromkeys(rejected_candidates))

    if not rejected_candidates:
        return None, None

    if input_row_count is None:
        return accepted_candidates[0], rejected_candidates[0]

    accepted_rows = {p: _get_csv_row_count(p) for p in accepted_candidates}
    rejected_rows = {p: _get_csv_row_count(p) for p in rejected_candidates}

    best_pair: tuple[Path, Path] | None = None
    best_pair_score = -1.0

    for acc_path in accepted_candidates:
        acc_count = accepted_rows[acc_path]
        if acc_count < 0:
            continue
        for rej_path in rejected_candidates:
            rej_count = rejected_rows[rej_path]
            if rej_count < 0:
                continue
            if (acc_count + rej_count) != input_row_count:
                continue

            pair_score = max(acc_path.stat().st_mtime, rej_path.stat().st_mtime)
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (acc_path, rej_path)

    if best_pair is None:
        return None, None

    return best_pair


def load_or_filter(
    compute_fn: Callable[[], tuple[pd.DataFrame, pd.DataFrame]],
    accepted_csv: str | Path,
    rejected_csv: str | Path,
    force_recompute: bool = False,
    input_row_count: int | None = None,
    params: dict[str, object] | None = None,
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
        input_row_count: Optional input row count for integrity check.
            When provided, cached files are used only if accepted+rejected equals this value.
        params: Optional parameter dict used to validate cache compatibility.
        print_report: Print loading/computing messages.
    
    Returns:
        Tuple of (accepted_df, rejected_df).
    """
    accepted_path = Path(accepted_csv)
    rejected_path = Path(rejected_csv)

    from ._checkpoint import CheckpointManager

    filter_stage_name = _strip_rowcount_suffix(accepted_path.stem)
    filter_checkpoint = CheckpointManager(filter_stage_name, accepted_path.parent)

    if params is not None and not force_recompute and not filter_checkpoint.validate_params(params):
        if print_report:
            print("[load_or_filter] Filter params changed, recomputing")
        force_recompute = True

    if force_recompute:
        filter_checkpoint.reset()

    if params is not None:
        filter_checkpoint.set_input_params(params)
    
    # Try to find existing files (including those with row count in filename)
    if not force_recompute:
        found_accepted, found_rejected = _find_existing_filter_csv(
            accepted_path,
            rejected_path,
            input_row_count=input_row_count,
        )
        
        if found_accepted and found_rejected:
            acc_rows = _get_csv_row_count(found_accepted)
            rej_rows = _get_csv_row_count(found_rejected)

            if print_report:
                print(f"[load_or_filter] Loading {found_accepted.name} ({acc_rows:,} rows) "
                      f"+ {found_rejected.name} ({rej_rows:,} rows) ✓")
            if not filter_checkpoint.is_complete():
                filter_checkpoint.set_complete(row_count=acc_rows)
            return pd.read_csv(found_accepted), pd.read_csv(found_rejected)

        if input_row_count is not None and print_report:
            print(
                "[load_or_filter] No cache pair matches input row count "
                f"({input_row_count:,}); recomputing"
            )
    
    if print_report:
        print(f"[load_or_filter] Computing {accepted_path.name}...")
    
    df_accepted, df_rejected = compute_fn()
    
    # Update paths with actual row counts
    n_acc = len(df_accepted)
    n_rej = len(df_rejected)
    
    accepted_path = accepted_path.parent / f"{_with_rowcount_suffix(accepted_path.stem, n_acc)}.csv"
    rejected_path = rejected_path.parent / f"{_with_rowcount_suffix(rejected_path.stem, n_rej)}.csv"
    
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_accepted.to_csv(accepted_path, index=False)
    df_rejected.to_csv(rejected_path, index=False)
    
    if print_report:
        print(f"[load_or_filter] Saved {accepted_path.name} ({len(df_accepted):,} accepted) "
              f"+ {rejected_path.name} ({len(df_rejected):,} rejected)")

    filter_checkpoint.set_complete(row_count=n_acc)
    
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
