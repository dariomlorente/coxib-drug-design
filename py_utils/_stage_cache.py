from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class StageCache:
    """
    Simple stage cache that saves DataFrames to CSV with metadata validation.
    
    Uses row count only for validation (no hashing). Cache files are stored
    alongside the stage output in a .cache subdirectory.
    """
    
    def __init__(self, base_dir: str | Path = "mol_files"):
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / ".cache" / "stages"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_metadata_path(self, stage_name: str) -> Path:
        """Get path to metadata JSON for a stage."""
        return self.cache_dir / f"{stage_name}_metadata.json"
    
    def get_csv_path(self, stage_name: str, suffix: str = "") -> Path:
        """Get path to CSV for a stage."""
        suffix = f"_{suffix}" if suffix else ""
        return self.cache_dir / f"{stage_name}{suffix}.csv"
    
    def exists(self, stage_name: str, expected_rows: int | None = None) -> bool:
        """
        Check if a cached stage exists and has the expected row count.
        
        Args:
            stage_name: Name of the stage (e.g., "veber_oxazolones")
            expected_rows: Expected row count (optional, used for validation)
        
        Returns:
            True if CSV exists and (optionally) matches expected row count
        """
        csv_path = self.get_csv_path(stage_name)
        if not csv_path.exists():
            return False
        
        if expected_rows is not None:
            try:
                # Quick row count check without loading full DataFrame
                with open(csv_path, 'r') as f:
                    # Count lines (header + data rows)
                    n_lines = sum(1 for _ in f)
                # Subtract header
                actual_rows = n_lines - 1 if n_lines > 0 else 0
                return actual_rows == expected_rows
            except Exception:
                return False
        
        return True
    
    def load(self, stage_name: str, suffix: str = "") -> pd.DataFrame | None:
        """
        Load cached DataFrame from CSV.
        
        Args:
            stage_name: Name of the stage
            suffix: Optional suffix for multiple outputs (e.g., "accepted", "rejected")
        
        Returns:
            DataFrame or None if not cached
        """
        csv_path = self.get_csv_path(stage_name, suffix)
        if not csv_path.exists():
            return None
        
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"⚠️  Failed to load cache for {stage_name}: {e}")
            return None
    
    def save(
        self,
        stage_name: str,
        df: pd.DataFrame,
        params: dict[str, Any],
        suffix: str = "",
    ) -> None:
        """
        Save DataFrame to CSV with metadata.
        
        Args:
            stage_name: Name of the stage
            df: DataFrame to save
            params: Parameters used to generate this stage
            suffix: Optional suffix for multiple outputs
        """
        csv_path = self.get_csv_path(stage_name, suffix)
        metadata_path = self.get_metadata_path(stage_name)
        
        # Save CSV
        df.to_csv(csv_path, index=False)
        
        # Save metadata
        metadata = {
            "stage_name": stage_name,
            "row_count": len(df),
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "suffix": suffix if suffix else None,
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save metadata for {stage_name}: {e}")
    
    def get_metadata(self, stage_name: str) -> dict[str, Any] | None:
        """Load metadata for a stage."""
        metadata_path = self.get_metadata_path(stage_name)
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None


def load_or_compute(
    stage_name: str,
    compute_fn: Any,
    cache: StageCache | None = None,
    force_recompute: bool = False,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    Load cached stage or compute and save it.
    
    Args:
        stage_name: Name of the stage (e.g., "veber_oxazolones")
        compute_fn: Function that returns DataFrame to cache
        cache: StageCache instance (created if None)
        force_recompute: Ignore cache and always recompute
        print_report: Print progress messages
    
    Returns:
        DataFrame from cache or from compute_fn
    """
    if cache is None:
        cache = StageCache()
    
    if not force_recompute and cache.exists(stage_name):
        if print_report:
            metadata = cache.get_metadata(stage_name)
            if metadata:
                print(f"[StageCache] Loading {stage_name} from cache "
                      f"({metadata['row_count']:,} rows)")
        return cache.load(stage_name)
    
    if print_report:
        print(f"[StageCache] Computing {stage_name}...")
    
    df = compute_fn()
    cache.save(stage_name, df, params={})
    
    if print_report:
        print(f"[StageCache] Saved {stage_name} to cache "
              f"({len(df):,} rows)")
    
    return df


def save_stage(
    stage_name: str,
    df: pd.DataFrame,
    cache: StageCache | None = None,
    params: dict[str, Any] | None = None,
    suffix: str = "",
    print_report: bool = True,
) -> None:
    """Save DataFrame as a named stage with metadata."""
    if cache is None:
        cache = StageCache()
    
    cache.save(stage_name, df, params or {}, suffix)
    
    if print_report:
        print(f"[StageCache] Saved {stage_name}{f'_{suffix}' if suffix else ''} "
              f"({len(df):,} rows)")
