"""
Checkpoint management module for robust pipeline resumption.

Handles metadata tracking for reaction and filter operations,
enabling resume after kernel crashes or interruptions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class CheckpointManager:
    """
    Manages checkpoint metadata for pipeline stages.

    Uses JSON metadata files to track:
    - Completion status (complete/in_progress/failed)
    - Completed IDs for reactions
    - Chunk progress for large datasets
    - Parameter hash to detect changes

    Example:
        >>> manager = CheckpointManager("Oxazolones", Path("mol_files/3. Oxazolones"))
        >>> manager.add_completed_ids("aldehyde", {"A1", "A2"})
        >>> manager.set_complete(row_count=4892, stats={...})
    """

    def __init__(
        self,
        stage_name: str,
        output_dir: Path,
        version: int = 1,
    ):
        """
        Initialize checkpoint manager for a pipeline stage.

        Args:
            stage_name: Name of the stage (e.g., "Oxazolones")
            output_dir: Directory where CSV files are stored
            version: Schema version for migration support
        """
        self.stage_name = stage_name
        self.version = version
        self.output_dir = output_dir  # Store original output directory
        self.path = output_dir / ".cache" / f"{stage_name}_checkpoint.json"

        if self.path.exists():
            self.data = self._load()
        else:
            self.data = self._init_empty()
            self._save()

    def _init_empty(self) -> dict[str, Any]:
        """Create empty checkpoint data structure."""
        return {
            "version": self.version,
            "stage": self.stage_name,
            "status": "in_progress",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "input_params": {},
            "params_hash": None,
            "progress": {
                "total_chunks": 0,
                "completed_chunks": 0,
                "last_chunk_time": 0.0,
            },
            "completed_ids": {
                "aldehyde": [],
                "oxazolone": [],
                "amine": [],
                "thiazolone": [],
            },
            "output": {
                "row_count": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            },
            "error": None,
        }

    def _load(self) -> dict[str, Any]:
        """Load checkpoint from JSON file."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Validate schema version
                if data.get("version") != self.version:
                    print(f"[CheckpointManager] Warning: version mismatch, treating as new")
                    return self._init_empty()
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"[CheckpointManager] Could not load checkpoint: {e}")
            return self._init_empty()

    def _save(self) -> None:
        """Atomic save of checkpoint data."""
        self.data["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".json.tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, separators=(",", ":"))
            temp_path.rename(self.path)  # Atomic on POSIX
        except IOError as e:
            print(f"[CheckpointManager] Could not save checkpoint: {e}")

    def compute_params_hash(self, params: dict[str, Any]) -> str:
        """Compute SHA256 hash of parameters for validation."""
        param_str = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(param_str.encode()).hexdigest()

    def set_input_params(self, params: dict[str, Any]) -> None:
        """Store input parameters and compute hash."""
        self.data["input_params"] = params
        self.data["params_hash"] = self.compute_params_hash(params)
        self._save()

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Check if parameters match the stored hash."""
        if self.data.get("params_hash") is None:
            return True  # No previous params to compare
        current_hash = self.compute_params_hash(params)
        return current_hash == self.data["params_hash"]

    def is_complete(self) -> bool:
        """Check if this stage is complete."""
        return self.data.get("status") == "complete"

    def is_in_progress(self) -> bool:
        """Check if this stage is in progress."""
        return self.data.get("status") == "in_progress"

    def has_error(self) -> bool:
        """Check if this stage has an error."""
        return self.data.get("status") == "failed" or self.data.get("error") is not None

    def get_status(self) -> str:
        """Get current status."""
        return self.data.get("status", "unknown")

    def get_completed_ids(self, reactant_type: str = "aldehyde") -> set[str]:
        """Get set of completed IDs for a given reactant type."""
        key_map = {
            "aldehyde": "aldehyde",
            "carboxylic": "aldehyde",  # E-P uses aldehyde ID as key
            "oxazolone": "oxazolone",
            "amine": "amine",
            "thiazolone": "thiazolone",
        }
        key = key_map.get(reactant_type, reactant_type)
        ids = self.data.get("completed_ids", {}).get(key, [])
        return set(ids)

    def add_completed_ids(self, reactant_type: str, ids: set[str]) -> None:
        """Add completed IDs for a reactant type."""
        key_map = {
            "aldehyde": "aldehyde",
            "carboxylic": "aldehyde",
            "oxazolone": "oxazolone",
            "amine": "amine",
            "thiazolone": "thiazolone",
        }
        key = key_map.get(reactant_type, reactant_type)

        if key not in self.data["completed_ids"]:
            self.data["completed_ids"][key] = []

        current = set(self.data["completed_ids"][key])
        current.update(ids)
        self.data["completed_ids"][key] = sorted(list(current))
        self._save()

    def update_progress(
        self,
        total_chunks: int | None = None,
        completed_chunks: int | None = None,
        last_chunk_time: float | None = None,
    ) -> None:
        """Update progress tracking."""
        if total_chunks is not None:
            self.data["progress"]["total_chunks"] = total_chunks
        if completed_chunks is not None:
            self.data["progress"]["completed_chunks"] = completed_chunks
        if last_chunk_time is not None:
            self.data["progress"]["last_chunk_time"] = last_chunk_time
        self._save()

    def get_progress(self) -> dict[str, Any]:
        """Get progress information."""
        return self.data.get("progress", {})

    def set_complete(self, row_count: int = 0, stats: dict[str, Any] | None = None) -> None:
        """Mark this stage as complete."""
        self.data["status"] = "complete"
        self.data["output"]["row_count"] = row_count
        if stats:
            self.data["output"]["cache_hits"] = stats.get("cache_hits", 0)
            self.data["output"]["cache_misses"] = stats.get("cache_misses", 0)
        self._save()

    def set_failed(self, error: str) -> None:
        """Mark this stage as failed with error message."""
        self.data["status"] = "failed"
        self.data["error"] = error
        self._save()

    def reset(self) -> None:
        """Reset checkpoint to initial state (force recompute)."""
        self.data = self._init_empty()
        self._save()

    def delete(self) -> None:
        """Delete checkpoint file."""
        if self.path.exists():
            self.path.unlink()

    @property
    def output_csv(self) -> Path:
        """Get canonical output CSV path ({Stage}_raw.csv)."""
        return self.output_dir / f"{self.stage_name}_raw.csv"

    @property
    def final_output_csv(self) -> Path:
        """Get final export CSV path ({Stage}.csv)."""
        return self.output_dir / f"{self.stage_name}.csv"


def _get_checkpoint(stage_name: str, output_dir: Path) -> CheckpointManager:
    """Convenience function to get or create a checkpoint manager."""
    return CheckpointManager(stage_name, output_dir)


def _get_stage_dir(stage_name: str) -> Path:
    """Get the directory for a stage based on name."""
    stage_dirs = {
        "Aldehydes": Path("mol_files/2. Building Blocks"),
        "Carboxylics": Path("mol_files/2. Building Blocks"),
        "Amines": Path("mol_files/2. Building Blocks"),
        "Oxazolones": Path("mol_files/3. Oxazolones"),
        "Imidazolones": Path("mol_files/4. Imidazolones"),
        "Thiazolones": Path("mol_files/5. Thiazolones"),
    }
    return stage_dirs.get(stage_name, Path(f"mol_files/{stage_name}"))
