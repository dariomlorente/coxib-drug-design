from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _parse_vina_poses(log_path: Path, n_modes: int = None) -> list[dict]:
    """
    Parse poses from a Vina log file.

    Parameters:
        log_path: Path to the Vina log file.
        n_modes: Maximum number of poses to return (None = return all).

    Returns:
        List of dicts with keys: rank, score, rmsd_lb, rmsd_ub.
    """
    poses = []
    sep_count = 0
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("-----") or line.startswith("Refine"):
                sep_count += 1
                continue
            if sep_count < 1 or not line:
                continue
            if "mode" in line.lower() or "affinity" in line.lower():
                continue
            cleaned = line.replace("|", " ")
            parts = cleaned.split()
            if len(parts) >= 4:
                try:
                    poses.append({
                        "rank": int(parts[0]),
                        "score": float(parts[1]),
                        "rmsd_lb": float(parts[2]),
                        "rmsd_ub": float(parts[3]),
                    })
                    if n_modes is not None and len(poses) >= n_modes:
                        break
                except ValueError:
                    continue
    return poses


def _parse_vina_log(log_path: Path) -> float | None:
    """Extract the best (most negative) affinity from a Vina log file.

    Docking scores are a ranking proxy, not an energy model.
    """
    poses = _parse_vina_poses(log_path)
    if poses:
        return poses[0]["score"]
    return None


def parse_docking_logs(
    results_dir: str | Path,
    logs_dir: str | Path,
    receptor_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse Vina log files and extract docking scores and pose instability.

    Returns two DataFrames:
        scores: ligand_id, receptor_id, cox_label, docking_score (best pose)
        instability: ligand_id, receptor_id, cox_label, pose_spread (max deviation
            across top 3 poses, measured as RMSD range or score range)

    Docking scores are a ranking proxy, not an energy model.
    """
    results_dir = Path(results_dir)
    logs_dir = Path(logs_dir)

    if receptor_map is None:
        receptor_map = {"6COX": "COX2", "3KK6": "COX1"}

    score_records = []
    instability_records = []

    log_files = list(logs_dir.glob("*.log"))
    for log_file in log_files:
        stem = log_file.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        ligand_id, receptor_id = parts

        cox_label = receptor_map.get(receptor_id, receptor_id)

        try:
            poses = _parse_vina_poses(log_file)
            if not poses:
                continue
            best_score = poses[0]["score"]
            score_records.append({
                "ligand_id": ligand_id,
                "receptor_id": receptor_id,
                "cox_label": cox_label,
                "docking_score": round(best_score, 2),
            })

            top3 = poses[:3]
            if len(top3) >= 2:
                score_spread = max(p["score"] for p in top3) - min(p["score"] for p in top3)
                rmsd_spread = max(p["rmsd_ub"] for p in top3) - min(p["rmsd_lb"] for p in top3)
                pose_spread = max(score_spread, rmsd_spread)
            else:
                pose_spread = 0.0
            instability_records.append({
                "ligand_id": ligand_id,
                "receptor_id": receptor_id,
                "cox_label": cox_label,
                "pose_spread": round(pose_spread, 3),
            })
        except Exception:
            continue

    if not score_records:
        print("[parse_docking_logs] No valid docking logs found")
        return pd.DataFrame(), pd.DataFrame()

    df_scores = pd.DataFrame(score_records)
    df_instability = pd.DataFrame(instability_records)
    print(f"[parse_docking_logs] Parsed {len(df_scores)} docking results")
    return df_scores, df_instability


def validate_docking_poses(
    mapping_csv: str | Path,
    logs_dir: str | Path,
    results_dir: str | Path,
    min_pdbqt_size: int = 100,
) -> dict:
    """
    Three-layer validation of docking completion.

    Layer 1 — Task-level integrity:
        For each row in mapping.csv, verify that both the log file
        ({ligand_id}_{receptor_id}.log) and pose file
        ({ligand_id}_{receptor_id}_out.pdbqt) exist.

    Layer 2 — Log content validation:
        Parse each log for a Vina pose table (best score).
        Identifies logs that exist but contain no valid results
        (e.g., crashed jobs, Vina errors).

    Layer 3 — Physical output integrity:
        Verify pose file exists, size > min_pdbqt_size bytes,
        and contains Vina markers (MODEL 1, REMARK VINA RESULT).
        Catches truncated or corrupted output files.

    Parameters:
        mapping_csv: Path to the docking mapping CSV.
        logs_dir: Directory containing Vina log files.
        results_dir: Directory containing Vina pose output files.
        min_pdbqt_size: Minimum expected pose file size in bytes
            (default 100 — catches empty or near-empty files).

    Returns:
        Dict with status, counts, and per-task details:
        {
            "status": "PASS" | "PARTIAL" | "FAIL",
            "total": int,              # expected tasks
            "complete": int,           # passed all 3 layers
            "missing": [(ligand_id, receptor_id), ...],
            "failed": [(ligand_id, receptor_id, reason), ...],
            "orphaned": [filename, ...],
            "summary": str,
        }

    Note:
        Does NOT raise exceptions on validation failure.
        Downstream cells should check the "status" field:
        - "PASS": proceed with all results
        - "PARTIAL": proceed with complete subset, exclude failed
        - "FAIL": no valid results (block downstream)
    """
    mapping_csv = Path(mapping_csv)
    logs_dir = Path(logs_dir)
    results_dir = Path(results_dir)

    if not mapping_csv.exists():
        raise ValueError(f"Mapping file not found: {mapping_csv}")
    if not logs_dir.is_dir():
        raise ValueError(f"Logs directory not found: {logs_dir}")
    if not results_dir.is_dir():
        raise ValueError(f"Results directory not found: {results_dir}")

    mapping_df = pd.read_csv(mapping_csv)
    expected = set()
    for _, row in mapping_df.iterrows():
        expected.add((str(row["ligand_id"]), str(row["receptor_id"])))

    # Layer 1: task-level file existence
    missing = []
    layer1_ok = set()
    for ligand_id, receptor_id in expected:
        log_file = logs_dir / f"{ligand_id}_{receptor_id}.log"
        pose_file = results_dir / f"{ligand_id}_{receptor_id}_out.pdbqt"
        if log_file.exists() and pose_file.exists():
            layer1_ok.add((ligand_id, receptor_id))
        else:
            missing.append((ligand_id, receptor_id))

    # Detect orphaned files (present but not in mapping)
    orphaned = []
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            parts = log_file.stem.rsplit("_", 1)
            if len(parts) == 2:
                pair = (parts[0], parts[1])
                if pair not in expected:
                    orphaned.append(log_file.name)

    # Layer 2: log content validation
    layer2_ok = set()
    failed = []
    for ligand_id, receptor_id in layer1_ok:
        log_file = logs_dir / f"{ligand_id}_{receptor_id}.log"
        poses = _parse_vina_poses(log_file)
        if poses:
            layer2_ok.add((ligand_id, receptor_id))
        else:
            failed.append((ligand_id, receptor_id, "no Vina poses in log"))

    # Layer 3: physical output integrity
    layer3_ok = set()
    for ligand_id, receptor_id in layer2_ok:
        pose_file = results_dir / f"{ligand_id}_{receptor_id}_out.pdbqt"
        if pose_file.stat().st_size < min_pdbqt_size:
            failed.append((
                ligand_id, receptor_id,
                f"pose file too small ({pose_file.stat().st_size} B)",
            ))
            continue

        content = pose_file.read_text()
        if "MODEL 1" not in content and "REMARK VINA RESULT" not in content:
            failed.append((ligand_id, receptor_id, "pose file missing Vina markers"))
            continue

        layer3_ok.add((ligand_id, receptor_id))

    # Determine tri-state status
    complete = len(layer3_ok)
    total = len(expected)
    if complete == total:
        status = "PASS"
    elif complete > 0:
        status = "PARTIAL"
    else:
        status = "FAIL"

    summary = (
        f"{status}: {complete}/{total} docking tasks validated. "
        f"Missing: {len(missing)}, Failed: {len(failed)}, Orphaned: {len(orphaned)}"
    )
    if missing:
        summary += f" | Missing tasks: {missing[:5]}{'...' if len(missing) > 5 else ''}"
    if failed:
        reasons = set(r[2] for r in failed)
        summary += f" | Failure reasons: {reasons}"

    result = {
        "status": status,
        "total": total,
        "complete": complete,
        "missing": sorted(missing),
        "failed": sorted(failed),
        "orphaned": sorted(orphaned),
        "summary": summary,
        "valid_tasks": sorted(layer3_ok),
    }

    print(f"[validate_docking] {summary}")
    return result


def extract_all_docking_scores(
    logs_dir: str | Path,
    receptor_map: dict | None = None,
    n_modes: int = 3,
) -> pd.DataFrame:
    """
    Extract all poses (up to n_modes) for each ligand-receptor pair.

    Parameters:
        logs_dir: Directory containing Vina log files.
        receptor_map: Dict mapping receptor_id -> cox_label.
        n_modes: Maximum number of poses to extract per log.

    Returns:
        DataFrame with columns: ligand_id, receptor_id, cox_label, pose_rank, docking_score.
    """
    logs_dir = Path(logs_dir)

    if receptor_map is None:
        receptor_map = {"6COX": "COX2", "3KK6": "COX1"}

    records = []
    for log_file in logs_dir.glob("*.log"):
        stem = log_file.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        ligand_id, receptor_id = parts
        cox_label = receptor_map.get(receptor_id, receptor_id)

        poses = _parse_vina_poses(log_file, n_modes=n_modes)
        for p in poses:
            records.append({
                "ligand_id": ligand_id,
                "receptor_id": receptor_id,
                "cox_label": cox_label,
                "pose_rank": p["rank"],
                "docking_score": round(p["score"], 2),
            })

    return pd.DataFrame(records)


# =============================================================================
# DockingResultParser
# =============================================================================


class DockingResultParser:
    """
    Parses, validates, and extracts scores from AutoDock Vina output files.

    Wraps parse_docking_logs(), validate_docking_poses(), and
    extract_all_docking_scores(). Stores receptor_map and n_modes at
    construction time so the same parser handles all receptors consistently.

    Parameters
    ----------
    receptor_map : dict[str, str] or None, optional
        Mapping from receptor_id to cox_label.
        Default: {"6COX": "COX2", "3KK6": "COX1"}.
    n_modes : int, optional
        Maximum number of poses to extract per log file. Default: 3.
    """

    def __init__(
        self,
        receptor_map: dict[str, str] | None = None,
        n_modes: int = 3,
    ) -> None:
        if receptor_map is None:
            receptor_map = {"6COX": "COX2", "3KK6": "COX1"}
        self.receptor_map = receptor_map
        self.n_modes = n_modes

    def parse(
        self,
        results_dir: str | Path,
        logs_dir: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse Vina log files and extract docking scores and pose instability.

        Delegates to parse_docking_logs() with stored receptor_map.

        Parameters
        ----------
        results_dir : str or Path
            Directory containing Vina pose output files.
        logs_dir : str or Path
            Directory containing Vina log files.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (df_scores, df_instability).
        """
        return parse_docking_logs(results_dir, logs_dir, receptor_map=self.receptor_map)

    def validate(
        self,
        mapping_csv: str | Path,
        logs_dir: str | Path,
        results_dir: str | Path,
        **kwargs: Any,
    ) -> dict:
        """
        Three-layer validation of docking completion.

        Delegates to validate_docking_poses().  All kwargs forwarded
        directly.

        Parameters
        ----------
        mapping_csv : str or Path
            Path to the docking mapping CSV.
        logs_dir : str or Path
            Directory containing Vina log files.
        results_dir : str or Path
            Directory containing Vina pose output files.
        **kwargs
            Additional keyword arguments forwarded to
            validate_docking_poses().

        Returns
        -------
        dict
            Validation result dict with status, counts, and details.
        """
        return validate_docking_poses(mapping_csv, logs_dir, results_dir, **kwargs)

    def extract(
        self,
        logs_dir: str | Path,
    ) -> pd.DataFrame:
        """
        Extract all docking scores (all poses) from Vina logs.

        Delegates to extract_all_docking_scores() with stored receptor_map
        and n_modes.

        Parameters
        ----------
        logs_dir : str or Path
            Directory containing Vina log files.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ligand_id, receptor_id, cox_label,
            pose_rank, docking_score.
        """
        return extract_all_docking_scores(
            logs_dir,
            receptor_map=self.receptor_map,
            n_modes=self.n_modes,
        )
