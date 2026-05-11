from __future__ import annotations

import json
import os.path
import subprocess
from pathlib import Path

import pandas as pd

from ._receptors import prepare_receptor
from ._ligands import prepare_ligands_multi_conf
from ._docking import init_hpc_dirs, write_mapping_csv, run_local_docking
from ._parsing import validate_docking_poses, extract_all_docking_scores
from ._scoring import compute_final_ranking, select_best_poses_by_geo_score


def find_samples_csv(clustering_dir: str | Path, series: str) -> Path:
    """
    Auto-detect the latest clustering output CSV for a given series.

    Parameters:
        clustering_dir: Path to the clustering outputs directory.
        series: Series name (e.g., "Thiazolones", "Imidazolones").

    Returns:
        Path to the latest matching CSV file.

    Raises:
        FileNotFoundError: If no matching CSV is found.
    """
    clustering_dir = Path(clustering_dir)
    matches = sorted(clustering_dir.glob(f"{series}_*samples.csv"))
    if not matches:
        raise FileNotFoundError(f"No {series}_*samples.csv found in {clustering_dir}")
    if len(matches) > 1:
        print(f"[warn] Multiple {series} files found, using latest: {matches[-1].name}")
    return matches[-1]


def get_docking_config(
    root: Path | None = None,
    inputs_dir: str = "03_docking_pdbqts_data/inputs",
    clustering_dir: str = "02_selected_mols_data/outputs",
) -> dict:
    """
    Get standard docking configuration.

    Parameters:
        root: Project root (defaults to Path.cwd()).
        inputs_dir: Relative path to inputs.
        clustering_dir: Relative path to clustering outputs.

    Returns:
        Dict with ROOT, COX2_PDB, COX1_PDB, COX2_LIGAND, COX1_LIGAND,
        RECEPTOR_MAP, receptor_pdb_map, INPUT_CSV_THIAZOLONES, INPUT_CSV_IMIDAZOLONES.
    """
    from pathlib import Path as _Path
    root = root or _Path.cwd()
    
    _inputs = root / inputs_dir
    _clustering = root / clustering_dir
    
    config = {
        "ROOT": root,
        "COX2_PDB": _inputs / "6COX.pdb",
        "COX1_PDB": _inputs / "3KK6.pdb",
        "COX2_LIGAND": "S58",
        "COX1_LIGAND": "FLC",
        "RECEPTOR_MAP": {"6COX": "COX2", "3KK6": "COX1"},
        "receptor_pdb_map": {"6COX": str(_inputs / "6COX.pdb"), "3KK6": str(_inputs / "3KK6.pdb")},
    }
    config["INPUT_CSV_THIAZOLONES"] = find_samples_csv(_clustering, "Thiazolones")
    config["INPUT_CSV_IMIDAZOLONES"] = find_samples_csv(_clustering, "Imidazolones")
    return config


def print_box_info(dirs: dict[str, Path], receptor_ids: list[str]) -> None:
    """
    Print box dimensions for each receptor.

    Parameters:
        dirs: Directory dict from init_hpc_dirs.
        receptor_ids: List of receptor IDs to display.
    """
    import json
    for rec_id in receptor_ids:
        box_path = dirs["receptors"] / f"{rec_id}_box.json"
        if box_path.exists():
            with open(box_path) as f:
                box = json.load(f)
            print(f"\n{rec_id}:")
            print(f"  Center: ({box['center_x']}, {box['center_y']}, {box['center_z']})")
            print(f"  Size: {box['size_x']} x {box['size_y']} x {box['size_z']} A")


def load_ligands(
    input_csvs: list[Path],
    id_col: str = "ID",
    smiles_col: str = "SMILES",
    qsar_col: str = "QSAR_score",
) -> pd.DataFrame:
    """Load and concatenate multiple ligand CSV files, print summary."""
    dfs = [pd.read_csv(p) for p in input_csvs]
    df = pd.concat(dfs, ignore_index=True)
    print(f"[Load ligands] {len(df)} compounds total")
    return df


def prepare_all_receptors(
    dirs: dict[str, Path],
    cfg: dict,
    box_size: float = 22.0,
) -> tuple[dict, dict]:
    """Prepare both COX-2 (6COX) and COX-1 (3KK6) receptors."""
    cox2_info = prepare_receptor(
        pdb_path=cfg["COX2_PDB"],
        receptor_dir=dirs["receptors"],
        receptor_id="6COX",
        ligand_resname=cfg["COX2_LIGAND"],
        box_size=box_size,
        override_box=dict(center_x=26.5, center_y=23.5, center_z=48.3,
                          size_x=22.0, size_y=22.0, size_z=22.0),
    )
    cox1_info = prepare_receptor(
        pdb_path=cfg["COX1_PDB"],
        receptor_dir=dirs["receptors"],
        receptor_id="3KK6",
        ligand_resname=cfg["COX1_LIGAND"],
        box_size=box_size,
    )
    print_box_info(dirs, ["6COX", "3KK6"])
    return cox2_info, cox1_info


def run_docking_workflow(
    df_ligands: pd.DataFrame,
    dirs: dict[str, Path],
    receptor_ids: list[str] | None = None,
    mapping_csv: Path | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Write mapping CSV and run local docking. Handles guard clauses."""
    if df_ligands.empty:
        print("[docking] No ligands available")
        return pd.DataFrame(), {}

    if receptor_ids is None:
        receptor_ids = ["6COX", "3KK6"]
    if mapping_csv is None:
        mapping_csv = dirs["docking_mapping"] / "mapping.csv"

    mapping_path = write_mapping_csv(
        df_ligands=df_ligands,
        receptor_ids=receptor_ids,
        mapping_csv=mapping_csv,
    )
    mapping_df = pd.read_csv(mapping_path)
    print(f"[docking mapping] {len(mapping_df)} tasks")

    if mapping_df.empty:
        print("[docking] No mapping available")
        return mapping_df, {}

    docking_stats = run_local_docking(mapping_df, dirs, **kwargs)
    return mapping_df, docking_stats


def validate_all_docking(
    mapping_csv: str | Path,
    dirs: dict[str, Path],
    receptor_ids: list[str],
) -> dict:
    """
    Validate docking results across all receptors.

    Parameters:
        mapping_csv: Path to mapping.csv.
        dirs: Directory dict.
        receptor_ids: List of receptor IDs.

    Returns:
        Aggregated validation dict.
    """
    mapping_csv = Path(mapping_csv)
    validation = {"status": "PASS", "total": 0, "complete": 0,
                 "missing": [], "failed": [], "orphaned": [],
                 "summary": "", "valid_tasks": []}
    for rec_id in receptor_ids:
        val = validate_docking_poses(
            mapping_csv=mapping_csv,
            logs_dir=dirs[f"docking_logs_{rec_id}"],
            results_dir=dirs[f"docking_poses_{rec_id}"],
        )
        validation["total"] += val["total"]
        validation["complete"] += val["complete"]
        validation["missing"].extend(val["missing"])
        validation["failed"].extend(val["failed"])
        validation["orphaned"].extend(val["orphaned"])
        validation["valid_tasks"].extend(val["valid_tasks"])
        if val["status"] == "FAIL":
            validation["status"] = "FAIL"
        elif val["status"] == "PARTIAL" and validation["status"] != "FAIL":
            validation["status"] = "PARTIAL"
    print(validation.get("summary", ""))
    if validation["status"] == "FAIL":
        print("No valid docking results. Run docking first.")
    elif validation["status"] == "PARTIAL":
        print(f"Partial results: {validation['complete']}/{validation['total']} tasks validated.")
        print("Downstream analysis will use only validated tasks.")
    return validation


def validate_and_extract(
    dirs: dict[str, Path],
    receptor_ids: list[str],
    receptor_map: dict[str, str],
    mapping_csv: Path | None = None,
    num_modes: int = 3,
) -> pd.DataFrame:
    """Validate docking results and extract all poses. Handles guard clauses."""
    if mapping_csv is None:
        mapping_csv = dirs["docking_mapping"] / "mapping.csv"

    validation = validate_all_docking(mapping_csv, dirs, receptor_ids)

    if validation.get("status") in ("PASS", "PARTIAL"):
        return extract_all_poses(dirs, receptor_ids, receptor_map, n_modes=num_modes)
    else:
        print("[parse poses] No valid docking results")
        return pd.DataFrame()


def extract_all_poses(
    dirs: dict[str, Path],
    receptor_ids: list[str],
    receptor_map: dict[str, str],
    n_modes: int = 3,
) -> pd.DataFrame:
    """
    Extract docking scores for all receptors.

    Parameters:
        dirs: Directory dict.
        receptor_ids: List of receptor IDs.
        receptor_map: Dict mapping receptor_id -> cox_label.
        n_modes: Maximum number of poses to extract per task.

    Returns:
        Concatenated DataFrame of all poses.
    """
    poses_list = []
    for rec_id in receptor_ids:
        df_poses = extract_all_docking_scores(
            logs_dir=dirs[f"docking_logs_{rec_id}"],
            receptor_map=receptor_map,
            n_modes=n_modes,
        )
        poses_list.append(df_poses)
    if poses_list:
        df_all = pd.concat(poses_list, ignore_index=True)
        print(f"[parse poses] {len(df_all)} poses extracted ({n_modes} per task max)")
        return df_all
    return pd.DataFrame()


def select_best_poses_across(
    df_all_poses: pd.DataFrame,
    dirs: dict[str, Path],
    receptor_ids: list[str],
    receptor_pdb_map: dict[str, str],
) -> pd.DataFrame:
    """
    Select best poses across all receptors using geometric scoring.

    Parameters:
        df_all_poses: DataFrame with all poses.
        dirs: Directory dict.
        receptor_ids: List of receptor IDs.
        receptor_pdb_map: Dict mapping receptor_id -> receptor PDB path.

    Returns:
        DataFrame with best pose per ligand-receptor pair.
    """
    if df_all_poses.empty:
        return pd.DataFrame()
    best_list = []
    for rec_id in receptor_ids:
        df_subset = df_all_poses[df_all_poses["receptor_id"] == rec_id]
        if not df_subset.empty:
            df_best = select_best_poses_by_geo_score(
                df_subset,
                results_dir=dirs[f"docking_poses_{rec_id}"],
                receptor_pdb_map=receptor_pdb_map,
            )
            best_list.append(df_best)
    if best_list:
        df_best = pd.concat(best_list, ignore_index=True)
        print(f"[best poses] {len(df_best)} ligand-receptor pairs evaluated")
        return df_best
    return pd.DataFrame()


def save_docking_scores(
    df_analysis: pd.DataFrame,
    output_path: str | Path = "03_docking_pdbqts_data/outputs/docking_scores.csv",
) -> Path:
    """Save docking analysis scores to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_analysis.to_csv(output_path, index=False)
    print(f"[save_docking_scores] Saved {len(df_analysis)} entries to {os.path.relpath(output_path)}")
    return output_path


def save_docking_poses(
    df_ranked: pd.DataFrame,
    dirs: dict[str, Path],
    receptor_ids: list[str],
    output_dir: str | Path,
) -> None:
    """
    Copy PDBQT files to output directory.

    Parameters:
        df_ranked: Ranked DataFrame with ligand_id column.
        dirs: Directory dict.
        receptor_ids: List of receptor IDs.
        output_dir: Destination directory for pose files.
    """
    if df_ranked.empty:
        print("[poses] No poses to save")
        return
    output_dir = Path(output_dir)
    for _, row in df_ranked.iterrows():
        ligand_id = row["ligand_id"]
        for rec_id in receptor_ids:
            src = dirs[f"docking_poses_{rec_id}"] / f"{ligand_id}_{rec_id}_out.pdbqt"
            if src.exists():
                dst_dir = output_dir / rec_id
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / f"{ligand_id}_{rec_id}.pdbqt"
                dst.write_bytes(src.read_bytes())
    print(f"[poses] Poses saved to {os.path.relpath(output_dir)}")


def save_all_outputs(
    df_ranked: pd.DataFrame,
    dirs: dict[str, Path],
    output_dir: str | Path,
    receptor_ids: list[str] | None = None,
) -> None:
    """Save docking scores CSV and pose files. Handles guard clauses."""
    if df_ranked.empty:
        print("[save] No data to save")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if receptor_ids is None:
        receptor_ids = ["6COX", "3KK6"]

    csv_path = output_dir / "docking_scores.csv"
    save_docking_scores(df_ranked, csv_path)

    poses_dir = output_dir / "poses"
    save_docking_poses(df_ranked, dirs, receptor_ids, poses_dir)


def render_top_poses(
    df_ranked: pd.DataFrame,
    dirs: dict[str, Path],
    receptor_id: str,
    receptor_pdb: Path,
    output_dir: Path,
    vis_script: Path | None = None,
    top_n: int = 5,
) -> None:
    """
    Render top poses using PyMOL visualization script.

    Parameters:
        df_ranked: Ranked DataFrame with ligand_id column.
        dirs: Directory dict.
        receptor_id: Receptor ID to visualize (e.g., "6COX").
        receptor_pdb: Path to receptor PDB file.
        output_dir: Output directory for PNG files.
        vis_script: Path to visualize_pose.py (defaults to module-relative).
        top_n: Number of top compounds to render.
    """
    import subprocess
    from pathlib import Path
    from IPython.display import Image, display
    
    if df_ranked.empty:
        print("[visualization] No ranking data available")
        return
    
    if vis_script is None:
        vis_script = Path(__file__).parent / "_visualize_pose.py"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    top_n = min(top_n, len(df_ranked))
    print(f"[visualization] Rendering top {top_n} compounds ({receptor_id})")
    
    for rank, (_, row) in enumerate(df_ranked.head(top_n).iterrows(), 1):
        ligand_id = row["ligand_id"]
        ligand_pdbqt = dirs[f"docking_poses_{receptor_id}"] / f"{ligand_id}_{receptor_id}_out.pdbqt"
        
        if not ligand_pdbqt.exists():
            print(f"[visualization] #{rank} {ligand_id}: pose not found, skipping")
            continue
        
        output_png = output_dir / f"{ligand_id}_{receptor_id}.png"
        result = subprocess.run(
            ["python", str(vis_script), "--ligand", str(ligand_pdbqt),
             "--receptor", str(receptor_pdb), "--output", str(output_png)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and output_png.exists():
            print(f"[visualization] #{rank} {ligand_id}: saved {output_png.name}")
            display(Image(filename=str(output_png)))
        else:
            print(f"[visualization] #{rank} {ligand_id}: failed: {result.stderr[-300:]}")
