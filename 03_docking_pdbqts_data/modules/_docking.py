from __future__ import annotations

import json
import os.path
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def init_hpc_dirs(
    base: str | Path = "03_docking_pdbqts_data/.interim",
    receptor_ids: list[str] | None = None,
) -> dict[str, Path]:
    """Create the job directory tree and return paths dict."""
    base = Path(base)
    dirs: dict[str, Path] = {
        "ligands": base / "ligands",
        "receptors": base / "receptors",
        "docking_mapping": base,
    }
    if receptor_ids:
        for rid in receptor_ids:
            dirs[f"docking_logs_{rid}"] = base / "docking" / rid / "logs"
            dirs[f"docking_poses_{rid}"] = base / "docking" / rid / "poses"
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"[init_hpc_dirs] Created {len(dirs)} directories under {os.path.relpath(base)}")
    return dirs


def write_mapping_csv(
    df_ligands: pd.DataFrame,
    receptor_ids: list[str],
    mapping_csv: str | Path,
    id_col: str = "ID",
) -> Path:
    """
    Create mapping.csv with all ligand × receptor combinations.

    Parameters:
        df_ligands: DataFrame with ligand IDs.
        receptor_ids: List of receptor identifiers.
        mapping_csv: Output path for mapping.csv.
        id_col: Name of the ligand ID column.

    Returns:
        Path to the generated mapping file.
    """
    mapping_csv = Path(mapping_csv)
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    task_id = 1
    for _, ligand_row in df_ligands.iterrows():
        for rec_id in receptor_ids:
            rows.append({
                "task_id": task_id,
                "ligand_id": str(ligand_row[id_col]),
                "receptor_id": rec_id,
            })
            task_id += 1

    mapping_df = pd.DataFrame(rows)
    mapping_df.to_csv(mapping_csv, index=False)
    print(f"[write_mapping_csv] {len(mapping_df)} docking tasks written to {os.path.relpath(mapping_csv)}")
    return mapping_csv


def _run_single_docking_task(task_args: dict) -> dict:
    """
    Run a single Vina docking task (module-level for picklability).

    Parameters:
        task_args: Dict with keys: vina_exe_path, ligand_pdbqt, receptor_pdbqt,
            box, seed, log_path, pose_path, timeout, exhaustiveness, num_modes,
            n_cpu, ligand_id, receptor_id.

    Returns:
        Dict with keys: status ('done' | 'failed'), ligand_id, receptor_id,
        and optionally 'error'.
    """
    vina_exe_path = task_args["vina_exe_path"]
    ligand_pdbqt = task_args["ligand_pdbqt"]
    receptor_pdbqt = task_args["receptor_pdbqt"]
    box = task_args["box"]
    seed = task_args["seed"]
    log_path = task_args["log_path"]
    pose_path = task_args["pose_path"]
    timeout = task_args["timeout"]
    exhaustiveness = task_args["exhaustiveness"]
    num_modes = task_args["num_modes"]
    n_cpu = task_args["n_cpu"]
    ligand_id = task_args["ligand_id"]
    receptor_id = task_args["receptor_id"]

    cmd = [
        vina_exe_path,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--cpu", str(n_cpu),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--out", str(pose_path),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        with open(log_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)

        if result.returncode == 0:
            return {"status": "done", "ligand_id": ligand_id, "receptor_id": receptor_id}
        return {
            "status": "failed",
            "ligand_id": ligand_id,
            "receptor_id": receptor_id,
            "error": result.stderr[:200],
        }
    except subprocess.TimeoutExpired:
        return {"status": "failed", "ligand_id": ligand_id, "receptor_id": receptor_id, "error": "timeout"}
    except Exception:
        return {"status": "failed", "ligand_id": ligand_id, "receptor_id": receptor_id, "error": "exception"}


def run_local_docking(
    mapping_df: pd.DataFrame,
    dirs: dict[str, Path],
    *,
    vina_exe: str = "vina",
    exhaustiveness: int = 16,
    num_modes: int = 3,
    n_cpu: int = 4,
    timeout: int = 300,
    force_redock: bool = False,
    seed: int = 42,
    n_workers: int | None = None,
) -> dict:
    """
    Run Vina docking locally for each task in the mapping DataFrame.

    Parameters:
        mapping_df: DataFrame with columns ligand_id, receptor_id.
        dirs: Directory dict from init_hpc_dirs or equivalent.
              Must contain keys: ligands, receptors, and
              docking_logs_{rid}, docking_poses_{rid} for each receptor.
        vina_exe: Path to Vina binary.
        exhaustiveness: Vina exhaustiveness parameter.
        num_modes: Maximum number of binding modes to output.
        n_cpu: Number of CPU cores for each Vina invocation.
        timeout: Per-docking timeout in seconds.
        force_redock: Re-run even if output files already exist.
        seed: Random seed.
        n_workers: Number of parallel Vina jobs. Each job uses n_cpu cores.
            When None, computed automatically as max(1, cpu_count // n_cpu)
            to fill all available cores. Set to 1 to disable parallelism.

    Returns:
        Dict with counts: completed, skipped, failed, total, total_time_min.
    """
    vina_exe_path = shutil.which(vina_exe) if "/" not in vina_exe else vina_exe
    if vina_exe_path is None:
        raise RuntimeError(
            f"AutoDock Vina not found at '{vina_exe}'. "
            "Install with: conda install -c conda-forge vina=1.2.5"
        )
    print(f"[docking] Using Vina at: {Path(vina_exe_path).name}")

    if mapping_df.empty:
        print("[docking] No mapping available")
        return {"completed": 0, "skipped": 0, "failed": 0, "total": 0, "total_time_min": 0.0}

    ligands_dir = dirs["ligands"]
    receptors_dir = dirs["receptors"]
    receptor_ids = mapping_df["receptor_id"].unique()

    box_configs: dict[str, dict] = {}
    for rec_id in receptor_ids:
        box_path = receptors_dir / f"{rec_id}_box.json"
        with open(box_path) as f:
            box_configs[rec_id] = json.load(f)

    # Ensure log/pose directories exist for every receptor in the mapping
    for rec_id in receptor_ids:
        dirs.get(f"docking_logs_{rec_id}", Path()).mkdir(parents=True, exist_ok=True)
        dirs.get(f"docking_poses_{rec_id}", Path()).mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_skipped = 0
    n_failed = 0
    n_total = len(mapping_df)
    start = time.time()

    # Build task list with pre-flight skip/fail checks
    task_list: list[dict] = []
    for _, task_row in mapping_df.iterrows():
        ligand_id = task_row["ligand_id"]
        receptor_id = task_row["receptor_id"]

        logs_dir = dirs[f"docking_logs_{receptor_id}"]
        poses_dir = dirs[f"docking_poses_{receptor_id}"]
        log_path = logs_dir / f"{ligand_id}_{receptor_id}.log"
        pose_path = poses_dir / f"{ligand_id}_{receptor_id}_out.pdbqt"

        if log_path.exists() and pose_path.exists() and not force_redock:
            n_skipped += 1
            continue

        ligand_pdbqt = ligands_dir / f"{ligand_id}.pdbqt"
        receptor_pdbqt = receptors_dir / f"{receptor_id}.pdbqt"
        box = box_configs[receptor_id]

        if not ligand_pdbqt.exists() or not receptor_pdbqt.exists():
            print(f"[docking] Missing input for {ligand_id} x {receptor_id}")
            n_failed += 1
            continue

        task_list.append({
            "vina_exe_path": vina_exe_path,
            "ligand_id": ligand_id,
            "receptor_id": receptor_id,
            "ligand_pdbqt": ligand_pdbqt,
            "receptor_pdbqt": receptor_pdbqt,
            "box": box,
            "log_path": log_path,
            "pose_path": pose_path,
            "timeout": timeout,
            "exhaustiveness": exhaustiveness,
            "num_modes": num_modes,
            "seed": seed,
        })

    # Determine parallelism level and per-job CPU allocation
    if n_workers is None:
        ncpu_avail = os.cpu_count() or 1
        n_workers = max(1, ncpu_avail // n_cpu)

    n_cpu_per_job = max(1, n_cpu // n_workers) if n_workers > 1 else n_cpu

    for task in task_list:
        task["n_cpu"] = n_cpu_per_job

    if n_workers <= 1:
        # Sequential fallback — identical behavior to the original loop
        for task in task_list:
            result = _run_single_docking_task(task)
            if result["status"] == "done":
                n_done += 1
                print(f"  [{n_done + n_skipped}/{n_total}] {result['ligand_id']} x {result['receptor_id']} done")
            else:
                n_failed += 1
                print(f"  FAILED: {result['ligand_id']} x {result['receptor_id']}: {result.get('error', '')}")
    else:
        # Parallel execution via thread pool
        print(f"[docking] Running {len(task_list)} tasks with {n_workers} workers ({n_cpu_per_job} CPU per job)")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_run_single_docking_task, task) for task in task_list]
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "done":
                    n_done += 1
                    print(f"  [{n_done + n_skipped}/{n_total}] {result['ligand_id']} x {result['receptor_id']} done")
                else:
                    n_failed += 1
                    print(f"  FAILED: {result['ligand_id']} x {result['receptor_id']}: {result.get('error', '')}")

    elapsed = time.time() - start
    elapsed_min = elapsed / 60.0
    print(f"\n[docking] {n_done} completed, {n_skipped} skipped, {n_failed} failed")
    print(f"[docking] Total time: {elapsed_min:.1f} min")

    return {
        "completed": n_done,
        "skipped": n_skipped,
        "failed": n_failed,
        "total": n_total,
        "total_time_min": round(elapsed_min, 1),
    }


def generate_docking_slurm(
    mapping_csv: str | Path,
    slurm_dir: str | Path,
    n_tasks: int,
    base_dir: str | Path = "03_docking_pdbqts_data/.interim/hpc",
    vina_exe: str = "vina",
    partition: str = "normal",
    account: str | None = None,
) -> Path:
    """
    Generate a SLURM array job script for AutoDock Vina docking.

    Parameters:
        mapping_csv: Path to mapping.csv with task_id, ligand_id, receptor_id.
        slurm_dir: Output directory for the SLURM script.
        n_tasks: Number of docking tasks (auto-fills SLURM --array range).
        base_dir: Base directory used for all HPC paths (default ".hpc").
        vina_exe: Vina executable name.
        partition: SLURM partition name.
        account: SLURM account (optional).

    Returns:
        Path to the generated .sh script.
    """
    slurm_dir = Path(slurm_dir)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script = slurm_dir / "docking_array.sh"
    base = Path(base_dir)

    try:
        base = base.relative_to(Path.cwd())
    except ValueError:
        base = Path(base_dir).name
    base_str = str(base)

    account_line = f"#SBATCH --account={account}" if account else ""

    content = f"""\
#!/bin/bash
#SBATCH --job-name=vina_docking
#SBATCH --partition={partition}
{account_line}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --array=1-{n_tasks}
#SBATCH --output={base}/docking/slurm/vina_%A_%a.out
#SBATCH --error={base}/docking/slurm/vina_%A_%a.err

# AutoDock Vina array job
# Each task reads one row from the mapping file and docks one ligand-receptor pair.

set -euo pipefail

# Read mapping for this task
MAPPING="{base}/docking/mapping.csv"
TASK_ID=${{SLURM_ARRAY_TASK_ID}}

# Parse the CSV (skip header, get row at TASK_ID)
ROW=$(awk -F',' -v tid="$TASK_ID" 'NR>1 && $1==tid {{print $0}}' "$MAPPING")
if [ -z "$ROW" ]; then
    echo "No mapping found for task_id=$TASK_ID"
    exit 1
fi

LIGAND_ID=$(echo "$ROW" | cut -d',' -f2)
RECEPTOR_ID=$(echo "$ROW" | cut -d',' -f3)

LIGAND_PDBQT="{base}/ligands/${{LIGAND_ID}}.pdbqt"
RECEPTOR_PDBQT="{base}/receptors/${{RECEPTOR_ID}}.pdbqt"
BOX_JSON="{base}/receptors/${{RECEPTOR_ID}}_box.json"
OUTPUT_LOG="{base}/docking/${{RECEPTOR_ID}}/logs/${{LIGAND_ID}}_${{RECEPTOR_ID}}.log"
OUTPUT_POSE="{base}/docking/${{RECEPTOR_ID}}/results/${{LIGAND_ID}}_${{RECEPTOR_ID}}_out.pdbqt"

# Skip if output already exists
if [ -f "$OUTPUT_POSE" ] && [ -f "$OUTPUT_LOG" ]; then
    echo "Skipping $LIGAND_ID x $RECEPTOR_ID (outputs exist)"
    exit 0
fi

# Read box parameters
CENTER_X=$(python -c "import json; b=json.load(open('$BOX_JSON')); print(b['center_x'])")
CENTER_Y=$(python -c "import json; b=json.load(open('$BOX_JSON')); print(b['center_y'])")
CENTER_Z=$(python -c "import json; b=json.load(open('$BOX_JSON')); print(b['center_z'])")
SIZE_X=$(python -c "import json; b=json.load(open('$BOX_JSON')); print(b['size_x'])")
SIZE_Y=$(python -c "import json; b=json.load(open('$BOX_JSON')); print(b['size_y'])")
SIZE_Z=$(python -c "import json; b=json.load(open('$BOX_JSON')); print(b['size_z'])")

echo "Docking $LIGAND_ID x $RECEPTOR_ID (task $TASK_ID)"

{vina_exe} \\
    --receptor "$RECEPTOR_PDBQT" \\
    --ligand "$LIGAND_PDBQT" \\
    --center_x "$CENTER_X" --center_y "$CENTER_Y" --center_z "$CENTER_Z" \\
    --size_x "$SIZE_X" --size_y "$SIZE_Y" --size_z "$SIZE_Z" \\
    --cpu 8 \\
    --exhaustiveness 16 \\
    --num_modes 20 \\
    --energy_range 5 \\
    --out "$OUTPUT_POSE" \\
    > "$OUTPUT_LOG" 2>&1

echo "Done: $LIGAND_ID x $RECEPTOR_ID"
"""
    with open(script, "w") as f:
        f.write(content)
    print(f"[generate_docking_slurm] Written {os.path.relpath(script)}")
    return script
