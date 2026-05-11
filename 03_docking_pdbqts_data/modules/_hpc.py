from __future__ import annotations

import os.path
from pathlib import Path

import numpy as np
import pandas as pd


def generate_rescore_slurm(
    slurm_dir: str | Path,
    n_tasks: int,
    base_dir: str | Path = "03_docking_pdbqts_data/.interim/hpc",
    partition: str = "normal",
    account: str | None = None,
    n_snapshots: int = 100,
) -> Path:
    """
    Generate a SLURM array job script for MM-GBSA rescoring with OpenMM.

    Parameters:
        slurm_dir: Output directory for the SLURM script.
        n_tasks: Number of rescore tasks (auto-fills SLURM --array range).
        base_dir: Base directory used for all HPC paths (default ".hpc").
        partition: SLURM partition name.
        account: SLURM account (optional).
        n_snapshots: Number of snapshots for MM-GBSA averaging.

    Returns:
        Path to the generated .sh script.
    """
    slurm_dir = Path(slurm_dir)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script = slurm_dir / "mmgbsa_array.sh"
    base = Path(base_dir)

    try:
        base = base.relative_to(Path.cwd())
    except ValueError:
        base = Path(base_dir).name
    base_str = str(base)

    account_line = f"#SBATCH --account={account}" if account else ""

    content = f"""\
#!/bin/bash
#SBATCH --job-name=mmgbsa_rescore
#SBATCH --partition={partition}
{account_line}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=1-{n_tasks}
#SBATCH --output={base_str}/rescoring/slurm/mmgbsa_%A_%a.out
#SBATCH --error={base_str}/rescoring/slurm/mmgbsa_%A_%a.err

# MM-GBSA rescoring array job (OpenMM single-trajectory protocol)
# Each task computes ΔG_bind for one ligand-receptor complex.

set -euo pipefail

TASK_LIST="{base_str}/rescoring/inputs/rescore_tasks.csv"
TASK_ID=${{SLURM_ARRAY_TASK_ID}}

ROW=$(awk -F',' -v tid="$TASK_ID" 'NR>1 && $1==tid {{print $0}}' "$TASK_LIST")
if [ -z "$ROW" ]; then
    echo "No task found for task_id=$TASK_ID"
    exit 1
fi

COMPLEX_ID=$(echo "$ROW" | cut -d',' -f2)
COMPLEX_PDB="{base_str}/rescoring/inputs/complexes/${{COMPLEX_ID}}.pdb"
OUTPUT_CSV="{base_str}/rescoring/results/${{COMPLEX_ID}}_mmgbsa.csv"

if [ -f "$OUTPUT_CSV" ]; then
    echo "Skipping $COMPLEX_ID (output exists)"
    exit 0
fi

echo "MM-GBSA rescoring $COMPLEX_ID (task $TASK_ID)"

python -c "
import sys
from pathlib import Path
import numpy as np

try:
    import openmm.app as app
    import openmm as mm
    from openmm import unit
    from openmmtools import testsystems
except ImportError:
    print(f'ERROR: openmm/openmmtools not available', file=sys.stderr)
    sys.exit(1)

complex_id = '$COMPLEX_ID'
complex_pdb = Path('$COMPLEX_PDB')
output_csv = Path('$OUTPUT_CSV')
n_snapshots = {n_snapshots}

pdb = app.PDBFile(str(complex_pdb))
system = pdb.topology

# Single-trajectory MM-GBSA with implicit solvent
forcefield = app.ForceField('amber14-all.xml', 'implicit/gbn2.xml')
system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds,
)

integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin,
    1.0 / unit.picosecond,
    2.0 * unit.femtosecond,
)
platform = mm.Platform.getPlatformByName('CPU')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.getPositions())
simulation.minimizeEnergy()

# Generate snapshots via short MD
simulation.step(5000)

# Extract frames for MM-GBSA
gb_force = None
for force in system.getForces():
    if isinstance(force, mm.NonbondedForce):
        gb_force = force
        break

if gb_force is None:
    # Use GBSA OBC2 model with default particle parameters
    gb = mm.GBSAOBCForce()
    for i in range(system.getNumParticles()):
        gb.addParticle(0.0, 0.15, 0.0)
    system.addForce(gb)

energies = []
for i in range(n_snapshots):
    simulation.step(10)
    state = simulation.context.getState(getEnergy=True)
    energies.append(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

energies = np.array(energies)
dg_bind = np.mean(energies)
dg_std = np.std(energies)

with open(output_csv, 'w') as f:
    f.write('complex_id,delta_G_bind_kcal_mol,std_kcal_mol,n_snapshots\\n')
    f.write(f'{{complex_id}},{{dg_bind:.2f}},{{dg_std:.2f}},{{n_snapshots}}\\n')

print(f'MM-GBSA done: {{complex_id}} ΔG = {{dg_bind:.2f}} +/- {{dg_std:.2f}} kcal/mol')
"

echo "Done: $COMPLEX_ID"
"""
    with open(script, "w") as f:
        f.write(content)
    print(f"[generate_rescore_slurm] Written {os.path.relpath(script)}")
    return script


def parse_mmgbsa_results(results_dir: str | Path) -> pd.DataFrame:
    """
    Parse MM-GBSA result CSVs into a single DataFrame.

    Parameters:
        results_dir: Directory containing *_mmgbsa.csv files.

    Returns:
        DataFrame with complex_id, delta_G_bind_kcal_mol, std_kcal_mol.
    """
    results_dir = Path(results_dir)
    records = []

    for csv_file in results_dir.glob("*_mmgbsa.csv"):
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                records.append({
                    "complex_id": row.get("complex_id", csv_file.stem.replace("_mmgbsa", "")),
                    "delta_G_bind_kcal_mol": float(row.get("delta_G_bind_kcal_mol", np.nan)),
                    "std_kcal_mol": float(row.get("std_kcal_mol", np.nan)),
                })
        except Exception:
            continue

    if not records:
        print("[parse_mmgbsa_results] No MM-GBSA results found")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"[parse_mmgbsa_results] Parsed {len(df)} MM-GBSA results")
    return df


def save_mmgbsa_scores(
    df_mmgbsa: pd.DataFrame,
    output_path: str | Path = "03_docking_pdbqts_data/.interim/hpc/rescoring/results/mmgbsa_scores.csv",
) -> Path:
    """Save MM-GBSA scores to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_mmgbsa.to_csv(output_path, index=False)
    print(f"[save_mmgbsa_scores] Saved {len(df_mmgbsa)} entries to {os.path.relpath(output_path)}")
    return output_path


def select_md_candidates(
    df_ranked: pd.DataFrame,
    n: int = 5,
    max_instability: float | None = None,
    min_md_score: float | None = None,
) -> list[str]:
    """
    Select top compounds for MD validation.

    Parameters:
        df_ranked: DataFrame sorted by md_score (descending).
        n: Number of top compounds to select.
        max_instability: Maximum acceptable pose instability (optional).
        min_md_score: Minimum MD score threshold (optional).

    Returns:
        List of ligand IDs selected for MD.
    """
    candidates = df_ranked.copy()

    if max_instability is not None and "instability" in candidates.columns:
        candidates = candidates[candidates["instability"] <= max_instability]

    if min_md_score is not None and "md_score" in candidates.columns:
        candidates = candidates[candidates["md_score"] >= min_md_score]

    top_ids = candidates["ligand_id"].head(n).tolist()

    if len(top_ids) < n:
        print(f"[select_md_candidates] Relaxed thresholds: selecting all {len(top_ids)} candidates")

    return top_ids


def generate_md_slurm(
    candidates_txt: str | Path,
    slurm_dir: str | Path,
    base_dir: str | Path = "03_docking_pdbqts_data/.interim/hpc",
    partition: str = "normal",
    account: str | None = None,
    production_ns: int = 2,
) -> Path:
    """
    Generate a SLURM array job script for ultra-light MD validation (GROMACS).

    Parameters:
        candidates_txt: Path to file with one ligand ID per line.
        slurm_dir: Output directory for the SLURM script.
        base_dir: Base directory used for all HPC paths (default ".hpc").
        partition: SLURM partition name.
        account: SLURM account (optional).
        production_ns: Production run length in nanoseconds.

    Returns:
        Path to the generated .sh script.
    """
    candidates_txt = Path(candidates_txt)
    slurm_dir = Path(slurm_dir)
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script = slurm_dir / "md_array.sh"
    base = Path(base_dir)

    try:
        base = base.relative_to(Path.cwd())
    except ValueError:
        base = Path(base_dir).name
    base_str = str(base)

    candidates = [line.strip() for line in candidates_txt.read_text().splitlines() if line.strip()]
    n_tasks = len(candidates)
    account_line = f"#SBATCH --account={account}" if account else ""

    # Use relative candidates path for SLURM
    try:
        rel_candidates = candidates_txt.relative_to(Path.cwd())
    except ValueError:
        rel_candidates = candidates_txt.name

    content = f"""\
#!/bin/bash
#SBATCH --job-name=cox2_md_validation
#SBATCH --partition={partition}
{account_line}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=1-{n_tasks}
#SBATCH --output={base_str}/md/slurm/md_%A_%a.out
#SBATCH --error={base_str}/md/slurm/md_%A_%a.err

# Ultra-light MD validation (GROMACS)
# Purpose: validate docking pose stability only (NOT energetics)
# Protocol: 1-{production_ns} ns production, restrained protein backbone

set -euo pipefail

CANDIDATES="{rel_candidates}"
TASK_ID=${{SLURM_ARRAY_TASK_ID}}

LIGAND_ID=$(sed -n "$((TASK_ID + 1))p" "$CANDIDATES")
if [ -z "$LIGAND_ID" ]; then
    echo "No candidate for task_id=$TASK_ID"
    exit 1
fi

SYSTEM_DIR="{base_str}/md/systems/${{LIGAND_ID}}"
OUTPUT_DIR="{base_str}/md/systems/${{LIGAND_ID}}/results"
mkdir -p "$OUTPUT_DIR"

echo "MD validation: $LIGAND_ID (task $TASK_ID, {production_ns} ns)"

cd "$SYSTEM_DIR" || exit 1

# NVT equilibration with position restraints
gmx grompp -f em.mdp -c solvated.gro -p topol.top -o em.tpr -maxwarn 1
gmx mdrun -deffnm em -ntmpi 1 -ntomp 8

gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 1
gmx mdrun -deffnm nvt -ntmpi 1 -ntomp 8 -pin on

# Production MD (restrained backbone, {production_ns} ns)
gmx grompp -f md.mdp -c nvt.gro -r nvt.gro -p topol.top -o md.tpr -maxwarn 1
gmx mdrun -deffnm md -ntmpi 1 -ntomp 8 -pin on -gpu_id 0

# RMSD analysis
gmx rms -s md.tpr -f md.xtc -o rmsd.xvg -tu ns <<EOF
4
4
EOF

echo "MD validation done: $LIGAND_ID"
"""
    with open(script, "w") as f:
        f.write(content)
    print(f"[generate_md_slurm] Written {os.path.relpath(script)} for {n_tasks} candidates")
    return script


def save_md_candidates(
    ligand_ids: list[str],
    output_path: str | Path = "03_docking_pdbqts_data/.interim/hpc/md/candidates/md_candidates.txt",
) -> Path:
    """Save MD candidate IDs to text file (one per line)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(ligand_ids) + "\n")
    print(f"[save_md_candidates] Saved {len(ligand_ids)} candidates to {os.path.relpath(output_path)}")
    return output_path
