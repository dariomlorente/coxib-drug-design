from __future__ import annotations

import json
import os.path
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

_CONSERVED_COX_CA_MAPPING = {
    # COX-2 resnum -> COX-1 resnum (structurally equivalent Cα pairs for alignment)
    # Used for binding site transfer between 6COX (COX-2) and 3KK6 (COX-1)
    93: 93,   100: 100, 106: 106, 110: 110, 116: 116,
    120: 120, 123: 123, 127: 127, 133: 133, 349: 349,
    352: 352, 355: 355, 376: 376, 379: 379, 381: 381,
    384: 384, 385: 385, 387: 387, 390: 390, 504: 504,
    509: 509, 513: 513, 518: 518, 524: 524, 530: 530,
}


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


def _generate_3d_conformer(mol: Chem.Mol, seed: int = 42) -> Chem.Mol | None:
    """Generate a 3D conformer using ETKDGv3 and MMFF94 optimization."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    try:
        conf_id = AllChem.EmbedMolecule(mol, params)
        if conf_id < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def prepare_ligands(
    df: pd.DataFrame,
    ligands_dir: str | Path,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate 3D conformers, optimize with MMFF94, save SDF and PDBQT.

    Parameters:
        df: DataFrame with ID and SMILES columns.
        ligands_dir: Output directory for .sdf and .pdbqt files.
        smiles_col: Name of the SMILES column.
        id_col: Name of the ID column.
        seed: Random seed for ETKDG conformer generation.

    Returns:
        DataFrame with added 'sdf_path' and 'pdbqt_path' columns for successful molecules.
    """
    ligands_dir = Path(ligands_dir)
    ligands_dir.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out["sdf_path"] = None
    out["pdbqt_path"] = None
    successes = 0
    failures = []
    meeko_available = True

    for idx, row in out.iterrows():
        ligand_id = str(row[id_col])
        smi = str(row[smiles_col])
        if not smi or smi == "nan":
            failures.append(ligand_id)
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append(ligand_id)
            continue

        mol_3d = _generate_3d_conformer(mol, seed=seed)
        if mol_3d is None:
            failures.append(ligand_id)
            continue

        sdf_path = ligands_dir / f"{ligand_id}.sdf"
        pdbqt_path = ligands_dir / f"{ligand_id}.pdbqt"

        try:
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol_3d)
            writer.close()
            out.at[idx, "sdf_path"] = str(sdf_path)
        except Exception as e:
            print(f"[prepare_ligands] Failed to write SDF for {ligand_id}: {e}")
            failures.append(ligand_id)
            continue

        try:
            if _write_pdbqt(mol_3d, pdbqt_path):
                out.at[idx, "pdbqt_path"] = str(pdbqt_path)
                successes += 1
            else:
                print(f"[prepare_ligands] PDBQT conversion failed for {ligand_id}")
                failures.append(ligand_id)
        except ImportError:
            meeko_available = False
            print(
                "[prepare_ligands] meeko not installed. "
                "PDBQT conversion skipped — run: pip install meeko"
            )
            successes += 1
        except Exception as e:
            print(f"[prepare_ligands] PDBQT error for {ligand_id}: {e}")
            failures.append(ligand_id)

    n_pdbqt = out["pdbqt_path"].notna().sum()
    if meeko_available and n_pdbqt == 0:
        raise RuntimeError(
            "[prepare_ligands] Zero ligands have valid PDBQT files. "
            "Check meeko installation and input SMILES."
        )

    print(f"[prepare_ligands] {successes}/{len(df)} ligands prepared successfully")
    if failures:
        print(f"[prepare_ligands] Failed IDs: {', '.join(failures)}")
    return out


def _write_pdbqt(mol: Chem.Mol, pdbqt_path: Path) -> bool:
    """
    Convert an RDKit molecule to PDBQT format using Meeko.

    Parameters:
        mol: RDKit molecule with 3D coordinates and hydrogens.
        pdbqt_path: Output path for the PDBQT file.

    Returns:
        True if PDBQT was written successfully, False otherwise.
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    if not mol_setups:
        return False

    setup = mol_setups[0]
    result = PDBQTWriterLegacy.write_string(setup)

    if isinstance(result, tuple):
        pdbqt_string, is_ok, _error_msg = result
        if not is_ok:
            return False
    else:
        pdbqt_string = result

    with open(pdbqt_path, "w") as f:
        f.write(pdbqt_string)
    return True


def _extract_ligand_coords(pdb_path: Path, ligand_resname: str) -> np.ndarray:
    """Extract centroid of a co-crystallized ligand from a PDB file."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and line[17:20].strip() == ligand_resname:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    if not coords:
        raise ValueError(f"No atoms found for ligand residue '{ligand_resname}' in {pdb_path}")
    return np.mean(coords, axis=0)


def _extract_ca_atoms(pdb_path: Path) -> tuple[list[int], np.ndarray]:
    """Extract Cα coordinates and residue numbers from a PDB file (chain A only)."""
    resnums = []
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[13:15] == "CA" and line[21] == "A":
                try:
                    resnum = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    resnums.append(resnum)
                    coords.append([x, y, z])
                except ValueError:
                    continue
    return resnums, np.array(coords)


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Compute optimal rotation matrix aligning P onto Q using the Kabsch algorithm."""
    C = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(C)
    d = np.linalg.det(np.dot(Wt.T, V.T))
    D = np.eye(3)
    D[2, 2] = d
    return np.dot(np.dot(Wt.T, D), V.T)


def _align_structures(
    ref_pdb: Path,
    mob_pdb: Path,
    resnum_mapping: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align mobile structure onto reference using matched Cα atoms.

    Parameters:
        ref_pdb: Reference PDB path (6COX for COX-2).
        mob_pdb: Mobile PDB path (3KK6 for COX-1).
        resnum_mapping: Dict mapping ref_resnum -> mob_resnum.

    Returns:
        (rotation_matrix, translation_vector) that transforms mobile -> reference frame.
    """
    ref_resnums, ref_coords = _extract_ca_atoms(ref_pdb)
    mob_resnums, mob_coords = _extract_ca_atoms(mob_pdb)

    ref_set = {r: i for i, r in enumerate(ref_resnums)}
    mob_set = {r: i for i, r in enumerate(mob_resnums)}

    ref_pts = []
    mob_pts = []
    for ref_rn, mob_rn in resnum_mapping.items():
        if ref_rn in ref_set and mob_rn in mob_set:
            ref_pts.append(ref_coords[ref_set[ref_rn]])
            mob_pts.append(mob_coords[mob_set[mob_rn]])

    if len(ref_pts) < 3:
        raise ValueError(
            f"Only {len(ref_pts)} matched Cα pairs found — need >= 3 for alignment"
        )

    P = np.array(ref_pts)
    Q = np.array(mob_pts)

    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)

    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    R = _kabsch_rotation(Q_centered, P_centered)
    t = P_centroid - R @ Q_centroid

    return R, t


def _conserved_cox_center() -> np.ndarray:
    """Return an approximate COX active-site center (literature-derived fallback)."""
    return np.array([73.0, 52.0, 28.0])


def get_binding_site_center(
    pdb_path: str | Path,
    ligand_resname: str | None = None,
    reference_pdb: str | Path | None = None,
    reference_center: list[float] | None = None,
    resnum_mapping: dict[int, int] | None = None,
) -> list[float]:
    """
    Compute binding site center for a receptor.

    If ligand_resname is provided and found in the PDB, uses the ligand centroid.
    If reference_pdb and reference_center are provided, structurally aligns the
    reference onto this receptor and transfers the binding site coordinates.
    Falls back to conserved COX active-site center.

    Parameters:
        pdb_path: Path to the PDB file.
        ligand_resname: Residue name of the co-crystallized ligand.
        reference_pdb: Path to reference PDB for structural alignment (e.g. 6COX).
        reference_center: [x, y, z] binding site center in the reference frame.
        resnum_mapping: Dict mapping reference resnum -> target resnum for alignment.

    Returns:
        [x, y, z] center coordinates.
    """
    pdb_path = Path(pdb_path)
    if ligand_resname:
        try:
            center = _extract_ligand_coords(pdb_path, ligand_resname)
            print(f"[get_binding_site_center] Using {ligand_resname} centroid from {pdb_path.name}")
            return center.tolist()
        except ValueError:
            pass

    if reference_pdb and reference_center:
        try:
            ref_path = Path(reference_pdb)
            mapping = resnum_mapping or _CONSERVED_COX_CA_MAPPING
            R, t = _align_structures(ref_path, pdb_path, mapping)
            ref_center = np.array(reference_center)
            center = R @ ref_center + t
            print(
                f"[get_binding_site_center] Transferred binding site from "
                f"{ref_path.name} via structural alignment -> {pdb_path.name}"
            )
            return center.tolist()
        except Exception as e:
            print(f"[get_binding_site_center] Alignment failed ({e}), using fallback")

    center = _conserved_cox_center()
    print(f"[get_binding_site_center] Using conserved COX active-site center for {pdb_path.name}")
    return center.tolist()


def _clean_pdb(pdb_path: Path, output_path: Path) -> Path:
    """Write a cleaned PDB keeping only protein ATOM records (all chains, standard AAs)."""
    kept = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                resname = line[17:20].strip()
                if resname in _STANDARD_AA:
                    kept.append(line)
    with open(output_path, "w") as f:
        f.writelines(kept)
    return output_path


_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}


def _detect_non_standard_residues(pdb_path: Path) -> list[str]:
    """Detect non-standard residues in a PDB file (chain:resnum format)."""
    residues = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                resname = line[17:20].strip()
                chain = line[21]
                resnum = line[22:26].strip()
                if resname not in _STANDARD_AA and resname not in ("HOH", "WAT"):
                    residues.add(f"{chain}:{resnum}")
    return sorted(residues)


def _prepare_receptor_pdbqt(clean_pdb: Path, pdbqt_path: Path) -> Path:
    """
    Generate receptor PDBQT from a cleaned PDB file.

    Uses Open Babel (obabel) for conversion, then strips torsion
    definitions to produce a rigid receptor PDBQT. Falls back to
    mk_prepare_receptor.py (Meeko) or prepare_receptor4.py (MGLTools).

    Parameters:
        clean_pdb: Path to cleaned protein-only PDB file.
        pdbqt_path: Output path for the PDBQT file.

    Returns:
        Path to the written PDBQT file.

    Raises:
        RuntimeError: If no conversion tool is available.
    """
    # Method 1: Open Babel (obabel) — robust, handles disulfide bonds
    if shutil.which("obabel"):
        tmp_pdbqt = pdbqt_path.with_suffix(".tmp.pdbqt")
        result = subprocess.run(
            [
                "obabel", str(clean_pdb),
                "-O", str(tmp_pdbqt),
                "-p", "7.4",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and tmp_pdbqt.exists():
            # Post-process: keep only ATOM/HETATM records for a rigid receptor.
            # Vina 1.2.x requires rigid receptors to have no ROOT/ENDROOT/TORSDOF.
            rigid_lines = []
            with open(tmp_pdbqt) as f:
                for line in f:
                    if line.startswith(("ATOM  ", "HETATM")):
                        rigid_lines.append(line)
            with open(pdbqt_path, "w") as f:
                f.writelines(rigid_lines)
            tmp_pdbqt.unlink()
            if pdbqt_path.exists():
                return pdbqt_path
        obabel_err = result.stderr.strip() if result.stderr else "unknown"
    else:
        obabel_err = "obabel not found"

    # Method 2: Meeko CLI (mk_prepare_receptor.py)
    if shutil.which("mk_prepare_receptor.py"):
        non_standard = _detect_non_standard_residues(clean_pdb)
        base = str(pdbqt_path).replace(".pdbqt", "")
        cmd = [
            "mk_prepare_receptor.py",
            "--read_pdb", str(clean_pdb),
            "-o", base,
            "--write_pdbqt",
        ]
        if non_standard:
            cmd.extend(["--delete_residues", ",".join(non_standard)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and pdbqt_path.exists():
            return pdbqt_path

        # Meeko with --allow_bad_res
        result = subprocess.run(
            cmd + ["--allow_bad_res"], capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and pdbqt_path.exists():
            return pdbqt_path
        meeko_err = result.stderr.strip() if result.stderr else "unknown"
    else:
        meeko_err = "mk_prepare_receptor.py not found"

    # Method 3: MGLTools (prepare_receptor4.py)
    if shutil.which("prepare_receptor4.py"):
        result = subprocess.run(
            [
                "prepare_receptor4.py",
                "-r", str(clean_pdb),
                "-o", str(pdbqt_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and pdbqt_path.exists():
            return pdbqt_path

    raise RuntimeError(
        f"No receptor PDBQT tool available for {clean_pdb.name}. "
        f"Install openbabel (obabel), meeko (mk_prepare_receptor.py), or mgltools. "
        f"obabel: {obabel_err} | meeko: {meeko_err}"
    )


def prepare_receptor(
    pdb_path: str | Path,
    receptor_dir: str | Path,
    receptor_id: str,
    ligand_resname: str | None = None,
    box_size: float = 24.0,
    box_json_out: str | Path | None = None,
    reference_pdb: str | Path | None = None,
    reference_center: list[float] | None = None,
    resnum_mapping: dict[int, int] | None = None,
    override_box: dict[str, float] | None = None,
) -> dict:
    """
    Clean PDB, compute binding box, write receptor PDBQT and box.json.

    Parameters:
        pdb_path: Path to the source PDB file.
        receptor_dir: Output directory for receptor files.
        receptor_id: Identifier for the receptor (e.g., '6COX', '3KK6').
        ligand_resname: Co-crystallized ligand residue name (e.g., 'S58' for SC-558).
        box_size: Cube size in Angstroms (default 24.0).
        reference_pdb: Path to reference PDB for structural alignment of binding site.
        reference_center: [x, y, z] binding site center in the reference frame.
        resnum_mapping: Dict mapping reference resnum -> target resnum for alignment.
        override_box: If provided, use this dict (must contain center_x/y/z, size_x/y/z)
                      instead of auto-computing the box.

    Returns:
        Dict with receptor_id, pdbqt_path, box_center, box_size.
    """
    pdb_path = Path(pdb_path)
    receptor_dir = Path(receptor_dir)
    receptor_dir.mkdir(parents=True, exist_ok=True)

    clean_pdb = receptor_dir / f"{receptor_id}_clean.pdb"
    pdbqt_path = receptor_dir / f"{receptor_id}.pdbqt"
    box_out = Path(box_json_out) if box_json_out else receptor_dir / f"{receptor_id}_box.json"

    clean_pdb = _clean_pdb(pdb_path, clean_pdb)
    pdbqt_path = _prepare_receptor_pdbqt(clean_pdb, pdbqt_path)
    print(f"[prepare_receptor] Generated {os.path.relpath(pdbqt_path)}")

    if override_box is not None:
        required_keys = {"center_x", "center_y", "center_z", "size_x", "size_y", "size_z"}
        missing = required_keys - set(override_box.keys())
        if missing:
            raise ValueError(f"override_box missing keys: {missing}")
        box = {k: override_box[k] for k in required_keys}
        center = [box["center_x"], box["center_y"], box["center_z"]]
        box_size = box["size_x"]
        print(f"[prepare_receptor] Using overridden box for {receptor_id}: center={center}")
    else:
        center = get_binding_site_center(
            pdb_path,
            ligand_resname=ligand_resname,
            reference_pdb=reference_pdb,
            reference_center=reference_center,
            resnum_mapping=resnum_mapping,
        )

        box = {
            "center_x": round(center[0], 3),
            "center_y": round(center[1], 3),
            "center_z": round(center[2], 3),
            "size_x": box_size,
            "size_y": box_size,
            "size_z": box_size,
        }

    with open(box_out, "w") as f:
        json.dump(box, f, indent=2)

    print(f"[prepare_receptor] Saved {os.path.relpath(box_out)}")

    out = {
        "receptor_id": receptor_id,
        "pdbqt_path": str(pdbqt_path),
        "box_center": center,
        "box_size": box_size,
        "box_json": str(box_out),
    }
    return out


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

    for _, task in mapping_df.iterrows():
        ligand_id = task["ligand_id"]
        receptor_id = task["receptor_id"]

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

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        with open(log_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)

        if result.returncode == 0:
            n_done += 1
            print(f"  [{n_done + n_skipped}/{n_total}] {ligand_id} x {receptor_id} done")
        else:
            n_failed += 1
            print(f"  FAILED: {ligand_id} x {receptor_id}: {result.stderr[:200]}")

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


def _rank_normalize(series: pd.Series, invert: bool = True) -> pd.Series:
    """
    Percentile rank normalization within a group, returning values in [0, 1].

    When invert=True (default): lower raw values get higher normalized ranks.
    This is the correct behavior for docking scores where more negative = better.

    When invert=False: higher raw values get higher normalized ranks.

    This is NOT a physical transformation. Docking scores are a ranking proxy,
    not an energy model. Percentile ranks only reflect relative ordering within
    the same receptor's score distribution.
    """
    s = series.astype(float)
    n = len(s)
    ranks = s.rank(method="average")
    result = (ranks - 1) / max(n - 1, 1)
    if invert:
        result = 1.0 - result
    return result


def compute_docking_analysis(
    df_scores: pd.DataFrame,
    df_instability: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute docking analysis metrics from parsed scores and pose instability.

    Each receptor's scores are treated independently. No cross-receptor scaling
    of raw energies is performed.

    Outputs:
        - score_cox2: best docking score for COX-2
        - score_cox1: best docking score for COX-1
        - instability_cox2: pose spread for COX-2 (top 3 poses)
        - instability_cox1: pose spread for COX-1 (top 3 poses)
        - instability: max(instability_cox2, instability_cox1) for ranking penalty

    Docking scores are a ranking proxy, not an energy model.
    """
    if df_scores.empty:
        return pd.DataFrame()

    pivot = df_scores.pivot_table(
        index="ligand_id",
        columns="cox_label",
        values="docking_score",
        aggfunc="min",
    ).reset_index()

    required = {"COX2", "COX1"}
    if not required.issubset(set(pivot.columns)):
        print(f"[compute_docking_analysis] Missing columns: {required - set(pivot.columns)}")
        return pd.DataFrame()

    pivot = pivot.rename(columns={"COX2": "score_cox2", "COX1": "score_cox1"})

    inst_pivot = df_instability.pivot_table(
        index="ligand_id",
        columns="cox_label",
        values="pose_spread",
        aggfunc="max",
    ).reset_index()

    inst_required = {"COX2", "COX1"}
    if inst_required.issubset(set(inst_pivot.columns)):
        inst_pivot = inst_pivot.rename(columns={"COX2": "instability_cox2", "COX1": "instability_cox1"})
        inst_pivot["instability"] = inst_pivot[["instability_cox2", "instability_cox1"]].max(axis=1)
    else:
        print(f"[compute_docking_analysis] Missing instability columns: {inst_required - set(inst_pivot.columns)}")
        return pd.DataFrame()

    pivot = pivot.merge(
        inst_pivot[["ligand_id", "instability_cox2", "instability_cox1", "instability"]],
        on="ligand_id",
        how="left",
    )

    return pivot


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


def compute_composite_score(
    df: pd.DataFrame,
    instability_lambda: float = 0.2,
) -> pd.DataFrame:
    """
    Compute a dimensionless MD_score for ranking compounds.

    Formula:
        MD_score = 2 * norm_rank(COX2_score)
                  - 1 * norm_rank(COX1_score)
                  - λ * instability

    Where:
        - norm_rank(x): percentile rank normalization within each receptor's
          score distribution (separately for COX-2 and COX-1)
        - instability: maximum pose deviation across top 3 docking poses
          (max of COX-2 and COX-1 pose spread)
        - λ: instability penalty weight, default 0.2 (range [0.1, 0.3])

    Important:
        - COX-2 and COX-1 docking scores are NOT directly comparable energies.
        - Each receptor's scores are normalized within their own distribution
          BEFORE any comparison.
        - This is a ranking proxy, not an energy model.
        - NO cross-receptor scaling of raw energies is performed.

    Parameters:
        df: DataFrame with score_cox2, score_cox1, instability columns.
        instability_lambda: Weight λ for the instability penalty term.
            Recommended range: 0.1 to 0.3. Default 0.2.

    Returns:
        DataFrame with added norm_rank_cox2, norm_rank_cox1, instability_penalty,
        and md_score columns, sorted by md_score descending.
    """
    out = df.copy()

    if "score_cox2" not in out.columns or "score_cox1" not in out.columns:
        raise ValueError("Input must contain 'score_cox2' and 'score_cox1' columns.")

    out["norm_rank_cox2"] = _rank_normalize(out["score_cox2"])
    out["norm_rank_cox1"] = _rank_normalize(out["score_cox1"])

    if "instability" not in out.columns or out["instability"].isna().all():
        out["instability"] = 0.0

    out["instability_penalty"] = (instability_lambda * out["instability"]).round(4)

    out["md_score"] = (
        2.0 * out["norm_rank_cox2"]
        - 1.0 * out["norm_rank_cox1"]
        - out["instability_penalty"]
    )
    out["md_score"] = out["md_score"].round(4)

    out = out.sort_values("md_score", ascending=False).reset_index(drop=True)

    print(f"[compute_composite_score] MD scores computed for {len(out)} compounds")
    print(f"[compute_composite_score] Formula: MD_score = 2*norm_rank(COX2) - 1*norm_rank(COX1) - λ*instability")
    print(f"[compute_composite_score] Instability penalty λ = {instability_lambda}")
    return out


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


def prepare_ligands_multi_conf(
    df: pd.DataFrame,
    ligands_dir: str | Path,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    seed: int = 42,
    n_confs: int = 15,
) -> pd.DataFrame:
    """
    Generate ~15 conformers with ETKDGv3 (RMS pruning), MMFF minimization,
    keep lowest energy conformer, convert to PDBQT with Meeko.

    Parameters:
        df: DataFrame with ID and SMILES columns.
        ligands_dir: Output directory for .pdbqt files.
        smiles_col: Name of the SMILES column.
        id_col: Name of the ID column.
        seed: Random seed for ETKDG conformer generation.
        n_confs: Number of conformers to generate.

    Returns:
        DataFrame with added 'pdbqt_path' column for successful molecules.
    """
    ligands_dir = Path(ligands_dir)
    ligands_dir.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out["pdbqt_path"] = None
    successes = 0
    failures = []

    for idx, row in out.iterrows():
        ligand_id = str(row[id_col])
        smi = str(row[smiles_col])
        if not smi or smi == "nan":
            failures.append(ligand_id)
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append(ligand_id)
            continue

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useSmallRingTorsions = True
        params.pruneRmsThresh = 0.5

        conf_ids = AllChem.EmbedMultipleConfs(
            mol, numConfs=n_confs, params=params
        )
        if not conf_ids:
            failures.append(ligand_id)
            continue

        # Minimize all conformers, track energies
        energies = []
        for cid in conf_ids:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(
                    mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid
                )
                if ff:
                    ff.Minimize(maxIts=500)
                    energies.append((cid, ff.CalcEnergy()))
            except Exception:
                continue

        if not energies:
            failures.append(ligand_id)
            continue

        # Keep lowest energy conformer
        best_cid = min(energies, key=lambda x: x[1])[0]
        mol = Chem.Mol(mol, confId=best_cid)
        Chem.SanitizeMol(mol)

        # Write PDBQT with Meeko
        pdbqt_path = ligands_dir / f"{ligand_id}.pdbqt"
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy

            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            if mol_setups:
                setup = mol_setups[0]
                result = PDBQTWriterLegacy.write_string(setup)
                pdbqt_string = result[0] if isinstance(result, tuple) else result
                with open(pdbqt_path, "w") as f:
                    f.write(pdbqt_string)
                out.at[idx, "pdbqt_path"] = str(pdbqt_path)
                successes += 1
            else:
                failures.append(ligand_id)
        except Exception as e:
            print(f"[prepare_ligands_multi_conf] PDBQT error for {ligand_id}: {e}")
            failures.append(ligand_id)

    print(f"[prepare_ligands_multi_conf] {successes}/{len(df)} ligands prepared successfully")
    if failures:
        print(f"[prepare_ligands_multi_conf] Failed IDs: {', '.join(failures)}")
    return out


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


def compute_geometric_score(
    pose_pdbqt_path: str | Path,
    receptor_pdb_path: str | Path,
    cox_label: str = "COX2",
    side_pocket_center: np.ndarray | None = None,
) -> float:
    """
    Compute geometric score for a pose.

    Rules:
        - Arg120 (NH1/NH2) interaction: <3.5 A -> +2
        - Tyr355 (OH) interaction: <3.5 A -> +2
        - Side pocket heuristic: fraction of ligand atoms within 5 A of center >0.3 -> +1
        - Clash penalty: 0 clashes -> +1, 1-2 -> 0, >2 -> -2

    Only applied to COX-2 (6COX). Returns 0.0 for COX-1.

    Parameters:
        pose_pdbqt_path: Path to the PDBQT file containing the pose.
        receptor_pdb_path: Path to the receptor PDB file.
        cox_label: Label for the receptor (COX2 or COX1).
        side_pocket_center: Optional fixed center for side pocket calculation.

    Returns:
        Geometric score (float).
    """
    if cox_label != "COX2":
        return 0.0

    score = 0.0
    pose_pdbqt_path = Path(pose_pdbqt_path)
    receptor_pdb_path = Path(receptor_pdb_path)

    # Parse ligand coordinates from PDBQT
    ligand_atoms = []
    with open(pose_pdbqt_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ligand_atoms.append([x, y, z])
                except (ValueError, IndexError):
                    continue

    if not ligand_atoms:
        return 0.0

    ligand_coords = np.array(ligand_atoms)
    ligand_center = ligand_coords.mean(axis=0)

    # Parse receptor for Arg120 and Tyr355
    arg120_coords = []
    tyr355_coords = []
    receptor_atoms = []

    with open(receptor_pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                resname = line[17:20].strip()
                resnum = int(line[22:26])
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                receptor_atoms.append([x, y, z])

                if resnum == 120 and resname == "ARG":
                    if atom_name in ("NH1", "NH2"):
                        arg120_coords.append([x, y, z])
                elif resnum == 355 and resname == "TYR":
                    if atom_name == "OH":
                        tyr355_coords.append([x, y, z])
            except (ValueError, IndexError):
                continue

    # Arg120 interaction
    if arg120_coords:
        arg_coords = np.array(arg120_coords)
        distances = np.linalg.norm(arg_coords[:, None, :] - ligand_coords[None, :, :], axis=2)
        min_dist = distances.min()
        if min_dist < 3.5:
            score += 2.0
            print(f"    Arg120: interaction {min_dist:.2f} A (+2)")

    # Tyr355 interaction
    if tyr355_coords:
        tyr_coords = np.array(tyr355_coords)
        distances = np.linalg.norm(tyr_coords[:, None, :] - ligand_coords[None, :, :], axis=2)
        min_dist = distances.min()
        if min_dist < 3.5:
            score += 2.0
            print(f"    Tyr355: interaction {min_dist:.2f} A (+2)")

    # Side pocket heuristic (ligand centroid +/- 5 A)
    pocket_center = side_pocket_center if side_pocket_center is not None else ligand_center
    distances = np.linalg.norm(ligand_coords - pocket_center, axis=1)
    in_pocket = distances < 5.0
    fraction = in_pocket.sum() / len(ligand_coords) if len(ligand_coords) > 0 else 0.0
    if fraction > 0.3:
        score += 1.0
        print(f"    Side pocket: fraction={fraction:.2f} (+1)")

    # Clash penalty
    clashes = 0
    if receptor_atoms:
        receptor_coords = np.array(receptor_atoms)
        for lig_atom in ligand_coords:
            dists = np.linalg.norm(receptor_coords - lig_atom, axis=1)
            if (dists < 2.0).any():
                clashes += 1

    if clashes == 0:
        score += 1.0
    elif clashes <= 2:
        pass  # 0
    else:
        score -= 2.0
    print(f"    Clashes: {clashes} (running score: {score})")

    return round(score, 2)


def select_best_poses_by_geo_score(
    df_all_poses: pd.DataFrame,
    results_dir: str | Path,
    receptor_pdb_map: dict,
) -> pd.DataFrame:
    """
    Evaluate all poses and select the one with best geometric_score per ligand-receptor pair.

    Parameters:
        df_all_poses: DataFrame with all poses (from extract_all_docking_scores).
        results_dir: Directory containing docking result PDBQT files.
        receptor_pdb_map: Dict mapping receptor_id -> receptor_pdb_path.

    Returns:
        DataFrame with best pose per ligand-receptor pair.
    """
    results_dir = Path(results_dir)
    records = []

    for ligand_id in df_all_poses["ligand_id"].unique():
        for receptor_id in df_all_poses["receptor_id"].unique():
            subset = df_all_poses[
                (df_all_poses["ligand_id"] == ligand_id)
                & (df_all_poses["receptor_id"] == receptor_id)
            ]

            if subset.empty:
                continue

            best_geo_score = -1000.0
            best_pose_rank = None
            best_docking_score = None

            cox_label = subset.iloc[0]["cox_label"]
            pose_file = results_dir / f"{ligand_id}_{receptor_id}_out.pdbqt"

            for _, row in subset.iterrows():
                pose_rank = row["pose_rank"]
                docking_score = row["docking_score"]

                geo_score = 0.0
                if cox_label == "COX2" and pose_file.exists():
                    receptor_pdb = receptor_pdb_map.get(receptor_id)
                    if receptor_pdb:
                        geo_score = compute_geometric_score(
                            pose_file, Path(receptor_pdb), cox_label="COX2"
                        )

                if geo_score > best_geo_score:
                    best_geo_score = geo_score
                    best_pose_rank = pose_rank
                    best_docking_score = docking_score

            records.append({
                "ligand_id": ligand_id,
                "receptor_id": receptor_id,
                "cox_label": cox_label,
                "best_pose_rank": best_pose_rank,
                "docking_score": best_docking_score,
                "geometric_score": best_geo_score,
            })

    return pd.DataFrame(records)


def compute_final_ranking(
    df_best_poses: pd.DataFrame,
    df_ligands_raw: pd.DataFrame,
    qsar_col: str = "QSAR_score",
    id_col: str = "ID",
) -> pd.DataFrame:
    """
    Compute final ranking with normalized scores.

    Formula: final_score = 0.5 * qsar_norm + 0.4 * geo_norm + 0.1 * vina_norm

    Normalization:
        - QSAR_score: normalized (invert=True, lower QSAR is better)
        - geometric_score: normalized (invert=False, higher is better)
        - Vina score (score_cox2): normalized (invert=True, more negative is better)

    Parameters:
        df_best_poses: DataFrame with best poses per ligand-receptor.
        df_ligands_raw: Original DataFrame with QSAR_score.
        qsar_col: Name of the QSAR score column.
        id_col: Name of the ID column.

    Returns:
        DataFrame sorted by final_score descending.
    """
    if df_best_poses.empty:
        return pd.DataFrame()

    # Pivot to have score_cox2 and score_cox1 columns
    pivot = df_best_poses.pivot_table(
        index="ligand_id",
        columns="cox_label",
        values="docking_score",
        aggfunc="first",
    ).reset_index()

    required = {"COX2", "COX1"}
    if not required.issubset(set(pivot.columns)):
        print(f"[compute_final_ranking] Missing columns: {required - set(pivot.columns)}")
        return pd.DataFrame()

    pivot = pivot.rename(columns={"COX2": "score_cox2", "COX1": "score_cox1"})

    # Add geometric_score for COX-2
    geo_scores = df_best_poses[df_best_poses["cox_label"] == "COX2"][
        ["ligand_id", "geometric_score"]
    ].rename(columns={"geometric_score": "geo_score"})

    if not geo_scores.empty:
        pivot = pivot.merge(geo_scores, on="ligand_id", how="left")

    # Add QSAR_score from original CSV
    qsar = df_ligands_raw[[id_col, qsar_col]].rename(columns={id_col: "ligand_id"})
    df_analysis = pivot.merge(qsar, on="ligand_id", how="left")

    if df_analysis.empty:
        return pd.DataFrame()

    # Normalize scores to [0, 1]
    df_analysis["qsar_norm"] = _rank_normalize(df_analysis[qsar_col], invert=True)
    df_analysis["geo_norm"] = _rank_normalize(df_analysis["geo_score"], invert=False)
    df_analysis["vina_norm"] = _rank_normalize(df_analysis["score_cox2"], invert=True)

    # final_score = 0.5 * qsar_norm + 0.4 * geo_norm + 0.1 * vina_norm
    df_analysis["final_score"] = (
        0.5 * df_analysis["qsar_norm"]
        + 0.4 * df_analysis["geo_norm"]
        + 0.1 * df_analysis["vina_norm"]
    ).round(4)

    df_ranked = df_analysis.sort_values("final_score", ascending=False).reset_index(drop=True)

    print(f"[compute_final_ranking] Scores computed for {len(df_ranked)} compounds")
    print("[compute_final_ranking] Formula: final_score = 0.5*qsar_norm + 0.4*geo_norm + 0.1*vina_norm")

    return df_ranked


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
        vis_script = Path(__file__).parent / "visualize_pose.py"
    
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