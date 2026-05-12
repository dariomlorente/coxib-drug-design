from __future__ import annotations

import json
import os.path
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


_CONSERVED_COX_CA_MAPPING = {
    # COX-2 resnum -> COX-1 resnum (structurally equivalent Cα pairs for alignment)
    # Used for binding site transfer between 6COX (COX-2) and 3KK6 (COX-1)
    93: 93,   100: 100, 106: 106, 110: 110, 116: 116,
    120: 120, 123: 123, 127: 127, 133: 133, 349: 349,
    352: 352, 355: 355, 376: 376, 379: 379, 381: 381,
    384: 384, 385: 385, 387: 387, 390: 390, 504: 504,
    509: 509, 513: 513, 518: 518, 524: 524, 530: 530,
}


_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}


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


# =============================================================================
# ReceptorPreparator
# =============================================================================


class ReceptorPreparator:
    """
    Cleans PDB files, locates binding sites, and generates receptor PDBQTs.

    Wraps prepare_receptor() and get_binding_site_center(). Stores
    receptor_dir and box_size at construction time so the same preparator
    can be reused across multiple receptors with consistent box dimensions.

    Parameters
    ----------
    receptor_dir : str or Path
        Output directory for receptor files (PDBQTs, box JSONs).
    box_size : float, optional
        Binding box cube size in Angstroms. Default: 24.0.
    """

    def __init__(
        self,
        receptor_dir: str | Path,
        box_size: float = 24.0,
    ) -> None:
        self.receptor_dir = receptor_dir
        self.box_size = box_size

    def prepare(
        self,
        pdb_path: str | Path,
        receptor_id: str,
        **kwargs: Any,
    ) -> dict:
        """
        Clean a PDB file, compute binding box, and generate receptor PDBQT.

        Delegates to prepare_receptor() with stored receptor_dir and
        box_size.  Remaining kwargs are forwarded directly.

        Parameters
        ----------
        pdb_path : str or Path
            Path to the source PDB file.
        receptor_id : str
            Identifier for the receptor (e.g. '6COX', '3KK6').
        **kwargs
            Additional keyword arguments forwarded to prepare_receptor()
            (e.g. ligand_resname, override_box, reference_pdb).

        Returns
        -------
        dict
            Dict with receptor_id, pdbqt_path, box_center, box_size.
        """
        return prepare_receptor(
            pdb_path,
            self.receptor_dir,
            receptor_id,
            box_size=self.box_size,
            **kwargs,
        )

    def get_center(
        self,
        pdb_path: str | Path,
        **kwargs: Any,
    ) -> list[float]:
        """
        Compute binding site center for a receptor.

        Delegates to get_binding_site_center().  All kwargs forwarded
        directly.

        Parameters
        ----------
        pdb_path : str or Path
            Path to the PDB file.
        **kwargs
            Additional keyword arguments forwarded to
            get_binding_site_center()
            (e.g. ligand_resname, reference_pdb, reference_center).

        Returns
        -------
        list[float]
            [x, y, z] center coordinates.
        """
        return get_binding_site_center(pdb_path, **kwargs)
