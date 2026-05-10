from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def convert_pdbqt_to_pdb(pdbqt_path: Path, pdb_path: Path) -> None:
    result = subprocess.run(
        ["obabel", str(pdbqt_path), "-O", str(pdb_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"obabel failed: {result.stderr}")


def render_pose(receptor_pdb: Path, ligand_pdb: Path, output_png: Path) -> None:
    import pymol
    from pymol import cmd

    pymol.finish_launching(["pymol", "-qc"])

    cmd.load(str(receptor_pdb), "receptor")
    cmd.load(str(ligand_pdb), "ligand")

    cmd.show("cartoon", "receptor")
    cmd.color("lightblue", "receptor")
    cmd.show("sticks", "ligand")
    cmd.color("magenta", "ligand")

    cmd.center("ligand")
    cmd.zoom("ligand", 3)

    cmd.ray(1200, 900)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(str(output_png), dpi=300, ray=1)

    cmd.delete("all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render docking pose with PyMOL")
    parser.add_argument("--ligand", required=True, help="Ligand PDBQT file")
    parser.add_argument("--receptor", required=True, help="Receptor PDB file")
    parser.add_argument("--output", required=True, help="Output PNG file")
    args = parser.parse_args()

    ligand_pdbqt = Path(args.ligand)
    receptor_pdb = Path(args.receptor)
    output_png = Path(args.output)

    if not ligand_pdbqt.exists():
        print(f"[visualize_pose] Ligand not found: {ligand_pdbqt}")
        sys.exit(1)
    if not receptor_pdb.exists():
        print(f"[visualize_pose] Receptor not found: {receptor_pdb}")
        sys.exit(1)

    tmp_pdb = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp_pdb = Path(f.name)

        convert_pdbqt_to_pdb(ligand_pdbqt, tmp_pdb)
        render_pose(receptor_pdb, tmp_pdb, output_png)
        print(f"[visualize_pose] Saved: {output_png}")

    except Exception as e:
        print(f"[visualize_pose] Error: {e}")
        sys.exit(1)

    finally:
        if tmp_pdb and tmp_pdb.exists():
            tmp_pdb.unlink()


if __name__ == "__main__":
    main()
