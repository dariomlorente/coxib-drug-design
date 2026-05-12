from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


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


# =============================================================================
# LigandPreparator
# =============================================================================


class LigandPreparator:
    """
    Prepares 3D ligand conformers and PDBQT files for docking.

    Wraps prepare_ligands() and prepare_ligands_multi_conf(). Stores seed,
    n_confs, smiles_col, and id_col at construction time so the same
    preparator can be reused across multiple compound series.

    Parameters
    ----------
    seed : int, optional
        Random seed for ETKDGv3 conformer generation. Default: 42.
    n_confs : int, optional
        Number of conformers for multi-conformer preparation. Default: 15.
    smiles_col : str, optional
        Column name for SMILES strings. Default: "SMILES".
    id_col : str, optional
        Column name for compound IDs. Default: "ID".
    """

    def __init__(
        self,
        seed: int = 42,
        n_confs: int = 15,
        smiles_col: str = "SMILES",
        id_col: str = "ID",
    ) -> None:
        self.seed = seed
        self.n_confs = n_confs
        self.smiles_col = smiles_col
        self.id_col = id_col

    def prepare(
        self,
        df: pd.DataFrame,
        ligands_dir: str | Path,
    ) -> pd.DataFrame:
        """
        Generate 3D conformers and PDBQT files.

        Delegates to prepare_ligands() with stored parameters.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with ID and SMILES columns.
        ligands_dir : str or Path
            Output directory for .sdf and .pdbqt files.

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'sdf_path' and 'pdbqt_path' columns.
        """
        return prepare_ligands(
            df, ligands_dir,
            smiles_col=self.smiles_col,
            id_col=self.id_col,
            seed=self.seed,
        )

    def prepare_multi_conf(
        self,
        df: pd.DataFrame,
        ligands_dir: str | Path,
    ) -> pd.DataFrame:
        """
        Generate multi-conformer 3D structures and PDBQT files.

        Delegates to prepare_ligands_multi_conf() with stored parameters.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with ID and SMILES columns.
        ligands_dir : str or Path
            Output directory for .pdbqt files.

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'pdbqt_path' column.
        """
        return prepare_ligands_multi_conf(
            df, ligands_dir,
            smiles_col=self.smiles_col,
            id_col=self.id_col,
            seed=self.seed,
            n_confs=self.n_confs,
        )
