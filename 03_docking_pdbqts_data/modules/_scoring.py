from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


# =============================================================================
# DockingScorer
# =============================================================================


class DockingScorer:
    """
    Computes composite docking scores and final compound rankings.

    Wraps compute_docking_analysis(), compute_composite_score(), and
    compute_final_ranking(). Stores instability_lambda at construction time.

    Parameters
    ----------
    instability_lambda : float, optional
        Weight for the instability penalty term in MD_score formula.
        Recommended range: 0.1–0.3. Default: 0.2.
    """

    def __init__(self, instability_lambda: float = 0.2) -> None:
        self.instability_lambda = instability_lambda
        self.df_analysis: pd.DataFrame | None = None
        self.df_composite: pd.DataFrame | None = None

    def analyse(
        self,
        df_scores: pd.DataFrame,
        df_instability: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute docking analysis metrics from parsed scores and instability.

        Delegates to compute_docking_analysis().  Stores the result as
        ``self.df_analysis``.

        Parameters
        ----------
        df_scores : pd.DataFrame
            DataFrame with ligand_id, cox_label, docking_score.
        df_instability : pd.DataFrame
            DataFrame with ligand_id, cox_label, pose_spread.

        Returns
        -------
        pd.DataFrame
            Pivot DataFrame with score_cox2, score_cox1, instability.
        """
        result = compute_docking_analysis(df_scores, df_instability)
        self.df_analysis = result
        return result

    def composite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute MD_score composite ranking.

        Delegates to compute_composite_score() with stored
        instability_lambda.  Stores the result as ``self.df_composite``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with score_cox2, score_cox1, instability columns.

        Returns
        -------
        pd.DataFrame
            DataFrame sorted by md_score descending.
        """
        result = compute_composite_score(df, instability_lambda=self.instability_lambda)
        self.df_composite = result
        return result

    def rank(
        self,
        df_best_poses: pd.DataFrame,
        df_ligands_raw: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute final ranking with normalised scores.

        Delegates to compute_final_ranking().  Only ``qsar_col`` and
        ``id_col`` are forwarded; all other kwargs are silently ignored
        (``compute_final_ranking`` has a fixed parameter list).

        Parameters
        ----------
        df_best_poses : pd.DataFrame
            DataFrame with best poses per ligand-receptor pair.
        df_ligands_raw : pd.DataFrame
            Original DataFrame with QSAR_score.
        **kwargs
            Only ``qsar_col`` and ``id_col`` are forwarded to
            compute_final_ranking().

        Returns
        -------
        pd.DataFrame
            DataFrame sorted by final_score descending.
        """
        accepted = {"qsar_col", "id_col"}
        safe_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return compute_final_ranking(df_best_poses, df_ligands_raw, **safe_kwargs)

    def run_full(
        self,
        df_scores: pd.DataFrame,
        df_instability: pd.DataFrame,
        df_ligands_raw: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Run the full docking scoring pipeline.

        Chains .analyse() -> .composite() -> .rank() in sequence.
        Intermediate results are stored as ``self.df_analysis`` and
        ``self.df_composite``.

        Parameters
        ----------
        df_scores : pd.DataFrame
            Scores DataFrame with ligand_id, cox_label, docking_score.
        df_instability : pd.DataFrame
            Instability DataFrame with ligand_id, cox_label, pose_spread.
        df_ligands_raw : pd.DataFrame
            Original DataFrame with QSAR_score.
        **kwargs
            Additional keyword arguments forwarded to .rank().

        Returns
        -------
        pd.DataFrame
            DataFrame sorted by final_score descending.
        """
        self.df_analysis = self.analyse(df_scores, df_instability)
        self.df_composite = self.composite(self.df_analysis)
        return self.rank(df_scores, df_ligands_raw, **kwargs)
