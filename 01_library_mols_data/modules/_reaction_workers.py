from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from ._utils import _get_cache_key, REACTION_CACHE


DEFAULT_CACHE_DIR = Path(REACTION_CACHE)

_REACTION_DESCRIPTOR_COLUMNS = {
    "MW",
    "HBD",
    "HBA",
    "RotB",
    "HvyAtm",
    "Rings",
    "HetAtm",
    "MR",
    "CAtm",
    "Atoms",
    "tPSA",
    "LogP",
}

_SMARTS_EP_REACTION = (
    "([#6:42][CX3H:41](=[O:91])).([#6:21][CX3:2](=[O:1])[OX1H0-,OX2H1,Cl,Br,I:92])."
    "([O:51]=[C:5]([OH1:90])[CH2:4][NH2:3])>>"
    "([O:51]=[C:5]1[O:1][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
    "([OH:90]).([OH:91]).([O:92])"
)
_SMARTS_AG_REACTION = (
    "([O:51]=[C:5]1[O:1][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
    "([NX3H2:10][*:11])>>"
    "([O:51]=[C:5]1[NX3:10]([*:11])[C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
    "([OX2H2:1])"
)
_SMARTS_SE_REACTION = (
    "([O:51]=[C:5]1[O:1][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
    "([C:93][C:94](=[O:85])[SX2H:10])>>"
    "([O:51]=[C:5]1[SX2:10][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42])."
    "([C:93][C:94](=[O:85])[OX2H:1])"
)


def _append_to_temp_csv(
    out_rows: list[dict],
    temp_csv_path: Path,
    is_first_chunk: bool = True,
) -> None:
    if not out_rows:
        return
    temp_df = pd.DataFrame(out_rows)
    mode = "w" if is_first_chunk else "a"
    header = is_first_chunk
    temp_df.to_csv(temp_csv_path, mode=mode, index=False, header=header)


def _preview_columns(columns: list[str], max_items: int = 6) -> str:
    if len(columns) <= max_items:
        return ", ".join(columns)
    shown = ", ".join(columns[:max_items])
    return f"{shown} ... (+{len(columns) - max_items} more)"


def _prepare_reaction_inputs(
    df: pd.DataFrame,
    smiles_col: str,
    id_col: str,
    price_col: str,
    reaction_name: str,
    print_report: bool,
) -> pd.DataFrame:
    required_cols = [smiles_col, id_col, price_col]
    required_unique = list(dict.fromkeys(required_cols))
    missing = [col for col in required_unique if col not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns for {reaction_name}: {missing_text}")

    dropped_cols = [col for col in df.columns if col not in required_unique]
    dropped_descriptors = [col for col in dropped_cols if col in _REACTION_DESCRIPTOR_COLUMNS]

    if print_report and dropped_descriptors:
        descriptor_text = _preview_columns(sorted(dropped_descriptors))
        print(f"[{reaction_name}] Removed descriptor columns before reaction: {descriptor_text}")

    if print_report and dropped_cols:
        print(
            f"[{reaction_name}] Using minimal reaction columns "
            f"{required_unique} (dropped {len(dropped_cols)} non-reactive columns)"
        )

    return df.loc[:, required_unique].copy()


def _cached_product_smiles(cached_result: object) -> list[str]:
    if not isinstance(cached_result, list):
        return []

    smiles: list[str] = []
    for item in cached_result:
        if isinstance(item, str):
            smiles.append(item)
            continue

        if isinstance(item, dict):
            maybe_smi = item.get("smiles") or item.get("SMILES")
            if isinstance(maybe_smi, str) and maybe_smi:
                smiles.append(maybe_smi)

    return list(dict.fromkeys(smiles))


def _format_reaction_stats(stats: dict, reaction_name: str) -> str:
    out = stats["output_rows"]
    inp_a = stats.get("input_aldehydes") or stats.get("input_oxazolones")
    inp_b = stats.get("input_carboxylics") or stats.get("input_amines")

    line = f"[{reaction_name}] out={out:,} rows | inp={inp_a:,}"
    if inp_b is not None:
        line += f"×{inp_b:,}"

    issues = {}
    for key, label in [
        ("skipped_price", "skipped"),
        ("no_product", "no_prod"),
        ("problematic", "probl"),
    ]:
        if stats.get(key, 0) > 0:
            issues[label] = stats[key]

    if issues:
        issue_str = ", ".join(f"{k}={v:,}" for k, v in issues.items())
        line += f" | ⚠️ {issue_str}"

    return line


def _process_ep_batch(
    work_items: list[tuple[int, int]],
    ald_data: list[tuple[str, str, float]],
    carb_data: list[tuple[str, str, float]],
    smiles_gly: str,
    keep_mol: bool = False,
) -> tuple[list[dict], dict[str, list[str]], dict[str, int]]:
    rxn_ep = rdChemReactions.ReactionFromSmarts(_SMARTS_EP_REACTION)
    mol_gly = Chem.MolFromSmiles(smiles_gly)

    out_rows: list[dict] = []
    new_cache: dict[str, list[str]] = {}
    stats = {"no_product": 0, "problematic": 0}

    for ald_idx, carb_idx in work_items:
        ald_smi, ald_id, ald_price = ald_data[ald_idx]
        carb_smi, carb_id, carb_price = carb_data[carb_idx]
        cache_key = _get_cache_key(ald_smi, carb_smi, smiles_gly)

        ald_mol = Chem.MolFromSmiles(ald_smi)
        carb_mol = Chem.MolFromSmiles(carb_smi)

        if ald_mol is None or carb_mol is None or mol_gly is None:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        try:
            products = rxn_ep.RunReactants((ald_mol, carb_mol, mol_gly))
        except Exception:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        if not products:
            new_cache[cache_key] = []
            stats["no_product"] += 1
            continue

        price_total = ald_price + carb_price
        seen: set[str] = set()
        product_smiles: list[str] = []

        for prod_tuple in products:
            try:
                prod_mol = prod_tuple[0]
                Chem.SanitizeMol(prod_mol)
                psmi = Chem.MolToSmiles(prod_mol)
                if psmi in seen:
                    continue
                seen.add(psmi)
                product_smiles.append(psmi)

                ald_num = ald_id.lstrip("A")
                carb_num = carb_id.lstrip("C")
                new_row: dict = {
                    "ID": f"A{ald_num}C{carb_num}",
                    "SMILES": psmi,
                    "PriceMol": price_total,
                }
                if keep_mol:
                    new_row["Mol"] = prod_mol
                out_rows.append(new_row)
            except Exception:
                stats["problematic"] += 1

        new_cache[cache_key] = product_smiles

    return out_rows, new_cache, stats


def _process_ag_batch(
    ox_chunk: list[tuple[str, str, float]],
    work_items: list[tuple[int, int]],
    am_data: list[tuple[str, str, float]],
    keep_mol: bool = False,
) -> tuple[list[dict], dict[str, list[str]], dict[str, int]]:
    rxn_ag = rdChemReactions.ReactionFromSmarts(_SMARTS_AG_REACTION)

    out_rows: list[dict] = []
    new_cache: dict[str, list[str]] = {}
    stats = {"no_product": 0, "problematic": 0}

    for ox_idx, am_idx in work_items:
        ox_smi, ox_id, ox_price = ox_chunk[ox_idx]
        am_smi, am_id, am_price = am_data[am_idx]
        cache_key = _get_cache_key(ox_smi, am_smi)

        ox_mol = Chem.MolFromSmiles(ox_smi)
        am_mol = Chem.MolFromSmiles(am_smi)

        if ox_mol is None or am_mol is None:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        try:
            products = rxn_ag.RunReactants((ox_mol, am_mol))
        except Exception:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        if not products:
            new_cache[cache_key] = []
            stats["no_product"] += 1
            continue

        price_total = ox_price + am_price
        seen: set[str] = set()
        product_smiles: list[str] = []

        for prod_tuple in products:
            try:
                prod_mol = prod_tuple[0]
                Chem.SanitizeMol(prod_mol)
                psmi = Chem.MolToSmiles(prod_mol)
                if psmi in seen:
                    continue
                seen.add(psmi)
                product_smiles.append(psmi)

                am_num = am_id.lstrip("N")
                new_row: dict = {
                    "ID": f"{ox_id}N{am_num}",
                    "SMILES": psmi,
                    "PriceMol": price_total,
                }
                if keep_mol:
                    new_row["Mol"] = prod_mol
                out_rows.append(new_row)
            except Exception:
                stats["problematic"] += 1

        new_cache[cache_key] = product_smiles

    return out_rows, new_cache, stats


def _process_se_batch(
    ox_chunk: list[tuple[str, str, float]],
    smiles_thioacetic: str,
    thioacetic_price_eq: float,
) -> tuple[list[dict], dict[str, list[str]], dict[str, int]]:
    rxn_se = rdChemReactions.ReactionFromSmarts(_SMARTS_SE_REACTION)
    mol_thioacetic = Chem.MolFromSmiles("[C:93][C:94](=[O:85])[SH:10]")

    out_rows: list[dict] = []
    new_cache: dict[str, list[str]] = {}
    stats = {"no_product": 0, "problematic": 0}

    for ox_smi, ox_id, ox_price in ox_chunk:
        cache_key = _get_cache_key(ox_smi, smiles_thioacetic)

        ox_mol = Chem.MolFromSmiles(ox_smi)

        if ox_mol is None or mol_thioacetic is None:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        try:
            products = rxn_se.RunReactants((ox_mol, mol_thioacetic))
        except Exception:
            new_cache[cache_key] = []
            stats["problematic"] += 1
            continue

        if not products:
            new_cache[cache_key] = []
            stats["no_product"] += 1
            continue

        seen: set[str] = set()
        product_smiles: list[str] = []

        for prod_tuple in products:
            try:
                prod_mol = prod_tuple[0]
                Chem.SanitizeMol(prod_mol)
                psmi = Chem.MolToSmiles(prod_mol)
                if psmi in seen:
                    continue
                seen.add(psmi)
                product_smiles.append(psmi)

                new_row: dict = {
                    "ID": f"{ox_id}S",
                    "SMILES": psmi,
                    "PriceMol": ox_price + thioacetic_price_eq,
                }
                out_rows.append(new_row)
            except Exception:
                stats["problematic"] += 1

        new_cache[cache_key] = product_smiles

    return out_rows, new_cache, stats
