from __future__ import annotations

from bisect import bisect_right
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from ._utils import _get_cache_key, _get_n_workers, _load_cache, _save_cache


DEFAULT_CACHE_DIR = Path("mol_files/3. Oxazolones/.cache")

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


def rxn_ErlenmeyerPlochl(
    df_aldehyde: pd.DataFrame,
    df_carboxylic: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col_aldehyde: str = "ID",
    id_col_carboxyl: str = "ID",
    price_col: str = "PriceMol",
    keep_mol: bool = False,
    print_report: bool = True,
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    n_workers: int | None = None,
    chunk_size: int = 25,
    output_csv: str | Path | None = None,
    checkpoint_csv: str | Path | None = None,
    checkpoint_manager: Any = None,
    max_price_mol: float | None = None,
) -> pd.DataFrame:
    """
    Erlenmeyer-Plöchl reaction: combines aldehydes with carboxylic acids
    (or derivatives) and glycine to form oxazolones.

    Uses chunked streaming to limit memory usage - processes aldehydes in
    batches of `chunk_size`, writing intermediate results to disk.

    Supports robust checkpoint-based resume using CheckpointManager.

    Parameters:
        df_aldehyde: DataFrame with aldehyde compounds.
        df_carboxylic: DataFrame with carboxylic acid/derivative compounds.
        smiles_col: Column containing SMILES (default: ``"SMILES"``).
        id_col_aldehyde: ID column in aldehyde DataFrame (default: ``"ID"``).
        id_col_carboxyl: ID column in carboxylic DataFrame (default: ``"ID"``).
        price_col: Column with prices (default: ``"PriceMol"``).
        keep_mol: Keep RDKit mol objects (default: False).
        print_report: Print statistics (default: True).
        use_cache: Use persistent cache (default: True).
        cache_file: Cache file path (default: auto-generated).
        n_workers: Number of parallel workers (default: cpu_count - 1).
        chunk_size: Number of aldehydes per chunk (default: 25).
        output_csv: If provided, write final result to this CSV file.
        checkpoint_csv: Legacy parameter (ignored if checkpoint_manager provided).
        checkpoint_manager: CheckpointManager instance for robust resume support.
        max_price_mol: Optional hard cutoff for product price per mole.
            Pairs with ``aldehyde_price + carboxylic_price > max_price_mol`` are skipped.

    Returns:
        DataFrame with oxazolone products. Outputs contain reaction-native columns
        (``ID``, ``SMILES``, ``PriceMol``).
    """
    from .pipeline import CheckpointManager

    if checkpoint_manager is None and output_csv is not None:
        output_path = Path(output_csv)
        stage_name = "Oxazolones"
        checkpoint_manager = CheckpointManager(stage_name, output_path.parent)

    for df, name, id_col in [
        (df_aldehyde, "aldehyde", id_col_aldehyde),
        (df_carboxylic, "carboxylic", id_col_carboxyl),
    ]:
        if smiles_col not in df.columns:
            raise ValueError(f"Missing '{smiles_col}' column in {name} DataFrame.")
        if id_col not in df.columns:
            raise ValueError(f"Missing '{id_col}' column in {name} DataFrame.")
        if price_col not in df.columns:
            raise ValueError(f"Missing '{price_col}' column in {name} DataFrame.")
    if max_price_mol is not None and max_price_mol < 0:
        raise ValueError("max_price_mol must be non-negative.")

    df_aldehyde = _prepare_reaction_inputs(
        df_aldehyde, smiles_col, id_col_aldehyde, price_col, "ErlenmeyerPlochl", print_report
    )
    df_carboxylic = _prepare_reaction_inputs(
        df_carboxylic, smiles_col, id_col_carboxyl, price_col, "ErlenmeyerPlochl", print_report
    )

    if cache_file is None:
        cache_file = DEFAULT_CACHE_DIR / "erlenmeyer_plochl_cache.json.gz"
    else:
        cache_file = Path(cache_file)

    cache = _load_cache(cache_file) if use_cache else {}
    cache_hits = 0
    cache_misses = 0
    new_cache_entries: dict[str, list[str]] = {}

    patt_aldehyde = Chem.MolFromSmarts("[#6:42][CX3H:41](=[O:91])")
    patt_carboxyl = Chem.MolFromSmarts("[#6:21][CX3:2](=[O:1])[OX1H0-,OX2H1,Cl,Br,I:92]")

    smiles_gly = "O=[C:5]([OH1:6])[CH2:4][NH2:8]"
    mol_gly = Chem.MolFromSmiles(smiles_gly)
    if mol_gly is None:
        raise ValueError("Failed to build glycine reagent Mol (invalid SMILES).")

    rxn_ep = rdChemReactions.ReactionFromSmarts(_SMARTS_EP_REACTION)
    if rxn_ep is None:
        raise ValueError("Failed to build reaction from SMARTS.")

    stats = {
        "input_aldehydes": len(df_aldehyde),
        "input_carboxylics": len(df_carboxylic),
        "invalid_aldehyde": 0,
        "invalid_carboxyl": 0,
        "not_aldehyde": 0,
        "not_carboxyl": 0,
        "no_product": 0,
        "problematic": 0,
        "skipped_price": 0,
        "output_rows": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    ald_smiles = df_aldehyde[smiles_col].astype(str)
    ald_mols = ald_smiles.map(Chem.MolFromSmiles)
    ald_valid = ald_mols.notna()
    ald_match = ald_valid & ald_mols.map(
        lambda m: m.HasSubstructMatch(patt_aldehyde) if m is not None else False
    )

    valid_aldehydes: list[tuple[str, str, float]] = [
        (smi, str(id_), float(price))
        for smi, id_, price in zip(
            df_aldehyde.loc[ald_match, smiles_col],
            df_aldehyde.loc[ald_match, id_col_aldehyde],
            df_aldehyde.loc[ald_match, price_col],
        )
    ]
    stats["invalid_aldehyde"] = int((~ald_valid).sum())
    stats["not_aldehyde"] = int((ald_valid & ~ald_match).sum())

    carb_smiles = df_carboxylic[smiles_col].astype(str)
    carb_mols = carb_smiles.map(Chem.MolFromSmiles)
    carb_valid = carb_mols.notna()
    carb_match = carb_valid & carb_mols.map(
        lambda m: m.HasSubstructMatch(patt_carboxyl) if m is not None else False
    )

    valid_carboxylics: list[tuple[str, str, float]] = [
        (smi, str(id_), float(price))
        for smi, id_, price in zip(
            df_carboxylic.loc[carb_match, smiles_col],
            df_carboxylic.loc[carb_match, id_col_carboxyl],
            df_carboxylic.loc[carb_match, price_col],
        )
    ]
    stats["invalid_carboxyl"] = int((~carb_valid).sum())
    stats["not_carboxyl"] = int((carb_valid & ~carb_match).sum())

    if checkpoint_csv is not None:
        temp_csv_path = Path(checkpoint_csv)
    elif checkpoint_manager is not None:
        temp_csv_path = checkpoint_manager.path.parent / ".tmp_ep_results.csv"
    elif output_csv is not None:
        temp_csv_path = Path(output_csv).parent / ".cache" / ".tmp_ep_results.csv"
    else:
        temp_csv_path = cache_file.parent / ".tmp_ep_results.csv"
    temp_csv_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ald_ids: set[str] = set()
    is_resuming = False
    total_chunks = (len(valid_aldehydes) + chunk_size - 1) // chunk_size

    if checkpoint_manager is not None:
        completed_ald_ids = checkpoint_manager.get_completed_ids("aldehyde")
        if len(completed_ald_ids) > 0:
            is_resuming = True
            if print_report:
                print(
                    f"[ErlenmeyerPlochl] Resuming from checkpoint: "
                    f"{len(completed_ald_ids):,} aldehydes already processed"
                )

    if not is_resuming and temp_csv_path.exists():
        try:
            checkpoint_df = pd.read_csv(temp_csv_path)
            if "ID" in checkpoint_df.columns:
                completed_ald_ids = set(
                    id_str.rsplit("C", 1)[0]
                    for id_str in checkpoint_df["ID"].unique()
                    if "C" in id_str
                )
                is_resuming = True
                if print_report:
                    print(
                        f"[ErlenmeyerPlochl] Resuming from CSV checkpoint: "
                        f"{len(completed_ald_ids):,} aldehydes already processed"
                    )
                if checkpoint_manager is None:
                    from .pipeline import CheckpointManager

                    output_path = Path(output_csv) if output_csv else cache_file.parent
                    checkpoint_manager = CheckpointManager("Oxazolones", output_path.parent)
                if completed_ald_ids:
                    checkpoint_manager.add_completed_ids("aldehyde", completed_ald_ids)
        except Exception as e:
            if print_report:
                print(f"[ErlenmeyerPlochl] Warning: Could not read CSV checkpoint: {e}")

    if is_resuming and completed_ald_ids and not temp_csv_path.exists():
        if print_report:
            print(
                "[ErlenmeyerPlochl] Checkpoint IDs found but temp CSV is missing; "
                "restarting stage from scratch"
            )
        if checkpoint_manager is not None:
            checkpoint_manager.reset()
        completed_ald_ids = set()
        is_resuming = False

    if is_resuming and completed_ald_ids:
        original_count = len(valid_aldehydes)
        valid_aldehydes = [
            (smi, ald_id, price) for smi, ald_id, price in valid_aldehydes
            if ald_id not in completed_ald_ids
        ]
        if print_report:
            print(
                f"[ErlenmeyerPlochl] Skipping {original_count - len(valid_aldehydes):,} completed aldehydes, "
                f"processing {len(valid_aldehydes):,} remaining"
            )

    if checkpoint_manager is not None:
        checkpoint_manager.update_progress(total_chunks=total_chunks)

    n_chunks = (
        (len(valid_aldehydes) + chunk_size - 1) // chunk_size if valid_aldehydes else 0
    )
    has_written_temp = temp_csv_path.exists() and temp_csv_path.stat().st_size > 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(valid_aldehydes))
        chunk_aldehydes = valid_aldehydes[chunk_start:chunk_end]

        if (
            print_report
            and n_chunks > 1
            and ((chunk_idx + 1) % 2 == 0 or chunk_idx == 0)
        ):
            print(
                f"[ErlenmeyerPlochl] Processing chunk {chunk_idx + 1}/{n_chunks} "
                f"({len(chunk_aldehydes)} aldehydes × {len(valid_carboxylics)} carboxylics)"
            )

        out_rows: list[dict] = []
        work_items: list[tuple[int, int]] = []
        new_cache_entries = {}

        for i, (ald_smi, ald_id, ald_price) in enumerate(chunk_aldehydes):
            for j, (carb_smi, carb_id, carb_price) in enumerate(valid_carboxylics):
                price_total = ald_price + carb_price
                if max_price_mol is not None and price_total > max_price_mol:
                    stats["skipped_price"] += 1
                    continue

                cache_key = _get_cache_key(ald_smi, carb_smi, smiles_gly)

                if use_cache and cache_key in cache:
                    cached_result = cache[cache_key]
                    cached_smiles = _cached_product_smiles(cached_result)
                    if cached_smiles:
                        ald_num = ald_id.lstrip("A")
                        carb_num = carb_id.lstrip("C")
                        for psmi in cached_smiles:
                            out_rows.append(
                                {
                                    "ID": f"A{ald_num}C{carb_num}",
                                    "SMILES": psmi,
                                    "PriceMol": price_total,
                                }
                            )
                    cache_hits += 1
                else:
                    work_items.append((i, j))
                    cache_misses += 1

        if print_report and chunk_idx == 0:
            total_pairs = len(valid_aldehydes) * len(valid_carboxylics)
            print(
                f"[ErlenmeyerPlochl] {total_pairs:,} total pairs "
                f"({n_chunks} chunks of ~{chunk_size} aldehydes)"
            )

        if work_items:
            n_workers = _get_n_workers(n_workers)

            if len(work_items) >= 100 and n_workers > 1:
                batch_size = max(1, len(work_items) // (n_workers * 10))
                batches = [
                    work_items[i : i + batch_size]
                    for i in range(0, len(work_items), batch_size)
                ]

                worker_fn = partial(
                    _process_ep_batch,
                    ald_data=chunk_aldehydes,
                    carb_data=valid_carboxylics,
                    smiles_gly=smiles_gly,
                    keep_mol=keep_mol,
                )

                import multiprocessing as mp

                with mp.Pool(processes=n_workers) as pool:
                    results = pool.map(worker_fn, batches)

                for batch_rows, batch_cache, batch_stats in results:
                    out_rows.extend(batch_rows)
                    new_cache_entries.update(batch_cache)
                    stats["no_product"] += batch_stats["no_product"]
                    stats["problematic"] += batch_stats["problematic"]
            else:
                batch_rows, batch_cache, batch_stats = _process_ep_batch(
                    work_items, chunk_aldehydes, valid_carboxylics, smiles_gly, keep_mol
                )
                out_rows.extend(batch_rows)
                new_cache_entries.update(batch_cache)
                stats["no_product"] += batch_stats["no_product"]
                stats["problematic"] += batch_stats["problematic"]

        if out_rows:
            _append_to_temp_csv(
                out_rows, temp_csv_path, is_first_chunk=not has_written_temp
            )
            has_written_temp = True

        if checkpoint_manager is not None and chunk_aldehydes:
            chunk_ald_ids = {ald_id for (_, ald_id, _) in chunk_aldehydes}
            checkpoint_manager.add_completed_ids("aldehyde", chunk_ald_ids)
            checkpoint_manager.update_progress(
                completed_chunks=chunk_idx + 1, last_chunk_time=0.0
            )

        if use_cache and new_cache_entries:
            cache.update(new_cache_entries)
            _save_cache(cache_file, cache)

    if temp_csv_path.exists():
        out_df = pd.read_csv(temp_csv_path)
        out_df = out_df[["ID", "SMILES", "PriceMol"]]

        if len(out_df) > 0:
            out_df = out_df.sort_values(
                by=["SMILES", "PriceMol"], ascending=[True, True]
            )
            out_df = out_df.drop_duplicates(subset=["SMILES"], keep="first").reset_index(
                drop=True
            )

        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(output_path, index=False)
            if print_report:
                print(f"[ErlenmeyerPlochl] Written to: {output_path}")

        temp_csv_path.unlink()
    else:
        out_df = pd.DataFrame()

    stats["output_rows"] = len(out_df)
    stats["cache_hits"] = cache_hits
    stats["cache_misses"] = cache_misses

    if checkpoint_manager is not None:
        checkpoint_manager.set_complete(row_count=len(out_df), stats=stats)
        if print_report:
            print(f"[ErlenmeyerPlochl] Checkpoint saved: {checkpoint_manager.path.name}")

    if print_report:
        print(_format_reaction_stats(stats, "ErlenmeyerPlochl"))

    return out_df


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


def rxn_AminolysisGFPc(
    df_oxazolones: pd.DataFrame,
    df_amines: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col_oxazolone: str = "ID",
    id_col_amine: str = "ID",
    price_col: str = "PriceMol",
    keep_mol: bool = False,
    print_report: bool = True,
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    n_workers: int | None = None,
    chunk_size: int | None = None,
    chunk_size_per_worker: int = 1,
    output_csv: str | Path | None = None,
    checkpoint_csv: str | Path | None = None,
    checkpoint_manager: Any = None,
    max_price_mol: float | None = None,
) -> pd.DataFrame:
    """
    Aminolysis-GFPc reaction: opens oxazolone rings with amines to form imidazolones.

    Uses chunked streaming to limit memory usage - processes oxazolones in
    batches of `chunk_size`, writing intermediate results to disk.

    Supports robust checkpoint-based resume using CheckpointManager.

    Parameters:
        df_oxazolones: DataFrame with oxazolone compounds.
        df_amines: DataFrame with primary amine compounds.
        smiles_col: Column containing SMILES (default: ``"SMILES"``).
        id_col_oxazolone: ID column in oxazolone DataFrame (default: ``"ID"``).
        id_col_amine: ID column in amine DataFrame (default: ``"ID"``).
        price_col: Column with prices (default: ``"PriceMol"``).
        keep_mol: Keep RDKit mol objects (default: False).
        print_report: Print statistics (default: True).
        use_cache: Use persistent cache (default: True).
        cache_file: Cache file path (default: auto-generated).
        n_workers: Number of parallel workers (default: cpu_count - 1).
        chunk_size: Number of oxazolones per chunk. If None, computed as
            ``chunk_size_per_worker * n_workers``.
        chunk_size_per_worker: Number of oxazolones per worker (default: 1).
        output_csv: If provided, write final result to this CSV file.
        checkpoint_csv: Legacy parameter for checkpoint-based resume.
        checkpoint_manager: CheckpointManager instance for robust resume support.
        max_price_mol: Optional hard cutoff for product price per mole.

    Returns:
        DataFrame with imidazolone products (``ID``, ``SMILES``, ``PriceMol``).
    """
    for df, name, id_col in [
        (df_oxazolones, "oxazolones", id_col_oxazolone),
        (df_amines, "amines", id_col_amine),
    ]:
        if smiles_col not in df.columns:
            raise ValueError(f"Missing '{smiles_col}' column in {name} DataFrame.")
        if id_col not in df.columns:
            raise ValueError(f"Missing '{id_col}' column in {name} DataFrame.")
        if price_col not in df.columns:
            raise ValueError(f"Missing '{price_col}' column in {name} DataFrame.")
    if max_price_mol is not None and max_price_mol < 0:
        raise ValueError("max_price_mol must be non-negative.")

    df_oxazolones = _prepare_reaction_inputs(
        df_oxazolones, smiles_col, id_col_oxazolone, price_col, "AminolysisGFPc", print_report
    )
    df_amines = _prepare_reaction_inputs(
        df_amines, smiles_col, id_col_amine, price_col, "AminolysisGFPc", print_report
    )

    if cache_file is None:
        cache_file = DEFAULT_CACHE_DIR / "aminolysis_gfpc_cache.json.gz"
    else:
        cache_file = Path(cache_file)

    cache = _load_cache(cache_file) if use_cache else {}
    cache_hits = 0
    cache_misses = 0
    new_cache_entries: dict[str, list[str]] = {}

    patt_amine = Chem.MolFromSmarts("[NX3H2:10][*:11]")
    if patt_amine is None:
        raise ValueError("Failed to build amine pattern from SMARTS.")

    stats = {
        "input_oxazolones": len(df_oxazolones),
        "input_amines": len(df_amines),
        "invalid_oxazolone": 0,
        "invalid_amine": 0,
        "not_oxazolone": 0,
        "not_amine": 0,
        "no_product": 0,
        "problematic": 0,
        "skipped_price": 0,
        "candidate_pairs_total": 0,
        "candidate_pairs_affordable": 0,
        "output_rows": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    if chunk_size is None:
        if n_workers is None:
            n_workers = _get_n_workers(None)
        chunk_size = chunk_size_per_worker * n_workers
        if print_report:
            print(
                f"[AminolysisGFPc] Using chunk_size={chunk_size} "
                f"({chunk_size_per_worker} oxazolones per worker × {n_workers} workers)"
            )

    valid_oxazolones: list[tuple[str, str, float]] = [
        (str(smi), str(id_), float(price))
        for smi, id_, price in zip(
            df_oxazolones[smiles_col],
            df_oxazolones[id_col_oxazolone],
            df_oxazolones[price_col],
        )
    ]
    stats["invalid_oxazolone"] = 0
    stats["not_oxazolone"] = 0

    am_smiles = df_amines[smiles_col].astype(str)
    am_mols = am_smiles.map(Chem.MolFromSmiles)
    am_valid = am_mols.notna()
    am_match = am_valid & am_mols.map(
        lambda m: m.HasSubstructMatch(patt_amine) if m is not None else False
    )

    valid_amines: list[tuple[str, str, float]] = [
        (smi, str(id_), float(price))
        for smi, id_, price in zip(
            df_amines.loc[am_match, smiles_col],
            df_amines.loc[am_match, id_col_amine],
            df_amines.loc[am_match, price_col],
        )
    ]
    valid_amines.sort(key=lambda item: item[2])
    amine_prices = [price for _, _, price in valid_amines]

    stats["invalid_amine"] = int((~am_valid).sum())
    stats["not_amine"] = int((am_valid & ~am_match).sum())

    del df_oxazolones, df_amines, am_smiles, am_mols, am_valid, am_match

    from .pipeline import CheckpointManager

    if checkpoint_manager is None and output_csv is not None:
        output_path = Path(output_csv)
        stage_name = "Imidazolones"
        checkpoint_manager = CheckpointManager(stage_name, output_path.parent)

    if checkpoint_csv is not None:
        temp_csv_path = Path(checkpoint_csv)
    elif checkpoint_manager is not None:
        temp_csv_path = checkpoint_manager.path.parent / ".tmp_ag_results.csv"
    elif output_csv is not None:
        temp_csv_path = Path(output_csv).parent / ".cache" / ".tmp_ag_results.csv"
    else:
        temp_csv_path = cache_file.parent / ".tmp_ag_results.csv"
    temp_csv_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ox_ids: set[str] = set()
    is_resuming = False
    total_chunks = (
        (len(valid_oxazolones) + chunk_size - 1) // chunk_size if valid_oxazolones else 0
    )

    if checkpoint_manager is not None:
        completed_ox_ids = checkpoint_manager.get_completed_ids("oxazolone")
        if len(completed_ox_ids) > 0:
            is_resuming = True
            if print_report:
                print(
                    f"[AminolysisGFPc] Resuming from checkpoint: "
                    f"{len(completed_ox_ids):,} oxazolones already processed"
                )

    if not is_resuming and temp_csv_path.exists():
        try:
            checkpoint_df = pd.read_csv(temp_csv_path)
            if "ID" in checkpoint_df.columns:
                completed_ox_ids = set(
                    checkpoint_df["ID"].str.replace(r"N\d+$", "", regex=True).unique()
                )
                is_resuming = True
                if print_report:
                    print(
                        f"[AminolysisGFPc] Resuming from CSV checkpoint: "
                        f"{len(completed_ox_ids):,} oxazolones already processed"
                    )
                if completed_ox_ids and checkpoint_manager is not None:
                    checkpoint_manager.add_completed_ids("oxazolone", completed_ox_ids)
        except Exception as e:
            if print_report:
                print(f"[AminolysisGFPc] Warning: Could not read CSV checkpoint: {e}")

    if is_resuming and completed_ox_ids and not temp_csv_path.exists():
        if print_report:
            print(
                "[AminolysisGFPc] Checkpoint IDs found but temp CSV is missing; "
                "restarting stage from scratch"
            )
        if checkpoint_manager is not None:
            checkpoint_manager.reset()
        completed_ox_ids = set()
        is_resuming = False

    if is_resuming and completed_ox_ids:
        original_count = len(valid_oxazolones)
        valid_oxazolones = [
            (smi, ox_id, price) for smi, ox_id, price in valid_oxazolones
            if ox_id not in completed_ox_ids
        ]
        if print_report:
            print(
                f"[AminolysisGFPc] Skipping {original_count - len(valid_oxazolones):,} completed oxazolones, "
                f"processing {len(valid_oxazolones):,} remaining"
            )

    candidate_pairs_total = len(valid_oxazolones) * len(valid_amines)
    if max_price_mol is None:
        candidate_pairs_affordable = candidate_pairs_total
    else:
        candidate_pairs_affordable = sum(
            bisect_right(amine_prices, max_price_mol - ox_price)
            for _, _, ox_price in valid_oxazolones
        )

    stats["candidate_pairs_total"] = candidate_pairs_total
    stats["candidate_pairs_affordable"] = candidate_pairs_affordable

    if print_report:
        print(
            f"[AminolysisGFPc] Pairing preflight: total={candidate_pairs_total:,}, "
            f"affordable={candidate_pairs_affordable:,}, price_cutoff={max_price_mol}"
        )

    if checkpoint_manager is not None:
        checkpoint_manager.update_progress(total_chunks=total_chunks)

    n_chunks = (
        (len(valid_oxazolones) + chunk_size - 1) // chunk_size if valid_oxazolones else 0
    )
    has_written_temp = temp_csv_path.exists() and temp_csv_path.stat().st_size > 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(valid_oxazolones))
        chunk_oxazolones = valid_oxazolones[chunk_start:chunk_end]

        if print_report and n_chunks > 1 and (chunk_idx + 1) % 500 == 0:
            print(
                f"[AminolysisGFPc] Processing chunk {chunk_idx + 1}/{n_chunks} "
                f"({chunk_idx * chunk_size:,} oxazolones processed)"
            )

        out_rows: list[dict] = []
        work_items: list[tuple[int, int]] = []
        new_cache_entries = {}

        for i, (ox_smi, ox_id, ox_price) in enumerate(chunk_oxazolones):
            if max_price_mol is None:
                affordable_count = len(valid_amines)
            else:
                affordable_count = bisect_right(amine_prices, max_price_mol - ox_price)
                stats["skipped_price"] += max(0, len(valid_amines) - affordable_count)

            if affordable_count <= 0:
                continue

            for j in range(affordable_count):
                am_smi, am_id, am_price = valid_amines[j]
                price_total = ox_price + am_price

                cache_key = _get_cache_key(ox_smi, am_smi)

                if use_cache and cache_key in cache:
                    cached_result = cache[cache_key]
                    cached_smiles = _cached_product_smiles(cached_result)
                    if cached_smiles:
                        am_num = am_id.lstrip("N")
                        for psmi in cached_smiles:
                            out_rows.append(
                                {
                                    "ID": f"{ox_id}N{am_num}",
                                    "SMILES": psmi,
                                    "PriceMol": price_total,
                                }
                            )
                    cache_hits += 1
                else:
                    work_items.append((i, j))
                    cache_misses += 1

        if work_items:
            n_workers = _get_n_workers(n_workers)

            if n_workers > 1:
                batch_size = max(1, len(work_items) // n_workers)
                batches = [
                    work_items[i : i + batch_size]
                    for i in range(0, len(work_items), batch_size)
                ]

                worker_fn = partial(_process_ag_batch, am_data=valid_amines, keep_mol=keep_mol)

                import multiprocessing as mp

                with mp.Pool(processes=n_workers) as pool:
                    results = pool.starmap(
                        worker_fn, [(chunk_oxazolones, batch) for batch in batches]
                    )

                for batch_rows, batch_cache, batch_stats in results:
                    out_rows.extend(batch_rows)
                    new_cache_entries.update(batch_cache)
                    stats["no_product"] += batch_stats["no_product"]
                    stats["problematic"] += batch_stats["problematic"]
            else:
                batch_rows, batch_cache, batch_stats = _process_ag_batch(
                    chunk_oxazolones, work_items, valid_amines, keep_mol
                )
                out_rows.extend(batch_rows)
                new_cache_entries.update(batch_cache)
                stats["no_product"] += batch_stats["no_product"]
                stats["problematic"] += batch_stats["problematic"]

        if out_rows:
            _append_to_temp_csv(
                out_rows, temp_csv_path, is_first_chunk=not has_written_temp
            )
            has_written_temp = True

        if checkpoint_manager is not None:
            checkpoint_manager.update_progress(
                completed_chunks=chunk_idx + 1, last_chunk_time=0.0
            )
            if chunk_oxazolones:
                chunk_ox_ids = {ox_id for _, ox_id, _ in chunk_oxazolones}
                checkpoint_manager.add_completed_ids("oxazolone", chunk_ox_ids)

        if use_cache and new_cache_entries:
            cache.update(new_cache_entries)
            if (chunk_idx + 1) % 50 == 0:
                _save_cache(cache_file, cache)

    if use_cache and cache:
        _save_cache(cache_file, cache)
        if print_report:
            print(f"[AminolysisGFPc] Cache saved to: {cache_file}")

    if temp_csv_path.exists():
        out_df = pd.read_csv(temp_csv_path)
        out_df = out_df[["ID", "SMILES", "PriceMol"]]

        if len(out_df) > 0:
            out_df = out_df.sort_values(
                by=["SMILES", "PriceMol"], ascending=[True, True]
            )
            out_df = out_df.drop_duplicates(subset=["SMILES"], keep="first").reset_index(
                drop=True
            )

        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(output_path, index=False)
            if print_report:
                print(f"[AminolysisGFPc] Written to: {output_path}")

        temp_csv_path.unlink()
    else:
        out_df = pd.DataFrame()

    stats["output_rows"] = len(out_df)
    stats["cache_hits"] = cache_hits
    stats["cache_misses"] = cache_misses

    if checkpoint_manager is not None:
        checkpoint_manager.set_complete(row_count=len(out_df), stats=stats)
        if print_report:
            print(f"[AminolysisGFPc] Checkpoint saved: {checkpoint_manager.path.name}")

    if print_report:
        print(_format_reaction_stats(stats, "AminolysisGFPc"))

    return out_df


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


def rxn_SulphurExchange(
    df_oxazolones: pd.DataFrame,
    thioacetic_price_eq: float,
    smiles_col: str = "SMILES",
    id_col: str = "ID",
    price_col: str = "PriceMol",
    use_cache: bool = True,
    cache_file: str | Path | None = None,
    print_report: bool = True,
    output_csv: str | Path | None = None,
    checkpoint_csv: str | Path | None = None,
    chunk_size: int = 1000,
    checkpoint_manager: Any = None,
    max_price_mol: float | None = None,
) -> pd.DataFrame:
    """
    Sulphur-Exchange reaction: converts oxazolones to thiazolones by replacing O with S.

    Uses chunked processing for large datasets with robust checkpoint-based resume.

    Parameters:
        df_oxazolones: DataFrame with oxazolone compounds.
        thioacetic_price_eq: Price per equivalent of thioacetic acid reagent.
        smiles_col: Column containing SMILES (default: ``"SMILES"``).
        id_col: ID column in oxazolone DataFrame (default: ``"ID"``).
        price_col: Column with prices (default: ``"PriceMol"``).
        use_cache: Use persistent cache (default: True).
        cache_file: Cache file path (default: auto-generated).
        print_report: Print statistics (default: True).
        output_csv: If provided, write final result to this CSV file.
        checkpoint_csv: Legacy parameter (ignored if checkpoint_manager provided).
        chunk_size: Number of oxazolones per chunk for checkpointing (default: 1000).
        checkpoint_manager: CheckpointManager instance for robust resume support.
        max_price_mol: Optional hard cutoff for product price per mole.

    Returns:
        DataFrame with thiazolone products (``ID``, ``SMILES``, ``PriceMol``).
    """
    if smiles_col not in df_oxazolones.columns:
        raise ValueError(f"Missing '{smiles_col}' column in oxazolones DataFrame.")
    if id_col not in df_oxazolones.columns:
        raise ValueError(f"Missing '{id_col}' column in oxazolones DataFrame.")
    if price_col not in df_oxazolones.columns:
        raise ValueError(f"Missing '{price_col}' column in oxazolones DataFrame.")
    if thioacetic_price_eq < 0:
        raise ValueError(
            f"Thioacetic price must be non-negative, got {thioacetic_price_eq}."
        )
    if max_price_mol is not None and max_price_mol < 0:
        raise ValueError("max_price_mol must be non-negative.")

    df_oxazolones = _prepare_reaction_inputs(
        df_oxazolones, smiles_col, id_col, price_col, "SulphurExchange", print_report
    )

    if cache_file is None:
        cache_file = DEFAULT_CACHE_DIR / "thiazolone_cache.json.gz"
    else:
        cache_file = Path(cache_file)

    cache = _load_cache(cache_file) if use_cache else {}
    cache_hits = 0
    cache_misses = 0
    new_cache_entries: dict[str, list[str]] = {}

    patt_oxazolone = Chem.MolFromSmarts(
        "[O:51]=[C:5]1[O:1][C:2]([#6:21])=[N:3]/[C:4]1=[C:41]\\[#6:42]"
    )
    if patt_oxazolone is None:
        raise ValueError("Failed to build oxazolone pattern from SMARTS.")

    smiles_thioacetic = "[C:93][C:94](=[O:85])[SX2H:10]"
    mol_thioacetic = Chem.MolFromSmiles("[C:93][C:94](=[O:85])[SH:10]")
    if mol_thioacetic is None:
        raise ValueError("Failed to build thioacetic acid reagent Mol (invalid SMILES).")

    rxn_se = rdChemReactions.ReactionFromSmarts(_SMARTS_SE_REACTION)
    if rxn_se is None:
        raise ValueError("Failed to build reaction from SMARTS.")

    stats = {
        "input_oxazolones": len(df_oxazolones),
        "thioacetic_price_eq": thioacetic_price_eq,
        "invalid_oxazolone": 0,
        "not_oxazolone": 0,
        "skipped_price": 0,
        "no_product": 0,
        "problematic": 0,
        "candidate_pairs_total": 0,
        "candidate_pairs_affordable": 0,
        "output_rows": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    valid_oxazolones: list[tuple[str, str, float]] = [
        (str(smi), str(id_), float(price))
        for smi, id_, price in zip(
            df_oxazolones[smiles_col],
            df_oxazolones[id_col],
            df_oxazolones[price_col],
        )
    ]
    stats["invalid_oxazolone"] = 0
    stats["not_oxazolone"] = 0

    from .pipeline import CheckpointManager

    if checkpoint_manager is None and output_csv is not None:
        output_path = Path(output_csv)
        stage_name = "Thiazolones"
        checkpoint_manager = CheckpointManager(stage_name, output_path.parent)

    if checkpoint_csv is not None:
        temp_csv_path = Path(checkpoint_csv)
    elif checkpoint_manager is not None:
        temp_csv_path = checkpoint_manager.path.parent / ".tmp_se_results.csv"
    elif output_csv is not None:
        temp_csv_path = Path(output_csv).parent / ".cache" / ".tmp_se_results.csv"
    else:
        temp_csv_path = cache_file.parent / ".tmp_se_results.csv"
    temp_csv_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ox_ids: set[str] = set()
    is_resuming = False
    total_chunks = (
        (len(valid_oxazolones) + chunk_size - 1) // chunk_size if valid_oxazolones else 0
    )

    if checkpoint_manager is not None:
        completed_ox_ids = checkpoint_manager.get_completed_ids("oxazolone")
        if len(completed_ox_ids) > 0:
            is_resuming = True
            if print_report:
                print(
                    f"[SulphurExchange] Resuming from checkpoint: "
                    f"{len(completed_ox_ids):,} oxazolones already processed"
                )

    if not is_resuming and temp_csv_path.exists():
        try:
            checkpoint_df = pd.read_csv(temp_csv_path)
            if "ID" in checkpoint_df.columns:
                completed_ox_ids = set(
                    checkpoint_df["ID"].str.rstrip("S").unique()
                )
                is_resuming = True
                if print_report:
                    print(
                        f"[SulphurExchange] Resuming from CSV checkpoint: "
                        f"{len(completed_ox_ids):,} oxazolones already processed"
                    )
                if completed_ox_ids and checkpoint_manager is not None:
                    checkpoint_manager.add_completed_ids("oxazolone", completed_ox_ids)
        except Exception as e:
            if print_report:
                print(f"[SulphurExchange] Warning: Could not read CSV checkpoint: {e}")

    if is_resuming and completed_ox_ids and not temp_csv_path.exists():
        if print_report:
            print(
                "[SulphurExchange] Checkpoint IDs found but temp CSV is missing; "
                "restarting stage from scratch"
            )
        if checkpoint_manager is not None:
            checkpoint_manager.reset()
        completed_ox_ids = set()
        is_resuming = False

    if is_resuming and completed_ox_ids:
        original_count = len(valid_oxazolones)
        valid_oxazolones = [
            (smi, ox_id, price) for smi, ox_id, price in valid_oxazolones
            if ox_id not in completed_ox_ids
        ]
        if print_report:
            print(
                f"[SulphurExchange] Skipping {original_count - len(valid_oxazolones):,} completed oxazolones, "
                f"processing {len(valid_oxazolones):,} remaining"
            )

    candidate_pairs_total = len(valid_oxazolones)
    if max_price_mol is None:
        candidate_pairs_affordable = candidate_pairs_total
    else:
        candidate_pairs_affordable = sum(
            1
            for _, _, ox_price in valid_oxazolones
            if (ox_price + thioacetic_price_eq) <= max_price_mol
        )

    stats["candidate_pairs_total"] = candidate_pairs_total
    stats["candidate_pairs_affordable"] = candidate_pairs_affordable

    if print_report:
        print(
            f"[SulphurExchange] Pairing preflight: total={candidate_pairs_total:,}, "
            f"affordable={candidate_pairs_affordable:,}, price_cutoff={max_price_mol}"
        )

    if checkpoint_manager is not None:
        checkpoint_manager.update_progress(total_chunks=total_chunks)

    out_rows: list[dict] = []
    work_items: list[int] = []

    for i, (ox_smi, ox_id, ox_price) in enumerate(valid_oxazolones):
        if max_price_mol is not None and (ox_price + thioacetic_price_eq) > max_price_mol:
            stats["skipped_price"] += 1
            continue

        cache_key = _get_cache_key(ox_smi, smiles_thioacetic)

        if use_cache and cache_key in cache:
            cached_result = cache[cache_key]
            cached_smiles = _cached_product_smiles(cached_result)
            if cached_smiles:
                price_total = ox_price + thioacetic_price_eq
                for psmi in cached_smiles:
                    out_rows.append(
                        {
                            "ID": f"{ox_id}S",
                            "SMILES": psmi,
                            "PriceMol": price_total,
                        }
                    )
            cache_hits += 1
        else:
            work_items.append(i)
            cache_misses += 1

    if print_report:
        print(
            f"[SulphurExchange] {len(valid_oxazolones):,} valid oxazolones: "
            f"{cache_hits:,} cache hits, {cache_misses:,} misses"
        )

    n_chunks = (len(work_items) + chunk_size - 1) // chunk_size if work_items else 0
    has_written_temp = temp_csv_path.exists() and temp_csv_path.stat().st_size > 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(work_items))
        chunk_items = work_items[chunk_start:chunk_end]

        if print_report and n_chunks > 1:
            print(
                f"[SulphurExchange] Processing chunk {chunk_idx + 1}/{n_chunks} "
                f"({len(chunk_items):,} items)"
            )

        n_workers = _get_n_workers(None)
        if len(chunk_items) >= 1000 and n_workers > 1:
            chunk_oxazolones = [valid_oxazolones[i] for i in chunk_items]
            batch_size = max(1, len(chunk_oxazolones) // n_workers)
            batches = [
                chunk_oxazolones[i : i + batch_size]
                for i in range(0, len(chunk_oxazolones), batch_size)
            ]

            worker_fn = partial(
                _process_se_batch,
                smiles_thioacetic=smiles_thioacetic,
                thioacetic_price_eq=thioacetic_price_eq,
            )

            import multiprocessing as mp

            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(worker_fn, batches)

            chunk_rows: list[dict] = []
            for batch_rows, batch_cache, batch_stats in results:
                chunk_rows.extend(batch_rows)
                new_cache_entries.update(batch_cache)
                stats["no_product"] += batch_stats["no_product"]
                stats["problematic"] += batch_stats["problematic"]
        else:
            if chunk_items:
                chunk_rows, batch_cache, batch_stats = _process_se_batch(
                    [valid_oxazolones[i] for i in chunk_items],
                    smiles_thioacetic,
                    thioacetic_price_eq,
                )
                new_cache_entries.update(batch_cache)
                stats["no_product"] += batch_stats["no_product"]
                stats["problematic"] += batch_stats["problematic"]
            else:
                chunk_rows = []

        if chunk_rows:
            out_rows.extend(chunk_rows)
            _append_to_temp_csv(
                chunk_rows, temp_csv_path, is_first_chunk=not has_written_temp
            )
            has_written_temp = True

        if checkpoint_manager is not None and chunk_items:
            chunk_ox_ids = {valid_oxazolones[i][1] for i in chunk_items}
            checkpoint_manager.add_completed_ids("oxazolone", chunk_ox_ids)
            checkpoint_manager.update_progress(
                completed_chunks=chunk_idx + 1, last_chunk_time=0.0
            )

        if use_cache and new_cache_entries and (chunk_idx + 1) % 100 == 0:
            cache.update(new_cache_entries)
            _save_cache(cache_file, cache)

    if use_cache and new_cache_entries:
        cache.update(new_cache_entries)
        _save_cache(cache_file, cache)

    if checkpoint_manager is not None:
        checkpoint_manager.set_complete(row_count=len(out_rows), stats=stats)
        if print_report:
            print(f"[SulphurExchange] Checkpoint saved: {checkpoint_manager.path.name}")

    out_df = pd.DataFrame(out_rows)

    if len(out_df) > 0:
        out_df = out_df[["ID", "SMILES", "PriceMol"]]
        out_df = out_df.sort_values(by=["SMILES", "PriceMol"], ascending=[True, True])
        out_df = out_df.drop_duplicates(subset=["SMILES"], keep="first").reset_index(
            drop=True
        )

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        if print_report:
            print(f"[SulphurExchange] Written to: {output_path}")

    stats["output_rows"] = len(out_df)
    stats["cache_hits"] = cache_hits
    stats["cache_misses"] = cache_misses

    if checkpoint_manager is not None:
        checkpoint_manager.set_complete(row_count=len(out_df), stats=stats)

    if print_report:
        print(_format_reaction_stats(stats, "SulphurExchange"))

    return out_df


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
