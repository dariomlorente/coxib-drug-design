from __future__ import annotations

import gzip
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import inchi

PUBCHEM_DELAY = 0.3
PUBCHEM_TIMEOUT = 10
PUBCHEM_RETRIES = 3
PUBCHEM_RETRY_DELAY = 1.0

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"

CACHE_DIR = "mol_files/0. Esteban BBs/.cache"


def load_inventory_cas(xlsx_path: str) -> list[str]:
    """
    Extract unique, non-null CAS numbers from Esteban's inventory spreadsheet.

    Parameters:
        xlsx_path: Path to Inventario.xlsx.

    Returns:
        Order-preserving list of unique CAS strings.
    """
    df = pd.read_excel(xlsx_path)

    if "CAS" not in df.columns:
        raise ValueError(f"Missing 'CAS' column in {xlsx_path}. Found: {list(df.columns)}")

    cas_values = df["CAS"].dropna().astype(str).str.strip()
    cas_values = [cas for cas in cas_values if cas and cas != "nan"]

    unique_cas = list(dict.fromkeys(cas_values))
    print(f"[Inventory] Loaded {len(unique_cas)} unique CAS numbers from {xlsx_path}")
    return unique_cas


def _lookup_cas_single(cas: str) -> str | None:
    """
    Query PubChem for a single CAS number and return its SMILES.

    Parameters:
        cas: CAS registry number string.

    Returns:
        SMILES string, or None if not found or request failed.
    """
    url = f"{PUBCHEM_BASE}/{cas}/property/SMILES/JSON"

    for attempt in range(PUBCHEM_RETRIES):
        try:
            resp = requests.get(url, timeout=PUBCHEM_TIMEOUT)

            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                if props:
                    return props[0]["SMILES"]
                return None

            if resp.status_code == 404:
                return None

            if attempt < PUBCHEM_RETRIES - 1:
                time.sleep(PUBCHEM_RETRY_DELAY)
                continue

            return None

        except requests.exceptions.Timeout:
            if attempt < PUBCHEM_RETRIES - 1:
                time.sleep(PUBCHEM_RETRY_DELAY)
            else:
                return None
        except requests.exceptions.RequestException:
            if attempt < PUBCHEM_RETRIES - 1:
                time.sleep(PUBCHEM_RETRY_DELAY)
            else:
                return None

    return None


def _load_smiles_cache(smi_path: str) -> list[tuple[str, str]]:
    """Load SMILES and CAS pairs from a .smi file."""
    resolved: list[tuple[str, str]] = []
    with open(smi_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2 and parts[1]:
                    resolved.append((parts[1], parts[0]))
    print(f"[CAS to SMILES] Loaded {len(resolved)} SMILES from cache: {smi_path}")
    return resolved


def _write_smiles_cache(smi_path: str, resolved: list[tuple[str, str]]) -> None:
    """Write resolved CAS to SMILES pairs to a .smi file."""
    os.makedirs(os.path.dirname(smi_path), exist_ok=True)
    with open(smi_path, "w") as f:
        f.write("# CAS to SMILES cache from PubChem PUG-REST\n")
        f.write(f"# {len(resolved)} compounds\n")
        for cas, smi in resolved:
            f.write(f"{smi}\t{cas}\n")
    print(f"[CAS to SMILES] Saved {len(resolved)} SMILES to {smi_path}")


def _load_json_cache(path: str) -> dict[str, str | None]:
    """Load gzip-compressed JSON cache."""
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _save_json_cache(path: str, data: dict[str, str | None]) -> None:
    """Save gzip-compressed JSON cache."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def _write_inchikey_cache(
    resolved: list[tuple[str, str]],
    inchikey_path: str,
) -> list[str]:
    """
    Write InChIKeys to cache file and return list of valid InChIKeys.

    Parameters:
        resolved: List of (CAS, SMILES) pairs.
        inchikey_path: Path to output .inchikey file.

    Returns:
        List of valid InChIKeys.
    """
    inchikeys: list[str] = []
    os.makedirs(os.path.dirname(inchikey_path), exist_ok=True)

    with open(inchikey_path, "w") as f:
        f.write("# CAS to InChIKey cache derived from PubChem SMILES\n")
        for cas, smi in resolved:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                ik = inchi.MolToInchiKey(mol)
                if ik:
                    inchikeys.append(ik)
                    f.write(f"{ik}\t{cas}\n")
            except Exception:
                continue

    print(f"[CAS to InChIKey] Saved {len(inchikeys)} InChIKeys to {inchikey_path}")
    return inchikeys


def cas_to_smiles(
    cas_list: list[str],
    output_path: str = "mol_files/0. Esteban BBs/Purchased.smi",
    inchikey_path: str = "mol_files/0. Esteban BBs/Purchased.inchikey",
    delay: float = PUBCHEM_DELAY,
    use_cache: bool = True,
    force_refresh: bool = False,
    print_report: bool = True,
) -> list[str]:
    """
    Convert CAS numbers to SMILES via PubChem PUG-REST API with caching.

    Cache is stored as a .smi file (SMILES<tab>CAS per line). On cache hit,
    the file is loaded directly. On miss or force_refresh, PubChem is queried
    with per-request delay to respect rate limits.

    A secondary gzip JSON cache stores individual CAS to SMILES mappings to
    enable incremental lookups on interrupted runs.

    Parameters:
        cas_list: List of CAS numbers to resolve.
        output_path: Path for the cached/output .smi file.
        inchikey_path: Path for the cached/output .inchikey file.
        delay: Seconds between PubChem requests (default: 0.3).
        use_cache: Load from existing .smi file if present (default: True).
        force_refresh: Ignore cache and re-query all CAS (default: False).
        print_report: Print progress messages (default: True).

    Returns:
        List of resolved SMILES strings (order matches input CAS list, with
        failed lookups omitted).
    """
    output_path_obj = Path(output_path)
    inchikey_path_obj = Path(inchikey_path)

    os.makedirs(output_path_obj.parent, exist_ok=True)

    cache_file = Path(CACHE_DIR) / "pubchem_lookup.json.gz"
    cache: dict[str, str | None] = {}
    if use_cache and not force_refresh and cache_file.exists():
        cache = _load_json_cache(str(cache_file))
    elif use_cache and not force_refresh and output_path_obj.exists():
        for cas, smi in _load_smiles_cache(str(output_path_obj)):
            cache[cas] = smi

    resolved: list[tuple[str, str]] = []
    not_found: list[str] = []

    cached_cas = {cas for cas in cache.keys()}

    if print_report:
        print(f"[CAS to SMILES] Resolving {len(cas_list)} CAS numbers via PubChem...")

    for i, cas in enumerate(cas_list):
        if cas in cached_cas:
            cached_value = cache[cas]
            if cached_value is not None:
                resolved.append((cas, cached_value))
            else:
                not_found.append(cas)
            continue

        smiles = _lookup_cas_single(cas)
        cache[cas] = smiles

        if smiles is not None:
            resolved.append((cas, smiles))
        else:
            not_found.append(cas)

        if print_report and (i + 1) % 100 == 0:
            print(
                f"  [{i + 1}/{len(cas_list)}] Resolved: {len(resolved)}, Not found: {len(not_found)}"
            )

        if i < len(cas_list) - 1:
            time.sleep(delay)

    _save_json_cache(str(cache_file), cache)
    _write_smiles_cache(str(output_path_obj), resolved)
    _write_inchikey_cache(resolved, str(inchikey_path_obj))

    if print_report:
        print(f"[CAS to SMILES] Completed: {len(resolved)}/{len(cas_list)} resolved")
        if not_found:
            print(f"⚠️ {len(not_found)} CAS numbers not found in PubChem")

    return [smi for _, smi in resolved]


def _compute_inchikeys_from_smiles(smiles_set: set[str]) -> set[str]:
    """
    Compute InChIKeys from a set of SMILES strings.

    Parameters:
        smiles_set: Set of SMILES strings.

    Returns:
        Set of valid InChIKey strings.
    """
    inchikeys: set[str] = set()
    for smi in smiles_set:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                ik = inchi.MolToInchiKey(mol)
                if ik:
                    inchikeys.add(ik)
            except Exception:
                pass
    return inchikeys


def _infer_bb_type_from_filename(filename: str) -> str:
    """
    Infer building-block type from the SDF filename.

    Parameters:
        filename: Filename or path string.

    Returns:
        Inferred type string (e.g., "Aldehydes", "CarboxylicAcids", "PrimaryAmines").
    """
    for token in ("Aldehydes", "CarboxylicAcids", "PrimaryAmines"):
        if token in filename:
            return token

    match = re.search(r"_(?P<bbtype>[A-Za-z]+)_\d+", filename)
    if match:
        return match.group("bbtype")

    return Path(filename).stem


def _extract_count_from_path(path_str: str) -> int | None:
    """
    Extract row count from SDF filename suffix.

    Parameters:
        path_str: Path or filename string.

    Returns:
        Parsed count, or None if no suffix is found.
    """
    if not path_str:
        return None

    name = Path(path_str).name
    match = re.search(r"_(\d+)(?:cmpds)?\.sdf$", name)
    if match:
        return int(match.group(1))
    return None


def _latest_match(base_dir: Path, pattern: str) -> str | None:
    """Return latest file matching pattern in base_dir."""
    matches = list(base_dir.glob(pattern))
    if not matches:
        return None
    return str(max(matches, key=lambda p: p.stat().st_mtime))


def plot_sdf_size_summary(
    enamine_paths: dict[str, str],
    purchased_paths: dict[str, str],
    tester_dir: str = "mol_files/1. Enamine SDFs",
    tester_prefix: str = "TESTER",
    types: list[str] | None = None,
    title: str = "SDF sizes by source and type",
) -> None:
    """
    Plot grouped bar chart of SDF sizes (ln compounds) by compound type.

    Parameters:
        enamine_paths: Mapping of type -> Enamine SDF path.
        purchased_paths: Mapping of type -> Purchased SDF path.
        tester_dir: Directory containing TESTER SDFs.
        tester_prefix: Prefix for TESTER files (default: "TESTER").
        types: Ordered list of type strings. Defaults to Aldehydes, CarboxylicAcids, PrimaryAmines.
        title: Plot title.

    Returns:
        None.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if types is None:
        types = ["Aldehydes", "CarboxylicAcids", "PrimaryAmines"]

    base_dir = Path(tester_dir)
    tester_paths = {
        t: _latest_match(base_dir, f"{tester_prefix}_{t}_*.sdf") for t in types
    }

    sources = {
        "EnamineBBStock": enamine_paths,
        "PurchasedByEsteban": purchased_paths,
        "TESTER": tester_paths,
    }

    x = np.arange(len(types))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10.5, 3.4))

    for i, (source, path_map) in enumerate(sources.items()):
        counts: list[float] = []
        for bb_type in types:
            path = path_map.get(bb_type)
            count = _extract_count_from_path(path) if path else None
            if count is None and path:
                print(f"⚠️ Could not parse count from {path}")
            counts.append(float(count) if count is not None else float("nan"))

        log_counts = [np.log(c) if c > 0 else float("nan") for c in counts]
        ax.bar(x + (i - 1) * width, log_counts, width=width, label=source)

    ax.set_xticks(x)
    ax.set_xticklabels(["Aldehydes", "Carboxylic Acids", "Primary Amines"])
    ax.set_ylabel("ln(compounds)")
    fig.suptitle(title, y=1.02)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


def filter_sdf_by_smiles(
    source_sdf_path: str,
    smiles_set: set[str],
    output_dir: str | None = None,
    output_prefix: str = "PurchasedByEsteban",
) -> tuple[int, int, str]:
    """
    Filter an SDF file by matching molecules against a set of SMILES via InChIKey.

    Reads all molecules from the source SDF, computes InChIKeys, and writes
    only matching molecules to the output SDF. All original properties
    (Catalog_ID, IUPAC Name, etc.) are preserved.

    Parameters:
        source_sdf_path: Path to the source SDF file.
        smiles_set: Set of SMILES strings to match against.
        output_dir: Output directory for the filtered SDF file (defaults to source directory).
        output_prefix: Output filename prefix (default: "PurchasedByEsteban").

    Returns:
        Tuple of (matched_count, total_scanned_count, output_path).
    """
    purchased_iks = _compute_inchikeys_from_smiles(smiles_set)

    if not purchased_iks:
        bb_type = _infer_bb_type_from_filename(os.path.basename(source_sdf_path))
        out_dir = Path(output_dir) if output_dir else Path(source_sdf_path).parent
        output_path_obj = out_dir / f"{output_prefix}_{bb_type}_0.sdf"
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(output_path_obj))
        writer.close()
        print("[SDF Filter] No valid InChIKeys to match against.")
        print(f"  Output:  {output_path_obj}")
        return 0, 0, str(output_path_obj)

    supp = Chem.SDMolSupplier(source_sdf_path, removeHs=False)

    matched: list[Chem.Mol] = []
    total = 0

    for mol in supp:
        if mol is None:
            continue
        total += 1
        try:
            ik = inchi.MolToInchiKey(mol)
            if ik and ik in purchased_iks:
                matched.append(mol)
        except Exception:
            pass

    bb_type = _infer_bb_type_from_filename(os.path.basename(source_sdf_path))
    out_dir = Path(output_dir) if output_dir else Path(source_sdf_path).parent

    output_path_obj = out_dir / f"{output_prefix}_{bb_type}_{len(matched)}.sdf"
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(output_path_obj))
    for mol in matched:
        writer.write(mol)
    writer.close()

    source_name = os.path.basename(source_sdf_path)
    print(f"[SDF Filter] {source_name}")
    print(f"  Scanned: {total} molecules")
    print(f"  Matched: {len(matched)} molecules")
    print(f"  Output:  {output_path_obj}")

    return len(matched), total, str(output_path_obj)
