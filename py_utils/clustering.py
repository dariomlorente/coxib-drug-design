from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

import pandas as pd


DEFAULT_IGNORE_COLS = [
    'SMILES',
    'Violation',
]


def _sanitize_token(value: str) -> str:
    token = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')
    return token or 'run'


def _normalize_ignore_cols(ignore_cols: list[str] | tuple[str, ...] | None) -> list[str]:
    base = list(DEFAULT_IGNORE_COLS if ignore_cols is None else ignore_cols)
    out: list[str] = []
    seen: set[str] = set()
    for col in base:
        name = str(col).strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(name)
    return out


def _format_ignore_arg(ignore_cols: list[str]) -> str:
    quoted = ','.join(f"'{col}'" for col in ignore_cols)
    return f'[{quoted}]'


def _sha256_file(path: Path, chunk_size: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _count_csv_rows(path: Path) -> int:
    with path.open('r', encoding='utf-8') as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _read_csv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _extract_rowcount_suffix(path: Path, prefix: str) -> int | None:
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)cmpds$')
    match = pattern.match(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def find_latest_clustering_input_csv(
    series_name: str,
    clustering_dir: str | Path = 'mol_files/7. Clustering',
) -> Path:
    """
    Find latest counted clustering-input CSV for one series.

    Parameters
    ----------
    series_name
        Family name, e.g. ``Imidazolones`` or ``Thiazolones``.
    clustering_dir
        Directory containing ``*_input_*cmpds.csv`` files.

    Returns
    -------
    Path
        Path to selected input CSV.
    """
    base_dir = Path(clustering_dir)
    if not base_dir.exists():
        raise ValueError(f'Clustering directory not found: {base_dir}')

    prefix = f'{series_name}_input'
    candidates: list[tuple[Path, int]] = []
    for path in base_dir.glob(f'{prefix}_*cmpds.csv'):
        count = _extract_rowcount_suffix(path, prefix)
        if count is not None:
            candidates.append((path, count))

    if not candidates:
        raise ValueError(f"No clustering input file found for '{series_name}' in {base_dir}.")

    chosen = max(candidates, key=lambda item: (item[1], item[0].stat().st_mtime))[0]
    return chosen


def load_phase2_clustering_input_paths(
    clustering_dir: str | Path = 'mol_files/7. Clustering',
) -> tuple[Path, Path]:
    """
    Load latest imidazolone/thiazolone clustering input paths.

    Parameters
    ----------
    clustering_dir
        Directory containing ``*_input_*cmpds.csv`` files.

    Returns
    -------
    tuple[Path, Path]
        Imidazolone and thiazolone input CSV paths.
    """
    imidazolones_path = find_latest_clustering_input_csv('Imidazolones', clustering_dir)
    thiazolones_path = find_latest_clustering_input_csv('Thiazolones', clustering_dir)
    return imidazolones_path, thiazolones_path


def validate_distinct_series_inputs(
    imidazolones_input_csv: str | Path,
    thiazolones_input_csv: str | Path,
    print_report: bool = True,
) -> None:
    """
    Validate that imidazolone and thiazolone input files are not identical.

    Parameters
    ----------
    imidazolones_input_csv
        Imidazolone input CSV path.
    thiazolones_input_csv
        Thiazolone input CSV path.
    print_report
        Print hash and row-count diagnostics.
    """
    imidazolones_path = Path(imidazolones_input_csv).expanduser().resolve()
    thiazolones_path = Path(thiazolones_input_csv).expanduser().resolve()

    if not imidazolones_path.exists():
        raise ValueError(f'Imidazolone input CSV not found: {imidazolones_path}')
    if not thiazolones_path.exists():
        raise ValueError(f'Thiazolone input CSV not found: {thiazolones_path}')

    imi_sha = _sha256_file(imidazolones_path)
    thi_sha = _sha256_file(thiazolones_path)
    imi_rows = _count_csv_rows(imidazolones_path)
    thi_rows = _count_csv_rows(thiazolones_path)

    if print_report:
        print(
            f'[validate_distinct_series_inputs] '
            f'Imidazolones: rows={imi_rows:,}, sha256={imi_sha[:12]}...'
        )
        print(
            f'[validate_distinct_series_inputs] '
            f'Thiazolones:  rows={thi_rows:,}, sha256={thi_sha[:12]}...'
        )

    if imi_sha == thi_sha:
        raise ValueError(
            'Imidazolone and thiazolone clustering inputs are byte-identical. '
            'These families must remain separated before Phase 3 clustering.'
        )


def _resolve_cluster_col(columns: list[str]) -> str:
    lowered = {col.lower(): col for col in columns}
    for candidate in ('cluster', 'clusters', 'cluster_id', 'kmeans_cluster', 'k_means_cluster'):
        if candidate in lowered:
            return lowered[candidate]

    fuzzy = [col for col in columns if 'cluster' in col.lower()]
    if fuzzy:
        return fuzzy[0]

    raise ValueError('No cluster column found in ALMOS output CSV.')


def _pick_distance_col(columns: list[str]) -> str | None:
    scored: list[tuple[int, str]] = []
    for col in columns:
        low = col.lower()
        score = 0
        if 'distance' in low:
            score += 2
        elif 'dist' in low:
            score += 1
        if 'centroid' in low:
            score += 2
        if 'cluster' in low:
            score += 1
        if score > 0:
            scored.append((score, col))

    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _resolve_point_selected_col(columns: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in ('point selected', 'point_selected'):
        if candidate in lowered:
            return lowered[candidate]

    fuzzy = [
        col
        for col in columns
        if 'selected' in col.lower() and 'rank' not in col.lower()
    ]
    if fuzzy:
        return fuzzy[0]

    return None


def _prepare_almos_input_dataframe(
    df_input: pd.DataFrame,
    name_col: str,
    smiles_col: str,
    ignore_cols: list[str],
    generated_name_col: str = 'ALMOS_ID',
) -> tuple[pd.DataFrame, str, list[str], bool, int]:
    out = df_input.copy()
    ignore_out = list(ignore_cols)

    duplicate_name_rows = int(out[name_col].astype(str).duplicated().sum())
    if duplicate_name_rows <= 0:
        return out, name_col, _normalize_ignore_cols(ignore_out), False, 0

    candidate_name_col = generated_name_col
    suffix = 1
    while candidate_name_col in out.columns:
        candidate_name_col = f'{generated_name_col}_{suffix}'
        suffix += 1

    base_ids = out[name_col].astype(str).str.strip()
    base_ids = base_ids.mask(base_ids == '', other='MOL')
    smiles_values = out[smiles_col].astype(str).fillna('')

    hashes = (base_ids + '|' + smiles_values).map(
        lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]
    )
    almos_ids = (base_ids + '__' + hashes).map(_sanitize_token)

    dup_index = almos_ids.groupby(almos_ids, sort=False).cumcount()
    almos_ids = almos_ids.where(
        dup_index == 0,
        almos_ids + '__d' + dup_index.astype(str),
    )

    out[candidate_name_col] = almos_ids
    if out[candidate_name_col].duplicated().any():
        raise ValueError(
            f'Could not generate a unique ALMOS name column ({candidate_name_col}).'
        )

    if name_col not in ignore_out:
        ignore_out.append(name_col)

    return out, candidate_name_col, _normalize_ignore_cols(ignore_out), True, duplicate_name_rows


def _merge_almos_cluster_results(
    df_input: pd.DataFrame,
    df_almos_results: pd.DataFrame,
    almos_name_col: str,
    cluster_col: str,
) -> pd.DataFrame:
    if almos_name_col not in df_almos_results.columns:
        raise ValueError(
            f"ALMOS clustered CSV is missing name column '{almos_name_col}'."
        )
    if cluster_col not in df_almos_results.columns:
        raise ValueError(
            f"ALMOS clustered CSV is missing cluster column '{cluster_col}'."
        )

    if df_almos_results[almos_name_col].astype(str).duplicated().any():
        raise ValueError(
            'ALMOS clustered CSV has duplicated names in the name column; '
            'cannot merge results back to the input table safely.'
        )

    merge_cols = [almos_name_col, cluster_col]
    for optional_col in ('Point selected', 'PC1', 'PC2', 'PC3'):
        if optional_col in df_almos_results.columns and optional_col not in merge_cols:
            merge_cols.append(optional_col)

    merged = df_input.merge(
        df_almos_results[merge_cols],
        on=almos_name_col,
        how='left',
        sort=False,
        validate='one_to_one',
    )

    missing_cluster_rows = int(merged[cluster_col].isna().sum())
    if missing_cluster_rows > 0:
        raise ValueError(
            f'Could not map ALMOS cluster labels back to {missing_cluster_rows:,} input rows.'
        )

    return merged


def validate_clustering_input(
    df: pd.DataFrame,
    name_col: str = 'ID',
    smiles_col: str = 'SMILES',
    ignore_cols: list[str] | tuple[str, ...] | None = None,
    min_descriptor_cols: int = 3,
    print_report: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate a descriptor CSV before ALMOS clustering.

    Parameters
    ----------
    df
        Input DataFrame.
    name_col
        Identifier column used as ``--name`` in ALMOS.
    smiles_col
        SMILES column name.
    ignore_cols
        Columns ignored by ALMOS during clustering.
    min_descriptor_cols
        Minimum descriptor columns required by ALMOS.
    print_report
        Print validation details.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Copy of input DataFrame and normalized ignore columns.
    """
    out = df.copy()
    normalized_ignore = _normalize_ignore_cols(ignore_cols)

    missing = [col for col in (name_col, smiles_col) if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for ALMOS: {', '.join(missing)}")

    if 'batch' in out.columns:
        raise ValueError("ALMOS input cannot contain a column named 'batch'.")

    excluded = {name_col, *normalized_ignore}
    descriptor_cols = [col for col in out.columns if col not in excluded]
    if len(descriptor_cols) < min_descriptor_cols:
        raise ValueError(
            f'Expected at least {min_descriptor_cols} descriptor columns after exclusions; '
            f'found {len(descriptor_cols)}.'
        )

    numeric_like_cols: list[str] = []
    for col in descriptor_cols:
        series = pd.to_numeric(out[col], errors='coerce')
        ratio = float(series.notna().mean())
        if ratio >= 0.95:
            numeric_like_cols.append(col)

    if len(numeric_like_cols) < min_descriptor_cols:
        raise ValueError(
            'ALMOS expects descriptor tables with numeric feature columns. '
            f'Numeric-like columns detected: {len(numeric_like_cols)}.'
        )

    if out[name_col].astype(str).duplicated().any() and print_report:
        dup_count = int(out[name_col].astype(str).duplicated().sum())
        print(
            f"⚠️ [validate_clustering_input] Found {dup_count:,} duplicate IDs in '{name_col}'."
        )

    if print_report:
        missing_ratio = out[numeric_like_cols].isna().mean().sort_values(ascending=False)
        high_missing = missing_ratio[missing_ratio > 0.30]
        if not high_missing.empty:
            top = ', '.join(f'{col} ({ratio:.1%})' for col, ratio in high_missing.head(5).items())
            print(
                '⚠️ [validate_clustering_input] Some descriptor columns exceed 30% NaN: '
                f'{top}'
            )

        print(
            f"[validate_clustering_input] rows={len(out):,}, "
            f"descriptor_cols={len(descriptor_cols):,}, "
            f"numeric_like_cols={len(numeric_like_cols):,}"
        )

    return out, normalized_ignore


def validate_clustering_input_csv(
    input_csv: str | Path,
    name_col: str = 'ID',
    smiles_col: str = 'SMILES',
    ignore_cols: list[str] | tuple[str, ...] | None = None,
    min_descriptor_cols: int = 3,
    print_report: bool = True,
) -> tuple[pd.DataFrame, list[str], Path]:
    """
    Load and validate a clustering input CSV.

    Parameters
    ----------
    input_csv
        Path to input CSV.
    name_col
        Identifier column used as ``--name`` in ALMOS.
    smiles_col
        SMILES column name.
    ignore_cols
        Columns ignored by ALMOS during clustering.
    min_descriptor_cols
        Minimum descriptor columns required by ALMOS.
    print_report
        Print validation details.

    Returns
    -------
    tuple[pd.DataFrame, list[str], Path]
        Validated DataFrame, normalized ignore columns, and resolved path.
    """
    path = Path(input_csv).expanduser().resolve()
    if not path.exists():
        raise ValueError(f'Input CSV not found: {path}')

    df = pd.read_csv(path)
    validated, normalized_ignore = validate_clustering_input(
        df,
        name_col=name_col,
        smiles_col=smiles_col,
        ignore_cols=ignore_cols,
        min_descriptor_cols=min_descriptor_cols,
        print_report=print_report,
    )
    return validated, normalized_ignore, path


def _build_almos_prefix(
    conda_env: str | None = None,
    python_executable: str | None = None,
) -> list[str]:
    if conda_env:
        return ['conda', 'run', '-n', conda_env, 'python', '-m', 'almos']

    python_cmd = python_executable or sys.executable
    return [python_cmd, '-m', 'almos']


def build_almos_cluster_command(
    input_csv: str | Path,
    name_col: str = 'ID',
    n_clusters: int | None = None,
    ignore_cols: list[str] | tuple[str, ...] | None = None,
    seed_clustered: int = 0,
    conda_env: str | None = None,
    python_executable: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """
    Build ALMOS clustering command.

    Parameters
    ----------
    input_csv
        Input descriptor CSV.
    name_col
        Name column passed to ``--name``.
    n_clusters
        Fixed number of clusters. If None, ALMOS elbow auto-selection is used.
    ignore_cols
        Columns ignored by ALMOS during clustering.
    seed_clustered
        KMeans seed passed to ALMOS.
    conda_env
        Optional conda environment name to run ALMOS.
    python_executable
        Python executable used when ``conda_env`` is None.
    extra_args
        Extra CLI arguments appended as-is.

    Returns
    -------
    list[str]
        ALMOS command list suitable for ``subprocess.run``.
    """
    input_path = Path(input_csv).expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f'Input CSV not found: {input_path}')

    if n_clusters is not None and n_clusters <= 0:
        raise ValueError('n_clusters must be greater than 0 when provided.')

    ignore_list = _normalize_ignore_cols(ignore_cols)

    cmd = _build_almos_prefix(conda_env=conda_env, python_executable=python_executable)
    cmd.extend(['--cluster', '--input', str(input_path), '--name', name_col])

    if n_clusters is not None:
        cmd.extend(['--n_clusters', str(int(n_clusters))])

    if ignore_list:
        cmd.extend(['--ignore', _format_ignore_arg(ignore_list)])

    cmd.extend(['--seed_clustered', str(int(seed_clustered))])

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_almos_cluster(
    input_csv: str | Path,
    run_dir: str | Path,
    name_col: str = 'ID',
    n_clusters: int | None = None,
    ignore_cols: list[str] | tuple[str, ...] | None = None,
    seed_clustered: int = 0,
    conda_env: str | None = None,
    python_executable: str | None = None,
    extra_args: list[str] | None = None,
    timeout_sec: int = 7200,
    print_report: bool = True,
) -> dict[str, Any]:
    """
    Execute ALMOS clustering and collect run artifacts.

    Parameters
    ----------
    input_csv
        Input descriptor CSV.
    run_dir
        Working directory for ALMOS run outputs.
    name_col
        Name column passed to ``--name``.
    n_clusters
        Fixed number of clusters. If None, ALMOS elbow auto-selection is used.
    ignore_cols
        Columns ignored by ALMOS during clustering.
    seed_clustered
        KMeans seed passed to ALMOS.
    conda_env
        Optional conda environment name to run ALMOS.
    python_executable
        Python executable used when ``conda_env`` is None.
    extra_args
        Extra CLI arguments appended as-is.
    timeout_sec
        Timeout for subprocess execution.
    print_report
        Print run report.

    Returns
    -------
    dict[str, Any]
        Command, logs, and generated CSV list.
    """
    run_path = Path(run_dir).expanduser().resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    cmd = build_almos_cluster_command(
        input_csv=input_csv,
        name_col=name_col,
        n_clusters=n_clusters,
        ignore_cols=ignore_cols,
        seed_clustered=seed_clustered,
        conda_env=conda_env,
        python_executable=python_executable,
        extra_args=extra_args,
    )

    if print_report:
        print(f"[run_almos_cluster] {' '.join(shlex.quote(part) for part in cmd)}")
        print(f'[run_almos_cluster] cwd={run_path}')

    completed = subprocess.run(
        cmd,
        cwd=run_path,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )

    stdout_path = run_path / 'almos_stdout.log'
    stderr_path = run_path / 'almos_stderr.log'
    stdout_path.write_text(completed.stdout or '', encoding='utf-8')
    stderr_path.write_text(completed.stderr or '', encoding='utf-8')

    if completed.returncode != 0:
        stderr_tail = '\n'.join((completed.stderr or '').splitlines()[-20:])
        stdout_tail = '\n'.join((completed.stdout or '').splitlines()[-20:])
        detail_chunks: list[str] = []
        if stderr_tail:
            detail_chunks.append(f'[stderr]\n{stderr_tail}')
        if stdout_tail:
            detail_chunks.append(f'[stdout]\n{stdout_tail}')
        details = '\n\n'.join(detail_chunks)
        raise ValueError(
            '[run_almos_cluster] ALMOS execution failed. '
            f'Return code: {completed.returncode}.\n{details}'
        )

    generated_csvs = sorted(
        run_path.rglob('*.csv'),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not generated_csvs:
        raise ValueError('[run_almos_cluster] ALMOS finished but no CSV files were generated.')

    if print_report:
        print(f'[run_almos_cluster] Generated CSV files: {len(generated_csvs):,}')

    return {
        'command': cmd,
        'run_dir': run_path,
        'stdout_path': stdout_path,
        'stderr_path': stderr_path,
        'generated_csvs': generated_csvs,
    }


def load_almos_clustered_dataframe(
    csv_path: str | Path,
    cluster_col: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load clustered ALMOS output and resolve cluster column.

    Parameters
    ----------
    csv_path
        Path to ALMOS clustered CSV.
    cluster_col
        Optional explicit cluster column name.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Loaded DataFrame and resolved cluster column.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f'Clustered CSV not found: {path}')

    df = pd.read_csv(path)
    resolved_cluster_col = cluster_col or _resolve_cluster_col(list(df.columns))

    if resolved_cluster_col not in df.columns:
        raise ValueError(f"Cluster column '{resolved_cluster_col}' not found in {path.name}.")

    return df, resolved_cluster_col


def _choose_clustered_csv(
    generated_csvs: list[Path],
    expected_rows: int,
) -> tuple[Path, str]:
    candidates: list[tuple[Path, int, str]] = []

    for path in generated_csvs:
        columns = _read_csv_columns(path)
        try:
            cluster_col = _resolve_cluster_col(columns)
        except ValueError:
            continue
        row_count = _count_csv_rows(path)
        candidates.append((path, row_count, cluster_col))

    if not candidates:
        raise ValueError('No ALMOS output CSV with cluster labels was found.')

    exact = [item for item in candidates if item[1] == expected_rows]
    if exact:
        exact.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
        chosen = exact[0]
        return chosen[0], chosen[2]

    candidates.sort(key=lambda item: (item[1], item[0].stat().st_mtime), reverse=True)
    chosen = candidates[0]
    return chosen[0], chosen[2]


def _choose_representatives_csv(
    generated_csvs: list[Path],
    clustered_csv: Path,
    expected_clusters: int | None,
) -> Path | None:
    candidates: list[tuple[Path, int]] = []

    for path in generated_csvs:
        if path == clustered_csv:
            continue
        columns = _read_csv_columns(path)
        try:
            _resolve_cluster_col(columns)
        except ValueError:
            continue
        row_count = _count_csv_rows(path)
        if row_count <= 0:
            continue
        candidates.append((path, row_count))

    if not candidates:
        return None

    if expected_clusters is not None:
        exact = [item for item in candidates if item[1] == expected_clusters]
        if exact:
            exact.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
            return exact[0][0]

    candidates.sort(key=lambda item: (item[1], item[0].stat().st_mtime))
    return candidates[0][0]


def select_cluster_representatives(
    df_clustered: pd.DataFrame,
    cluster_col: str,
    name_col: str = 'ID',
    price_col: str = 'PriceMol',
    qed_col: str = 'QED',
) -> pd.DataFrame:
    """
    Select one representative molecule per cluster.

    Parameters
    ----------
    df_clustered
        DataFrame containing cluster labels.
    cluster_col
        Cluster label column.
    name_col
        Molecule identifier column.
    price_col
        Price column used as tie-breaker (ascending).
    qed_col
        QED column used as primary rank (descending).

    Returns
    -------
    pd.DataFrame
        One row per cluster.
    """
    if cluster_col not in df_clustered.columns:
        raise ValueError(f"Missing cluster column '{cluster_col}'.")

    work = df_clustered.copy()
    distance_col = _pick_distance_col(list(work.columns))
    point_selected_col = _resolve_point_selected_col(list(work.columns))

    work['_price_rank'] = pd.to_numeric(
        work.get(price_col, pd.Series(index=work.index)),
        errors='coerce',
    )
    work['_price_rank'] = work['_price_rank'].fillna(float('inf'))

    if point_selected_col is not None:
        selected_numeric = pd.to_numeric(work[point_selected_col], errors='coerce')
        if selected_numeric.notna().any():
            work['_selected_rank'] = selected_numeric.fillna(0.0)
        else:
            selected_text = work[point_selected_col].astype(str).str.strip().str.lower()
            work['_selected_rank'] = selected_text.isin({'1', 'true', 'yes', 'y'}).astype(int)

        sort_cols = [cluster_col, '_selected_rank', '_price_rank']
        ascending = [True, False, True]
        reason = f'point_selected:{point_selected_col}'
    elif distance_col is not None:
        work['_distance_rank'] = pd.to_numeric(
            work[distance_col],
            errors='coerce',
        ).fillna(float('inf'))
        sort_cols = [cluster_col, '_distance_rank', '_price_rank']
        ascending = [True, True, True]
        reason = f'centroid_distance:{distance_col}'
    else:
        work['_qed_rank'] = pd.to_numeric(
            work.get(qed_col, pd.Series(index=work.index)),
            errors='coerce',
        )
        work['_qed_rank'] = work['_qed_rank'].fillna(float('-inf'))
        sort_cols = [cluster_col, '_qed_rank', '_price_rank']
        ascending = [True, False, True]
        reason = 'qed_price_rank'

    if name_col in work.columns:
        sort_cols.append(name_col)
        ascending.append(True)

    ordered = work.sort_values(sort_cols, ascending=ascending, kind='stable')
    reps = ordered.drop_duplicates(subset=[cluster_col], keep='first').copy()
    reps.insert(len(reps.columns), 'RepresentativeRule', reason)

    drop_cols = [
        col
        for col in ('_price_rank', '_distance_rank', '_qed_rank', '_selected_rank')
        if col in reps.columns
    ]
    reps = reps.drop(columns=drop_cols)
    return reps.reset_index(drop=True)


def select_top_n_per_cluster(
    df_clustered: pd.DataFrame,
    cluster_col: str,
    top_n: int = 3,
    name_col: str = 'ID',
    price_col: str = 'PriceMol',
    qed_col: str = 'QED',
) -> pd.DataFrame:
    """
    Select top-N candidates per cluster for group discussion.

    Parameters
    ----------
    df_clustered
        DataFrame containing cluster labels.
    cluster_col
        Cluster label column.
    top_n
        Number of candidates kept per cluster.
    name_col
        Molecule identifier column.
    price_col
        Price column used as tie-breaker (ascending).
    qed_col
        QED column used as primary rank (descending).

    Returns
    -------
    pd.DataFrame
        Top-N molecules per cluster with ``SelectionRank`` column.
    """
    if top_n <= 0:
        raise ValueError('top_n must be greater than 0.')
    if cluster_col not in df_clustered.columns:
        raise ValueError(f"Missing cluster column '{cluster_col}'.")

    work = df_clustered.copy()
    work['_qed_rank'] = pd.to_numeric(
        work.get(qed_col, pd.Series(index=work.index)),
        errors='coerce',
    )
    work['_qed_rank'] = work['_qed_rank'].fillna(float('-inf'))
    work['_price_rank'] = pd.to_numeric(
        work.get(price_col, pd.Series(index=work.index)),
        errors='coerce',
    )
    work['_price_rank'] = work['_price_rank'].fillna(float('inf'))

    sort_cols = [cluster_col, '_qed_rank', '_price_rank']
    ascending = [True, False, True]
    if name_col in work.columns:
        sort_cols.append(name_col)
        ascending.append(True)

    ordered = work.sort_values(sort_cols, ascending=ascending, kind='stable').copy()
    ordered['SelectionRank'] = ordered.groupby(cluster_col, sort=False).cumcount() + 1
    selected = ordered.loc[ordered['SelectionRank'] <= top_n].copy()

    drop_cols = [col for col in ('_qed_rank', '_price_rank') if col in selected.columns]
    selected = selected.drop(columns=drop_cols)
    return selected.reset_index(drop=True)


def summarize_clusters(
    df_clustered: pd.DataFrame,
    cluster_col: str,
    price_col: str = 'PriceMol',
    qed_col: str = 'QED',
) -> pd.DataFrame:
    """
    Build per-cluster summary statistics.

    Parameters
    ----------
    df_clustered
        DataFrame containing cluster labels.
    cluster_col
        Cluster label column.
    price_col
        Price column name.
    qed_col
        QED column name.

    Returns
    -------
    pd.DataFrame
        Cluster-level summary table.
    """
    if cluster_col not in df_clustered.columns:
        raise ValueError(f"Missing cluster column '{cluster_col}'.")

    work = df_clustered.copy()
    work[price_col] = pd.to_numeric(
        work.get(price_col, pd.Series(index=work.index)),
        errors='coerce',
    )
    work[qed_col] = pd.to_numeric(
        work.get(qed_col, pd.Series(index=work.index)),
        errors='coerce',
    )

    summary = (
        work.groupby(cluster_col, dropna=False)
        .agg(
            ClusterSize=(cluster_col, 'size'),
            PriceMolMin=(price_col, 'min'),
            PriceMolMedian=(price_col, 'median'),
            PriceMolMax=(price_col, 'max'),
            QEDMin=(qed_col, 'min'),
            QEDMedian=(qed_col, 'median'),
            QEDMax=(qed_col, 'max'),
        )
        .reset_index()
        .sort_values(cluster_col, kind='stable')
        .reset_index(drop=True)
    )
    return summary


def save_clustering_outputs(
    series_name: str,
    df_clustered: pd.DataFrame,
    df_representatives: pd.DataFrame,
    df_shortlist: pd.DataFrame,
    df_summary: pd.DataFrame,
    cluster_col: str,
    top_n: int,
    output_dir: str | Path = 'mol_files/7. Clustering/ALMOS',
    metadata: dict[str, Any] | None = None,
    print_report: bool = True,
) -> dict[str, Path]:
    """
    Save ALMOS clustering outputs using row-count suffix naming.

    Parameters
    ----------
    series_name
        Family name, e.g. ``Imidazolones`` or ``Thiazolones``.
    df_clustered
        Full clustered dataset.
    df_representatives
        One representative row per cluster.
    df_shortlist
        Top-N rows per cluster for discussion.
    df_summary
        Cluster-level summary statistics.
    cluster_col
        Cluster label column.
    top_n
        Top-N setting used for shortlist.
    output_dir
        Output directory for ALMOS files.
    metadata
        Optional metadata written as JSON.
    print_report
        Print saved file paths.

    Returns
    -------
    dict[str, Path]
        Saved output file paths.
    """
    if cluster_col not in df_clustered.columns:
        raise ValueError(f"Missing cluster column '{cluster_col}'.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = _sanitize_token(series_name)
    n_rows = len(df_clustered)
    n_clusters = int(df_clustered[cluster_col].nunique(dropna=True))

    paths = {
        'clustered': out_dir / f'{token}_clusters_k{n_clusters}_{n_rows}cmpds.csv',
        'representatives': out_dir
        / f'{token}_representatives_k{n_clusters}_{len(df_representatives)}cmpds.csv',
        'shortlist': out_dir
        / f'{token}_shortlist_top{top_n}_k{n_clusters}_{len(df_shortlist)}cmpds.csv',
        'summary': out_dir / f'{token}_cluster_summary_k{n_clusters}.csv',
        'metadata': out_dir / f'{token}_cluster_run_k{n_clusters}_{n_rows}cmpds.json',
    }

    df_clustered.to_csv(paths['clustered'], index=False)
    df_representatives.to_csv(paths['representatives'], index=False)

    root_dir = out_dir.parent
    root_dir.mkdir(parents=True, exist_ok=True)
    samples_cols = [c for c in df_representatives.columns if c not in ('ALMOS_ID', 'Cluster', 'Point selected', 'PC1', 'PC2', 'PC3', 'RepresentativeRule')]
    samples_path = root_dir / f'{token}_{n_clusters}_samples.csv'
    df_representatives[samples_cols].to_csv(samples_path, index=False)
    paths['samples'] = samples_path

    df_shortlist.to_csv(paths['shortlist'], index=False)
    df_summary.to_csv(paths['summary'], index=False)

    meta_payload = dict(metadata or {})
    meta_payload.update(
        {
            'series_name': series_name,
            'cluster_col': cluster_col,
            'n_rows': n_rows,
            'n_clusters': n_clusters,
            'top_n': top_n,
            'saved_at_utc': datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'outputs': {key: str(path) for key, path in paths.items() if key != 'metadata'},
        }
    )
    paths['metadata'].write_text(
        json.dumps(meta_payload, indent=2, sort_keys=True),
        encoding='utf-8',
    )

    if print_report:
        print(f"[save_clustering_outputs] {paths['clustered']}")
        print(f"[save_clustering_outputs] {paths['representatives']}")
        print(f"[save_clustering_outputs] {paths['samples']}")
        print(f"[save_clustering_outputs] {paths['shortlist']}")
        print(f"[save_clustering_outputs] {paths['summary']}")
        print(f"[save_clustering_outputs] {paths['metadata']}")

    return paths


def cluster_with_almos(
    series_name: str,
    input_csv: str | Path,
    n_clusters: int | None = None,
    top_n_per_cluster: int = 3,
    name_col: str = 'ID',
    smiles_col: str = 'SMILES',
    ignore_cols: list[str] | tuple[str, ...] | None = None,
    seed_clustered: int = 0,
    output_dir: str | Path = 'mol_files/7. Clustering/ALMOS',
    conda_env: str | None = None,
    python_executable: str | None = None,
    extra_args: list[str] | None = None,
    timeout_sec: int = 7200,
    print_report: bool = True,
) -> dict[str, Path]:
    """
    Run ALMOS clustering for one molecular family and export canonical outputs.

    Parameters
    ----------
    series_name
        Family name, e.g. ``Imidazolones`` or ``Thiazolones``.
    input_csv
        Input descriptor CSV.
    n_clusters
        Fixed number of clusters. If None, ALMOS elbow auto-selection is used.
    top_n_per_cluster
        Number of shortlist molecules per cluster.
    name_col
        Name column passed to ALMOS ``--name``.
    smiles_col
        SMILES column name for validation.
    ignore_cols
        Columns ignored by ALMOS during clustering.
    seed_clustered
        KMeans seed passed to ALMOS.
    output_dir
        Output directory for saved files.
    conda_env
        Optional conda environment used to execute ALMOS.
    python_executable
        Python executable used when ``conda_env`` is None.
    extra_args
        Extra CLI arguments appended as-is.
    timeout_sec
        Timeout for ALMOS subprocess execution.
    print_report
        Print execution report.

    Returns
    -------
    dict[str, Path]
        Saved output file paths.
    """
    df_input, normalized_ignore, input_path = validate_clustering_input_csv(
        input_csv,
        name_col=name_col,
        smiles_col=smiles_col,
        ignore_cols=ignore_cols,
        print_report=print_report,
    )

    (
        df_almos_input,
        almos_name_col,
        almos_ignore_cols,
        generated_name_used,
        duplicate_name_rows,
    ) = _prepare_almos_input_dataframe(
        df_input=df_input,
        name_col=name_col,
        smiles_col=smiles_col,
        ignore_cols=normalized_ignore,
    )

    output_root = Path(output_dir).expanduser().resolve()
    run_stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    run_dir = output_root / '.runs' / f"{_sanitize_token(series_name)}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    almos_input_path = run_dir / f'{input_path.stem}_almos_input.csv'
    df_almos_input.to_csv(almos_input_path, index=False)

    if generated_name_used and print_report:
        print(
            f"⚠️ [cluster_with_almos] {series_name}: found {duplicate_name_rows:,} duplicated "
            f"values in '{name_col}'. Generated unique '{almos_name_col}' for ALMOS."
        )

    run_result = run_almos_cluster(
        input_csv=almos_input_path,
        run_dir=run_dir,
        name_col=almos_name_col,
        n_clusters=n_clusters,
        ignore_cols=almos_ignore_cols,
        seed_clustered=seed_clustered,
        conda_env=conda_env,
        python_executable=python_executable,
        extra_args=extra_args,
        timeout_sec=timeout_sec,
        print_report=print_report,
    )

    generated_csvs: list[Path] = run_result['generated_csvs']
    clustered_csv, detected_cluster_col = _choose_clustered_csv(
        generated_csvs,
        expected_rows=len(df_input),
    )
    df_clustered, cluster_col = load_almos_clustered_dataframe(
        clustered_csv,
        cluster_col=detected_cluster_col,
    )

    df_clustered = _merge_almos_cluster_results(
        df_input=df_almos_input,
        df_almos_results=df_clustered,
        almos_name_col=almos_name_col,
        cluster_col=cluster_col,
    )

    if generated_name_used and almos_name_col not in (name_col, 'ALMOS_ID'):
        if 'ALMOS_ID' not in df_clustered.columns:
            df_clustered = df_clustered.rename(columns={almos_name_col: 'ALMOS_ID'})
            almos_name_col = 'ALMOS_ID'

    if not generated_name_used and almos_name_col != name_col:
        df_clustered = df_clustered.rename(columns={almos_name_col: name_col})
        almos_name_col = name_col

    cluster_count = int(df_clustered[cluster_col].nunique(dropna=True))
    if n_clusters is not None and cluster_count != n_clusters and print_report:
        print(
            f'⚠️ [cluster_with_almos] Requested n_clusters={n_clusters}, '
            f'but detected {cluster_count} clusters in ALMOS output.'
        )

    reps_from_almos = _choose_representatives_csv(
        generated_csvs,
        clustered_csv=clustered_csv,
        expected_clusters=n_clusters,
    )

    df_representatives = select_cluster_representatives(
        df_clustered,
        cluster_col=cluster_col,
        name_col=name_col,
    )
    df_shortlist = select_top_n_per_cluster(
        df_clustered,
        cluster_col=cluster_col,
        top_n=top_n_per_cluster,
        name_col=name_col,
    )
    df_summary = summarize_clusters(df_clustered, cluster_col=cluster_col)

    metadata = {
        'input_csv': str(input_path),
        'input_sha256': _sha256_file(input_path),
        'input_rows': len(df_input),
        'almos_input_csv': str(almos_input_path),
        'almos_input_sha256': _sha256_file(almos_input_path),
        'requested_n_clusters': n_clusters,
        'detected_n_clusters': cluster_count,
        'name_col_requested': name_col,
        'name_col_used': almos_name_col,
        'generated_name_used': generated_name_used,
        'duplicated_name_rows': duplicate_name_rows,
        'ignore_cols': normalized_ignore,
        'ignore_cols_used': almos_ignore_cols,
        'run_dir': str(run_result['run_dir']),
        'command': run_result['command'],
        'clustered_csv_from_almos': str(clustered_csv),
        'representatives_csv_from_almos': (
            str(reps_from_almos) if reps_from_almos else None
        ),
        'generated_csvs': [str(path) for path in generated_csvs],
        'stdout_log': str(run_result['stdout_path']),
        'stderr_log': str(run_result['stderr_path']),
    }

    outputs = save_clustering_outputs(
        series_name=series_name,
        df_clustered=df_clustered,
        df_representatives=df_representatives,
        df_shortlist=df_shortlist,
        df_summary=df_summary,
        cluster_col=cluster_col,
        top_n=top_n_per_cluster,
        output_dir=output_root,
        metadata=metadata,
        print_report=print_report,
    )

    if print_report:
        print(
            f"[cluster_with_almos] {series_name}: rows={len(df_clustered):,}, "
            f'clusters={cluster_count}, shortlist_rows={len(df_shortlist):,}'
        )

    return outputs


def cluster_inputs(
    imidazolones_input_csv: str | Path,
    thiazolones_input_csv: str | Path,
    n_clusters_imidazolones: int | None = None,
    n_clusters_thiazolones: int | None = None,
    top_n_per_cluster: int = 3,
    output_dir: str | Path = 'mol_files/7. Clustering/ALMOS',
    conda_env: str | None = None,
    print_report: bool = True,
) -> dict[str, dict[str, Path]]:
    """
    Cluster both Phase-2 product families with ALMOS.

    Parameters
    ----------
    imidazolones_input_csv
        Phase-2 clustering input for imidazolones.
    thiazolones_input_csv
        Phase-2 clustering input for thiazolones.
    n_clusters_imidazolones
        Fixed cluster count for imidazolones (None = elbow auto-selection).
    n_clusters_thiazolones
        Fixed cluster count for thiazolones (None = elbow auto-selection).
    top_n_per_cluster
        Number of shortlist molecules per cluster.
    output_dir
        Output directory for ALMOS files.
    conda_env
        Optional conda environment used to execute ALMOS.
    print_report
        Print execution report.

    Returns
    -------
    dict[str, dict[str, Path]]
        Saved output paths for each family.
    """
    validate_distinct_series_inputs(
        imidazolones_input_csv=imidazolones_input_csv,
        thiazolones_input_csv=thiazolones_input_csv,
        print_report=print_report,
    )

    imi_outputs = cluster_with_almos(
        series_name='Imidazolones',
        input_csv=imidazolones_input_csv,
        n_clusters=n_clusters_imidazolones,
        top_n_per_cluster=top_n_per_cluster,
        output_dir=output_dir,
        conda_env=conda_env,
        print_report=print_report,
    )
    thi_outputs = cluster_with_almos(
        series_name='Thiazolones',
        input_csv=thiazolones_input_csv,
        n_clusters=n_clusters_thiazolones,
        top_n_per_cluster=top_n_per_cluster,
        output_dir=output_dir,
        conda_env=conda_env,
        print_report=print_report,
    )

    return {
        'imidazolones': imi_outputs,
        'thiazolones': thi_outputs,
    }


def run_phase3_clustering(
    n_clusters_imidazolones: int | None = None,
    n_clusters_thiazolones: int | None = None,
    top_n_per_cluster: int = 3,
    clustering_input_dir: str | Path = 'mol_files/7. Clustering',
    output_dir: str | Path = 'mol_files/7. Clustering/ALMOS',
    conda_env: str | None = None,
    print_report: bool = True,
) -> dict[str, dict[str, Path]]:
    """
    Run Phase-3 ALMOS clustering using latest Phase-2 clustering inputs.

    Parameters
    ----------
    n_clusters_imidazolones
        Fixed cluster count for imidazolones (None = elbow auto-selection).
    n_clusters_thiazolones
        Fixed cluster count for thiazolones (None = elbow auto-selection).
    top_n_per_cluster
        Number of shortlist molecules per cluster.
    clustering_input_dir
        Directory containing ``*_input_*cmpds.csv`` files.
    output_dir
        Output directory for ALMOS files.
    conda_env
        Optional conda environment used to execute ALMOS.
    print_report
        Print execution report.

    Returns
    -------
    dict[str, dict[str, Path]]
        Saved output paths for each family.
    """
    imidazolones_path, thiazolones_path = load_phase2_clustering_input_paths(clustering_input_dir)

    if print_report:
        print(f'[run_phase3_clustering] Imidazolones input: {imidazolones_path}')
        print(f'[run_phase3_clustering] Thiazolones input:  {thiazolones_path}')

    return cluster_inputs(
        imidazolones_input_csv=imidazolones_path,
        thiazolones_input_csv=thiazolones_path,
        n_clusters_imidazolones=n_clusters_imidazolones,
        n_clusters_thiazolones=n_clusters_thiazolones,
        top_n_per_cluster=top_n_per_cluster,
        output_dir=output_dir,
        conda_env=conda_env,
        print_report=print_report,
    )
