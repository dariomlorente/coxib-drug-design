from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pandas as pd

from ._paths import CLUSTERING_INTERIM, ALMOS_DIR, OUTPUTS

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


def _relative_repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return path.name


def _sanitize_metadata_paths(payload: Any) -> Any:
    if isinstance(payload, Path):
        return _relative_repo_path(payload)
    if isinstance(payload, list):
        return [_sanitize_metadata_paths(item) for item in payload]
    if isinstance(payload, dict):
        return {str(key): _sanitize_metadata_paths(value) for key, value in payload.items()}
    if isinstance(payload, str) and payload.startswith('/'):
        return _relative_repo_path(Path(payload))
    return payload


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
            f"\u26a0\ufe0f [validate_clustering_input] Found {dup_count:,} duplicate IDs in '{name_col}'."
        )

    if print_report:
        missing_ratio = out[numeric_like_cols].isna().mean().sort_values(ascending=False)
        high_missing = missing_ratio[missing_ratio > 0.30]
        if not high_missing.empty:
            top = ', '.join(f'{col} ({ratio:.1%})' for col, ratio in high_missing.head(5).items())
            print(
                '\u26a0\ufe0f [validate_clustering_input] Some descriptor columns exceed 30% NaN: '
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
        input_rel = _relative_repo_path(input_csv)
        print(f"[run_almos_cluster] --input {input_rel}")
        print(f"[run_almos_cluster] cwd={_relative_repo_path(run_path)}")

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
