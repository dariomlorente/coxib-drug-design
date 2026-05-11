from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ._paths import CLUSTERING_INTERIM, ALMOS_DIR, OUTPUTS
from ._clustering_core import (
    DEFAULT_IGNORE_COLS,
    _sanitize_token,
    _relative_repo_path,
    _sanitize_metadata_paths,
    _sha256_file,
    _count_csv_rows,
    _extract_rowcount_suffix,
    _resolve_cluster_col,
    _pick_distance_col,
    _resolve_point_selected_col,
    _prepare_almos_input_dataframe,
    _merge_almos_cluster_results,
    _choose_clustered_csv,
    _choose_representatives_csv,
    validate_clustering_input_csv,
    run_almos_cluster,
    load_almos_clustered_dataframe,
)


def find_latest_clustering_input_csv(
    series_name: str,
    clustering_dir: str | Path = CLUSTERING_INTERIM,
) -> Path:
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
    clustering_dir: str | Path = CLUSTERING_INTERIM,
) -> tuple[Path, Path]:
    imidazolones_path = find_latest_clustering_input_csv('Imidazolones', clustering_dir)
    thiazolones_path = find_latest_clustering_input_csv('Thiazolones', clustering_dir)
    return imidazolones_path, thiazolones_path


def validate_distinct_series_inputs(
    imidazolones_input_csv: str | Path,
    thiazolones_input_csv: str | Path,
    print_report: bool = True,
) -> None:
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


def select_cluster_representatives(
    df_clustered: pd.DataFrame,
    cluster_col: str,
    name_col: str = 'ID',
    price_col: str = 'PriceMol',
    qed_col: str = 'QED',
) -> pd.DataFrame:
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
    output_dir: str | Path = ALMOS_DIR,
    metadata: dict[str, Any] | None = None,
    print_report: bool = True,
) -> dict[str, Path]:
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

    root_dir = Path(OUTPUTS)
    root_dir.mkdir(parents=True, exist_ok=True)
    samples_cols = [c for c in df_representatives.columns if c not in ('ALMOS_ID', 'Cluster', 'Point selected', 'PC1', 'PC2', 'PC3', 'RepresentativeRule')]
    samples_path = root_dir / f'{token}_{n_clusters}samples.csv'
    df_representatives[samples_cols].to_csv(samples_path, index=False)
    paths['samples'] = samples_path

    df_shortlist.to_csv(paths['shortlist'], index=False)
    df_summary.to_csv(paths['summary'], index=False)

    meta_payload = _sanitize_metadata_paths(dict(metadata or {}))
    meta_payload.update(
        {
            'series_name': series_name,
            'cluster_col': cluster_col,
            'n_rows': n_rows,
            'n_clusters': n_clusters,
            'top_n': top_n,
            'saved_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'outputs': {
                key: _relative_repo_path(path)
                for key, path in paths.items()
                if key != 'metadata'
            },
        }
    )
    paths['metadata'].write_text(
        json.dumps(meta_payload, indent=2, sort_keys=True),
        encoding='utf-8',
    )

    if print_report:
        for key in ['clustered', 'representatives', 'samples', 'shortlist', 'summary', 'metadata']:
            print(f"[save_clustering_outputs] {_relative_repo_path(paths[key])}")

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
    output_dir: str | Path = ALMOS_DIR,
    conda_env: str | None = None,
    python_executable: str | None = None,
    extra_args: list[str] | None = None,
    timeout_sec: int = 7200,
    print_report: bool = True,
) -> dict[str, Path]:
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
    run_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = output_root / '.runs' / f"{_sanitize_token(series_name)}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    almos_input_path = run_dir / f'{input_path.stem}_almos_input.csv'
    df_almos_input.to_csv(almos_input_path, index=False)

    if generated_name_used and print_report:
        print(
            f"\u26a0\ufe0f [cluster_with_almos] {series_name}: found {duplicate_name_rows:,} duplicated "
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
            f'\u26a0\ufe0f [cluster_with_almos] Requested n_clusters={n_clusters}, '
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
    output_dir: str | Path = ALMOS_DIR,
    conda_env: str | None = None,
    print_report: bool = True,
) -> dict[str, dict[str, Path]]:
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
    clustering_input_dir: str | Path = '02_selected_mols_data/.interim/clustering',
    output_dir: str | Path = ALMOS_DIR,
    conda_env: str | None = None,
    print_report: bool = True,
) -> dict[str, dict[str, Path]]:
    imidazolones_path, thiazolones_path = load_phase2_clustering_input_paths(clustering_input_dir)

    if print_report:
        print(f'[run_phase3_clustering] Imidazolones input: {_relative_repo_path(imidazolones_path)}')
        print(f'[run_phase3_clustering] Thiazolones input:  {_relative_repo_path(thiazolones_path)}')

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
