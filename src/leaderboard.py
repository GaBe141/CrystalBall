"""Aggregate per-series rankings into a global leaderboard.

Scans the processed_dir for any CSVs ending with 'rankings.csv',
extracts model, rank, and weighted_score, and computes per-model
aggregates (mean/median rank, count, mean weighted score). Saves
leaderboard.csv and leaderboard.png under visuals_dir.
"""
from __future__ import annotations

import os
from glob import glob
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def _find_ranking_files(processed_dir: str) -> List[str]:
    # Collect all rankings files (shallow and recursive)
    patterns = [
        os.path.join(processed_dir, "*rankings.csv"),
        os.path.join(processed_dir, "**", "*rankings.csv"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob(pat, recursive=True))
    # Filter duplicates and non-files
    files = [f for f in sorted(set(files)) if os.path.isfile(f)]
    return files


def aggregate_rankings(processed_dir: str, visuals_dir: str) -> str:
    files = _find_ranking_files(processed_dir)
    if not files:
        return ""

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # normalize columns
            cols = {c.lower(): c for c in df.columns}
            if 'model' not in cols and 'Model' in df.columns:
                df.rename(columns={'Model': 'model'}, inplace=True)
            if 'rank' not in df.columns and 'Rank' in df.columns:
                df.rename(columns={'Rank': 'rank'}, inplace=True)
            if 'weighted_score' not in df.columns and 'Weighted_Score' in df.columns:
                df.rename(columns={'Weighted_Score': 'weighted_score'}, inplace=True)
            if 'model' in df.columns and 'rank' in df.columns:
                # optionally attach a series identifier from filename for future slicing
                df['__source__'] = os.path.basename(f)
                # coerce types
                df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
                if 'weighted_score' in df.columns:
                    df['weighted_score'] = pd.to_numeric(df['weighted_score'], errors='coerce')
                frames.append(df[['model', 'rank'] + (['weighted_score'] if 'weighted_score' in df.columns else []) + ['__source__']])
        except Exception:
            continue

    if not frames:
        return ""

    allr = pd.concat(frames, ignore_index=True)
    # aggregate
    agg = allr.groupby('model').agg(
        mean_rank=('rank', 'mean'),
        median_rank=('rank', 'median'),
        count=('rank', 'count'),
        mean_weighted_score=('weighted_score', 'mean') if 'weighted_score' in allr.columns else ('rank', 'size'),
    ).reset_index()
    agg = agg.sort_values(['mean_rank', 'median_rank', 'model']).reset_index(drop=True)

    # save csv
    os.makedirs(visuals_dir, exist_ok=True)
    out_csv = os.path.join(visuals_dir, 'leaderboard.csv')
    agg.to_csv(out_csv, index=False)

    # save simple barplot of mean_rank
    try:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(agg))))
        ax.barh(agg['model'], agg['mean_rank'], color='steelblue')
        ax.invert_yaxis()
        ax.set_xlabel('Mean Rank (lower is better)')
        ax.set_title('Global Model Leaderboard')
        plt.tight_layout()
        out_png = os.path.join(visuals_dir, 'leaderboard.png')
        fig.savefig(out_png, bbox_inches='tight', dpi=300)
        plt.close(fig)
    except Exception:
        out_png = ""

    return out_csv
