from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from .schemas import (
    ExecutiveSummary,
    LLMConsensusItem,
    ModelMetrics,
    ReportArtifacts,
    SeriesSummary,
)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def build_executive_summary(processed_dir: str, exports_dir: str) -> Dict[str, str]:
    """Build a robust executive summary from processed outputs.

    Reads per-series *_rankings.csv files, matches visuals and exports, and
    produces two artifacts:
    - executive_summary.json: validated JSON with key metrics and artifacts
    - executive_summary.md: a skeleton Markdown outline for human review
    Returns a dict with paths of generated outputs.
    """
    series_summaries: List[SeriesSummary] = []

    rankings = [f for f in os.listdir(processed_dir) if f.endswith('_rankings.csv')]
    vis_dir = os.path.join(processed_dir, 'visualizations')
    vis_files = set(os.listdir(vis_dir)) if os.path.isdir(vis_dir) else set()
    # Ensure exports directory exists so we can write outputs
    if not os.path.isdir(exports_dir):
        os.makedirs(exports_dir, exist_ok=True)
    export_files = set(os.listdir(exports_dir)) if os.path.isdir(exports_dir) else set()

    # Load LLM consensus folder (if exists)
    llm_dir = os.path.join(exports_dir, 'llm_consensus')
    llm_index: Dict[str, str] = {}
    if os.path.isdir(llm_dir):
        # map base -> csv path
        for f in os.listdir(llm_dir):
            if f.endswith('_llm_consensus.csv'):
                base = f.replace('_rankings_llm_consensus.csv', '')
                llm_index[base] = os.path.join(llm_dir, f)

    for rf in rankings:
        base = rf.replace('_rankings.csv', '')
        path = os.path.join(processed_dir, rf)
        try:
            df = pd.read_csv(path)
        except Exception:
            # Skip corrupted/unreadable
            continue
        # Basic metrics extraction
        metrics: List[ModelMetrics] = []
        for _, row in df.iterrows():
            metrics.append(
                ModelMetrics(
                    name=str(row.get('model', 'unknown')),
                    family=str(row.get('family')) if pd.notna(row.get('family')) else None,
                    mean_rank=_safe_float(row.get('mean_rank')),
                    weighted_score=_safe_float(row.get('weighted_score')),
                    cv_mae=_safe_float(row.get('cv_mae')),
                    cv_rmse=_safe_float(row.get('cv_rmse')),
                )
            )
        # Determine top model (by weighted_score, fallback to mean_rank ascending)
        top: ModelMetrics | None = None
        if metrics:
            # Prefer highest weighted_score when available
            with_scores = [m for m in metrics if m.weighted_score is not None]
            if with_scores:
                top = max(with_scores, key=lambda m: m.weighted_score)  # type: ignore[arg-type]
            else:
                with_ranks = [m for m in metrics if m.mean_rank is not None]
                top = min(with_ranks, key=lambda m: m.mean_rank) if with_ranks else metrics[0]

        # Collect artifacts
        visuals = [
            os.path.join(processed_dir, 'visualizations', f)
            for f in vis_files
            if f.startswith(base) and f.lower().endswith(('.png', '.pdf'))
        ]
        exports: Dict[str, List[str]] = {}
        for f in export_files:
            if f.startswith(base) and '_analysis.' in f:
                ext = f.split('.')[-1].lower()
                exports.setdefault(ext, []).append(os.path.join(exports_dir, f))

        # Attach LLM consensus top-3 if present
        llm_top3: List[LLMConsensusItem] = []
        llm_csv = llm_index.get(base)
        if llm_csv and os.path.exists(llm_csv):
            try:
                cdf = pd.read_csv(llm_csv)
                # expected columns: model, consensus_score, consensus_rank, votes, score_variance, confidence
                cdf = cdf.sort_values('consensus_rank').head(3)
                for _, r in cdf.iterrows():
                    try:
                        llm_top3.append(LLMConsensusItem(
                            model=str(r.get('model')),
                            consensus_score=float(r.get('consensus_score')) if r.get('consensus_score') is not None else 0.0,
                            consensus_rank=int(r.get('consensus_rank')) if r.get('consensus_rank') is not None else 0,
                            votes=int(r.get('votes')) if r.get('votes') is not None else 0,
                            score_variance=float(r.get('score_variance')) if r.get('score_variance') is not None else None,
                            confidence=float(r.get('confidence')) if r.get('confidence') is not None else None,
                        ))
                    except Exception:
                        continue
            except Exception:
                pass

        series_summaries.append(
            SeriesSummary(
                base_name=base,
                metrics=metrics,
                top_model=top,
                artifacts=ReportArtifacts(visuals=visuals, exports=exports),
                llm_top3=llm_top3,
            )
        )

    exec_summary = ExecutiveSummary(
        total_series=len(series_summaries),
        series=series_summaries,
    )

    # Write JSON and Markdown skeleton
    out_json = os.path.join(exports_dir, 'executive_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        # Use Pydantic's JSON-compatible dump to safely encode datetimes, etc.
        json.dump(exec_summary.model_dump(mode='json'), f, ensure_ascii=False, indent=2)

    out_md = os.path.join(exports_dir, 'executive_summary.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('# Executive Summary\n\n')
        f.write(f'- Generated series: {exec_summary.total_series}\n')
        f.write('- Highlights:\n')
        f.write('  - Top models per series and key metrics\n\n')
        for s in exec_summary.series:
            f.write(f'## {s.base_name}\n')
            if s.top_model:
                f.write(f'- Top model: {s.top_model.name}\n')
                if s.top_model.weighted_score is not None:
                    f.write(f'  - Weighted score: {s.top_model.weighted_score:.3f}\n')
                if s.top_model.mean_rank is not None:
                    f.write(f'  - Mean rank: {s.top_model.mean_rank:.3f}\n')
                if s.top_model.cv_mae is not None:
                    f.write(f'  - CV MAE: {s.top_model.cv_mae:.3f}\n')
                if s.top_model.cv_rmse is not None:
                    f.write(f'  - CV RMSE: {s.top_model.cv_rmse:.3f}\n')
            if s.llm_top3:
                f.write('- LLM consensus (top 3):\n')
                for item in s.llm_top3:
                    conf = f", conf={item.confidence:.2f}" if item.confidence is not None else ""
                    f.write(f"  - #{item.consensus_rank}: {item.model} (score={item.consensus_score:.3f}, votes={item.votes}{conf})\n")
            if s.artifacts.visuals:
                f.write(f'- Visuals: {len(s.artifacts.visuals)}\n')
            if s.artifacts.exports:
                f.write('- Exports: ' + ', '.join(f"{k}({len(v)})" for k, v in s.artifacts.exports.items()) + '\n')
            f.write('\n')

    return {"json": out_json, "markdown": out_md}
