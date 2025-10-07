"""Series analysis pipeline functions.

This module contains the logic previously embedded in main.analyze_file,
exposed as reusable functions with smaller responsibilities.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import ranking, stats_robust, utils


def sanitize_filename(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(name))


@dataclass
class AnalysisArtifacts:
    outputs: List[str]
    models: Dict[str, Dict]


def prepare_series(df: pd.DataFrame, logger, target_col: Optional[str] = None) -> Tuple[Optional[pd.Series], Optional[str], Optional[str]]:
    time_col = utils.detect_time_column(df)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        for col in df.columns:
            try:
                coerced = pd.to_numeric(df[col], errors='coerce')
                if coerced.notnull().sum() > max(3, int(len(df) * 0.1)):
                    df[col] = coerced
                    numeric_cols.append(col)
            except Exception:
                continue
    if target_col is None:
        target_col = utils.detect_cpi_column(df) or (numeric_cols[0] if numeric_cols else None)
        if target_col and target_col not in df.columns:
            found = next((c for c in df.columns if 'cpi' in str(c).lower()), None)
            target_col = found or (numeric_cols[0] if numeric_cols else None)
    if not target_col:
        return None, None, None
    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        except Exception:
            time_col = None
    if time_col:
        try:
            df = df.dropna(subset=[time_col, target_col])
        except Exception:
            return None, None, None
        df = df.sort_values(time_col)
        df.set_index(time_col, inplace=True)
        try:
            series = pd.to_numeric(df[target_col], errors='coerce').dropna().astype(float)
        except Exception:
            return None, None, None
        # Ensure monotonic, unique datetime index to avoid downstream reindex errors
        try:
            series = series.sort_index()
            if series.index.duplicated().any():
                # Keep last observation for duplicate timestamps
                series = series[~series.index.duplicated(keep='last')]
        except Exception:
            # Best-effort; continue if index operations fail
            pass
    else:
        try:
            df = df.dropna(subset=[target_col])
        except Exception:
            return None, None, None
        df = df.reset_index(drop=True)
        series = pd.to_numeric(df[target_col], errors='coerce').dropna().astype(float)
        if series.empty:
            return None, None, None
        series.index = pd.RangeIndex(start=0, stop=len(series))
    return series, time_col, target_col


def compute_candidates(df: pd.DataFrame, series: pd.Series, target_col: str) -> Optional[pd.DataFrame]:
    candidates_df = None
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cand_cols = [c for c in numeric_cols if c != target_col]
    if not cand_cols:
        for col in df.columns:
            if col == target_col:
                continue
            try:
                coerced = pd.to_numeric(df[col], errors='coerce')
                if coerced.notnull().sum() > max(3, int(len(df) * 0.1)):
                    df[col] = coerced
                    cand_cols.append(col)
            except Exception:
                continue
    if cand_cols:
        cand_df = df[[c for c in cand_cols if c in df.columns]]
        try:
            cand_df = cand_df.reindex(series.index).astype(float)
        except Exception:
            cand_df = cand_df.astype(float)
        cand_df = cand_df.dropna(axis=1, how='all')
        if not cand_df.empty:
            candidates_df = cand_df
    return candidates_df


def run_diagnostics(series: pd.Series, exog_df: Optional[pd.DataFrame], proc_dir: str, safe_base: str, logger) -> List[str]:
    outputs: List[str] = []
    diagnostics = {}
    try:
        diagnostics['adf'] = utils.adf_test(series)
        diagnostics['kpss'] = utils.kpss_test(series)
        diagnostics['acf_pacf'] = utils.acf_pacf(series, nlags=24)
        stl_res = utils.stl_decompose(series, period=None)
        if isinstance(stl_res, dict) and 'error' in stl_res:
            diagnostics['stl'] = stl_res
        else:
            try:
                diagnostics['stl'] = {
                    'seasonal': stl_res['seasonal'].tolist(),
                    'trend': stl_res['trend'].tolist(),
                    'resid': stl_res['resid'].tolist(),
                }
            except Exception:
                diagnostics['stl'] = str(stl_res)
        lb = utils.ljung_box_test(series, lags=[10, 20], return_df=True)
        try:
            diagnostics['ljung_box'] = lb.to_dict() if hasattr(lb, 'to_dict') else str(lb)
        except Exception:
            diagnostics['ljung_box'] = str(lb)
        # Robust analytical diagnostics (stationarity, serial correlation, heteroskedasticity, outliers, breaks)
        try:
            diagnostics['robust'] = stats_robust.run_robust_diagnostics(series)
        except Exception:
            diagnostics['robust'] = {'error': 'robust diagnostics failed'}
        if exog_df is not None and not exog_df.empty:
            try:
                gc = utils.granger_causality_tests(series, exog_df, maxlag=6, verbose=False)
                diagnostics['granger'] = gc.replace({np.nan: None}).to_dict(orient='records') if hasattr(gc, 'replace') else str(gc)
            except Exception:
                diagnostics['granger'] = {'error': 'granger failed'}
    except Exception:
        logger.exception("Diagnostics failed")
    diag_path = os.path.join(proc_dir, f"{safe_base}_diagnostics.json")
    try:
        with open(diag_path, 'w', encoding='utf-8') as fh:
            json.dump(diagnostics, fh, default=str, indent=2)
        outputs.append(diag_path)
    except Exception:
        logger.exception("Failed to write diagnostics")
    return outputs


def plot_model_outputs(series: pd.Series, base_name: str, target_col: Optional[str], vis_dir: str, models: Dict[str, Dict], logger) -> List[str]:
    outputs: List[str] = []
    for name, res in models.items():
        if not res or res.get("forecast") is None:
            continue
        try:
            f = res["forecast"]
            # Use object-oriented API for explicit memory management
            fig, ax = plt.subplots(figsize=(10, 5))
            try:
                ax.plot(series.index, series.values, label="Actual")
                if res.get("fitted") is not None:
                    try:
                        ax.plot(res["fitted"].index, res["fitted"].values, label=f"{name} fitted", linestyle="--")
                    except Exception:
                        pass
                try:
                    if isinstance(series.index, pd.DatetimeIndex) and not isinstance(f.index, pd.DatetimeIndex):
                        freq = pd.infer_freq(series.index)
                        if freq is not None:
                            start = series.index[-1]
                            future_index = pd.date_range(start=start + pd.tseries.frequencies.to_offset(freq), periods=len(f), freq=freq)
                            f = pd.Series(getattr(f, 'values', f), index=future_index)
                except Exception:
                    pass
                try:
                    ax.plot(f.index, getattr(f, 'values', f), label=f"{name} forecast")
                except Exception:
                    ax.plot(range(len(f)), f, label=f"{name} forecast")
                # draw conformal intervals if we have residuals
                try:
                    fitted_vals = res.get('fitted')
                    if fitted_vals is not None and isinstance(series.index, type(f.index)):
                        # residuals from training
                        train_idx = series.index.intersection(getattr(fitted_vals, 'index', series.index))
                        if len(train_idx) >= 5:
                            resid = series.loc[train_idx] - fitted_vals.loc[train_idx]
                            ci = stats_robust.conformal_intervals_safe(resid, pd.Series(getattr(f, 'values', f), index=f.index), alpha=0.1)
                            ax.fill_between(ci.index, ci['lower'].values, ci['upper'].values, color='gray', alpha=0.15, label=f"{name} conf. band")
                except Exception:
                    pass
                safe_target = sanitize_filename(target_col) if target_col else "target"
                ax.set_title(f"{base_name}: {target_col} - {name} forecast")
                ax.legend()
                fig.tight_layout()
                out_png = os.path.join(vis_dir, f"{sanitize_filename(base_name)}_{safe_target}_{name}_forecast.png")
                tmp_png = out_png + '.tmp'
                try:
                    fig.savefig(tmp_png, format='png')
                    try:
                        os.replace(tmp_png, out_png)
                    except Exception:
                        os.rename(tmp_png, out_png)
                    outputs.append(out_png)
                except Exception:
                    try:
                        if os.path.exists(tmp_png):
                            os.remove(tmp_png)
                    except Exception:
                        pass
                    raise
            finally:
                plt.close(fig)  # Ensure figure memory is released
        except Exception:
            logger.exception("Plotting failed for %s", name)
    return outputs


def compute_rankings_and_adherence(models: Dict[str, Dict], series: pd.Series, vis_dir: str, proc_dir: str, safe_base: str, logger) -> List[str]:
    outputs: List[str] = []
    try:
        model_rankings = ranking.compute_model_rankings(models)
        rankings_path = os.path.join(proc_dir, f"{safe_base}_rankings.csv")
        model_rankings.to_csv(rankings_path, index=False)
        outputs.append(rankings_path)
        ranking_fig = ranking.visualize_rankings(model_rankings, title=f"Model Performance Rankings - {safe_base}")
        ranking_viz_path = os.path.join(vis_dir, f"{safe_base}_rankings.png")
        ranking_fig.savefig(ranking_viz_path, bbox_inches='tight', dpi=300)
        plt.close()
        outputs.append(ranking_viz_path)
        adherence_path = ranking.create_adherence_report(models=models, actual=series, output_dir=vis_dir, base_name=safe_base)
        outputs.append(adherence_path)
    except Exception:
        logger.exception("Ranking/adherence failed")
    return outputs
