"""Visualization utilities for CrystalBall

Creates Matplotlib static plots for quick inspection and Plotly interactive
plots for enterprise embedding. Exports CSV/Parquet/JSON artifacts suitable for
Power BI ingestion.

Functions:
- save_matplotlib_timeseries(df, out_path, title)
- save_plotly_timeseries(df, out_path, title)
- export_for_powerbi(df, out_prefix)
- visualize_model_results(results_dict, out_dir, series_name)
"""
from __future__ import annotations

import os
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_matplotlib_timeseries(df: pd.DataFrame, out_path: str, title: str = None):
    """Save a static Matplotlib timeseries plot to PNG.

    df should be wide-format: columns are series names, index is time.
    """
    _ensure_dir(os.path.dirname(out_path))
    plt.style.use('seaborn-v0_8-whitegrid')
    # Use object-oriented API for explicit memory management
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        for col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=2.0)
        if isinstance(df.index, pd.DatetimeIndex):
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        ax.legend(loc='best', frameon=True)
        ax.grid(alpha=0.3)
        if title:
            ax.set_title(title, fontsize=14)
        ax.set_xlabel('Date' if isinstance(df.index, pd.DatetimeIndex) else 'Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
    finally:
        plt.close(fig)  # Ensure figure memory is released


def save_plotly_timeseries(df: pd.DataFrame, out_path: str, title: str = None):
    """Save an interactive Plotly HTML file.

    The produced HTML can be embedded in dashboards that support HTML.
    """
    _ensure_dir(os.path.dirname(out_path))
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.update_layout(
        title={'text': title or '', 'x': 0.01, 'xanchor': 'left', 'font': {'size': 18}},
        xaxis_title='Date' if isinstance(df.index, pd.DatetimeIndex) else 'Index',
        yaxis_title='Value',
        template='plotly_white',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': -0.25, 'xanchor': 'left', 'x': 0},
        hovermode='x unified',
        font={'size': 12}
    )
    if isinstance(df.index, pd.DatetimeIndex):
        fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    fig.write_html(out_path, include_plotlyjs='cdn')


def export_for_powerbi(df: pd.DataFrame, out_prefix: str):
    """Export CSV, Parquet and JSON artifacts with clear schema for Power BI."""
    _ensure_dir(os.path.dirname(out_prefix) if os.path.dirname(out_prefix) else '.')
    csv_path = f"{out_prefix}.csv"
    parquet_path = f"{out_prefix}.parquet"
    json_path = f"{out_prefix}.json"

    # Reset index to expose time column explicitly
    export_df = df.reset_index()
    # Ensure a column name for index
    if export_df.columns[0] == 'index':
        export_df = export_df.rename(columns={'index': 'time'})
    export_df.to_csv(csv_path, index=False)
    try:
        export_df.to_parquet(parquet_path, index=False)
    except Exception:
        # Parquet optional; fail silently but leave CSV/JSON
        pass
    export_df.to_json(json_path, orient='records', date_format='iso')
    return {'csv': csv_path, 'parquet': parquet_path if os.path.exists(parquet_path) else None, 'json': json_path}


def visualize_model_results(results: Dict[str, Dict], out_dir: str, series_name: str):
    """Create plots and exports for a single series using models results.

    results: dict mapping model name -> result dict (with 'forecast' pd.Series and optionally 'fitted')
    """
    _ensure_dir(out_dir)
    # Build a combined DataFrame
    frames = []
    for name, res in results.items():
        try:
            if res.get('fitted') is not None:
                frames.append(pd.DataFrame({f'{name}_fitted': res['fitted']}))
            if res.get('forecast') is not None:
                # forecast might be Series or DataFrame
                fc = res['forecast']
                if isinstance(fc, pd.Series):
                    frames.append(pd.DataFrame({f'{name}_forecast': fc}))
                elif isinstance(fc, pd.DataFrame):
                    # flatten columns
                    for c in fc.columns:
                        frames.append(pd.DataFrame({f'{name}_forecast_{c}': fc[c]}))
        except Exception:
            continue

    if not frames:
        return {}

    df_combined = pd.concat(frames, axis=1)
    # Save artifacts
    png = os.path.join(out_dir, f"{series_name}_timeseries.png")
    html = os.path.join(out_dir, f"{series_name}_timeseries.html")
    prefix = os.path.join(out_dir, f"{series_name}_export")

    save_matplotlib_timeseries(df_combined, png, title=series_name)
    save_plotly_timeseries(df_combined, html, title=series_name)
    exports = export_for_powerbi(df_combined, prefix)

    return {'png': png, 'html': html, **exports}
