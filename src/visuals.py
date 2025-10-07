"""Generate train/test visualizations and landscape PDFs for processed series.

This utility scans the processed data directory, detects the time and target
columns, splits the series into training and holdout using the configured
test_size_fraction, renders an overlay chart, saves a PNG in the visuals
directory, and writes a matching landscape PDF with clear labels.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

from src import utils
from src.analysis import sanitize_filename
from src.config import load_config
from src.logutil import get_logger


def _prepare_series_from_df(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Detect time and target columns and return a clean series with index."""
    time_col = utils.detect_time_column(df)
    target_col = utils.detect_cpi_column(df)
    if target_col is None:
        # fallback: first numeric
        num_cols = df.select_dtypes(include='number').columns.tolist()
        target_col = num_cols[0] if num_cols else None
    if target_col is None:
        return None, None

    if time_col:
        # Use safe datetime parser to avoid warnings and improve robustness
        df[time_col] = utils.to_datetime_safe(df[time_col], errors='coerce')
        data = pd.to_numeric(df[target_col], errors='coerce')
        s = pd.Series(data.values, index=df[time_col])
        s = s.dropna()
        s = s.sort_index()
    else:
        data = pd.to_numeric(df[target_col], errors='coerce').dropna()
        s = pd.Series(data.values)
        s.index = pd.RangeIndex(start=0, stop=len(s))
    if s.empty:
        return None, None
    return s.astype(float), target_col


def _plot_train_test(series: pd.Series, split_idx: int, title: str, out_png: str) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6.5))
    # Plot full series
    plt.plot(series.index, series.values, color="#1f77b4", label="Actual (Full)", linewidth=1.2)
    # Overlay train and holdout in distinct styles
    train_x, train_y = series.index[:split_idx], series.values[:split_idx]
    test_x, test_y = series.index[split_idx:], series.values[split_idx:]
    plt.plot(train_x, train_y, color="#2ca02c", linewidth=2.2, label=f"Training ({len(train_y)})")
    if len(test_y) > 0:
        plt.plot(test_x, test_y, color="#d62728", linewidth=2.2, label=f"Holdout ({len(test_y)})")

    # Light background band for training region if datetime or numeric index
    try:
        xmin, xmax = (train_x[0], train_x[-1]) if len(train_x) else (None, None)
        if xmin is not None and xmax is not None:
            plt.axvspan(xmin, xmax, color="#2ca02c", alpha=0.08, label="Training window")
    except Exception:
        pass

    if isinstance(series.index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax = plt.gca()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    plt.title(title, fontsize=14)
    plt.xlabel("Date" if isinstance(series.index, pd.DatetimeIndex) else "Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    tmp = out_png + ".tmp"
    plt.savefig(tmp, dpi=180, format='png')
    plt.close()
    try:
        os.replace(tmp, out_png)
    except Exception:
        os.rename(tmp, out_png)


def _latin1_safe(text: str) -> str:
    # Replace common unicode punctuation and coerce to latin-1-safe string
    replacements = {
        '\u2014': '-',  # em dash
        '\u2013': '-',  # en dash
        '\u2019': "'",  # curly apostrophe
        '\u2018': "'",
        '\u201c': '"',
        '\u201d': '"',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')


def _pdf_landscape_with_image(title: str, img_path: str, out_pdf: str) -> None:
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    # Title
    pdf.set_font('helvetica', 'B', 16)
    # Use recommended positioning API to avoid deprecation of ln parameter
    try:
        from fpdf.enums import XPos, YPos  # Available in newer fpdf2
        pdf.cell(0, 10, _latin1_safe(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    except Exception:
        # Fallback for older fpdf2 versions
        pdf.cell(0, 10, _latin1_safe(title), ln=True)
    pdf.ln(3)
    # Insert image scaled to page width with margins
    page_w = pdf.w - 20  # 10mm margin each side
    y = pdf.get_y()
    try:
        pdf.image(img_path, x=10, y=y, w=page_w)
    except Exception:
        # fallback: no-scale insert
        pdf.image(img_path, x=10, y=y)
    pdf.output(out_pdf)


def generate_train_test_visuals() -> List[str]:
    cfg = load_config()
    logger = get_logger("crystalball.visuals")
    processed_dir = cfg.paths.processed_dir
    visuals_dir = cfg.paths.visuals_dir
    exports_dir = cfg.paths.exports_dir
    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(exports_dir, exist_ok=True)

    test_frac = getattr(cfg.settings, 'test_size_fraction', 0.2)
    outputs: List[str] = []
    # Gather candidate processed files
    files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.lower().endswith(('.csv', '.xlsx'))]
    for path in files:
        try:
            if path.lower().endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception:
            continue

        series, target_col = _prepare_series_from_df(df)
        if series is None or target_col is None or len(series) < 10:
            continue

        split_idx = max(1, int(round(len(series) * (1 - float(test_frac)))))
        base_name = os.path.splitext(os.path.basename(path))[0]
        safe_base = sanitize_filename(base_name)

        title = f"{base_name} — {target_col} (Train vs Holdout)"
        out_png = os.path.join(visuals_dir, f"{safe_base}_{sanitize_filename(target_col)}_train_test.png")
        _plot_train_test(series, split_idx, title, out_png)
        outputs.append(out_png)

        out_pdf = os.path.join(exports_dir, f"{safe_base}_{sanitize_filename(target_col)}_train_test.pdf")
        _pdf_landscape_with_image(title, out_png, out_pdf)
        outputs.append(out_pdf)

        logger.info("Generated visuals: %s and %s", out_png, out_pdf)

    return outputs


def main() -> None:
    outputs = generate_train_test_visuals()
    if outputs:
        print("\nGenerated the following visual artifacts:")
        for p in outputs:
            print(f"  - {p}")
    else:
        print("No eligible series found for visualization.")


if __name__ == "__main__":
    main()
