from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import pandas as pd

# Ensure project imports work when frozen and when run from source
try:
    from src import visualize
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src import visualize


def _detect_time_and_values(df: pd.DataFrame, time_col: Optional[str], value_cols: Optional[List[str]]) -> Tuple[str, List[str]]:
    # Use explicit if provided and valid
    if time_col and time_col in df.columns:
        tcol = time_col
    else:
        # Heuristics for time column detection
        candidates = [c for c in df.columns if c.lower() in (
            'date', 'time', 'period', 'month', 'year', 'title')]
        tcol = candidates[0] if candidates else None
        if tcol is None:
            # Try parse-like: pick column with highest datetime parse success
            best_col, best_rate = None, 0.0
            for c in df.columns:
                parsed = pd.to_datetime(df[c], errors='coerce')
                rate = parsed.notna().mean()
                if rate > best_rate:
                    best_col, best_rate = c, rate
            if best_col and best_rate >= 0.6:
                tcol = best_col
            else:
                # Fallback: try first column
                tcol = df.columns[0]

    # Value columns
    if value_cols:
        vcols = [c for c in value_cols if c in df.columns]
        if not vcols:
            # fallback to numeric
            vcols = df.select_dtypes(include='number').columns.tolist()
    else:
        vcols = df.select_dtypes(include='number').columns.tolist()
        if tcol in vcols:
            vcols.remove(tcol)
        if not vcols:
            # take all except tcol
            vcols = [c for c in df.columns if c != tcol]
    return tcol, vcols


def _interactive_pick_paths(logger: Optional[logging.Logger] = None) -> Tuple[str, str]:
    # Minimal interactive file/directory picker using tkinter
    try:
        if logger:
            logger.info('Opening file picker... If a window does not appear, check the taskbar or alt-tab.')
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        csv_path = filedialog.askopenfilename(
            title='Select input CSV',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if not csv_path:
            raise SystemExit('No CSV selected.')
        default_out = os.path.join(os.path.dirname(csv_path), 'visualizations')
        out_dir = filedialog.askdirectory(title='Select output directory (Cancel for default)') or default_out
        return csv_path, out_dir
    except Exception as e:
        msg = f'Interactive selection failed; please pass --csv and --out-dir. Details: {e}'
        if logger:
            logger.error(msg)
        raise SystemExit(msg)


def _console_prompt_paths(default_out: Optional[str] = None) -> Tuple[str, str]:
    csv_path = input('Enter path to CSV: ').strip().strip('"')
    if not csv_path:
        raise SystemExit('No CSV path provided.')
    suggested = default_out or os.path.join(os.path.dirname(csv_path), 'visualizations')
    out_dir = input(f'Enter output directory [{suggested}]: ').strip().strip('"')
    if not out_dir:
        out_dir = suggested
    return csv_path, out_dir


def _init_logger(out_dir: Optional[str], verbose: bool = True, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('CrystalBallVisualize')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Avoid duplicate handlers if re-initialized
    if logger.handlers:
        return logger
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File handler
    try:
        base_dir = out_dir or os.path.dirname(getattr(sys, 'executable', '') or os.getcwd())
        os.makedirs(base_dir, exist_ok=True)
        lf = log_file or os.path.join(base_dir, 'crystalball_visualize.log')
        fh = logging.FileHandler(lf, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.debug('Log file: %s', lf)
    except Exception:
        # Logging to file is best-effort
        pass
    return logger


def main():
    p = argparse.ArgumentParser(description="CrystalBall visualization CLI")
    p.add_argument("--csv", help="Input CSV with time column + value columns")
    p.add_argument("--time-col", required=False, default=None, help="Time column name (auto-detected if omitted)")
    p.add_argument("--value-cols", nargs="+", required=False, default=None, help="One or more value columns (auto-detected if omitted)")
    p.add_argument("--out-dir", required=False, default=None, help="Output directory (defaults to <csv_dir>/visualizations)")
    p.add_argument("--title", default=None, help="Plot title")
    p.add_argument("--interactive", action='store_true', help="Open file pickers if --csv/--out-dir not provided")
    p.add_argument("--verbose", action='store_true', help="Enable verbose logging")
    p.add_argument("--log-file", default=None, help="Optional log file path")
    args = p.parse_args()

    start = time.time()
    # Decide input/output
    csv_path: Optional[str] = args.csv
    out_dir: Optional[str] = args.out_dir
    if not csv_path:
        # Offer console prompt by default; GUI if --interactive
        if args.interactive:
            # Temp logger to console until out_dir known
            tmp_logger = _init_logger(out_dir=None, verbose=True, log_file=None)
            tmp_logger.info('No --csv provided. Launching interactive selection (GUI)...')
            try:
                csv_path, out_dir = _interactive_pick_paths(tmp_logger)
            except SystemExit:
                tmp_logger.info('Falling back to console prompts...')
                csv_path, out_dir = _console_prompt_paths(out_dir)
        else:
            print('No --csv provided. Enter values at the prompts (or Ctrl+C to cancel).')
            csv_path, out_dir = _console_prompt_paths(out_dir)

    if not os.path.exists(csv_path):
        raise SystemExit(f"Input CSV not found: {csv_path}")

    out_dir = out_dir or os.path.join(os.path.dirname(csv_path), 'visualizations')
    os.makedirs(out_dir, exist_ok=True)

    logger = _init_logger(out_dir=out_dir, verbose=args.verbose, log_file=args.log_file)
    logger.info('Starting visualization job')
    logger.info('CSV: %s', csv_path)
    logger.info('Out dir: %s', out_dir)

    # Load data
    logger.info('Step 1/5: Reading CSV...')
    df = pd.read_csv(csv_path)
    logger.info('Loaded %d rows, %d columns', len(df), len(df.columns))

    # Detect columns
    logger.info('Step 2/5: Detecting time/value columns...')
    time_col = args.time_col
    value_cols = args.value_cols
    tcol, vcols = _detect_time_and_values(df, time_col, value_cols)
    logger.info('Detected time_col=%s, value_cols=%s', tcol, ', '.join(vcols) if vcols else '[]')
    if tcol not in df.columns:
        logger.error('Could not determine time column. Available: %s', list(df.columns))
        raise SystemExit(1)
    if not vcols:
        logger.error('No value columns detected.')
        raise SystemExit(1)

    # Parse time
    logger.info('Step 3/5: Parsing time column and shaping data...')
    parsed_time = pd.to_datetime(df[tcol], errors="coerce")
    if parsed_time.isna().all():
        try:
            years = pd.to_numeric(df[tcol], errors='coerce')
            parsed_time = pd.to_datetime(years, format='%Y', errors='coerce')
        except Exception:
            pass
    df[tcol] = parsed_time
    df = df.dropna(subset=[tcol]).set_index(tcol)
    wide = df[vcols].copy()

    base = os.path.splitext(os.path.basename(csv_path))[0]
    title = args.title or base
    png = os.path.join(out_dir, f"{base}_timeseries.png")
    html = os.path.join(out_dir, f"{base}_timeseries.html")
    export_prefix = os.path.join(out_dir, f"{base}_export")

    # Generate outputs
    logger.info('Step 4/5: Saving Matplotlib PNG... -> %s', png)
    visualize.save_matplotlib_timeseries(wide, png, title=title)
    logger.info('Saved PNG')
    logger.info('Saving Plotly HTML... -> %s', html)
    visualize.save_plotly_timeseries(wide, html, title=title)
    logger.info('Saved HTML')
    logger.info('Step 5/5: Exporting CSV/Parquet/JSON... -> %s*', export_prefix)
    exports = visualize.export_for_powerbi(wide, export_prefix)
    logger.info('Exports complete')

    elapsed = time.time() - start
    logger.info('All done in %.2f seconds', elapsed)

    print('Detected:')
    print(f"  time_col: {tcol}")
    print(f"  value_cols: {', '.join(vcols)}")
    print('Wrote:')
    print(f"  PNG:   {png}")
    print(f"  HTML:  {html}")
    print(f"  CSV:   {exports['csv']}")
    if exports.get('parquet'):
        print(f"  PARQUET: {exports['parquet']}")
    print(f"  JSON:  {exports['json']}")

    # End of main


if __name__ == "__main__":
    main()
