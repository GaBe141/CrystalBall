"""Utility functions for CrystalBall data processing.

This module provides small, testable helpers for loading and cleaning
datasets, detecting time/CPI-like columns, and summarizing statistics.
Keep these functions pure where possible so they're easy to unit-test.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


# --- Bulk CSV Loader and Cleaner ---
import glob


def read_csv_optimized(path: str, time_col: Optional[str] = None) -> pd.DataFrame:
    """Reads a CSV with memory optimization by downcasting numeric types
    and converting low-cardinality string columns to 'category' dtype.
    """
    df = pd.read_csv(path, low_memory=False, parse_dates=[time_col] if time_col else None)
    # Downcast numeric types to save memory
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    # Heuristic: if unique values are less than 50% of total, categorize
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / len(df[col]) < 0.5:
            df[col] = df[col].astype("category")
    return df


def bulk_load_and_clean_raw_csv(raw_dir, processed_dir, logger=None):
    csv_files = glob.glob(os.path.join(raw_dir, '*.csv'))
    summary = []
    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            # Use the new optimized reader
            df = read_csv_optimized(fpath)
        except Exception:
            try:
                # Fallback for encoding issues
                df = pd.read_csv(fpath, encoding='latin1', engine='python')
            except Exception as e:
                if logger: logger.error(f"Failed to load {fname}: {e}")
                continue
        # Clean column names
        df.columns = [str(c).strip().replace('\n', ' ').replace('\r', '').replace('"', '').replace("'", '').replace(' ', '_').replace('%','pct').replace('/','_').replace(':','_').replace(',','_') for c in df.columns]
        # Try to coerce all columns to numeric where possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
        # Drop fully empty columns
        df = df.dropna(axis=1, how='all')
        # Save cleaned file
        out_path = os.path.join(processed_dir, fname)
        try:
            df.to_csv(out_path, index=False, encoding='utf-8')
            if logger: logger.info(f"Cleaned file saved: {out_path}")
        except Exception as e:
            if logger: logger.error(f"Failed to save cleaned {fname}: {e}")
            continue
        # Collect summary stats
        summary.append({
            'file': fname,
            'rows': len(df),
            'cols': list(df.columns),
            'shape': df.shape
        })
    return summary


# --- Datetime parsing helpers ---
def to_datetime_safe(values, errors: str = 'coerce') -> pd.Series:
    """Parse datetimes without noisy warnings across pandas versions.

    Tries pandas>=2.2 'format="mixed"' first (suppresses the
    "Could not infer format" warning), and falls back to the default
    behavior for older versions.
    """
    try:
        # pandas >= 2.2 supports format='mixed' to avoid per-element warnings
        return pd.to_datetime(values, errors=errors, format='mixed')
    except Exception:
        # Fallback for older pandas
        return pd.to_datetime(values, errors=errors)


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame.

    Raises ValueError for unsupported file types.
    """
    # Prefer a tolerant loader that can discover tables embedded behind prose
    if filepath.lower().endswith('.csv'):
        df = pd.read_csv(filepath)
        # quick pass: if this looks like a normal table, return it
        if not df.select_dtypes(include='number').empty:
            return df
        # otherwise try a header-discovery scan on the raw CSV
        try:
            raw = pd.read_csv(filepath, header=None, dtype=str)
            tbl = _find_table_in_frame(raw)
            if tbl is not None:
                return tbl
        except Exception:
            pass
        return df

    if filepath.lower().endswith(('.xlsx', '.xls')):
        # read all sheets without an assumed header and try to detect the tabular block
        try:
            sheets = pd.read_excel(filepath, sheet_name=None, header=None, dtype=str)
            for name, sheet in sheets.items():
                try:
                    # drop fully-empty rows/cols early
                    sheet_clean = sheet.replace(r'^\s*$', np.nan, regex=True).dropna(how='all', axis=0).dropna(how='all', axis=1)
                    tbl = _find_table_in_frame(sheet_clean)
                    if tbl is not None:
                        return tbl
                except Exception:
                    continue
        except Exception:
            # fall back to default reader for compatibility
            try:
                return pd.read_excel(filepath)
            except Exception:
                raise

    raise ValueError(f"Unsupported or unreadable file type: {filepath}")


def _find_table_in_frame(df_raw: pd.DataFrame, min_numeric_cols: int = 1, min_numeric_rows: int = 3, max_header_search: int = 10) -> Optional[pd.DataFrame]:
    """Try to locate a tabular block inside a DataFrame that has no header.

    Heuristic: scan the first `max_header_search` rows as possible header rows. For each
    candidate header row, treat the following rows as the body and check whether the body
    contains at least `min_numeric_cols` columns with >= `min_numeric_rows` numeric values.
    If found, return a typed DataFrame with detected header names and body rows.
    """
    if df_raw is None or df_raw.empty:
        return None
    df = df_raw.copy()
    # normalize blank-like cells
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
    nrows = len(df)
    if nrows < 2:
        return None

    max_search = min(max_header_search, max(0, nrows - 2))
    for header_row in range(0, max_search + 1):
        # candidate header
        header_vals: List[str] = [str(x).strip() if pd.notna(x) else '' for x in df.iloc[header_row].tolist()]
        # body is rows after header_row
        body = df.iloc[header_row + 1 :].copy()
        if body.empty:
            continue
        # assign column names from header, replace empty names with generic ones
        col_names = []
        for i, h in enumerate(header_vals):
            name = h if h else f"col_{i}"
            # ensure unique
            if name in col_names:
                name = f"{name}_{i}"
            col_names.append(name)
        body.columns = col_names

        # count numeric-like columns
        numeric_cols = 0
        for c in body.columns:
            try:
                coerced = pd.to_numeric(body[c], errors='coerce')
                if coerced.notnull().sum() >= min_numeric_rows:
                    numeric_cols += 1
            except Exception:
                continue
        if numeric_cols >= min_numeric_cols:
            # coerce numeric columns where possible and return
            for c in body.columns:
                # coerce non-numeric values to NaN so downstream numeric logic is consistent
                body[c] = pd.to_numeric(body[c], errors='coerce')
            # drop rows that are entirely NA
            body = body.dropna(how='all')
            body = body.reset_index(drop=True)
            return body
    return None


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Perform lightweight cleaning used by the CLI:
    - drop unnamed columns
    - strip and normalize column names
    - drop rows that are entirely NA
    - forward-fill missing values
    Returns a new DataFrame (does not modify the input in-place).
    """
    df = df.copy()
    # remove typical exported unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # normalize column names
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.dropna(how='all')
    df = df.ffill()
    return df


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Find a candidate time/date column name.

    Strategy:
    1. Look for common name tokens (date, month, year, time, period).
    2. Try to parse candidate columns to datetime; accept the first with >0 parsed values.
    3. Fall back to any column that already has a datetime dtype.
    Returns the column name or None.
    """
    candidates = [c for c in df.columns if any(k in c for k in ['date', 'month', 'year', 'time', 'period'])]
    for col in candidates:
        try:
            parsed = to_datetime_safe(df[col], errors='coerce')
            if parsed.notnull().sum() > 0:
                return col
        except Exception:
            continue
    # If no token-based candidate found, attempt to parse any column with many valid datetimes
    for col in df.columns:
        try:
            parsed = to_datetime_safe(df[col], errors='coerce')
            if parsed.notnull().sum() >= max(3, int(len(df) * 0.2)):
                return col
        except Exception:
            continue
    # dtype-based fallback
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def detect_cpi_column(df: pd.DataFrame) -> Optional[str]:
    """Detect a CPI-like numeric column by name tokens.
    Returns the column name or None.
    """
    for col in df.columns:
        if any(k in col for k in ['cpi', 'consumer_price_index', 'price_index', 'inflation']):
            return col
    return None


def summarize_stats(df: pd.DataFrame) -> Dict:
    """Return a small dictionary with useful dataset summaries.
    """
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'numeric_summary': df.select_dtypes(include='number').describe().to_dict(),
        'non_numeric_counts': df.select_dtypes(exclude='number').nunique().to_dict(),
    }
    return stats


def adf_test(series: pd.Series, maxlag: Optional[int] = None, autolag: str = 'AIC') -> Dict:
    """Run Augmented Dickey-Fuller test and return results as a dict.

    Returns keys: adf_stat, pvalue, usedlag, nobs, crit_values, icbest (if present), stationary (bool)
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception as e:
        return {'error': f'adfuller import failed: {e}'}

    try:
        s = series.dropna().astype(float)
        if s.empty:
            return {'error': 'empty series'}
        res = adfuller(s, maxlag=maxlag, autolag=autolag)
        adf_stat, pvalue, usedlag, nobs = res[0], res[1], res[2], res[3]
        crit = res[4] if len(res) > 4 else {}
        icbest = res[5] if len(res) > 5 else None
        stationary = pvalue < 0.05
        return {'adf_stat': float(adf_stat), 'pvalue': float(pvalue), 'usedlag': int(usedlag), 'nobs': int(nobs),
                'crit_values': crit, 'icbest': icbest, 'stationary': bool(stationary)}
    except Exception as e:
        return {'error': str(e)}


def kpss_test(series: pd.Series, regression: str = 'c', nlags: str = 'auto') -> Dict:
    """Run KPSS test and return results as a dict.

    Note: KPSS null is stationarity; we set 'stationary' True when pvalue > 0.05 (fail to reject).
    """
    try:
        from statsmodels.tsa.stattools import kpss
    except Exception as e:
        return {'error': f'kpss import failed: {e}'}

    try:
        s = series.dropna().astype(float)
        if s.empty:
            return {'error': 'empty series'}
        stat, pvalue, lags, crit = kpss(s, regression=regression, nlags=nlags)
        stationary = pvalue > 0.05
        return {'kpss_stat': float(stat), 'pvalue': float(pvalue), 'lags': int(lags), 'crit_values': crit, 'stationary': bool(stationary)}
    except Exception as e:
        return {'error': str(e)}


def acf_pacf(series: pd.Series, nlags: int = 40, fft: bool = True) -> Dict:
    """Compute ACF and PACF arrays up to nlags and return them.

    Returns dict with keys 'acf', 'pacf' (lists) and optional 'confint'.
    """
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except Exception as e:
        return {'error': f'acf/pacf import failed: {e}'}

    try:
        s = series.dropna().astype(float)
        if s.empty:
            return {'error': 'empty series'}
        acf_vals = acf(s, nlags=nlags, fft=fft, unbiased=False)
        pacf_vals = pacf(s, nlags=nlags)
        return {'acf': list(map(float, acf_vals)), 'pacf': list(map(float, pacf_vals))}
    except Exception as e:
        return {'error': str(e)}


def stl_decompose(series: pd.Series, period: Optional[int] = None, robust: bool = False) -> Dict:
    """Perform STL decomposition and return seasonal, trend, resid as Series.

    If period is None, attempt to infer frequency (e.g., monthly=12) and fallback to 1.
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except Exception as e:
        return {'error': f'STL import failed: {e}'}

    try:
        s = series.dropna().astype(float)
        if s.empty:
            return {'error': 'empty series'}
        # determine period
        if period is None:
            try:
                if isinstance(s.index, pd.DatetimeIndex):
                    freq = pd.infer_freq(s.index)
                    if freq is not None:
                        # crude mapping for common frequencies
                        if freq.lower().startswith('m'):
                            period = 12
                        elif freq.lower().startswith('q'):
                            period = 4
                        elif freq.lower().startswith('a') or freq.lower().startswith('y'):
                            period = 1
                        else:
                            period = 12
                    else:
                        period = 12
                else:
                    period = 12
            except Exception:
                period = 12

        stl = STL(s, period=period, robust=robust)
        res = stl.fit()
        return {'seasonal': res.seasonal, 'trend': res.trend, 'resid': res.resid}
    except Exception as e:
        return {'error': str(e)}


def ljung_box_test(series: pd.Series, lags: List[int] = [10], return_df: bool = True) -> Any:
    """Run Ljung-Box test for autocorrelation. Returns result DataFrame or array depending on statsmodels version.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except Exception as e:
        return {'error': f'acorr_ljungbox import failed: {e}'}

    try:
        s = series.dropna().astype(float)
        if s.empty:
            return {'error': 'empty series'}
        res = acorr_ljungbox(s, lags=lags, return_df=return_df)
        return res
    except Exception as e:
        return {'error': str(e)}


def granger_causality_tests(target: pd.Series, exog: pd.DataFrame, maxlag: int = 4, verbose: bool = False) -> pd.DataFrame:
    """Run Granger causality tests for each column in `exog` against `target`.

    Returns a DataFrame with columns: feature, best_pvalue, best_lag, pvalues_by_lag (dict)
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception as e:
        return pd.DataFrame([{'error': f'grangercausalitytests import failed: {e}'}])

    rows = []
    t = target.dropna().astype(float)
    for col in exog.columns:
        try:
            x = exog[col].dropna().astype(float)
            joined = pd.concat([t, x], axis=1).dropna()
            if joined.shape[0] < max(10, maxlag + 2):
                rows.append({'feature': col, 'best_pvalue': np.nan, 'best_lag': None, 'pvalues_by_lag': {}})
                continue
            data = joined.values
            # grangercausalitytests expects a 2-column array with [y, x]
            res = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)
            pvals = {}
            for lag, out in res.items():
                # out[0] is tuple of test results; pick the F-test p-value if available
                try:
                    test_res = out[0]
                    # depending on statsmodels version structure varies; try to extract p-value robustly
                    if isinstance(test_res, dict):
                        p = test_res.get('ssr_ftest', (None, None))[1]
                    else:
                        # older versions return tuples
                        p = None
                except Exception:
                    p = None
                pvals[int(lag)] = p
            # choose best lag by minimum p-value (ignoring None)
            best_lag = None
            best_p = None
            for lag, p in pvals.items():
                if p is None:
                    continue
                if best_p is None or (p < best_p):
                    best_p = p
                    best_lag = lag
            rows.append({'feature': col, 'best_pvalue': best_p, 'best_lag': best_lag, 'pvalues_by_lag': pvals})
        except Exception:
            rows.append({'feature': col, 'best_pvalue': np.nan, 'best_lag': None, 'pvalues_by_lag': {}})

    return pd.DataFrame(rows).sort_values('best_pvalue')


def rolling_correlation(target: pd.Series, exog: pd.Series, window: int = 12, min_periods: int = 3) -> pd.Series:
    """Compute rolling Pearson correlation between target and exog series.

    Returns a Series aligned to the input index with NaNs where insufficient data.
    """
    try:
        t = target.astype(float)
        x = exog.astype(float)
        joined = pd.concat([t, x], axis=1).dropna()
        if joined.empty:
            return pd.Series(dtype=float)
        rc = joined.iloc[:, 0].rolling(window=window, min_periods=min_periods).corr(joined.iloc[:, 1])
        return rc
    except Exception:
        return pd.Series(dtype=float)


def compute_affinity(target: pd.Series, candidates: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    """Compute affinity scores between a target series and candidate exogenous series.

    Returns a DataFrame with columns: feature, pearson_abs, spearman_abs, max_crosscorr, score
    where score is a simple weighted aggregation (pearson*0.5 + spearman*0.3 + max_crosscorr*0.2).
    The input `candidates` should be a DataFrame aligned or index-overlapping with the target.
    """
    # align indices
    common_idx = target.index.intersection(candidates.index)
    t = target.loc[common_idx].dropna()
    cand = candidates.loc[common_idx].copy()
    rows = []
    for col in cand.columns:
        c = cand[col].dropna()
        joined = pd.concat([t, c], axis=1).dropna()
        if joined.shape[0] < 3:
            rows.append({'feature': col, 'pearson_abs': 0.0, 'spearman_abs': 0.0, 'max_crosscorr': 0.0, 'score': 0.0})
            continue
        x = joined.iloc[:, 0].astype(float)
        y = joined.iloc[:, 1].astype(float)
        try:
            pear = abs(x.corr(y))
        except Exception:
            pear = 0.0
        try:
            from scipy.stats import spearmanr

            sp = abs(spearmanr(x, y).correlation)
        except Exception:
            sp = 0.0

        # cross-correlation up to max_lag (normalized)
        try:
            # use bfill/ffill explicitly to avoid deprecated fillna(method=...)
            xc = [abs(x.corr(y.shift(lag).bfill().ffill())) for lag in range(0, min(max_lag, len(joined) - 1) + 1)]
            max_xc = float(np.nanmax(xc)) if len(xc) > 0 else 0.0
        except Exception:
            max_xc = 0.0

        score = 0.5 * pear + 0.3 * sp + 0.2 * max_xc
        rows.append({'feature': col, 'pearson_abs': float(pear), 'spearman_abs': float(sp), 'max_crosscorr': float(max_xc), 'score': float(score)})

    return pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)


def top_affine_features(target: pd.Series, candidates: pd.DataFrame, top_n: int = 5, **kwargs) -> list:
    """Return a list of top-n feature names from candidates ranked by affinity to target."""
    df = compute_affinity(target, candidates, **kwargs)
    return df['feature'].tolist()[:top_n]


def fit_arima_series(series: pd.Series, test_size: int = 0, order: Optional[tuple] = None,
                     auto_order: bool = False, max_p: int = 2, max_d: int = 2, max_q: int = 2,
                     exog: Optional[pd.DataFrame] = None, exog_future: Optional[pd.DataFrame] = None) -> Dict:
    """Fit an ARIMA model on a univariate series and return forecasts and basic metrics.

    Parameters
    - series: pd.Series (indexed by time or integers). NaNs are dropped.
    - test_size: number of points to hold out for testing (0 = none)
    - order: (p,d,q) tuple to use; ignored if auto_order=True
    - auto_order: if True, perform a small grid search over p,d,q in [0..max_*] and pick lowest AIC

    Returns a dict with keys: order, model (the fitted result), forecast (pd.Series), conf_int (DataFrame),
    mae, rmse (if test provided), and error (if failed).
    """
    result = {
        'order': None,
        'model': None,
        'forecast': None,
        'conf_int': None,
        'mae': None,
        'rmse': None,
        'error': None,
    }

    s = series.dropna().astype(float)
    if s.empty or len(s) < 5:
        result['error'] = 'Series too short for ARIMA modelling'
        return result

    if test_size and test_size >= len(s):
        result['error'] = 'test_size must be smaller than series length'
        return result

    try:
        if auto_order:
            best_aic = np.inf
            best_order = None
            for p in range(0, max_p + 1):
                for d in range(0, max_d + 1):
                    for q in range(0, max_q + 1):
                        try:
                            train_for_search = s.iloc[:-test_size] if test_size else s
                            exog_for_search = None
                            if exog is not None:
                                # align exog to s index/length
                                exog_aligned = exog.loc[exog.index.intersection(s.index)] if not exog.index.equals(s.index) else exog
                                exog_for_search = exog_aligned.iloc[:len(train_for_search)]
                            m = ARIMA(train_for_search, order=(p, d, q), exog=exog_for_search)
                            res = m.fit()
                            if res.aic < best_aic:
                                best_aic = res.aic
                                best_order = (p, d, q)
                        except Exception:
                            continue
            if best_order is None:
                result['error'] = 'Auto-order selection failed'
                return result
            order = best_order

        if order is None:
            order = (1, 1, 1)
        result['order'] = order

        train = s.iloc[:-test_size] if test_size else s
        test = s.iloc[-test_size:] if test_size else None

        # prepare exog aligned to train/test if provided
        train_exog = None
        test_exog = None
        if exog is not None:
            exog_aligned = exog.loc[exog.index.intersection(s.index)] if not exog.index.equals(s.index) else exog
            train_exog = exog_aligned.iloc[:len(train)]
            if test is not None:
                test_exog = exog_aligned.iloc[len(train):len(train) + len(test)]
        model = ARIMA(train, order=order, exog=train_exog)
        fitted = model.fit()
        result['model'] = fitted

        # in-sample fitted values (align with train index)
        try:
            fitted_vals = pd.Series(fitted.predict(start=0, end=len(train) - 1), index=train.index)
        except Exception:
            # fallback to fittedvalues attribute
            try:
                fitted_vals = pd.Series(fitted.fittedvalues, index=train.index)
            except Exception:
                fitted_vals = None
        result['fitted'] = fitted_vals
        if fitted_vals is not None:
            train_mae = (fitted_vals - train).abs().mean()
            train_rmse = ((fitted_vals - train) ** 2).mean() ** 0.5
            result['train_mae'] = float(train_mae)
            result['train_rmse'] = float(train_rmse)

        steps = len(test) if test is not None else 10
        forecast_exog = test_exog if test_exog is not None else exog_future
        try:
            forecast_res = fitted.get_forecast(steps=steps, exog=forecast_exog)
        except Exception:
            forecast_res = fitted.get_forecast(steps=steps)
        forecast_mean = pd.Series(forecast_res.predicted_mean,
                                  index=(test.index if test is not None else
                                         pd.RangeIndex(start=len(train), stop=len(train) + steps)))
        conf_int = forecast_res.conf_int()

        result['forecast'] = forecast_mean
        result['conf_int'] = conf_int

        if test is not None:
            # align indices
            f = forecast_mean
            t = test
            # ensure same length
            if len(f) != len(t):
                result['error'] = 'Forecast/test length mismatch'
                return result
            mae = (f - t).abs().mean()
            rmse = ((f - t) ** 2).mean() ** 0.5
            result['mae'] = float(mae)
            result['rmse'] = float(rmse)

        return result
    except Exception as e:
        result['error'] = str(e)
        return result


def fit_prophet_series(series: pd.Series, test_size: int = 0, exog: Optional[pd.DataFrame] = None) -> Dict:
    """Fit a Prophet model on a univariate series (requires prophet package).

    The function will attempt to import Prophet; if it's not installed, returns an error in the result.
    """
    result = {'model': None, 'forecast': None, 'conf_int': None, 'mae': None, 'rmse': None, 'error': None}
    try:
        from prophet import Prophet
    except Exception:
        result['error'] = 'prophet not installed'
        return result

    s = series.dropna().astype(float)
    if s.empty or len(s) < 5:
        result['error'] = 'Series too short for Prophet modeling'
        return result

    # ensure datetime index
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s = s.copy()
            s.index = pd.to_datetime(s.index)
        except Exception:
            # can't coerce index
            result['error'] = 'Series index cannot be converted to datetime for Prophet'
            return result

    try:
        df = s.reset_index()
        df.columns = ['ds', 'y']
        train_df = df.iloc[:-test_size] if test_size else df
        test_df = df.iloc[-test_size:] if test_size else None

        m = Prophet()
        # register regressors if provided
        exog_aligned = None
        if exog is not None:
            try:
                exog_aligned = exog.copy()
                # try to align by index to s
                if not exog_aligned.index.equals(s.index):
                    exog_aligned.index = pd.to_datetime(exog_aligned.index)
                # add regressors to the prophet model
                for col in exog_aligned.columns:
                    m.add_regressor(col)
                # prepare exog reset for merging
                exog_reset = exog_aligned.reset_index()
                exog_reset = exog_reset.rename(columns={exog_reset.columns[0]: 'ds'})
                # merge regressor columns into train/test frames
                train_df = pd.merge(train_df, exog_reset, on='ds', how='left')
                if test_df is not None:
                    test_df = pd.merge(test_df, exog_reset, on='ds', how='left')
            except Exception:
                exog_aligned = None

        m.fit(train_df)

        # prepare future frame
        if test_df is not None:
            future = test_df[['ds'] + [c for c in train_df.columns if c not in ['ds', 'y']]]
        # if exog_aligned provided, try to create future with its last rows; otherwise use default
        elif exog_aligned is not None:
            # take next 10 rows from exog_aligned if available
            future_df = m.make_future_dataframe(periods=10)
            try:
                future = pd.merge(future_df, exog_reset, on='ds', how='left')
            except Exception:
                future = future_df
        else:
            future = m.make_future_dataframe(periods=10)

        forecast = m.predict(future)
        # align forecast series
        forecast_index = pd.to_datetime(forecast['ds'])
        forecast_series = pd.Series(forecast['yhat'].values, index=forecast_index)
        result['model'] = m
        result['forecast'] = forecast_series
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            ci = forecast[['yhat_lower', 'yhat_upper']].copy()
            ci.index = forecast_index
            result['conf_int'] = ci

        # in-sample fitted values: predict on training frame
        try:
            fitted_train = m.predict(train_df)
            fitted_series = pd.Series(fitted_train['yhat'].values, index=pd.to_datetime(fitted_train['ds']))
            result['fitted'] = fitted_series
            # compute train metrics (align lengths)
            tv = train_df['y'].values
            pv = fitted_series.values
            if len(tv) == len(pv):
                result['train_mae'] = float(abs(pv - tv).mean())
                result['train_rmse'] = float(((pv - tv) ** 2).mean() ** 0.5)
        except Exception:
            result['fitted'] = None

        if test_df is not None:
            # compute metrics on test
            actual = test_df['y'].values
            pred = forecast['yhat'].values
            mae = (abs(pred - actual)).mean()
            rmse = ((pred - actual) ** 2).mean() ** 0.5
            result['mae'] = float(mae)
            result['rmse'] = float(rmse)

        return result
    except Exception as e:
        result['error'] = str(e)
        return result


def fit_exponential_smoothing(series: pd.Series, test_size: int = 0, trend: Optional[str] = 'add',
                              seasonal: Optional[str] = None, seasonal_periods: Optional[int] = None,
                              damped_trend: bool = False) -> Dict:
    """Fit Holt-Winters Exponential Smoothing (ETS) and return forecasts and metrics.
    """
    result = {'model': None, 'forecast': None, 'conf_int': None, 'mae': None, 'rmse': None, 'error': None}
    s = series.dropna().astype(float)
    if s.empty or len(s) < 3:
        result['error'] = 'Series too short for ETS'
        return result

    if test_size and test_size >= len(s):
        result['error'] = 'test_size must be smaller than series length'
        return result

    try:
        train = s.iloc[:-test_size] if test_size else s
        test = s.iloc[-test_size:] if test_size else None

        model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal,
                                     seasonal_periods=seasonal_periods, damped_trend=damped_trend)
        fitted = model.fit(optimized=True)
        result['model'] = fitted

        # in-sample fitted
        try:
            fitted_vals = pd.Series(fitted.fittedvalues, index=train.index)
            result['fitted'] = fitted_vals
            result['train_mae'] = float((fitted_vals - train).abs().mean())
            result['train_rmse'] = float(((fitted_vals - train) ** 2).mean() ** 0.5)
        except Exception:
            result['fitted'] = None

        steps = len(test) if test is not None else 10
        forecast = pd.Series(fitted.forecast(steps), index=(test.index if test is not None else
                                                           pd.RangeIndex(start=len(train), stop=len(train) + steps)))
        result['forecast'] = forecast

        # statsmodels ExponentialSmoothing doesn't provide conf_int directly; skip for now
        if test is not None:
            mae = (forecast - test).abs().mean()
            rmse = ((forecast - test) ** 2).mean() ** 0.5
            result['mae'] = float(mae)
            result['rmse'] = float(rmse)

        return result
    except Exception as e:
        result['error'] = str(e)
        return result


def fit_theta_method(series: pd.Series, test_size: int = 0, h: int = 10, short_series_fallback: str = 'linear') -> Dict:
    """Simple Theta method implementation:
    - uses linear regression extrapolation (theta=2) and Simple Exponential Smoothing (theta=0)
    - combines them by averaging the forecasts.
    This is a pragmatic, lightweight approximation of the Hyndman Theta method.
    """
    import logging
    # result uses 'warning' rather than hard error for short series so pipeline can still include a forecast
    result = {'model': None, 'forecast': None, 'mae': None, 'rmse': None, 'error': None, 'warning': None, 'debug': {}}
    s = series.dropna().astype(float)
    result['debug']['input_length'] = len(s)
    result['debug']['test_size'] = test_size
    result['debug']['short_series_fallback'] = short_series_fallback
    # record a compact representation of the original index for diagnostics
    try:
        result['debug']['series_index_type'] = type(series.index).__name__
    except Exception:
        result['debug']['series_index_type'] = 'unknown'

    # If series very short, emit a fallback naive forecast (repeat last value) instead of failing
    if s.empty:
        result['error'] = 'Empty series for Theta'
        result['debug']['reason'] = 'empty'
        return result

    if test_size and test_size >= len(s):
        result['error'] = 'test_size must be smaller than series length'
        result['debug']['reason'] = 'test_size >= series length'
        return result

    try:
        train = s.iloc[:-test_size] if test_size else s
        test = s.iloc[-test_size:] if test_size else None
        result['debug']['train_length'] = len(train)
        result['debug']['test_length'] = len(test) if test is not None else 0
        # record index types/representations for easier debugging
        try:
            result['debug']['train_index_type'] = type(train.index).__name__
            result['debug']['test_index_type'] = type(test.index).__name__ if test is not None else None
        except Exception:
            pass

        n = len(train)
        # fallback for very small n: produce naive last-value forecast
        if n < 5:
            # Provide several conservative fallback strategies to choose from.
            result['warning'] = f'Series short for Theta — using fallback: {short_series_fallback}'
            result['debug']['reason'] = 'short_series_fallback'
            steps = len(test) if test is not None else h
            vals = None
            try:
                if short_series_fallback == 'last':
                    vals = np.array([float(train.values[-1])] * steps)
                    result['model'] = {'method': 'naive_last'}
                elif short_series_fallback == 'mean':
                    m = float(np.nanmean(train.values))
                    vals = np.array([m] * steps)
                    result['model'] = {'method': 'naive_mean', 'mean': m}
                elif short_series_fallback == 'linear':
                    # linear extrapolation when at least 2 points available, otherwise fallback to last
                    if n >= 2:
                        x_ins = np.arange(n)
                        coef = np.polyfit(x_ins, train.values, 1)
                        slope, intercept = float(coef[0]), float(coef[1])
                        vals = intercept + slope * np.arange(n, n + steps)
                        result['model'] = {'method': 'naive_linear', 'lr_coef': coef}
                    else:
                        vals = np.array([float(train.values[-1])] * steps)
                        result['model'] = {'method': 'naive_last_fallback_from_linear'}
                elif short_series_fallback == 'zero':
                    vals = np.array([0.0] * steps)
                    result['model'] = {'method': 'naive_zero'}
                else:
                    # unknown option: behave like 'last'
                    vals = np.array([float(train.values[-1])] * steps)
                    result['model'] = {'method': 'naive_last_unknown_option', 'option': short_series_fallback}
            except Exception as e:
                result['debug']['fallback_exception'] = str(e)
                # final fallback: last value
                try:
                    vals = np.array([float(train.values[-1])] * steps)
                    result['model'] = {'method': 'naive_last_on_exception'}
                except Exception:
                    result['error'] = 'Fallback generation failed'
                    result['debug']['reason'] = 'fallback_generation_failed'
                    return result

            out_index = test.index if test is not None else pd.RangeIndex(start=0, stop=steps)
            forecast_series = pd.Series(vals, index=out_index)
            result['forecast'] = forecast_series
            # compute positional metrics using values to avoid index type comparison issues
            if test is not None and len(test) > 0:
                try:
                    tv = test.values
                    pv = np.asarray(vals)[: len(tv)]
                    result['mae'] = float(np.mean(np.abs(pv - tv)))
                    result['rmse'] = float(np.sqrt(np.mean((pv - tv) ** 2)))
                except Exception as e:
                    result['debug']['mae_rmse_exception'] = str(e)
            return result

        # main Theta-like implementation
        x = np.arange(n)
        coef = np.polyfit(x, train.values, 1)
        slope, intercept = float(coef[0]), float(coef[1])
        steps = len(test) if test is not None else h
        lr_vals = intercept + slope * np.arange(n, n + steps)

        # SES: fit and forecast the same number of steps
        ses = SimpleExpSmoothing(train).fit(optimized=True)
        ses_forecast = np.asarray(ses.forecast(steps))

        # combine by value arrays to avoid index-mismatch issues
        if len(lr_vals) != len(ses_forecast):
            # align lengths defensively
            m = min(len(lr_vals), len(ses_forecast))
            lr_vals = lr_vals[:m]
            ses_forecast = ses_forecast[:m]

        combined_vals = 0.5 * (lr_vals + ses_forecast)

        # produce a forecast Series — prefer to preserve original test index if present
        out_index = test.index if test is not None else pd.RangeIndex(start=n, stop=n + len(combined_vals))
        result['forecast'] = pd.Series(combined_vals, index=out_index)
        result['debug']['lr_forecast_length'] = len(lr_vals)
        result['debug']['ses_forecast_length'] = len(ses_forecast)
        result['debug']['combined_index'] = str(out_index)
        result['model'] = {'lr_coef': coef, 'ses_model': str(ses)}

        # fitted (in-sample) — build by values to avoid index ordering issues
        try:
            lr_in_sample = intercept + slope * x
            ses_fitted_vals = np.asarray(ses.fittedvalues)
            # align in-sample lengths defensively
            m_ins = min(len(lr_in_sample), len(ses_fitted_vals))
            combined_fitted_vals = 0.5 * (lr_in_sample[:m_ins] + ses_fitted_vals[:m_ins])
            fitted_index = train.index[:m_ins]
            result['fitted'] = pd.Series(combined_fitted_vals, index=fitted_index)
            result['train_mae'] = float(np.mean(np.abs(combined_fitted_vals - train.values[:m_ins])))
            result['train_rmse'] = float(np.sqrt(np.mean((combined_fitted_vals - train.values[:m_ins]) ** 2)))
        except Exception as e:
            result['fitted'] = None
            result['debug']['fitted_exception'] = str(e)

        # compute test metrics by positional arrays (avoid index alignment problems)
        if test is not None and len(test) > 0:
            try:
                tv = test.values
                pv = combined_vals[: len(tv)]
                result['mae'] = float(np.mean(np.abs(pv - tv)))
                result['rmse'] = float(np.sqrt(np.mean((pv - tv) ** 2)))
            except Exception as e:
                result['debug']['mae_rmse_exception'] = str(e)

        return result
    except Exception as e:
        result['error'] = str(e)
        result['debug']['outer_exception'] = str(e)
        logging.exception('Theta method failed')
        return result


def fit_croston(series: pd.Series, alpha: float = 0.1, n_forecast: int = 1, test_size: int = 0) -> Dict:
    """Basic Croston method for intermittent demand forecasting.

    Returns forecast (length n_forecast) as constant a/p where a is demand estimate and p is interval estimate.
    """
    result = {'model': None, 'forecast': None, 'mae': None, 'rmse': None, 'error': None}
    s = series.fillna(0).astype(float)
    if s.empty:
        result['error'] = 'Empty series'
        return result

    if test_size and test_size >= len(s):
        result['error'] = 'test_size must be smaller than series length'
        return result

    try:
        train = s.iloc[:-test_size] if test_size else s
        test = s.iloc[-test_size:] if test_size else None

        # Initialize using first non-zero
        arr = train.values
        nz_idx = np.where(arr > 0)[0]
        if len(nz_idx) == 0:
            # all zeros
            forecast = pd.Series([0.0] * n_forecast, index=(test.index if test is not None else pd.RangeIndex(0, n_forecast)))
            result['forecast'] = forecast
            return result

        first = nz_idx[0]
        a = arr[first]
        p = first + 1
        tau = 0
        for t in range(first + 1, len(arr)):
            tau += 1
            if arr[t] > 0:
                # update estimates
                a = alpha * arr[t] + (1 - alpha) * a
                p = alpha * (tau) + (1 - alpha) * p
                tau = 0

        croston_forecast = a / p if p != 0 else 0.0
        forecast = pd.Series([croston_forecast] * n_forecast, index=(test.index if test is not None else pd.RangeIndex(0, n_forecast)))
        result['model'] = {'a': a, 'p': p}
        # no in-sample fitted for Croston (constant forecast)
        result['fitted'] = None
        result['forecast'] = forecast

        if test is not None and len(test) >= len(forecast):
            f = forecast.iloc[:len(test)]
            mae = (f - test).abs().mean()
            rmse = ((f - test) ** 2).mean() ** 0.5
            result['mae'] = float(mae)
            result['rmse'] = float(rmse)

        return result
    except Exception as e:
        result['error'] = str(e)
        return result


def example_util():
    """Kept for compatibility with existing tests (returns None)."""
    return None


def fit_statsforecast_model(series: pd.Series, test_size: int = 0, model_name: str = 'auto', exog: Optional[pd.DataFrame] = None,
                           exog_future: Optional[pd.DataFrame] = None) -> Dict:
    """Optional wrapper for Nixtla's statsforecast models.

    This function is optional and will return result['error']='statsforecast not installed' if the package
    isn't available. It returns the same result dict pattern as other helpers.
    """
    result = {'model': None, 'forecast': None, 'conf_int': None, 'fitted': None,
              'train_mae': None, 'train_rmse': None, 'mae': None, 'rmse': None, 'error': None}
    try:
        # lazy import to keep dependency optional
        from statsforecast import StatsForecast
        from statsforecast.models import ETS, AutoARIMA, Croston, Theta
    except Exception:
        result['error'] = 'statsforecast not installed'
        return result

    s = series.dropna().astype(float)
    if s.empty or len(s) < 3:
        result['error'] = 'Series too short for statsforecast'
        return result

    try:
        # statsforecast expects a long DataFrame with columns: unique_id, ds, y
        df = s.reset_index()
        df.columns = ['ds', 'y']
        df['unique_id'] = 'series_1'
        # build model selection
        if model_name == 'auto':
            model = AutoARIMA()
        elif model_name.lower() == 'theta':
            model = Theta()
        elif model_name.lower() == 'croston':
            model = Croston()
        elif model_name.lower() == 'ets':
            model = ETS()
        else:
            model = AutoARIMA()

        sf = StatsForecast(df=df, models=[model], freq=None)
        # train/test split
        train_df = df.iloc[:-test_size] if test_size else df
        test_df = df.iloc[-test_size:] if test_size else None
        sf.fit(train_df)

        # Forecast
        h = len(test_df) if test_df is not None else 10
        fc = sf.predict(h=h)
        # fc is a DataFrame with columns including 'y_hat' or similar; try to extract
        try:
            forecast_vals = fc['y_hat'].astype(float).values
        except Exception:
            # fallback to first numeric column
            numcols = fc.select_dtypes(include='number').columns
            forecast_vals = fc[numcols[0]].astype(float).values

        forecast_index = test_df['ds'] if test_df is not None else pd.RangeIndex(start=len(train_df), stop=len(train_df) + h)
        forecast_series = pd.Series(forecast_vals, index=pd.to_datetime(forecast_index))
        result['forecast'] = forecast_series
        result['model'] = sf

        # statsforecast doesn't expose fitted values easily; skip fitted
        result['fitted'] = None

        if test_df is not None:
            actual = test_df['y'].values
            pred = forecast_series.values[:len(actual)]
            result['mae'] = float(abs(pred - actual).mean())
            result['rmse'] = float(((pred - actual) ** 2).mean() ** 0.5)

        return result
    except Exception as e:
        result['error'] = str(e)
        return result


def fit_mlforecast_model(series: pd.Series, covariates: Optional[pd.DataFrame] = None, lags: list = [1, 12],
                         test_size: int = 0, n_estimators: int = 100, learning_rate: float = 0.1) -> Dict:
    """Optional wrapper for Nixtla's mlforecast (LightGBM) approach.

    This function will return result['error']='mlforecast not installed' if the package is missing.
    It constructs simple lag features from the series and provided covariates and fits a LightGBM model.
    """
    result = {'model': None, 'forecast': None, 'conf_int': None, 'fitted': None,
              'train_mae': None, 'train_rmse': None, 'mae': None, 'rmse': None, 'error': None}
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        result['error'] = 'mlforecast or lightgbm not installed'
        return result

    s = series.dropna().astype(float)
    if s.empty or len(s) < max(lags) + 5:
        result['error'] = 'Series too short for mlforecast'
        return result

    try:
        # Prepare long-format DataFrame required by mlforecast
        df = s.reset_index()
        df.columns = ['ds', 'y']
        df['unique_id'] = 'series_1'

        # attach covariates if provided (must be aligned by index/ds)
        if covariates is not None:
            cov = covariates.copy()
            if not cov.index.equals(s.index):
                cov.index = pd.to_datetime(cov.index)
            cov_reset = cov.reset_index()
            cov_reset = cov_reset.rename(columns={cov_reset.columns[0]: 'ds'})
            df = pd.merge(df, cov_reset, on='ds', how='left')

        # define regressor model
        model = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        # MLForecast usage would require a features list; to keep this lightweight we fit a simple lag-based model
        # But to avoid heavy wiring, use a manual lag-feature approach here
        df_feat = df.copy()
        for lag in lags:
            df_feat[f'lag_{lag}'] = df_feat['y'].shift(lag)
        df_feat = df_feat.dropna().reset_index(drop=True)

        train_df = df_feat.iloc[:-test_size] if test_size else df_feat
        test_df = df_feat.iloc[-test_size:] if test_size else None

        feature_cols = [c for c in df_feat.columns if c.startswith('lag_') or c not in ['ds', 'y', 'unique_id']]
        X_train = train_df[feature_cols]
        y_train = train_df['y']
        X_test = test_df[feature_cols] if test_df is not None else None

        model.fit(X_train, y_train)
        result['model'] = model

        # fitted
        fitted_vals = pd.Series(model.predict(X_train), index=pd.to_datetime(train_df['ds']))
        result['fitted'] = fitted_vals
        result['train_mae'] = float((fitted_vals - train_df['y']).abs().mean())
        result['train_rmse'] = float(((fitted_vals - train_df['y']) ** 2).mean() ** 0.5)

        # forecast
        if test_df is not None and X_test is not None:
            preds = model.predict(X_test)
            forecast_series = pd.Series(preds, index=pd.to_datetime(test_df['ds']))
            result['forecast'] = forecast_series
            result['mae'] = float(abs(preds - test_df['y']).mean())
            result['rmse'] = float(((preds - test_df['y']) ** 2).mean() ** 0.5)
        else:
            # build simple last-window forecast for horizon 10
            last = df_feat.iloc[-1:]
            preds = []
            current = last['y'].values[0]
            for i in range(10):
                # naive persistence or using model on rolling lags
                preds.append(current)
            forecast_series = pd.Series(preds, index=pd.RangeIndex(start=len(df_feat), stop=len(df_feat) + 10))
            result['forecast'] = forecast_series

        return result
    except Exception as e:
        result['error'] = str(e)
        return result
