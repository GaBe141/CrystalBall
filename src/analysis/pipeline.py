"""High-level pipeline orchestration for CrystalBall."""
from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import evaluation, stats_robust
from ..core import utils
from ..automation.git_gateway import (
    auto_push_on_execution,
    push_on_milestone,
    push_on_results_generated,
    AutoPushContext
)
from .analysis import (
    compute_candidates,
    compute_rankings_and_adherence,
    plot_model_outputs,
    prepare_series,
    run_diagnostics,
    sanitize_filename,
)
from ..core.config import load_config
from .ensemble import cv_weighted_ensemble
from ..core.logutil import get_logger
from ..models.models_semistructural import (
    fit_ml_enhanced_state_space,
    fit_semi_structural_gap_trend,
    suite_of_models_weighted,
)


@auto_push_on_execution("File analysis completed")
def analyze_file(path: str) -> Dict:
    cfg = load_config()
    logger = get_logger(f"crystalball.{sanitize_filename(os.path.basename(path))}")
    base_name = os.path.splitext(os.path.basename(path))[0]
    safe_base = sanitize_filename(base_name)
    result: Dict = {"path": path, "safe_base": safe_base, "status": "started", "outputs": [], "error": None}
    try:
        df = utils.load_dataset(path)
        df = utils.clean_df(df)
        series, time_col, target_col = prepare_series(df, logger)
        if series is None:
            result["status"] = "skipped"
            return result

        # feature candidates and affinity
        exog_df = None
        candidates_df = compute_candidates(df.copy(), series, target_col)
        if candidates_df is not None and not candidates_df.empty:
            try:
                affinity_df = utils.compute_affinity(series, candidates_df)
                affinity_csv = os.path.join(cfg.paths.processed_dir, f"{safe_base}_affinity.csv")
                affinity_df.to_csv(affinity_csv, index=False)
                result['outputs'].append(affinity_csv)
                top_feats = affinity_df["feature"].tolist()[: cfg.settings.max_affinity_features]
                exog_df = candidates_df[top_feats].copy() if top_feats else None
            except Exception:
                logger.exception("Affinity computation failed")

        # diagnostics
        result['outputs'].extend(
            run_diagnostics(series, exog_df, cfg.paths.processed_dir, safe_base, logger)
        )

        # modeling
        models: Dict[str, Dict] = {}
        test_size = max(1, int(len(series) * cfg.settings.test_size_fraction))
        try:
            models["arima"] = utils.fit_arima_series(series, test_size=test_size, auto_order=True, exog=exog_df)
        except Exception:
            logger.exception("ARIMA failed")
        try:
            models["ets"] = utils.fit_exponential_smoothing(series, test_size=test_size, trend="add")
        except Exception:
            logger.exception("ETS failed")
        try:
            models["theta"] = utils.fit_theta_method(series, test_size=test_size)
        except Exception:
            logger.exception("Theta failed")
        try:
            zeros_fraction = (series == 0).mean()
            if zeros_fraction > 0.3:
                models["croston"] = utils.fit_croston(series, n_forecast=10, test_size=test_size)
        except Exception:
            logger.exception("Croston failed")

        # Semi-structural hybrid (FINEX-inspired)
        try:
            ss = fit_semi_structural_gap_trend(series, test_size=test_size, exog=exog_df)
            if isinstance(ss, dict) and ss.get('forecast') is not None:
                models['semi_structural'] = ss
        except Exception:
            logger.exception("Semi-structural hybrid failed")

        # ML-enhanced state-space (optional torch)
        try:
            mlh = fit_ml_enhanced_state_space(series, test_size=test_size, exog=exog_df)
            if isinstance(mlh, dict) and mlh.get('forecast') is not None:
                models['ml_enhanced_ss'] = mlh
        except Exception:
            logger.exception("ML-enhanced state-space failed")

        # SAMIRA (State-space Adaptive Multi-variate Integrated Regression Analysis)
        try:
            from ..models.model_samira import fit_samira_model
            samira_res = fit_samira_model(series, test_size=test_size, exog=exog_df)
            if isinstance(samira_res, dict) and samira_res.get('forecast') is not None:
                models['samira'] = samira_res
        except Exception:
            logger.exception("SAMIRA model failed")

        # optional wrappers
        try:
            sf_res = utils.fit_statsforecast_model(series, test_size=test_size, model_name="auto", exog=exog_df)
            if isinstance(sf_res, dict) and sf_res.get("error") is None:
                models["statsforecast"] = sf_res
        except Exception:
            logger.exception("statsforecast wrapper failed")
        try:
            ml_res = utils.fit_mlforecast_model(series, covariates=exog_df, lags=[1, 12], test_size=test_size)
            if isinstance(ml_res, dict) and ml_res.get("error") is None:
                models["mlforecast"] = ml_res
        except Exception:
            logger.exception("mlforecast wrapper failed")

        # plots
        result['outputs'].extend(
            plot_model_outputs(series, base_name, target_col, cfg.paths.visuals_dir, models, logger)
        )

        # Suite-of-models weighted ensemble
        try:
            combo = suite_of_models_weighted(models)
            if combo and combo.get('forecast') is not None:
                models['suite_weighted'] = {
                    'forecast': combo['forecast'],
                    'fitted': None,
                    'mae': None,
                    'rmse': None,
                }
        except Exception:
            logger.exception("Suite-of-models ensemble failed")

        # ranking
        result['outputs'].extend(
            compute_rankings_and_adherence(models, series, cfg.paths.visuals_dir, cfg.paths.processed_dir, safe_base, logger)
        )
        # compute lightweight rolling-origin CV metrics per model (stateless forecast closures)
        try:
            horizon = max(1, int(len(series) * cfg.settings.test_size_fraction))
            # Build closures that take a train series and return np.array/pd.Series of h forecasts
            def _fc_from_model_key(key: str):
                def _fc(train: pd.Series, h: int):
                    # Re-fit quickly using utils helpers per key; minimal features, no exog for speed
                    try:
                        if key == 'arima':
                            r = utils.fit_arima_series(train, test_size=h, auto_order=True)
                        elif key == 'ets':
                            r = utils.fit_exponential_smoothing(train, test_size=h, trend='add')
                        elif key == 'theta':
                            r = utils.fit_theta_method(train, test_size=h)
                        elif key == 'croston':
                            r = utils.fit_croston(train, n_forecast=h)
                        elif key == 'samira':
                            from ..models.model_samira import fit_samira_model
                            r = fit_samira_model(train, test_size=h, exog=None)  # No exog for CV speed
                        else:
                            # fallback: naive last value
                            last = float(train.iloc[-1])
                            return pd.Series([last] * h, index=pd.RangeIndex(0, h))
                        fc = r.get('forecast')
                        if fc is None:
                            last = float(train.iloc[-1])
                            return pd.Series([last] * h, index=pd.RangeIndex(0, h))
                        return pd.Series(getattr(fc, 'values', fc))
                    except Exception:
                        last = float(train.iloc[-1])
                        return pd.Series([last] * h, index=pd.RangeIndex(0, h))
                return _fc

            model_keys = [k for k in models if k in ('arima', 'ets', 'theta', 'croston')]
            closures = {k: _fc_from_model_key(k) for k in model_keys}
            cv_metrics = evaluation.compute_model_cv_metrics(series, horizon=horizon, n_folds=min(5, max(2, len(series)//max(1, horizon))), models_to_eval=closures)
            rankings_csv = os.path.join(cfg.paths.processed_dir, f"{safe_base}_rankings.csv")
            if os.path.exists(rankings_csv) and cv_metrics:
                try:
                    df_rank = pd.read_csv(rankings_csv)
                    df_rank2 = evaluation.merge_cv_into_rankings(df_rank, cv_metrics)
                    # optional blended score: downweight high cv_rmse
                    if 'weighted_score' in df_rank2.columns:
                        inv = 1.0 / df_rank2['cv_rmse'].replace([0, np.inf, -np.inf], np.nan)
                        inv = inv / inv.sum(skipna=True)
                        df_rank2['cv_blend_score'] = 0.5 * df_rank2['weighted_score'] + 0.5 * inv.fillna(0.0)
                    df_rank2.to_csv(rankings_csv, index=False)
                except Exception:
                    logger.exception("Failed to merge CV metrics into rankings for %s", safe_base)
            # export accuracy-based weights and weighting matrix
            try:
                if cv_metrics:
                    wdf = evaluation.accuracy_to_weights(cv_metrics, method="inv_rmse")
                    weights_csv = os.path.join(cfg.paths.processed_dir, f"{safe_base}_weights.csv")
                    wdf.to_csv(weights_csv, index=False)
                    result['outputs'].append(weights_csv)
                    wmat = evaluation.weights_to_matrix(wdf)
                    if not wmat.empty:
                        wmat_csv = os.path.join(cfg.paths.processed_dir, f"{safe_base}_weights_matrix.csv")
                        wmat.to_csv(wmat_csv)
                        result['outputs'].append(wmat_csv)
            except Exception:
                logger.exception("Failed to compute/export accuracy-based weights for %s", safe_base)
            # ensemble using cv_rmse weights on available test forecasts
            try:
                if cv_metrics:
                    rmse_series = pd.Series({k: v[1] for k, v in cv_metrics.items() if v[1] is not None})
                    # collect latest forecasts from models dict where available
                    forecasts = {}
                    for k, r in models.items():
                        if k in rmse_series.index and isinstance(r, dict) and r.get('forecast') is not None:
                            forecasts[k] = r['forecast']
                    if forecasts and not rmse_series.empty:
                        ens = cv_weighted_ensemble(forecasts, rmse_series)
                        ens_path = os.path.join(cfg.paths.processed_dir, f"{safe_base}_ensemble_cv_weighted.csv")
                        ens.to_csv(ens_path, header=['forecast'])
                        result['outputs'].append(ens_path)
            except Exception:
                logger.exception("Failed to compute CV-weighted ensemble for %s", safe_base)
        except Exception:
            logger.exception("CV metrics computation failed for %s", safe_base)
        # attach bootstrap CIs to rankings
        try:
            rankings_csv = os.path.join(cfg.paths.processed_dir, f"{safe_base}_rankings.csv")
            updated = stats_robust.attach_bootstrap_cis_to_rankings(rankings_csv, series, models, test_size=test_size)
            if updated:
                result['outputs'].append(rankings_csv)
        except Exception:
            logger.exception("Failed to attach bootstrap CIs to rankings")

        # model summary light
        rows = []
        for name, r in models.items():
            if not isinstance(r, dict):
                continue
            rows.append({
                "model": name,
                "train_mae": r.get("train_mae"),
                "train_rmse": r.get("train_rmse"),
                "mae": r.get("mae"),
                "rmse": r.get("rmse"),
                "error": r.get("error"),
            })
        df_models = pd.DataFrame(rows)
        try:
            ms_path = os.path.join(cfg.paths.processed_dir, f"{safe_base}_model_summary.csv")
            df_models.to_csv(ms_path, index=False)
            result['outputs'].append(ms_path)
        except Exception:
            logger.exception("Failed to write model summary")

        result['status'] = 'success'
        return result
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        return result


@auto_push_on_execution("Batch analysis completed")
def analyze_all(
    max_workers: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    cfg = load_config()
    logger = get_logger("crystalball.pipeline")
    if not os.path.isdir(cfg.paths.raw_data_dir):
        logger.warning("Raw data directory not found: %s", cfg.paths.raw_data_dir)
        return []

    # Clean raw CSVs to processed folder (idempotent)
    logger.info("Loading and cleaning raw CSV files...")
    summary = utils.bulk_load_and_clean_raw_csv(cfg.paths.raw_data_dir, cfg.paths.processed_dir, logger=logger)
    logger.info("Cleaned %d CSV files", len(summary))

    # Analyze processed files in parallel
    files = [os.path.join(cfg.paths.processed_dir, f) for f in os.listdir(cfg.paths.processed_dir) if f.lower().endswith((".csv", ".xlsx"))]
    if limit:
        files = files[:limit]
    
    logger.info("Processing %d files with parallel workers (max_workers=%s)", len(files), max_workers)
    results: List[Dict] = []
    
    with AutoPushContext(f"Processing {len(files)} files"):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_file = {executor.submit(analyze_file, file_path): file_path for file_path in files}
        
        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(
                    "[%d/%d] Successfully processed %s -> status: %s",
                    i + 1,
                    len(files),
                    os.path.basename(file_path),
                    result.get('status', 'unknown'),
                )
            except Exception:
                logger.exception(
                    "[%d/%d] FAILED to process %s",
                    i + 1,
                    len(files),
                    os.path.basename(file_path),
                )
                # Add a failure record
                results.append({
                    'path': file_path,
                    'status': 'error',
                    'error': 'Process execution failed',
                    'outputs': []
                })

        # Push milestone after processing all files
        push_on_milestone(f"Processed {len(files)} files - {len([r for r in results if r.get('status') == 'success'])} successful")

    # Write summary
    try:
        summary_path = os.path.join(cfg.paths.processed_dir, "processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, default=str, indent=2)
        logger.info("Processing summary written: %s", summary_path)
        push_on_results_generated()  # Trigger push when results are written
    except Exception:
        logger.exception("Failed to write processing summary")
    return results
