import json
import logging
import os

from src import export
from src.automation.git_gateway import (
    auto_push_on_execution,
    push_on_milestone,
    AutoPushContext,
    start_auto_push_daemon
)
from src.core.config import load_config
from src.analysis.diagnostics import (
    enable_warning_capture,
    guarded_call,
    install_global_exception_logger,
    log_environment,
    preflight_verify,
)
from src.core.logutil import get_logger
from src.analysis.pipeline import analyze_all

# Paths are provided by config; keep this file thin and CLI-focused.

@auto_push_on_execution("File analysis completed")
def analyze_file(path):
    # Backward compatibility: delegate to new pipeline implementation
    from src.analysis.pipeline import analyze_file as _analyze
    return _analyze(path)

def main():
    """Programmatic entry point.

    Configures diagnostics and runs the full analysis.
    """
    logger = get_logger("crystalball.main")
    enable_warning_capture()
    install_global_exception_logger(logger)
    # Log environment report into processed/exports for traceability if possible
    try:
        cfg = load_config()
        env_path = os.path.join(cfg.paths.processed_dir, "exports", "environment.json")
    except Exception:
        env_path = None
    log_environment(logger, write_json_path=env_path)
    preflight_verify(logger)
    # Run pipeline
    return guarded_call(logger, analyze_all)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CrystalBall Time Series Analysis")
    parser.add_argument("--dashboard", action="store_true", help="Run the Streamlit dashboard")
    parser.add_argument("--export", nargs="*", choices=['pdf', 'pptx', 'docx', 'xlsx'], help="Export analysis to PDF, PowerPoint, Word, and/or Excel")
    # API fetch options
    parser.add_argument("--api-provider", type=str, help="API provider name (e.g., 'dummy')")
    parser.add_argument("--api-series", type=str, help="Series identifier to fetch from the provider")
    parser.add_argument("--api-output", type=str, help="Optional output directory for fetched CSV (defaults to data/raw/api)")
    parser.add_argument("--validate", nargs="*", help="Run model validation across processed datasets; optionally pass model names")
    parser.add_argument("--model-readmes", action="store_true", help="Write per-model README docs")
    parser.add_argument("--llm-consensus", nargs="*", help="Run LLM consensus (providers: heuristic, openai, anthropic, google, azure, mistral, cohere)")
    args = parser.parse_args()

    logger = get_logger("crystalball.cli")
    enable_warning_capture()
    install_global_exception_logger(logger)
    # Attempt to compute env path early for CLI runs
    try:
        _cfg = load_config()
        _env_json = os.path.join(_cfg.paths.processed_dir, "exports", "environment.json")
    except Exception:
        _env_json = None
    log_environment(logger, write_json_path=_env_json)
    preflight_verify(logger)

    if args.dashboard:
        # Launch the Streamlit app properly
        try:
            from src.run_dashboard import main as run_dash
            guarded_call(logger, run_dash)
        except Exception:
            # Fallback to direct call (may not start Streamlit server)
            from src.dashboard import display_dashboard
            guarded_call(logger, display_dashboard)
    elif args.export is not None:
        cfg = load_config()
        os.makedirs(cfg.paths.exports_dir, exist_ok=True)
        formats = args.export if args.export else ['pdf', 'pptx', 'docx', 'xlsx']
        exported = guarded_call(
            logger, export.batch_export_reports, cfg.paths.processed_dir, cfg.paths.exports_dir, formats
        ) or {}
        for fmt, files in exported.items():
            print(f"\nExported {fmt.upper()} files:")
            for f in files:
                print(f"  - {f}")
        # Clear duplicates keeping most recent
        try:
            clearance = guarded_call(logger, export.clear_duplicate_exports, cfg.paths.exports_dir, formats) or {}
            removed = clearance.get('removed', [])
            if removed:
                print(f"\nRemoved {len(removed)} duplicate older files:")
                for f in removed:
                    print(f"  - {f}")
        except Exception:
            # Non-fatal if clearance fails
            logger.exception("Duplicate clearance failed")
        # Build/update executive summary artifacts
        try:
            from src.summary import build_executive_summary
            paths = guarded_call(logger, build_executive_summary, cfg.paths.processed_dir, cfg.paths.exports_dir)
            if paths:
                md = paths.get('markdown')
                if md:
                    print(f"\nExecutive summary written: {paths['json']} and {md}")
                else:
                    print(f"\nExecutive summary written: {paths['json']}")
        except Exception:
            logger.exception("Executive summary build failed")
        # Write an exports manifest for quick inspection
        try:
            manifest_path = os.path.join(cfg.paths.exports_dir, 'exports_manifest.json')
            _ = guarded_call(logger, export.generate_exports_manifest, cfg.paths.exports_dir, manifest_path, formats)
            if os.path.exists(manifest_path):
                print(f"\nExports manifest written: {manifest_path}")
        except Exception:
            logger.exception("Exports manifest generation failed")
    elif args.api_provider and args.api_series:
        from src.api import fetch_and_write_csv
        cfg = load_config()
        default_out = os.path.join(cfg.paths.raw_data_dir, 'api')
        out_dir = args.api_output or default_out
        os.makedirs(out_dir, exist_ok=True)
        csv_path = guarded_call(logger, fetch_and_write_csv, args.api_provider, args.api_series, out_dir)
        print(f"API data fetched and written to: {csv_path}")
    elif args.validate is not None:
        # Validation harness
        from src.validation import run_validation
        models = args.validate if len(args.validate) > 0 else None
        out = guarded_call(get_logger("crystalball.validation"), lambda: run_validation(models=models))
        if out:
            print(json.dumps(out, indent=2))
    elif args.model_readmes:
        from src.validation import write_model_readmes
        paths = guarded_call(get_logger("crystalball.modeldocs"), write_model_readmes)
        if paths:
            print(f"Wrote {len(paths)} model README files")
    elif args.llm_consensus is not None:
        from src.llm_consensus import run_llm_consensus
        providers = args.llm_consensus if len(args.llm_consensus) > 0 else None
        out = guarded_call(logger, run_llm_consensus, providers)
        if out:
            print(f"\nLLM consensus written to: {out['out_dir']}\nOverview: {out['overview']}")
    else:
        # Keep a basic config for third-party loggers and run analysis guarded
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        guarded_call(logger, analyze_all)
        # After analysis, aggregate rankings into a leaderboard
        try:
            cfg = load_config()
            from src.leaderboard import aggregate_rankings
            lb_path = guarded_call(logger, aggregate_rankings, cfg.paths.processed_dir, cfg.paths.visuals_dir)
            if lb_path:
                print(f"\nGlobal leaderboard written: {lb_path}")
        except Exception:
            # non-fatal if aggregation fails
            logger.exception("Leaderboard aggregation failed")
        # Build/update executive summary artifacts after analysis
        try:
            from src.summary import build_executive_summary
            paths = guarded_call(logger, build_executive_summary, cfg.paths.processed_dir, cfg.paths.exports_dir)
            if paths:
                md = paths.get('markdown')
                if md:
                    print(f"\nExecutive summary written: {paths['json']} and {md}")
                else:
                    print(f"\nExecutive summary written: {paths['json']}")
        except Exception:
            logger.exception("Executive summary build failed")
