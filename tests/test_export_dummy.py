import os
import shutil
import tempfile

from src.export import batch_export_reports
from src.tools.generate_dummy_results import generate_dummy_results


def test_batch_export_with_dummy_results():
    tmpdir = tempfile.mkdtemp(prefix="cb_dummy_")
    try:
        results_dir = os.path.join(tmpdir, "processed")
        visuals_dir = os.path.join(results_dir, "visualizations")
        exports_dir = os.path.join(results_dir, "exports")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
        os.makedirs(exports_dir, exist_ok=True)

        # Generate 3 dummy series
        bases = generate_dummy_results(results_dir, visuals_dir, ["alpha", "beta", "gamma"], n_models=4)

        # Run exporter for all formats
        exported = batch_export_reports(results_dir, exports_dir, ["pdf", "pptx", "docx", "xlsx"])

        # Verify each format has at least one file and names correspond to series
        for fmt, files in exported.items():
            assert files, f"No files exported for {fmt}"
            for base in bases:
                # at least one file per base
                assert any(base in os.path.basename(f) for f in files), f"Missing {fmt} for {base}"

        # Check actual files exist on disk
        for fmt, files in exported.items():
            for f in files:
                assert os.path.exists(f), f"Exported file missing: {f}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
