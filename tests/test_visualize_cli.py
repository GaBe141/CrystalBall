import os
import sys
import csv
import subprocess
import tempfile
from datetime import datetime, timedelta


def _write_sample_csv(path: str, rows: int = 20):
    start = datetime(2020, 1, 1)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "series_a", "series_b"])  # time + two numeric cols
        for i in range(rows):
            d = start + timedelta(days=i)
            w.writerow([d.strftime("%Y-%m-%d"), i, rows - i])


def test_visualize_cli_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "sample.csv")
        out_dir = os.path.join(tmp, "viz")
        _write_sample_csv(csv_path)

        # Run the CLI via Python to ensure correct interpreter and module paths
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            os.path.join("tools", "visualize_cli.py"),
            "--csv",
            csv_path,
            "--out-dir",
            out_dir,
            "--title",
            "Test Series",
        ]
        proc = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
        if proc.returncode != 0:
            # Bubble up useful logs to debug failures in CI
            print("STDOUT:\n" + proc.stdout)
            print("STDERR:\n" + proc.stderr)
        assert proc.returncode == 0

        base = os.path.splitext(os.path.basename(csv_path))[0]
        # Expected artifacts
        png = os.path.join(out_dir, f"{base}_timeseries.png")
        html = os.path.join(out_dir, f"{base}_timeseries.html")
        export_csv = os.path.join(out_dir, f"{base}_export.csv")
        export_json = os.path.join(out_dir, f"{base}_export.json")
        # parquet is optional; don't assert it exists

        assert os.path.exists(png), f"PNG not found: {png}"
        assert os.path.exists(html), f"HTML not found: {html}"
        assert os.path.exists(export_csv), f"CSV export not found: {export_csv}"
        assert os.path.exists(export_json), f"JSON export not found: {export_json}"
