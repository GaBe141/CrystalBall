CrystalBall workspace helpers

If VS Code runs `main.py` from the wrong folder (for example from `praxis-engine`), add these workspace settings to force the terminal to run in the file's folder and prefer a local virtual environment:

- `.vscode/settings.json` contains:

  {
    "python.terminal.executeInFileDir": true,
    "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
  }

Quick steps to set up a local venv and run `main.py` in PowerShell (Windows):

1. Open PowerShell in the `CrystalBall` folder.
2. Create a venv: `python -m venv .venv`
3. Activate it: `.\.venv\Scripts\Activate.ps1`
4. Install dependencies: `pip install -r requirements.txt` (or your project's requirements)
5. Run the script from the CrystalBall workspace: `.\.venv\Scripts\python.exe src\main.py`

Alternatively, in VS Code press Ctrl+Shift+P -> "Python: Select Interpreter" and choose the workspace `.venv` interpreter. Then use "Run Python File in Terminal" and the file will execute in its folder using that interpreter.

Developer hygiene

- Install dev tools: `pip install -r dev-requirements.txt`
- Enable pre-commit hooks: `pre-commit install`
- Run checks locally:
  - Ruff: `ruff check . && ruff format --check .`
  - Mypy: `mypy .`
  - Tests: `pytest -q`
# CrystalBall Data Processing System

A local project for processing data.
\n+## Interactive Dashboard
\n+Run the Streamlit dashboard after an analysis run to explore results:\n\n1. Ensure dependencies are installed in your venv:\n   - `pip install -r requirements.txt`\n2. Launch the dashboard:\n   - `python -m src.run_dashboard`\n3. The app will open in your browser with:\n   - Per-series model rankings (table + Plotly chart)\n   - Forecast adherence report image per series\n   - Forecast plot gallery\n   - Global leaderboard (CSV + Plotly bar)\n\nTip: If you prefer, you can also run directly:\n   - `streamlit run src/dashboard.py`
