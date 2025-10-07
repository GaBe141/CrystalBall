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

A local project for processing data with automated GitHub backup integration.

## 🚀 Auto-Push Git Gateway

CrystalBall includes an intelligent Git Gateway system that automatically backs up your work to GitHub during analysis runs.

### Features

- **Automatic Pushes**: Triggers git pushes during code execution
- **Smart Rate Limiting**: Prevents excessive commits (5-minute minimum intervals)
- **File Watching**: Monitors data/results folders for changes
- **Context-Aware**: Different triggers for analysis completion, data changes, errors
- **Background Daemon**: Scheduled pushes for pending changes

### Usage

**Enhanced Main Script (Recommended)**:
```bash
python main_with_autopush.py
```

**CLI Management**:
```bash
# Show status
python tools/git_gateway_cli.py status

# Start/stop daemon
python tools/git_gateway_cli.py start
python tools/git_gateway_cli.py stop

# Force push with custom message
python tools/git_gateway_cli.py push -m "Custom milestone"

# Mark milestone
python tools/git_gateway_cli.py milestone "Analysis complete"

# File watcher mode
python tools/git_gateway_cli.py watch
```

**Programmatic Integration**:
```python
from src.git_gateway import auto_push_on_execution, push_on_milestone

@auto_push_on_execution("Custom analysis completed")
def my_analysis():
    # Your analysis code here
    pass

# Manual milestone
push_on_milestone("Important checkpoint reached")
```

### Auto-Push Triggers

- ✅ **File Analysis**: After each file is processed
- ✅ **Batch Analysis**: After full batch completion  
- ✅ **Results Generation**: When CSV/JSON outputs are created
- ✅ **Data Changes**: When input data files are modified
- ✅ **Visualizations**: When plots/charts are generated
- ✅ **Error Recovery**: Captures work even when errors occur
- ✅ **Session Milestones**: Start/end of analysis sessions
- ✅ **Scheduled Backup**: Every 10 minutes if changes exist

### Configuration

Edit `config/git_gateway_config.py` to customize:
- Push frequency and rate limits
- File patterns to monitor
- Commit message templates
- Safety constraints

## 🔮 Advanced Prediction Models

### SAMIRA (State-space Adaptive Multi-variate Integrated Regression Analysis)

CrystalBall includes the sophisticated SAMIRA model for enterprise-grade time series forecasting:

**Features:**
- **State-space representation** with Kalman filtering for optimal estimation
- **Adaptive learning** with forgetting factors for non-stationary data
- **Multivariate support** with time-varying regression coefficients
- **Uncertainty quantification** with 95% confidence intervals
- **Component decomposition** (trend, seasonal, exogenous effects)

**Usage:**
```python
from src.models.model_samira import fit_samira_model

# Fit SAMIRA model
result = fit_samira_model(
    series=your_timeseries,
    test_size=12,
    exog=exogenous_variables,  # Optional
    trend_components=2,        # Level + slope
    seasonal_period=12,        # Auto-detected if None
    adaptation_rate=0.95       # Higher = more adaptive
)

# Access results
forecast = result['forecast']
components = result['components']
confidence_intervals = result['confidence_intervals']
```

**Specialized Visualizations:**
```python
from src.visualization.samira_plots import create_samira_report

# Generate comprehensive SAMIRA report
plots = create_samira_report(
    result, 
    original_series, 
    output_dir="results/",
    series_name="unemployment"
)
```

**Model Capabilities:**
- ✅ **Structural breaks** - Adapts to regime changes
- ✅ **Time-varying coefficients** - Captures evolving relationships  
- ✅ **Multiple seasonalities** - Annual, quarterly, weekly patterns
- ✅ **Robust estimation** - Handles missing data and outliers
- ✅ **Real-time learning** - Updates parameters as new data arrives

### Configuration

Edit `config/git_gateway_config.py` to customize:
- Push frequency and rate limits
- File patterns to monitor
- Commit message templates
- Safety constraints
\n+## Interactive Dashboard
\n+Run the Streamlit dashboard after an analysis run to explore results:\n\n1. Ensure dependencies are installed in your venv:\n   - `pip install -r requirements.txt`\n2. Launch the dashboard:\n   - `python -m src.run_dashboard`\n3. The app will open in your browser with:\n   - Per-series model rankings (table + Plotly chart)\n   - Forecast adherence report image per series\n   - Forecast plot gallery\n   - Global leaderboard (CSV + Plotly bar)\n\nTip: If you prefer, you can also run directly:\n   - `streamlit run src/dashboard.py`
