# CrystalBall Directory Structure

## 📁 Organized Code Structure

```
src/
├── core/                      # Core utilities and configuration
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── utils.py              # Core utility functions
│   ├── logutil.py            # Logging utilities
│   └── schemas.py            # Data schemas and types
│
├── analysis/                  # Analysis pipeline and algorithms
│   ├── __init__.py
│   ├── pipeline.py           # Main analysis pipeline
│   ├── analysis.py           # Core analysis functions
│   ├── diagnostics.py        # Diagnostic utilities
│   ├── evaluation.py         # Model evaluation
│   ├── ensemble.py           # Ensemble methods
│   ├── validation.py         # Data validation
│   └── stats_robust.py       # Robust statistics
│
├── models/                    # Machine learning models
│   ├── __init__.py
│   ├── base.py               # Base model classes
│   ├── wrappers.py           # Model wrappers
│   ├── models_semistructural.py  # Semi-structural models
│   ├── advanced_models.py    # Advanced ML models
│   ├── model_*.py            # Individual model implementations
│   └── model_registry.py     # Model registry
│
├── visualization/             # Plotting and visualization
│   ├── __init__.py
│   ├── visualize.py          # Core visualization utilities
│   └── visuals.py            # Advanced plotting functions
│
├── automation/                # Automation and integration
│   ├── __init__.py
│   └── git_gateway.py        # Git auto-push system
│
├── [remaining root files]     # Other modules
│   ├── api.py                # API interface
│   ├── dashboard.py          # Streamlit dashboard
│   ├── export.py             # Data export utilities
│   ├── main.py               # Main entry point
│   └── ...
│
tools/                         # Command-line tools and utilities
├── git_gateway_cli.py        # Git gateway management
├── auto_push_watcher.py      # File system watcher
├── visualize_cli.py          # Visualization CLI
├── generate_dummy_results.py # Test data generation
└── synthetic.py              # Synthetic data tools

tests/                         # Test suite
├── test_*.py                 # Unit tests
├── test_fallbacks.py         # Fallback method tests
└── test_theta_fallbacks.py   # Theta method tests

config/                        # Configuration files
├── config.yaml               # Main configuration
└── git_gateway_config.py     # Git gateway settings

data/                          # Data directories
├── raw/                      # Raw input data
├── processed/                # Processed data
└── processed/visualizations/ # Generated plots
```

## 🎯 Benefits of New Structure

### **Logical Grouping**
- **Core**: Essential utilities used across the project
- **Analysis**: All analysis-related functionality in one place
- **Models**: Clean separation of ML models and algorithms  
- **Visualization**: Dedicated space for plotting code
- **Automation**: Integration and automation tools

### **Clear Dependencies** 
- Core utilities can be imported by any module
- Analysis modules work together cohesively
- Models are self-contained and reusable
- Visualization is separate from business logic

### **Scalability**
- Easy to add new models in `models/`
- Analysis pipeline components in logical location
- Automation tools centralized
- Clear separation of concerns

## 🔧 Import Pattern Examples

```python
# Core utilities (available everywhere)
from src.core import load_config, get_logger
from src.core.utils import load_dataset

# Analysis pipeline
from src.analysis import analyze_file, analyze_all
from src.analysis.pipeline import analyze_file

# Models
from src.models.base import BaseModel
from src.models import model_registry

# Visualization  
from src.visualization.visualize import create_timeseries_plot

# Automation
from src.automation.git_gateway import auto_push_on_execution
```

This structure makes CrystalBall more maintainable, scalable, and intuitive for new developers!