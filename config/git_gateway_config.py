# Git Gateway Configuration for CrystalBall Auto-Push System

# Push frequency settings (in seconds)
MINIMUM_PUSH_INTERVAL = 300  # 5 minutes minimum between pushes
SCHEDULED_PUSH_INTERVAL = 600  # 10 minutes for scheduled background pushes
FILE_CHANGE_COOLDOWN = 30  # 30 seconds between pushes for same file

# Push triggers configuration
PUSH_TRIGGERS = {
    # Function execution triggers
    "analyze_file": True,           # Push after individual file analysis
    "analyze_all": True,            # Push after batch analysis
    "model_training": True,         # Push after model training
    "results_generation": True,     # Push when results are written
    
    # File system triggers
    "data_changes": True,           # Push when data files change
    "config_changes": True,         # Push when config files change
    "source_changes": False,        # Don't auto-push source code changes
    
    # Milestone triggers
    "session_start": True,          # Push at session start
    "session_complete": True,       # Push at session completion
    "error_occurred": True,         # Push when errors occur
    
    # Scheduled triggers
    "background_daemon": True,      # Enable background scheduled pushes
    "file_watcher": True,          # Enable file system watching
}

# Git commit message templates
COMMIT_MESSAGES = {
    "analyze_file": "Auto-commit: File analysis completed - {filename}",
    "analyze_all": "Auto-commit: Batch analysis completed - {count} files",
    "results_generated": "Auto-commit: Analysis results generated",
    "data_change": "Auto-commit: Data updated - {filename}",
    "milestone": "Auto-commit: Milestone - {description}",
    "error": "Auto-commit: Error occurred - {error_summary}",
    "scheduled": "Auto-commit: Scheduled backup",
    "session_start": "Auto-commit: Analysis session started",
    "session_complete": "Auto-commit: Analysis session completed",
    "default": "Auto-commit: {timestamp}"
}

# Directories to watch for changes
WATCH_DIRECTORIES = [
    "data/processed",
    "data/processed/visualizations",
    "config",
    # "src",  # Uncomment to watch source code changes
]

# File patterns to trigger pushes
TRIGGER_PATTERNS = {
    # Results and data files
    "results": ["*.csv", "*.json", "*.xlsx", "*.parquet"],
    "visualizations": ["*.png", "*.html", "*.svg", "*.pdf"],
    "configs": ["*.yaml", "*.yml", "*.toml", "*.ini"],
    "models": ["*.pkl", "*.joblib", "*.h5"],
    
    # Source code (if enabled)
    "source": ["*.py", "*.ipynb"],
}

# Git operations timeout (in seconds)
GIT_TIMEOUT = {
    "status": 10,
    "add": 30,
    "commit": 30,
    "push": 60,
}

# Rate limiting and safety
SAFETY_LIMITS = {
    "max_pushes_per_hour": 20,      # Maximum pushes per hour
    "max_commits_per_day": 100,     # Maximum commits per day
    "require_changes": True,        # Only push if there are actual changes
    "verify_remote": True,          # Verify remote exists before pushing
}