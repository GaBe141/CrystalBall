"""Top-level convenience imports for legacy tests.

This re-exports selected modules so tests using `from src import utils, pipeline` keep working.
"""
from .core import utils as utils  # noqa: F401
from .analysis import pipeline as pipeline  # noqa: F401
from .analysis import stats_robust as stats_robust  # noqa: F401
from . import ranking as ranking  # noqa: F401

# Back-compat: allow `from src import visualize` used by CLI/tests
from .visualization import visualize as visualize  # noqa: F401
