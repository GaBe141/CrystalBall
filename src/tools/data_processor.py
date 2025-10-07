"""Data processing utilities CLI-friendly wrapper.

Moved from `src/dataprocessor.py` to a tools subpackage for clearer structure.
"""
from __future__ import annotations

import logging

from .. import utils
from ..config import load_config


def process_all_csv(logger: logging.Logger | None = None):
    """Use shared utils cleaner to process all raw CSVs into processed dir."""
    cfg = load_config()
    summ = utils.bulk_load_and_clean_raw_csv(
        cfg.paths.raw_data_dir, cfg.paths.processed_dir, logger=logger
    )
    return {"summary": summ, "errors": []}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger("data_processor")
    result = process_all_csv(logger=_logger)
    print("Summary:", result["summary"]) 
    print("Errors:", result["errors"])
