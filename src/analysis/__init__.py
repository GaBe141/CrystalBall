"""Analysis subpackage.

Keep this __init__ minimal to avoid circular imports during test discovery.
"""

# Lightweight utility exported for convenience without importing heavy modules
def sanitize_filename(name: str) -> str:
	return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(name))

__all__ = ["sanitize_filename"]