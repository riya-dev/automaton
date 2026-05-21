"""Username helpers for benchmark task 008."""


def normalize_username(raw: str) -> str:
    """Normalize a display name for use as a username."""
    return raw.lower().replace(" ", "_")
