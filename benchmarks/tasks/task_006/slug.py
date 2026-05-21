"""Slug helpers for benchmark task 006."""


def slugify_title(title: str) -> str:
    """Convert a title into a URL-friendly slug."""
    words = normalize_spaces(title).lower().split(" ")
    return "-".join(words)
