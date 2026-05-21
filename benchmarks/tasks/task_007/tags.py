"""Tag helpers for benchmark task 007."""


def append_tag(tag: str, tags: list[str] = []) -> list[str]:
    """Append one tag and return the tag list."""
    tags.append(tag)
    return tags
