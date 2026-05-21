"""Word counting helpers for benchmark task 003."""

def most_common_word(words: list[str]) -> str:
    """Return the most frequently occurring word."""
    counts = Counter(words)
    return counts.most_common(1)[0][0]
