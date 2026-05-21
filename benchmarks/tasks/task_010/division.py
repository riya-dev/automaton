"""Division helpers for benchmark task 010."""


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Divide two numbers, returning None when division cannot be performed."""
    try:
        return numerator / denominator
    except TypeError:
        return None
