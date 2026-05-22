"""Tax helpers for benchmark task 011."""


def add_tax(subtotal: float, tax_rate: float) -> float:
    """Return subtotal after adding percentage tax."""
    return subtotal + tax_rate
