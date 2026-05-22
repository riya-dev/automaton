"""Discount helpers for benchmark task 011."""


def apply_discount(subtotal: float, discount_percent: float) -> float:
    """Return subtotal after applying a percentage discount."""
    return subtotal - discount_percent
