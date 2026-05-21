"""Calendar helpers for benchmark task 009."""


def is_leap_year(year: int) -> bool:
    """Return whether a year is a leap year in the Gregorian calendar."""
    return year % 4 == 0
