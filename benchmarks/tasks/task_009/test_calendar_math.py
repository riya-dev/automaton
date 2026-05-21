from calendar_math import is_leap_year


def test_regular_leap_year() -> None:
    assert is_leap_year(2024) is True


def test_century_year_not_divisible_by_400_is_not_leap_year() -> None:
    assert is_leap_year(1900) is False


def test_century_year_divisible_by_400_is_leap_year() -> None:
    assert is_leap_year(2000) is True
