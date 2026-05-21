from division import safe_divide


def test_safe_divide_returns_quotient() -> None:
    assert safe_divide(6, 3) == 2


def test_safe_divide_returns_none_for_zero_denominator() -> None:
    assert safe_divide(6, 0) is None
