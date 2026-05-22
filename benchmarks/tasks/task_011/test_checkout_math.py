from discounts import apply_discount
from taxes import add_tax


def test_apply_discount_uses_percentage() -> None:
    assert apply_discount(80.0, 25.0) == 60.0


def test_apply_discount_handles_zero_discount() -> None:
    assert apply_discount(37.5, 0.0) == 37.5


def test_add_tax_uses_percentage() -> None:
    assert add_tax(80.0, 10.0) == 88.0


def test_add_tax_handles_zero_tax() -> None:
    assert add_tax(37.5, 0.0) == 37.5
