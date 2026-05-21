from sequence import next_number


def test_next_number_returns_successor() -> None:
    assert next_number(4) == 5
