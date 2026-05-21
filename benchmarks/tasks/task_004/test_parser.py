from parser import parse_count


def test_parse_count_returns_integer_value() -> None:
    result = parse_count(" 42 ")

    assert result == 42
    assert isinstance(result, int)
