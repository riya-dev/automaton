from usernames import normalize_username


def test_normalize_username_collapses_whitespace() -> None:
    assert normalize_username("  Ada   Lovelace  ") == "ada_lovelace"
