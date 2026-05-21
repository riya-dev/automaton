from ranges import last_index


def test_last_index_returns_zero_based_final_index() -> None:
    assert last_index(["a", "b", "c"]) == 2


def test_last_index_returns_negative_one_for_empty_list() -> None:
    assert last_index([]) == -1
