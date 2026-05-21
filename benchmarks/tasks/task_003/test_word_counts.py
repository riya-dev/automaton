from word_counts import most_common_word


def test_most_common_word_returns_highest_frequency_word() -> None:
    assert most_common_word(["red", "blue", "red", "green"]) == "red"
