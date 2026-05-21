from tags import append_tag


def test_append_tag_does_not_reuse_default_list() -> None:
    first = append_tag("urgent")
    second = append_tag("review")

    assert first == ["urgent"]
    assert second == ["review"]
