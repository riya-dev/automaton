from slug import slugify_title


def test_slugify_title_normalizes_title_text() -> None:
    assert slugify_title("  Hello   World  ") == "hello-world"
