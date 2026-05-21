from permissions import can_edit_document


def test_owner_can_edit_document() -> None:
    assert can_edit_document(is_owner=True, is_admin=False) is True


def test_admin_can_edit_document() -> None:
    assert can_edit_document(is_owner=False, is_admin=True) is True


def test_unprivileged_user_cannot_edit_document() -> None:
    assert can_edit_document(is_owner=False, is_admin=False) is False
