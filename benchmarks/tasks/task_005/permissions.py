"""Permission helpers for benchmark task 005."""


def can_edit_document(is_owner: bool, is_admin: bool) -> bool:
    """Return whether a user can edit a document."""
    return is_owner and is_admin
