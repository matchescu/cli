from collections.abc import Mapping, Sequence
from typing import Any


def field_text(record: Mapping[str, Any], field: str) -> str:
    """Return a field's value as text; missing and ``None`` become ``""``."""
    value = record.get(field)
    return "" if value is None else str(value)


def field_tokens(record: Mapping[str, Any], field: str) -> frozenset[str]:
    """Return the lowercased whitespace tokens of one field."""
    return frozenset(field_text(record, field).lower().split())


def record_tokens(record: Mapping[str, Any], fields: Sequence[str]) -> frozenset[str]:
    """Return the lowercased whitespace tokens of several fields together."""
    return frozenset().union(*(field_tokens(record, f) for f in fields))
