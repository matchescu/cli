from matchescu.typing import EntityReferenceIdentifier as RefId


def ref_id_sort_key(ref_id: RefId) -> tuple:
    """Sort key ordering reference ids by source, then label."""
    return ref_id.source, ref_id.label
