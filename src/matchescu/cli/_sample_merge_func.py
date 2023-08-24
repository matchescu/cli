from typing import Iterable

from matchescu.types import Record


def merge_as_sets(a: Record, b: Record) -> Record:
    common_item_count = min(len(a), len(b))
    max_item_count = max(len(a), len(b))

    result = []
    for i in range(common_item_count):
        a_field = a[i]
        b_field = b[i]

        # set reunion preserving order
        merge_val = {val: None for val in a_field} if isinstance(a_field, (list, tuple, set, dict)) else {a_field: None}
        merge_val.update(
            {val: None for val in b_field} if isinstance(b_field, (list, tuple, set, dict)) else {b_field: None}
        )
        result.append(tuple(val for val in merge_val))

    if common_item_count < len(a):
        for i in range(common_item_count, max_item_count):
            result.append(a[i])
    if common_item_count < len(b):
        for i in range(common_item_count, max_item_count):
            result.append(b[i])
    return tuple(val for val in result)
