from enum import StrEnum

import click


class InputOrder(StrEnum):
    normal = "normal"
    reverse = "reverse"
    both = "both"


def validate_input_order_option(_, param, value):
    """Validate that we take at most 3 distinct input orders as parameters."""
    if not value:
        return value

    if len(set(value)) != len(value):
        raise click.BadParameter("input orders must be distinct.", param=param)
    if len(value) > 3:
        raise click.BadParameter("can specify at most 3 input orders", param=param)

    return value
