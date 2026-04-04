from dataclasses import dataclass, field
from typing import cast

from click import Context


@dataclass(frozen=True)
class CommonCliOptions:
    verbose: bool = field(default=False)


def get_options(ctx: Context) -> CommonCliOptions:
    return cast(CommonCliOptions, ctx.obj)
