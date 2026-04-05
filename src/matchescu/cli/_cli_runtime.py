from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from click import Context


@dataclass(frozen=True)
class CommonCliOptions:
    verbose: bool = field(default=False)
    root_dir: Path|None = field(default=None)


def get_options(ctx: Context) -> CommonCliOptions:
    return cast(CommonCliOptions, ctx.obj)
