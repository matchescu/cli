from dataclasses import dataclass, field
from typing import Generic

from matchescu.cli.config import TConfig
from matchescu.cli.runtime import CommonCliOptions


@dataclass(frozen=True)
class EvalOptions(CommonCliOptions, Generic[TConfig]):
    config: TConfig | None = field(default=None)
