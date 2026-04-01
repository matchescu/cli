from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommonCliOptions:
    verbose: bool = field(default=False)
