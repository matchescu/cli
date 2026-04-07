import os
from pathlib import Path


def make_absolute_path(path: str | os.PathLike, parent_dir: str | os.PathLike | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path

    parent = Path(parent_dir).absolute()
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    return parent / path
