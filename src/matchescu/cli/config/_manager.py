import json
from os import PathLike
from pathlib import Path
from typing import TypeVar, Generic, Optional

from pydantic import TypeAdapter, BaseModel


TConfig = TypeVar("TConfig", bound=BaseModel)

class JSONConfig(Generic[TConfig]):
    def __init__(self, path: str | PathLike, config_type: type[TConfig]):
        self._path = Path(path)
        self._schema = config_type
        self._config: Optional[TConfig] = None

    def load(self) -> "JSONConfig":
        with open(self._path, "r") as f:
            obj = json.load(f)
            self._config = self._schema.model_validate(obj)
        return self

    @property
    def config_obj(self) -> TConfig:
        return self._config
