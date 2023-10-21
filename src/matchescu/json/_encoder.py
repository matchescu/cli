from json import JSONEncoder
from typing import Any

import numpy as np


class MatchescuEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.int64):
            return int(o)
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        return JSONEncoder.default(self, o)
