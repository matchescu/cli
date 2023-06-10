import json
from datetime import datetime

import matchescu_cli._global_flags as flags


def _print(*args):
    if not flags.VERBOSE:
        return
    print(*args)


class MatchescuJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)