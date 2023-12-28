import functools
import time
import datetime

from matchescu.logs import get_logger


def timer(text: str):
    def timer_decorator(f):

        @functools.wraps(f)
        def __wrap(*args, **kwargs):
            log = get_logger(text)
            start = time.time()
            try:
                return f(*args, **kwargs)
            finally:
                delta = datetime.timedelta(seconds=time.time() - start)
                log.info("duration=%(duration)s", {"duration": delta})

        return __wrap

    return timer_decorator
