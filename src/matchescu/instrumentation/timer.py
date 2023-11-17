import functools
import logging
import sys
import time
import datetime


def timer(text: str, log: logging.Logger = None):
    if log is None:
        log = logging.getLogger(timer.__name__)
        log.setLevel(logging.INFO)
        log.addHandler(logging.StreamHandler(stream=sys.stdout))

    def timer_decorator(f):
        @functools.wraps(f)
        def __wrap(*args, **kwargs):
            start = time.time()
            try:
                log.info("starting %s", text)
                return f(*args, **kwargs)
            finally:
                delta = datetime.timedelta(seconds=time.time() - start)
                log.info("%s took %s", text, delta)
        return __wrap

    return timer_decorator
