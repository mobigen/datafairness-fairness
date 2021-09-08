from functools import wraps
import time


def with_elapsed(func):
    @wraps(func)
    def elapsed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'[elapsed time] {func.__name__}: {time.time() - start}')
        return result
    return elapsed
