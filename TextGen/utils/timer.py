from functools import wraps
import time


def timer(func):
    """
    Decorator to measure the time elapsed during the execution of a function.
    """

    @wraps(func)
    def timer_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        print("Time elapsed: {:.4f}s".format(end_time - start_time))

        return result

    return timer_wrapper
