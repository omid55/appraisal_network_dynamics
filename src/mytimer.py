from time import time
from typing import Text


class Timer():
    def __init__(self, message: Text = None):
        if message:
            self.message = message
        else:
            self.message = 'It took {elapsed_time:.2f} {unit}.'

    def __enter__(self):
        self.start = time()
        return None

    def __exit__(self, type, value, traceback):
        elapsed_time = time() - self.start
        if elapsed_time < 60:
            unit = 'seconds'
        elif elapsed_time < 3600:
            unit = 'minutes'
            elapsed_time /= 60.0
        else:
            unit = 'hours'
            elapsed_time /= 3600.0
        print(
            self.message.format(elapsed_time=elapsed_time, unit=unit))
        
