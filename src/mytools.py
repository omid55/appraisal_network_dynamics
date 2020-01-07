# Omid55

from time import time
from typing import Text
import sys


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


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()


# def get_parameters_string_format():
#     vars_dict = vars()
#     for var_name, var_value in vars_dict.items():
#     if str.isalpha(var_name[0]) and str.isupper(var_name):
#         print(var_name)
