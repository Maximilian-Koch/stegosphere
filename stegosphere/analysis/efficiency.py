#Measures time and memory usage for efficiency

import timeit
import tracemalloc

class Timer:
    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.diff = self.end - self.start 
