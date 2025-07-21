"""
Module for measuring elapsed and used memory of processes.
"""

import time
import tracemalloc
import functools
import sys

__all__ = ['Timer', 'MemoryProfiler', 'EfficiencyProfiler', 'profile_function']

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

class MemoryProfiler:
    def __enter__(self):
        tracemalloc.start()
        self.start_snapshot = tracemalloc.take_snapshot()
        return self

    def __exit__(self, *args):
        self.end_snapshot = tracemalloc.take_snapshot()
        stats = self.end_snapshot.compare_to(self.start_snapshot, 'lineno')
        self.peak_memory = tracemalloc.get_traced_memory()[1]
        self.stats = stats
        tracemalloc.stop()

class EfficiencyProfiler:
    def __enter__(self):
        self.timer = Timer()
        self.mem = MemoryProfiler()
        self.timer.__enter__()
        self.mem.__enter__()
        return self

    def __exit__(self, *args):
        self.mem.__exit__(*args)
        self.timer.__exit__(*args)
        self.elapsed = self.timer.elapsed
        self.peak_memory = self.mem.peak_memory
        self.memory_stats = self.mem.stats

def profile_function(func):
    """
    Decorator to time and measure memory of a function call.
    Returns:
        result, profiling_info (dict)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with EfficiencyProfiler() as p:
            result = func(*args, **kwargs)
        info = {
            'elapsed': p.elapsed,
            'peak_memory': p.peak_memory,
            #'memory_stats': p.memory_stats
        }
        return result, info
    return wrapper

