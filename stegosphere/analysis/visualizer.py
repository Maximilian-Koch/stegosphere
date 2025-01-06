import matplotlib.pyplot as plt
import numpy as np

from stegosphere.analysis import imperceptibility as imp

def difference_plot(array_1, array_2, absolute=False):
    nums, counts = imp.difference_counts(array_1, array_2, absolute)

    plt.bar(nums, counts, edgecolor='black')

    plt.xlabel('Difference', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Difference frequency', fontsize=14)
    plt.xticks(nums)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()
