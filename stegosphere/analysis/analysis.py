#tool for automated analysis and analysis pipeline

"""
capacity: measures how much data can be stored in a cover
detectability: performs tests how well it can be recognized
efficiency: measures time and memory efficiency
imperceptibility: measures perceivable distortions of cover
robustness: performs attacks
accuracy: measures extraction accuracy

visualizer: visualize differences etc.
"""

from stegosphere.analysis import capacity
from stegosphere.analysis import detectability
from stegosphere.analysis import efficiency
from stegosphere.analysis import imperceptibility
from stegosphere.analysis import robustness
from stegosphere.analysis import accuracy

from stegosphere.analysis import visualizer
