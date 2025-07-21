"""
The containers package defines the file handling of various file types.
"""

from stegosphere.containers.container import Container
import lazy_loader as lazy

from pathlib import Path
import re

pkg_dir = Path(__file__).parent

submodules = [p.name
    for p in pkg_dir.iterdir()
    if p.is_dir() and re.match(r'^[A-Za-z0-9]', p.name)]

__getattr__, __dir__, _ = lazy.attach(__name__, submodules)
