"""
The analysis package is a collection of various tools for steganalysis.
"""


import lazy_loader as lazy

from pathlib import Path
pkg_dir = Path(__file__).parent
submodules = [p.stem for p in pkg_dir.iterdir()
     if p.is_file() and p.suffix == '.py' and p.stem != '__init__']

__getattr__, __dir__, _ = lazy.attach(__name__, submodules)