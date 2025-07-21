"""
The tools package provides various additional functionalities, which are
encryption, Error correcting codes, compression, and payload distribution across multiple files.
"""

import lazy_loader as lazy

from pathlib import Path
pkg_dir = Path(__file__).parent
submodules = [p.stem for p in pkg_dir.iterdir()
     if p.is_file() and p.suffix == '.py' and p.stem != '__init__']

__getattr__, __dir__, _ = lazy.attach(__name__, submodules)