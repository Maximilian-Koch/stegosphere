import lazy_loader as lazy

submodules = ['ttf_CustomTable']
submod_attrs = {
    'ttf': ['TTFContainer']
}

__getattr__, __dir__, __all__ = lazy.attach(__name__, submodules=submodules, submod_attrs=submod_attrs)