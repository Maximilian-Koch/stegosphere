import lazy_loader as lazy

submod_attrs = {
    'image': ['ImageContainer']
}

__getattr__, __dir__, __all__ = lazy.attach(__name__, submod_attrs=submod_attrs)