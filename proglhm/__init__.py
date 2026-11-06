"""ProGLHM: genomic language representations and hybrid promoter modelling.

Copyright (c) 2026 ProGLHM contributors. All rights reserved.
"""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "ProGLHM":
        from .model import ProGLHM
        return ProGLHM
    if name == "load_model":
        from .checkpoint import load_model
        return load_model
    raise AttributeError(name)


__all__ = ["ProGLHM", "load_model", "__version__"]
