"""Tensorflow Branch."""
try:
    from garage.tf._functions import paths_to_tensors
    __all__ = ['paths_to_tensors']
except ImportError:
    __all__ = []
