"""
Backend abstraction layer.

All numerical operations in the library should import from here
to remain backend-agnostic (JAX vs NumPy).
"""

from ._backend import _active_backend, get_backend, get_backend_name, set_backend

__all__ = ["get_backend", "get_backend_name", "set_backend", "_active_backend"]