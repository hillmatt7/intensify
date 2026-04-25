"""Backend selection and dispatch."""

import warnings

from . import jax_backend, numpy_backend

_active_backend: object | None = None
_backend_name: str = "jax"  # default


class _BackendProxy:
    """Thin wrapper that always delegates to the currently active backend.

    Lets modules cache ``bt = get_backend()`` at import time while still
    picking up later ``set_backend()`` switches — every attribute access
    resolves through ``_resolve_backend()`` on demand.
    """

    __slots__ = ()

    def __getattr__(self, name: str):
        return getattr(_resolve_backend(), name)

    def __repr__(self) -> str:
        return f"<BackendProxy active={_backend_name}>"


_PROXY = _BackendProxy()


def _resolve_backend() -> object:
    """Return the concrete backend module, initializing on first call."""
    global _active_backend, _backend_name
    if _active_backend is None:
        if jax_backend.is_available():
            jax_backend.enable_x64()
            _active_backend = jax_backend.backend
            _backend_name = "jax"
        elif numpy_backend.is_available():
            _active_backend = numpy_backend.backend
            _backend_name = "numpy"
        else:
            raise RuntimeError(
                "Neither JAX nor NumPy is available. Please install one of them."
            )
    return _active_backend


def set_backend(name: str) -> None:
    """Set the active backend: 'jax' or 'numpy'."""
    global _active_backend, _backend_name
    name = name.lower()
    if name == "jax":
        if not jax_backend.is_available():
            warnings.warn(
                "JAX backend requested but JAX is not available. Falling back to NumPy.",
                RuntimeWarning,
            )
            _active_backend = numpy_backend.backend
            _backend_name = "numpy"
        else:
            jax_backend.enable_x64()
            _active_backend = jax_backend.backend
            _backend_name = "jax"
    elif name == "numpy":
        _active_backend = numpy_backend.backend
        _backend_name = "numpy"
    else:
        raise ValueError(f"Unknown backend '{name}'. Use 'jax' or 'numpy'.")


def get_backend() -> object:
    """Return a proxy to the active backend.

    The returned proxy delegates every attribute access to the concrete
    backend at the moment of access, so ``bt = get_backend()`` cached in
    module scope stays valid across later ``set_backend()`` calls.
    """
    _resolve_backend()
    return _PROXY


def get_backend_name() -> str:
    """Return the name of the active backend ('jax' or 'numpy')."""
    get_backend()  # Ensure initialized
    return _backend_name


# Expose the active backend's namespace at module level for convenience
# All functions/tools will be accessed via get_backend()