"""Backend shim — JAX backend was removed in 0.3.0 (Rust port).

Pre-0.3.0 the library supported JAX/numpy backend swapping via
``set_backend("jax")`` / ``set_backend("numpy")``. The Rust port makes
the JAX backend obsolete: every numerical hot path now routes through
the compiled `intensify._libintensify` extension. This module remains
only as a backward-compatible no-op shim so existing user code (and
tests) calling ``get_backend()`` / ``set_backend()`` keeps working.

If your code relied on JAX-specific behavior (e.g. autodiff over a
custom kernel), use the Rust path's analytic gradient or the
`tests/_reference/` JAX oracle.
"""

from __future__ import annotations

import numpy as np

# Expose numpy under the legacy `bt = get_backend()` interface. Most
# pre-0.3.0 callsites used `bt.array`, `bt.asarray`, `bt.exp`, `bt.sum`,
# etc. — all standard numpy. JAX-specific calls (`bt.lax.scan`,
# `bt.random.PRNGKey`) are no longer supported and raise AttributeError.


def get_backend():
    """Return the active backend module (always numpy now)."""
    return np


def get_backend_name() -> str:
    """Return 'numpy' — only backend supported post-0.3.0."""
    return "numpy"


def set_backend(name: str) -> None:
    """No-op for ``set_backend('numpy')``; raises for any other value
    (JAX backend was removed in 0.3.0)."""
    if name != "numpy":
        raise ValueError(
            f"Backend {name!r} is not supported in intensify ≥ 0.3.0. "
            "Only the numpy backend remains; the Rust core handles all "
            "performance-critical paths."
        )
