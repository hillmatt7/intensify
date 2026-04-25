"""Dispatch shim for the Rust core (`intensify._libintensify`).

Single source of truth for {Python user-facing API → Rust implementation}.
Modeled on Nautilus Trader's pattern of importing from the compiled
extension at module load time and raising a clear, actionable ImportError
if the extension is missing.

JAX is NOT a fallback. If the user's environment doesn't have the Rust
extension built, this shim raises ImportError pointing at the [fast] extra
or a binary wheel install. The frozen JAX reference oracle in
`tests/_reference/` is dev-only and never imported here.

Phase 0: shim infrastructure + loud import. Phase 1+ adds dispatch helpers
that route (process, kernel, fit_decay) tuples to specific Rust functions.
"""

from __future__ import annotations

try:
    from intensify import _libintensify as _ext
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "intensify requires the compiled Rust extension `_libintensify`.\n"
        "\n"
        "Install a binary wheel:\n"
        "    pip install intensify\n"
        "\n"
        "Or build from source (requires a Rust toolchain):\n"
        "    pip install 'intensify[fast]'\n"
        "\n"
        f"Original ImportError: {e}"
    ) from e


# Re-export the compiled submodules for downstream use. Each will be
# populated with #[pyfunction]/#[pyclass] entries in subsequent phases.
kernels = _ext.kernels
likelihood = _ext.likelihood
simulation = _ext.simulation
diagnostics = _ext.diagnostics


def has_rust_kernel(name: str) -> bool:
    """Phase 1+: returns True if the Rust kernel exists. Phase 0 stub."""
    return hasattr(kernels, name)


def has_rust_likelihood(name: str) -> bool:
    """Phase 1+: returns True if the Rust likelihood function exists. Phase 0 stub."""
    return hasattr(likelihood, name)


__all__ = [
    "_ext",
    "diagnostics",
    "has_rust_kernel",
    "has_rust_likelihood",
    "kernels",
    "likelihood",
    "simulation",
]
