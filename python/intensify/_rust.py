"""Dispatch shim for the Rust core (`intensify._libintensify`).

Single source of truth for {Python user-facing API → Rust implementation}.
Modeled on Nautilus Trader's pattern of importing from the compiled
extension at module load time and raising a clear, actionable ImportError
if the extension is missing.

JAX is NOT a fallback. If the user's environment doesn't have the Rust
extension built, this shim raises ImportError pointing at the [fast] extra
or a binary wheel install. The frozen JAX reference oracle in
`tests/_reference/` is dev-only and never imported here.
"""

from __future__ import annotations

import numpy as np

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


# Re-export the compiled submodules.
kernels = _ext.kernels
likelihood = _ext.likelihood
simulation = _ext.simulation
diagnostics = _ext.diagnostics


# -----------------------------------------------------------------------------
# Dispatch predicates: when does the Rust path apply?
# -----------------------------------------------------------------------------


def has_rust_uni_exp_path(process) -> bool:
    """True if the Rust univariate exp Hawkes path applies to `process`.

    The Rust uni_exp likelihood handles only ExponentialKernel without
    `allow_signed`. Other kernels (Nonparametric, signed exp,
    SumExponential) currently fall through to the existing JAX/numpy
    paths until Phase 3 ports them.
    """
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    if not isinstance(process, UnivariateHawkes):
        return False
    if not isinstance(process.kernel, ExponentialKernel):
        return False
    return not process.kernel.allow_signed


def has_rust_uni_powerlaw_path(process) -> bool:
    """True if the Rust univariate power-law Hawkes path applies."""
    from intensify.core.kernels.power_law import PowerLawKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    if not isinstance(process, UnivariateHawkes):
        return False
    return isinstance(process.kernel, PowerLawKernel)


def has_rust_uni_nonparametric_path(process) -> bool:
    """True if the Rust univariate nonparametric Hawkes path applies."""
    from intensify.core.kernels.nonparametric import NonparametricKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    if not isinstance(process, UnivariateHawkes):
        return False
    return isinstance(process.kernel, NonparametricKernel)


def has_rust_uni_sumexp_path(process) -> bool:
    """True if the Rust univariate sum-of-exponentials Hawkes path applies."""
    from intensify.core.kernels.sum_exponential import SumExponentialKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    if not isinstance(process, UnivariateHawkes):
        return False
    return isinstance(process.kernel, SumExponentialKernel)


def has_rust_uni_approx_powerlaw_path(process) -> bool:
    """True if the Rust univariate ApproxPowerLaw Hawkes path applies."""
    from intensify.core.kernels.approx_power_law import ApproxPowerLawKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    if not isinstance(process, UnivariateHawkes):
        return False
    return isinstance(process.kernel, ApproxPowerLawKernel)


def has_rust_marked_exp_path(process) -> bool:
    """True if the Rust marked Hawkes (ExponentialKernel + builtin mark
    influence kind) path applies. Callable mark_influence falls through.
    """
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.marked_hawkes import MarkedHawkes

    if not isinstance(process, MarkedHawkes):
        return False
    if not isinstance(process.kernel, ExponentialKernel) or process.kernel.allow_signed:
        return False
    return process._mark_influence_kind in ("linear", "log", "power")


def mv_shared_beta(process) -> float | None:
    """Return the shared decay β if every cell in the kernel matrix is an
    ExponentialKernel with the same β; else None.
    """
    from intensify.core.kernels.exponential import ExponentialKernel

    betas: list[float] = []
    for row in process.kernel_matrix:
        for k in row:
            if not isinstance(k, ExponentialKernel) or k.allow_signed:
                return None
            betas.append(float(k.beta))
    if not betas:
        return None
    first = betas[0]
    return first if all(abs(b - first) < 1e-12 for b in betas) else None


def has_rust_mv_recursive_path(process, fit_decay: bool) -> bool:
    """True if the Rust mv_exp_recursive (β-fixed) path applies.

    Requires:
      * `process` is MultivariateHawkes
      * every cell of `kernel_matrix` is ExponentialKernel without allow_signed
      * the same β across all cells
      * `fit_decay = False` (recursive path requires β locked)
    """
    from intensify.core.processes.hawkes import MultivariateHawkes

    if not isinstance(process, MultivariateHawkes):
        return False
    if fit_decay:
        return False
    return mv_shared_beta(process) is not None


def has_rust_mv_dense_path(process) -> bool:
    """True if the Rust mv_exp_dense (joint-decay, β fitted per cell) path
    applies. Requires every cell to be `ExponentialKernel` without
    `allow_signed`. β can differ across cells (the dense path fits each
    independently).
    """
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.hawkes import MultivariateHawkes

    if not isinstance(process, MultivariateHawkes):
        return False
    for row in process.kernel_matrix:
        for k in row:
            if not isinstance(k, ExponentialKernel) or k.allow_signed:
                return False
    return True


# -----------------------------------------------------------------------------
# Layout adapters between intensify Python and Rust mv_exp coefficient vectors
# -----------------------------------------------------------------------------
#
# Python (existing): [μ_0..μ_{M-1}, (α, β) interleaved per cell, row-major]
#                    length M + 2·M·M.  β slots are at indices M + 2k + 1.
# Rust:              [μ_0..μ_{M-1}, α row-major]
#                    length M + M·M.  β is fixed (passed to constructor).


def mv_initial_rust_coeffs(process) -> np.ndarray:
    """Pack a MultivariateHawkes process's current parameters into the
    Rust coeff layout `[μ, α row-major]`."""
    M = process.n_dims
    out = np.empty(M + M * M, dtype=np.float64)
    out[:M] = np.asarray(process.mu, dtype=np.float64).ravel()
    for i in range(M):
        for j in range(M):
            out[M + i * M + j] = float(process.kernel_matrix[i][j].alpha)
    return out


def mv_apply_rust_coeffs(process, x: np.ndarray, beta: float) -> None:
    """Write Rust-layout coefficients back into a MultivariateHawkes process,
    preserving the fixed β on every cell."""
    from intensify.core.kernels.exponential import ExponentialKernel

    M = process.n_dims
    process.mu = np.array(x[:M], dtype=np.float64)
    new_kernels: list[list[ExponentialKernel]] = []
    for i in range(M):
        row: list[ExponentialKernel] = []
        for j in range(M):
            alpha = float(x[M + i * M + j])
            row.append(ExponentialKernel(alpha=alpha, beta=beta))
        new_kernels.append(row)
    process.kernel_matrix = new_kernels


__all__ = [
    "_ext",
    "diagnostics",
    "has_rust_marked_exp_path",
    "has_rust_mv_dense_path",
    "has_rust_mv_recursive_path",
    "has_rust_uni_approx_powerlaw_path",
    "has_rust_uni_exp_path",
    "has_rust_uni_nonparametric_path",
    "has_rust_uni_powerlaw_path",
    "has_rust_uni_sumexp_path",
    "kernels",
    "likelihood",
    "mv_apply_rust_coeffs",
    "mv_initial_rust_coeffs",
    "mv_shared_beta",
    "simulation",
]
