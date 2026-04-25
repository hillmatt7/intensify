"""Flat parameter vector for :class:`MultivariateHawkes` with exponential kernels."""

from __future__ import annotations

import numpy as np

from ..kernels.exponential import ExponentialKernel
from ..processes.hawkes import MultivariateHawkes


def multivariate_hawkes_initial_vector(process: MultivariateHawkes) -> np.ndarray:
    M = process.n_dims
    mu = np.asarray(process.mu, dtype=float).ravel()
    if mu.size != M:
        raise ValueError("mu length mismatch")
    parts: list[float] = list(mu)
    for m in range(M):
        for k in range(M):
            ker = process.kernel_matrix[m][k]
            if not isinstance(ker, ExponentialKernel) or ker.allow_signed:
                raise TypeError(
                    "Multivariate MLE expects ExponentialKernel without allow_signed in each cell"
                )
            parts.extend([float(ker.alpha), float(ker.beta)])
    return np.array(parts, dtype=float)


def multivariate_hawkes_bounds(process: MultivariateHawkes) -> list[tuple[float | None, float | None]]:
    M = process.n_dims
    b: list[tuple[float | None, float | None]] = [(1e-8, None)] * M
    for _ in range(M * M):
        b.append((1e-8, 0.999))  # alpha
        b.append((1e-8, None))  # beta
    return b


def multivariate_hawkes_param_names(process: MultivariateHawkes) -> list[str]:
    M = process.n_dims
    names = [f"mu_{m}" for m in range(M)]
    for m in range(M):
        for k in range(M):
            names.append(f"alpha_{m}_{k}")
            names.append(f"beta_{m}_{k}")
    return names


def multivariate_hawkes_apply_vector(process: MultivariateHawkes, x: np.ndarray) -> None:
    M = process.n_dims
    x = np.asarray(x, dtype=float).ravel()
    need = M + 2 * M * M
    if x.size != need:
        raise ValueError(f"expected {need} params, got {x.size}")
    process.mu = np.array(x[0:M], dtype=float)
    kernels: list[list[ExponentialKernel]] = []
    idx = M
    for _m in range(M):
        row: list[ExponentialKernel] = []
        for _k in range(M):
            a = float(x[idx])
            b = float(x[idx + 1])
            idx += 2
            row.append(ExponentialKernel(alpha=a, beta=b))
        kernels.append(row)
    process.kernel_matrix = kernels


def multivariate_hawkes_extract_alphas(x: np.ndarray, M: int) -> np.ndarray:
    """Alpha block shape (M, M) from flat vector."""
    x = np.asarray(x, dtype=float).ravel()
    alphas = np.zeros((M, M))
    idx = M
    for m in range(M):
        for k in range(M):
            alphas[m, k] = float(x[idx])
            idx += 2
    return alphas
