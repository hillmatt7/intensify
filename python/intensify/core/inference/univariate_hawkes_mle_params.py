"""MLE parameter vector packing for UnivariateHawkes + concrete kernels."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..kernels import (
    ApproxPowerLawKernel,
    ExponentialKernel,
    Kernel,
    NonparametricKernel,
    PowerLawKernel,
    SumExponentialKernel,
)


class _MuKernelProcess(Protocol):
    """Process carrying Hawkes baseline and kernel (univariate or marked)."""

    mu: float
    kernel: Kernel


def hawkes_mle_initial_vector(process: _MuKernelProcess) -> np.ndarray:
    """Initial optimization vector for SciPy / pattern matching."""
    mu = float(process.mu)
    k = process.kernel
    if isinstance(k, ExponentialKernel):
        return np.array([mu, k.alpha, k.beta], dtype=float)
    if isinstance(k, SumExponentialKernel):
        return np.array([mu, *k.alphas, *k.betas], dtype=float)
    if isinstance(k, PowerLawKernel):
        return np.array([mu, k.alpha, k.beta, k.c], dtype=float)
    if isinstance(k, ApproxPowerLawKernel):
        return np.array([mu, k.alpha, k.beta_pow, k.beta_min], dtype=float)
    if isinstance(k, NonparametricKernel):
        return np.array([mu, *k.values], dtype=float)
    raise TypeError(
        f"MLE vectorization not implemented for kernel type {type(k).__name__}"
    )


def hawkes_mle_bounds(process: _MuKernelProcess) -> list[tuple[float | None, float | None]]:
    """Box constraints for L-BFGS-B."""
    k = process.kernel
    if isinstance(k, ExponentialKernel):
        if getattr(k, "allow_signed", False):
            return [(1e-8, None), (-0.999, 0.999), (1e-8, None)]
        return [(1e-8, None), (1e-8, 0.999), (1e-8, None)]
    if isinstance(k, SumExponentialKernel):
        K = k.n_components
        b = [(1e-8, None)]  # mu
        for _ in range(K):
            b.append((1e-8, 0.999))  # each alpha; stationarity enforced in apply
        for _ in range(K):
            b.append((1e-8, None))  # beta
        return b
    if isinstance(k, PowerLawKernel):
        return [(1e-8, None), (1e-8, None), (1e-8, None), (1e-6, None)]
    if isinstance(k, ApproxPowerLawKernel):
        return [(1e-8, None), (1e-8, 0.999), (1e-8, None), (1e-8, None)]
    if isinstance(k, NonparametricKernel):
        m = [(1e-8, None)]
        for _ in range(k.n_bins):
            m.append((0.0, None))
        return m
    raise TypeError(
        f"MLE bounds not implemented for kernel type {type(k).__name__}"
    )


def hawkes_mle_param_names(process: _MuKernelProcess) -> list[str]:
    k = process.kernel
    if isinstance(k, ExponentialKernel):
        return ["mu", "alpha", "beta"]
    if isinstance(k, SumExponentialKernel):
        K = k.n_components
        return ["mu"] + [f"alpha_{i}" for i in range(K)] + [f"beta_{i}" for i in range(K)]
    if isinstance(k, PowerLawKernel):
        return ["mu", "alpha", "beta", "c"]
    if isinstance(k, ApproxPowerLawKernel):
        return ["mu", "alpha", "beta_pow", "beta_min"]
    if isinstance(k, NonparametricKernel):
        return ["mu"] + [f"value_{i}" for i in range(k.n_bins)]
    raise TypeError(
        f"MLE param names not implemented for kernel type {type(k).__name__}"
    )


def hawkes_mle_apply_vector(process: _MuKernelProcess, x: np.ndarray) -> None:
    """Update process in-place from flat vector x (NumPy)."""
    x = np.asarray(x, dtype=float).ravel()
    k = process.kernel
    if isinstance(k, ExponentialKernel):
        mu, alpha, beta = x
        signed = bool(getattr(k, "allow_signed", False))
        a_f = float(alpha)
        if signed and -1e-6 < a_f < 1e-6:
            a_f = 1e-6
        process.mu = float(mu)
        process.kernel = ExponentialKernel(
            alpha=a_f, beta=float(beta), allow_signed=signed,
        )
        return
    if isinstance(k, SumExponentialKernel):
        K = k.n_components
        mu = float(x[0])
        alphas = [float(x[1 + i]) for i in range(K)]
        betas = [float(x[1 + K + i]) for i in range(K)]
        s = sum(alphas)
        if s >= 0.999:
            scale = 0.998 / s
            alphas = [a * scale for a in alphas]
        process.mu = mu
        process.kernel = SumExponentialKernel(alphas=alphas, betas=betas)
        return
    if isinstance(k, PowerLawKernel):
        mu, alpha, beta, c = x
        process.mu = float(mu)
        process.kernel = PowerLawKernel(
            alpha=float(alpha), beta=float(beta), c=float(c)
        )
        return
    if isinstance(k, ApproxPowerLawKernel):
        mu, alpha, beta_pow, beta_min = x
        process.mu = float(mu)
        process.kernel = ApproxPowerLawKernel(
            alpha=float(alpha),
            beta_pow=float(beta_pow),
            beta_min=float(beta_min),
            r=k.r,
            n_components=k.n_components,
        )
        # keep L1 = alpha <= 1
        if process.kernel.alpha >= 0.999:
            process.kernel.alpha = 0.998
        return
    if isinstance(k, NonparametricKernel):
        mu = float(x[0])
        vals = [float(x[1 + i]) for i in range(k.n_bins)]
        width_sum = sum(
            max(0.0, v) * (k.edges[i + 1] - k.edges[i]) for i, v in enumerate(vals)
        )
        if width_sum >= 0.999:
            scale = 0.998 / width_sum if width_sum > 0 else 1.0
            vals = [max(0.0, v * scale) for v in vals]
        process.mu = mu
        process.kernel = NonparametricKernel(edges=list(k.edges), values=vals)
        return
    raise TypeError(
        f"MLE apply not implemented for kernel type {type(k).__name__}"
    )
