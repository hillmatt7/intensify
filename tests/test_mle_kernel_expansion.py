"""MLE kernel expansion: MarkedHawkes + NonlinearHawkes now accept every
univariate-kernel type, not only ExponentialKernel. These tests confirm the
fits complete without NotImplementedError and record FitResult invariants.

Parameter recovery on short simulated series is noisy; we check structural
properties (finite log-likelihood, non-negative branching ratio, kernel
type preserved) rather than tight numeric bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

import intensify as its


def _mk_events(seed: int = 0, T: float = 10.0, n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = rng.exponential(T / n, size=n)
    return np.cumsum(dt)[np.cumsum(dt) < T]


MARKED_KERNELS = [
    its.ExponentialKernel(alpha=0.2, beta=1.5),
    its.SumExponentialKernel(alphas=[0.15, 0.1], betas=[1.0, 3.0]),
    its.PowerLawKernel(alpha=0.2, beta=1.5, c=0.5),
    its.ApproxPowerLawKernel(alpha=0.2, beta_pow=1.2, beta_min=0.5, r=2.0, n_components=5),
]


@pytest.mark.parametrize("kernel", MARKED_KERNELS, ids=lambda k: type(k).__name__)
def test_marked_hawkes_mle_accepts_all_kernels(kernel):
    events = _mk_events(seed=1)
    marks = np.abs(np.random.default_rng(2).normal(1.0, 0.3, size=len(events)))
    T = float(events[-1]) + 0.1

    model = its.MarkedHawkes(mu=0.3, kernel=kernel)
    result = model.fit(events, marks, T=T)

    assert np.isfinite(result.log_likelihood)
    assert result.branching_ratio_ is not None
    assert result.branching_ratio_ >= 0.0
    assert type(result.process.kernel) is type(kernel)


NONLINEAR_KERNELS = [
    its.ExponentialKernel(alpha=0.25, beta=1.5),
    its.ExponentialKernel(alpha=0.2, beta=1.5, allow_signed=True),
    its.SumExponentialKernel(alphas=[0.15, 0.1], betas=[1.0, 3.0]),
    its.PowerLawKernel(alpha=0.2, beta=1.5, c=0.5),
]


@pytest.mark.parametrize("kernel", NONLINEAR_KERNELS, ids=lambda k: f"{type(k).__name__}_signed{getattr(k, 'allow_signed', False)}")
def test_nonlinear_hawkes_mle_accepts_all_kernels(kernel):
    events = _mk_events(seed=3)
    T = float(events[-1]) + 0.1

    model = its.NonlinearHawkes(mu=0.3, kernel=kernel, link_function="softplus")
    result = model.fit(events, T=T)

    assert np.isfinite(result.log_likelihood)
    assert result.branching_ratio_ is not None
    assert type(result.process.kernel) is type(kernel)
    if getattr(kernel, "allow_signed", False):
        assert result.process.kernel.allow_signed is True


def test_regularization_string_shorthand_resolves():
    from intensify.core.inference.mle import _resolve_regularization
    from intensify.core.regularizers import ElasticNet, L1

    assert _resolve_regularization(None) is None
    assert isinstance(_resolve_regularization("l1"), L1)
    assert isinstance(_resolve_regularization("L1"), L1)
    assert isinstance(_resolve_regularization("elasticnet"), ElasticNet)
    assert isinstance(_resolve_regularization("elastic_net"), ElasticNet)

    # instance passthrough
    custom = L1(strength=0.5)
    assert _resolve_regularization(custom) is custom

    with pytest.raises(ValueError):
        _resolve_regularization("ridge")


def test_backend_proxy_tracks_set_backend():
    """Module-level `bt = get_backend()` caches should update after set_backend()."""
    from intensify.core.kernels import base as base_mod

    current = its.get_backend_name()
    try:
        its.set_backend("numpy")
        assert its.get_backend_name() == "numpy"
        # Functional check: array call through the cached proxy returns a
        # NumPy ndarray (not a JAX DeviceArray) after the switch.
        arr = base_mod.bt.zeros(3)
        assert isinstance(arr, np.ndarray)

        its.set_backend("jax")
        assert its.get_backend_name() == "jax"
        arr_jax = base_mod.bt.zeros(3)
        assert not isinstance(arr_jax, np.ndarray)
    finally:
        its.set_backend(current)
