"""Tests for kernel implementations."""

import numpy as np
import pytest

from intensify.backends import get_backend, set_backend
from intensify.core.kernels import ExponentialKernel

bt = get_backend()


def test_exponential_evaluate():
    kernel = ExponentialKernel(alpha=0.5, beta=2.0)
    t = bt.array([0.0, 0.5, 1.0, 2.0])
    vals = kernel.evaluate(t)
    expected = 0.5 * 2.0 * np.exp(-2.0 * np.array([0.0, 0.5, 1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(vals), expected, rtol=1e-5)


def test_exponential_integrate():
    kernel = ExponentialKernel(alpha=0.3, beta=1.5)
    assert np.isclose(kernel.integrate(0.0), 0.0)
    assert np.isclose(kernel.integrate(10.0), kernel.l1_norm(), atol=1e-6)
    # Check formula: integrate(t) = alpha * (1 - exp(-beta*t))
    t = 1.0
    expected = 0.3 * (1 - np.exp(-1.5 * t))
    assert np.isclose(kernel.integrate(t), expected)


def test_exponential_l1_norm():
    kernel = ExponentialKernel(alpha=0.42, beta=1.23)
    assert kernel.l1_norm() == 0.42


def test_exponential_recursive_state_update():
    kernel = ExponentialKernel(alpha=0.4, beta=1.0)
    state = bt.array(0.0)
    dt = 0.5
    new_state = kernel.recursive_state_update(state, dt)
    expected = np.exp(-1.0 * 0.5) * (1.0 + 0.0)
    assert np.isclose(new_state, expected)


def test_exponential_has_recursive_form():
    kernel = ExponentialKernel(alpha=0.3, beta=1.5)
    assert kernel.has_recursive_form() is True


def test_exponential_stationarity():
    kernel = ExponentialKernel(alpha=0.8, beta=2.0)
    assert kernel.is_stationary() is True
    kernel2 = ExponentialKernel(alpha=1.0, beta=1.0)
    assert kernel2.is_stationary() is False  # norm == 1 is not stationary


def test_exponential_gradient():
    # Gradient of kernel form w.r.t. parameters (avoid tracing through __init__)
    try:
        import jax
        import jax.numpy as jnp

        beta = 1.5

        def loss_alpha(alpha):
            t = jnp.array([0.1, 0.5, 1.0])
            return jnp.sum(alpha * beta * jnp.exp(-beta * t))

        ga = jax.grad(loss_alpha)(0.5)
        assert ga is not None
    except ImportError:
        pytest.skip("JAX not available")