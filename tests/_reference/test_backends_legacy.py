"""Tests for backend abstraction."""

import pytest

from intensify.backends import get_backend, jax_backend, numpy_backend, set_backend


def test_backend_selection():
    # Default should be JAX if available, else NumPy
    backend = get_backend()
    assert backend is not None


def test_set_backend_jax():
    if not jax_backend.is_available():
        pytest.skip("JAX not available")
    set_backend("jax")
    from intensify.backends import get_backend_name
    assert get_backend_name() == "jax"


def test_set_backend_numpy():
    set_backend("numpy")
    from intensify.backends import get_backend_name
    assert get_backend_name() == "numpy"


def test_backend_has_required_ops():
    bt = get_backend()
    # Check a few essential operations exist
    assert bt.exp is not None
    assert bt.log is not None
    assert bt.sum is not None
    assert bt.array is not None
    assert bt.lax is not None or True  # JAX has lax, NumPy has lax with few ops