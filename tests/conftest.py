"""Pytest configuration."""

import matplotlib
import pytest

matplotlib.use("Agg")  # Non-interactive backend for tests


@pytest.fixture(autouse=True)
def _reset_intensify_backend():
    """Avoid backend leakage between tests (e.g. after test_backends sets numpy)."""
    from intensify.backends import set_backend

    try:
        set_backend("jax")
    except Exception:
        set_backend("numpy")
    yield
    try:
        set_backend("jax")
    except Exception:
        set_backend("numpy")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", 'slow: marks tests as slow (deselect with \'-m "not slow"\')'
    )