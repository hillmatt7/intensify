"""Pytest configuration."""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for tests

# tests/_reference/ contains the frozen JAX reference oracles + legacy tests
# preserved for the cross-validation suite. The legacy `test_*_legacy.py`
# files import APIs deleted in 0.3.0 (the JAX backend, the JIT cache); skip
# them at collection.
collect_ignore_glob = ["_reference/test_*_legacy.py"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", 'slow: marks tests as slow (deselect with \'-m "not slow"\')'
    )