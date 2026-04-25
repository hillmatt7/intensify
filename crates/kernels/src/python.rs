//! PyO3 bindings for `intensify-kernels`. Empty in Phase 0.

use pyo3::prelude::*;

/// Module initializer registered by the aggregator crate.
#[pymodule]
pub fn kernels(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 1: register `ExponentialKernel` here.
    Ok(())
}
