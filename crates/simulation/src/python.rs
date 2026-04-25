//! PyO3 bindings for `intensify-simulation`. Empty in Phase 0.

use pyo3::prelude::*;

#[pymodule]
pub fn simulation(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
