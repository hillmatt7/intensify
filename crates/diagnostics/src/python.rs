//! PyO3 bindings for `intensify-diagnostics`.

use pyo3::prelude::*;

#[pymodule]
pub fn diagnostics(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
