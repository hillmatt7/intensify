//! PyO3 bindings for `intensify-likelihood`. Empty in Phase 0.

use pyo3::prelude::*;

#[pymodule]
pub fn likelihood(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 1: register `uni_exp_neg_ll_with_grad` and
    // `mv_exp_recursive_neg_ll_with_grad` here.
    Ok(())
}
