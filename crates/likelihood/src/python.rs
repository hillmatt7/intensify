//! PyO3 bindings for `intensify-likelihood`.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::uni_exp::{uni_exp_neg_ll, uni_exp_neg_ll_with_grad};

/// Univariate exp Hawkes neg-log-likelihood + closed-form gradient.
///
/// Returns `(neg_loglik, grad_array)` where `grad_array` has shape (3,)
/// in (μ, α, β) order. `times` must be sorted on `[0, t_horizon]`.
#[pyfunction]
#[pyo3(name = "uni_exp_neg_ll_with_grad")]
fn py_uni_exp_neg_ll_with_grad<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
) -> PyResult<(f64, Bound<'py, numpy::PyArray1<f64>>)> {
    let t = times.as_slice()?;
    let (val, grad) = uni_exp_neg_ll_with_grad(t, t_horizon, mu, alpha, beta);
    Ok((val, numpy::IntoPyArray::into_pyarray(grad.to_vec(), py)))
}

/// Value-only entry point. Cheaper than the value+grad call by a few
/// percent; used by the std-error finite-difference loop.
#[pyfunction]
#[pyo3(name = "uni_exp_neg_ll")]
fn py_uni_exp_neg_ll(
    times: PyReadonlyArray1<'_, f64>,
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
) -> PyResult<f64> {
    let t = times.as_slice()?;
    Ok(uni_exp_neg_ll(t, t_horizon, mu, alpha, beta))
}

#[pymodule]
pub fn likelihood(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_uni_exp_neg_ll_with_grad, m)?)?;
    m.add_function(wrap_pyfunction!(py_uni_exp_neg_ll, m)?)?;
    Ok(())
}
