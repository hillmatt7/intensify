//! PyO3 bindings for `intensify-likelihood`.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyValueError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::mv_exp_dense::{mv_exp_dense_neg_ll, mv_exp_dense_neg_ll_with_grad};
use crate::mv_exp_recursive::MvExpRecursiveLogLik;
use crate::uni_exp::{uni_exp_neg_ll, uni_exp_neg_ll_with_grad};

// ---------------------------------------------------------------------------
// Univariate exp Hawkes
// ---------------------------------------------------------------------------

/// Univariate exp Hawkes neg-log-likelihood + closed-form gradient.
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
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let t = times.as_slice()?;
    let (val, grad) = uni_exp_neg_ll_with_grad(t, t_horizon, mu, alpha, beta);
    Ok((val, grad.to_vec().into_pyarray(py)))
}

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

// ---------------------------------------------------------------------------
// Multivariate exp Hawkes (β fixed)
// ---------------------------------------------------------------------------

/// Multivariate exp Hawkes negative log-likelihood model with a fixed
/// (shared) decay β. Mirrors tick's `ModelHawkesExpKernLogLik` but
/// returns the unnormalized loss (no division by `n_total_jumps`).
///
/// Construct once with `(timestamps, end_time, decay)`; weights are
/// precomputed eagerly. Then call `loss_and_grad(coeffs, out)` repeatedly
/// inside the L-BFGS-B loop. β is fixed; only μ and α are fit.
///
/// Coefficient layout: `[μ_0..μ_{M-1}, α_{0,:}, α_{1,:}, ..., α_{M-1,:}]`,
/// length `M + M·M`.
#[pyclass(
    module = "intensify._libintensify.likelihood",
    name = "MvExpRecursiveLogLik"
)]
pub struct PyMvExpRecursiveLogLik {
    inner: MvExpRecursiveLogLik,
}

#[pymethods]
impl PyMvExpRecursiveLogLik {
    /// Construct from a list of per-dim event arrays.
    #[new]
    #[pyo3(signature = (timestamps, end_time, decay))]
    fn py_new(
        timestamps: &Bound<'_, PyList>,
        end_time: f64,
        decay: f64,
    ) -> PyResult<Self> {
        let mut ts = Vec::with_capacity(timestamps.len());
        for item in timestamps.iter() {
            // Accept any 1-D float64 NumPy array (or list-like).
            let arr = item
                .extract::<PyReadonlyArray1<f64>>()
                .map_err(|_| PyTypeError::new_err(
                    "each element of `timestamps` must be a 1-D float64 NumPy array",
                ))?;
            if !arr.as_array().is_standard_layout() {
                return Err(PyValueError::new_err(
                    "timestamps arrays must be C-contiguous (use np.ascontiguousarray)",
                ));
            }
            ts.push(arr.as_slice()?.to_vec());
        }
        let inner = MvExpRecursiveLogLik::new(ts, end_time, decay)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn n_dims(&self) -> usize {
        self.inner.n_dims()
    }

    #[getter]
    fn end_time(&self) -> f64 {
        self.inner.end_time()
    }

    #[getter]
    fn decay(&self) -> f64 {
        self.inner.decay()
    }

    #[getter]
    fn n_total_jumps(&self) -> usize {
        self.inner.n_total_jumps()
    }

    #[getter]
    fn n_coeffs(&self) -> usize {
        self.inner.n_coeffs()
    }

    /// Compute neg-log-likelihood and gradient. `out` is mutated in place.
    /// Returns the loss as a Python float.
    #[pyo3(name = "loss_and_grad")]
    fn py_loss_and_grad(
        &self,
        coeffs: PyReadonlyArray1<'_, f64>,
        out: &Bound<'_, PyArray1<f64>>,
    ) -> PyResult<f64> {
        let coeffs_slice = coeffs.as_slice()?;
        // SAFETY: caller is responsible for not aliasing `out` with `coeffs`.
        // PyArray1::as_slice_mut is unsafe; we obtain a mutable slice via
        // `readwrite()` which guards against aliasing at the Python level.
        let mut out_rw = out.readwrite();
        let out_slice = out_rw.as_slice_mut()?;
        self.inner
            .loss_and_grad(coeffs_slice, out_slice)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Value-only entry point (~5–10% cheaper than `loss_and_grad`).
    #[pyo3(name = "loss")]
    fn py_loss(&self, coeffs: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let coeffs_slice = coeffs.as_slice()?;
        self.inner
            .loss(coeffs_slice)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "MvExpRecursiveLogLik(n_dims={}, end_time={}, decay={}, n_total_jumps={})",
            self.inner.n_dims(),
            self.inner.end_time(),
            self.inner.decay(),
            self.inner.n_total_jumps(),
        )
    }
}

// ---------------------------------------------------------------------------
// Multivariate exp Hawkes — joint-decay (β fitted per cell)
// ---------------------------------------------------------------------------

/// Multivariate exp Hawkes neg-log-likelihood with **per-cell β**
/// (joint-decay mode). Closed-form analytic gradient.
///
/// Returns `(neg_loglik, grad_mu, grad_alpha, grad_beta)` where:
/// - `grad_mu` shape `(M,)`
/// - `grad_alpha`, `grad_beta` shape `(M·M,)` row-major: index `m·M + k`
///   is the gradient w.r.t. cell (m, k).
#[pyfunction]
#[pyo3(name = "mv_exp_dense_neg_ll_with_grad")]
fn py_mv_exp_dense_neg_ll_with_grad<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    sources: PyReadonlyArray1<'py, i64>,
    end_time: f64,
    n_dims: usize,
    mu: PyReadonlyArray1<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
    beta: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let (val, gm, ga, gb) = mv_exp_dense_neg_ll_with_grad(
        times.as_slice()?,
        sources.as_slice()?,
        end_time,
        n_dims,
        mu.as_slice()?,
        alpha.as_slice()?,
        beta.as_slice()?,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        val,
        gm.into_pyarray(py),
        ga.into_pyarray(py),
        gb.into_pyarray(py),
    ))
}

/// Value-only entry point (~30% faster than the value+grad call).
#[pyfunction]
#[pyo3(name = "mv_exp_dense_neg_ll")]
fn py_mv_exp_dense_neg_ll(
    times: PyReadonlyArray1<'_, f64>,
    sources: PyReadonlyArray1<'_, i64>,
    end_time: f64,
    n_dims: usize,
    mu: PyReadonlyArray1<'_, f64>,
    alpha: PyReadonlyArray1<'_, f64>,
    beta: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    mv_exp_dense_neg_ll(
        times.as_slice()?,
        sources.as_slice()?,
        end_time,
        n_dims,
        mu.as_slice()?,
        alpha.as_slice()?,
        beta.as_slice()?,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
pub fn likelihood(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_uni_exp_neg_ll_with_grad, m)?)?;
    m.add_function(wrap_pyfunction!(py_uni_exp_neg_ll, m)?)?;
    m.add_function(wrap_pyfunction!(py_mv_exp_dense_neg_ll_with_grad, m)?)?;
    m.add_function(wrap_pyfunction!(py_mv_exp_dense_neg_ll, m)?)?;
    m.add_class::<PyMvExpRecursiveLogLik>()?;
    Ok(())
}
