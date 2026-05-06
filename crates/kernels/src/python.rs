//! PyO3 bindings for `intensify-kernels`.
//!
//! Exposes `ExponentialKernel` as a Python class. Mirrors the Python
//! API defined in `python/intensify/core/kernels/exponential.py`:
//! `evaluate`, `integrate`, `integrate_vec`, `l1_norm`, `scale`,
//! plus the recursive-form hooks (`recursive_state_update`, `recursive_decay`,
//! `recursive_absorb`, `recursive_intensity_excitation`).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::exponential::ExponentialKernel;
use crate::nonparametric::NonparametricKernel;
use crate::power_law::PowerLawKernel;

#[pymethods]
impl ExponentialKernel {
    /// Construct: `ExponentialKernel(alpha, beta, *, allow_signed=False)`.
    #[new]
    #[pyo3(signature = (alpha, beta, *, allow_signed = false))]
    fn py_new(alpha: f64, beta: f64, allow_signed: bool) -> PyResult<Self> {
        Self::new(alpha, beta, allow_signed).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    #[pyo3(name = "alpha")]
    fn py_alpha(&self) -> f64 {
        self.alpha
    }

    #[getter]
    #[pyo3(name = "beta")]
    fn py_beta(&self) -> f64 {
        self.beta
    }

    #[getter]
    #[pyo3(name = "allow_signed")]
    fn py_allow_signed(&self) -> bool {
        self.allow_signed
    }

    /// φ(t) = α·β·exp(-β·t). Accepts a NumPy 1-D array of lags.
    #[pyo3(name = "evaluate")]
    fn py_evaluate<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let t_slice = t.as_slice()?;
        let result: Vec<f64> = t_slice.iter().map(|&ti| self.evaluate(ti)).collect();
        Ok(result.into_pyarray(py))
    }

    /// ∫₀^t φ(τ)dτ = α·(1 - exp(-β·t)). Scalar input.
    #[pyo3(name = "integrate")]
    fn py_integrate(&self, t: f64) -> f64 {
        self.integrate(t)
    }

    /// Vectorized integrate: applies `integrate(t_i)` element-wise.
    #[pyo3(name = "integrate_vec")]
    fn py_integrate_vec<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let t_slice = t.as_slice()?;
        let result: Vec<f64> = t_slice.iter().map(|&ti| self.integrate(ti)).collect();
        Ok(result.into_pyarray(py))
    }

    /// L1 norm = |α|.
    #[pyo3(name = "l1_norm")]
    fn py_l1_norm(&self) -> f64 {
        self.l1_norm()
    }

    /// Multiply α in-place by `factor`.
    #[pyo3(name = "scale")]
    fn py_scale(&mut self, factor: f64) {
        self.scale(factor);
    }

    /// Whether the kernel admits the O(N) recursive likelihood path.
    #[pyo3(name = "has_recursive_form")]
    fn py_has_recursive_form(&self) -> bool {
        self.has_recursive_form()
    }

    /// R_i = exp(-β·dt)·(1 + R_{i-1}).
    #[pyo3(name = "recursive_state_update")]
    fn py_recursive_state_update(&self, state: f64, dt: f64) -> f64 {
        self.recursive_state_update(state, dt)
    }

    /// Decay-only step: R_dec = exp(-β·dt)·R.
    #[pyo3(name = "recursive_decay")]
    fn py_recursive_decay(&self, state: f64, dt: f64) -> f64 {
        self.recursive_decay(state, dt)
    }

    /// Post-event absorb: R⁺ = R + 1.
    #[pyo3(name = "recursive_absorb")]
    fn py_recursive_absorb(&self, state: f64) -> f64 {
        self.recursive_absorb(state)
    }

    /// α·β·R.
    #[pyo3(name = "recursive_intensity_excitation")]
    fn py_recursive_intensity_excitation(&self, state: f64) -> f64 {
        self.recursive_intensity_excitation(state)
    }

    fn __repr__(&self) -> String {
        let sig = if self.allow_signed {
            ", allow_signed=True"
        } else {
            ""
        };
        format!(
            "ExponentialKernel(alpha={}, beta={}{})",
            self.alpha, self.beta, sig
        )
    }
}

// ---------------------------------------------------------------------------
// PowerLawKernel
// ---------------------------------------------------------------------------

#[pymethods]
impl PowerLawKernel {
    /// `PowerLawKernel(alpha, beta, c=1.0)`
    #[new]
    #[pyo3(signature = (alpha, beta, c = 1.0))]
    fn py_new(alpha: f64, beta: f64, c: f64) -> PyResult<Self> {
        Self::new(alpha, beta, c).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    #[pyo3(name = "alpha")]
    fn py_alpha(&self) -> f64 {
        self.alpha
    }

    #[getter]
    #[pyo3(name = "beta")]
    fn py_beta(&self) -> f64 {
        self.beta
    }

    #[getter]
    #[pyo3(name = "c")]
    fn py_c(&self) -> f64 {
        self.c
    }

    /// φ(t) = α·(t + c)^{-(1+β)}.
    #[pyo3(name = "evaluate")]
    fn py_evaluate<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let t_slice = t.as_slice()?;
        let result: Vec<f64> = t_slice.iter().map(|&ti| self.evaluate(ti)).collect();
        Ok(result.into_pyarray(py))
    }

    /// Scalar integrate.
    #[pyo3(name = "integrate")]
    fn py_integrate(&self, t: f64) -> f64 {
        self.integrate(t)
    }

    /// Vectorized integrate.
    #[pyo3(name = "integrate_vec")]
    fn py_integrate_vec<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let t_slice = t.as_slice()?;
        let result: Vec<f64> = t_slice.iter().map(|&ti| self.integrate(ti)).collect();
        Ok(result.into_pyarray(py))
    }

    #[pyo3(name = "l1_norm")]
    fn py_l1_norm(&self) -> f64 {
        self.l1_norm()
    }

    #[pyo3(name = "scale")]
    fn py_scale(&mut self, factor: f64) {
        self.scale(factor);
    }

    #[pyo3(name = "has_recursive_form")]
    fn py_has_recursive_form(&self) -> bool {
        self.has_recursive_form()
    }

    fn __repr__(&self) -> String {
        format!(
            "PowerLawKernel(alpha={}, beta={}, c={})",
            self.alpha, self.beta, self.c
        )
    }
}

// ---------------------------------------------------------------------------
// NonparametricKernel
// ---------------------------------------------------------------------------

#[pymethods]
impl NonparametricKernel {
    /// `NonparametricKernel(edges, values)` where edges has length K+1
    /// (edges[0]=0, strictly increasing) and values has length K (≥0).
    #[new]
    fn py_new(
        edges: PyReadonlyArray1<'_, f64>,
        values: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        Self::new(edges.as_slice()?.to_vec(), values.as_slice()?.to_vec())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    #[pyo3(name = "edges")]
    fn py_edges<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.edges.clone().into_pyarray(py)
    }

    #[getter]
    #[pyo3(name = "values")]
    fn py_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values.clone().into_pyarray(py)
    }

    #[getter]
    #[pyo3(name = "n_bins")]
    fn py_n_bins(&self) -> usize {
        self.n_bins()
    }

    #[pyo3(name = "evaluate")]
    fn py_evaluate<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let t_slice = t.as_slice()?;
        let result: Vec<f64> = t_slice.iter().map(|&ti| self.evaluate(ti)).collect();
        Ok(result.into_pyarray(py))
    }

    #[pyo3(name = "integrate")]
    fn py_integrate(&self, t: f64) -> f64 {
        self.integrate(t)
    }

    #[pyo3(name = "integrate_vec")]
    fn py_integrate_vec<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let t_slice = t.as_slice()?;
        let result: Vec<f64> = t_slice.iter().map(|&ti| self.integrate(ti)).collect();
        Ok(result.into_pyarray(py))
    }

    #[pyo3(name = "l1_norm")]
    fn py_l1_norm(&self) -> f64 {
        self.l1_norm()
    }

    #[pyo3(name = "scale")]
    fn py_scale(&mut self, factor: f64) {
        self.scale(factor);
    }

    #[pyo3(name = "has_recursive_form")]
    fn py_has_recursive_form(&self) -> bool {
        self.has_recursive_form()
    }

    fn __repr__(&self) -> String {
        format!(
            "NonparametricKernel(edges={:?}, values={:?})",
            self.edges, self.values
        )
    }
}

/// Module initializer: registered by the aggregator crate.
#[pymodule]
pub fn kernels(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExponentialKernel>()?;
    m.add_class::<PowerLawKernel>()?;
    m.add_class::<NonparametricKernel>()?;
    Ok(())
}
