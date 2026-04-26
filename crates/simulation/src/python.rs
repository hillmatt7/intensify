//! PyO3 bindings for `intensify-simulation`.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::cluster::{simulate_mv_exp_branching, simulate_uni_exp_branching};
use crate::thinning::{simulate_mv_exp_hawkes, simulate_uni_exp_hawkes};

/// Univariate ExponentialKernel Hawkes via Ogata thinning.
/// Returns a 1-D float64 NumPy array of sorted event times.
#[pyfunction]
#[pyo3(name = "simulate_uni_exp_hawkes")]
fn py_simulate_uni_exp_hawkes<'py>(
    py: Python<'py>,
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let events = simulate_uni_exp_hawkes(t_horizon, mu, alpha, beta, seed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(events.into_pyarray(py))
}

/// Multivariate ExponentialKernel Hawkes (shared β) via Ogata thinning.
/// Returns a Python list of M 1-D float64 NumPy arrays.
#[pyfunction]
#[pyo3(name = "simulate_mv_exp_hawkes")]
fn py_simulate_mv_exp_hawkes<'py>(
    py: Python<'py>,
    t_horizon: f64,
    mu: PyReadonlyArray1<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
    beta: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyList>> {
    let histories = simulate_mv_exp_hawkes(
        t_horizon,
        mu.as_slice()?,
        alpha.as_slice()?,
        beta,
        seed,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let list = PyList::empty(py);
    for events in histories {
        list.append(events.into_pyarray(py))?;
    }
    Ok(list)
}

/// Univariate ExponentialKernel Hawkes via branching (Galton-Watson).
/// Faster than thinning for near-critical α; same statistical distribution.
#[pyfunction]
#[pyo3(name = "simulate_uni_exp_branching")]
fn py_simulate_uni_exp_branching<'py>(
    py: Python<'py>,
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let events = simulate_uni_exp_branching(t_horizon, mu, alpha, beta, seed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(events.into_pyarray(py))
}

/// Multivariate ExponentialKernel Hawkes (shared β) via branching.
#[pyfunction]
#[pyo3(name = "simulate_mv_exp_branching")]
fn py_simulate_mv_exp_branching<'py>(
    py: Python<'py>,
    t_horizon: f64,
    mu: PyReadonlyArray1<'py, f64>,
    alpha: PyReadonlyArray1<'py, f64>,
    beta: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyList>> {
    let histories = simulate_mv_exp_branching(
        t_horizon, mu.as_slice()?, alpha.as_slice()?, beta, seed,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let list = PyList::empty(py);
    for events in histories {
        list.append(events.into_pyarray(py))?;
    }
    Ok(list)
}

#[pymodule]
pub fn simulation(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_simulate_uni_exp_hawkes, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_mv_exp_hawkes, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_uni_exp_branching, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_mv_exp_branching, m)?)?;
    Ok(())
}
