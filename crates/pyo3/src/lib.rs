//! Aggregator crate. Produces a single Python extension `intensify._libintensify`
//! that re-exports submodules from each domain crate.
//!
//! Modeled on `nautilus_trader/crates/pyo3/src/lib.rs`: each domain crate
//! defines a `#[pymodule]` initializer in `src/python.rs` (or `src/python/mod.rs`),
//! and this crate `wrap_pymodule!`s them and registers them under
//! `intensify._libintensify.<domain>` in `sys.modules`.

use pyo3::prelude::*;

const MODULE_NAME: &str = "intensify._libintensify";

#[pymodule]
fn _libintensify(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sys = PyModule::import(py, "sys")?;
    let modules = sys.getattr("modules")?;

    // kernels
    let n = "kernels";
    let submodule = pyo3::wrap_pymodule!(intensify_kernels::python::kernels);
    m.add_wrapped(submodule)?;
    modules.set_item(format!("{MODULE_NAME}.{n}"), m.getattr(n)?)?;

    // likelihood
    let n = "likelihood";
    let submodule = pyo3::wrap_pymodule!(intensify_likelihood::python::likelihood);
    m.add_wrapped(submodule)?;
    modules.set_item(format!("{MODULE_NAME}.{n}"), m.getattr(n)?)?;

    // simulation
    let n = "simulation";
    let submodule = pyo3::wrap_pymodule!(intensify_simulation::python::simulation);
    m.add_wrapped(submodule)?;
    modules.set_item(format!("{MODULE_NAME}.{n}"), m.getattr(n)?)?;

    // diagnostics
    let n = "diagnostics";
    let submodule = pyo3::wrap_pymodule!(intensify_diagnostics::python::diagnostics);
    m.add_wrapped(submodule)?;
    modules.set_item(format!("{MODULE_NAME}.{n}"), m.getattr(n)?)?;

    Ok(())
}
