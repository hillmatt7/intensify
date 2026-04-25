//! Core types for intensify: errors and shared parameter layouts.
//!
//! This crate intentionally has no math; it exists so that `kernels`,
//! `likelihood`, `simulation`, and `diagnostics` can share types without
//! depending on each other.

use thiserror::Error;

/// Errors raised by intensify Rust crates. Mapped to Python exceptions
/// in the `pyo3` aggregator crate.
#[derive(Debug, Error)]
pub enum IntensifyError {
    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    #[error("non-monotone events: index {index} ({prev} > {curr})")]
    NonMonotoneEvents {
        index: usize,
        prev: f64,
        curr: f64,
    },

    #[error("event {value} outside [0, {horizon}]")]
    EventOutOfHorizon { value: f64, horizon: f64 },

    #[error("kernel {kernel} does not support {op}")]
    Unsupported {
        kernel: &'static str,
        op: &'static str,
    },
}

pub type Result<T> = std::result::Result<T, IntensifyError>;
