//! Kernel evaluators for point processes.
//!
//! Phase 1 ships `ExponentialKernel` (the only kernel needed for uni_exp
//! and mv_exp_recursive likelihoods). Phase 3 adds power-law,
//! sum-exponential, nonparametric, and approx-power-law variants.

pub mod exponential;

pub use exponential::ExponentialKernel;

#[cfg(feature = "python")]
pub mod python;
