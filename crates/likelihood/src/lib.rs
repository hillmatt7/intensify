//! Log-likelihood evaluators with closed-form gradients.
//!
//! Phase 0: stub. Phase 1 ships `uni_exp` + `mv_exp_recursive`. Phase 2
//! adds `mv_exp_dense` + `general`. Phase 3 adds `marked` + `nonlinear`.

#[cfg(feature = "python")]
pub mod python;
