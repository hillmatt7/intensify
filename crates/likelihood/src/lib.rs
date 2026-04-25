//! Log-likelihood evaluators with closed-form gradients.
//!
//! Phase 1 ships `uni_exp` + `mv_exp_recursive`. Phase 2 adds
//! `mv_exp_dense` + `general`. Phase 3 adds `marked` + `nonlinear`.

pub mod uni_exp;

pub use uni_exp::{uni_exp_neg_ll, uni_exp_neg_ll_with_grad};

#[cfg(feature = "python")]
pub mod python;
