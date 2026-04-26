//! Log-likelihood evaluators with closed-form gradients.
//!
//! Phase 1 ships `uni_exp` + `mv_exp_recursive`. Phase 2 adds
//! `mv_exp_dense` + `general`. Phase 3 adds `marked` + `nonlinear`.

pub mod marked_uni_exp;
pub mod mv_exp_dense;
pub mod mv_exp_recursive;
pub mod uni_approx_powerlaw;
pub mod uni_exp;
pub mod uni_nonparametric;
pub mod uni_powerlaw;
pub mod uni_sumexp;

pub use marked_uni_exp::{marked_uni_exp_neg_ll, marked_uni_exp_neg_ll_with_grad};
pub use mv_exp_dense::{mv_exp_dense_neg_ll, mv_exp_dense_neg_ll_with_grad};
pub use mv_exp_recursive::MvExpRecursiveLogLik;
pub use uni_approx_powerlaw::{
    uni_approx_powerlaw_neg_ll, uni_approx_powerlaw_neg_ll_with_grad,
};
pub use uni_exp::{uni_exp_neg_ll, uni_exp_neg_ll_with_grad};
pub use uni_nonparametric::{uni_nonparametric_neg_ll, uni_nonparametric_neg_ll_with_grad};
pub use uni_powerlaw::{uni_powerlaw_neg_ll, uni_powerlaw_neg_ll_with_grad};
pub use uni_sumexp::{uni_sumexp_neg_ll, uni_sumexp_neg_ll_with_grad};

#[cfg(feature = "python")]
pub mod python;
