//! Point-process simulators (Ogata thinning).

pub mod thinning;

pub use thinning::{simulate_mv_exp_hawkes, simulate_uni_exp_hawkes};

#[cfg(feature = "python")]
pub mod python;
