//! Point-process simulators (Ogata thinning + branching/cluster).

pub mod cluster;
pub mod thinning;

pub use cluster::{simulate_mv_exp_branching, simulate_uni_exp_branching};
pub use thinning::{simulate_mv_exp_hawkes, simulate_uni_exp_hawkes};

#[cfg(feature = "python")]
pub mod python;
