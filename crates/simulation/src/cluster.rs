//! Branching (Galton-Watson / cluster) simulation for Hawkes processes
//! with exponential kernels.
//!
//! Algorithm:
//!   1. Immigrants arrive as Poisson(μ·T) at uniform times in `[0, T]`.
//!   2. Each event (immigrant or offspring) generates Poisson(L1) offspring,
//!      where L1 is the kernel's L1 norm (= α for ExponentialKernel).
//!   3. Offspring times are parent + Exponential(β) delays.
//!   4. Drop offspring outside `[0, T]`. Sort.
//!
//! For multivariate: offspring from a parent in dim j go to dim k via the
//! kernel matrix entry (k, j) — count Poisson(α_{k,j}), delays Exp(β_{k,j}).

use intensify_core::{IntensifyError, Result};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp, Poisson};

/// Univariate branching simulation for ExponentialKernel.
pub fn simulate_uni_exp_branching(
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
    seed: u64,
) -> Result<Vec<f64>> {
    if t_horizon <= 0.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "t_horizon must be positive; got {t_horizon}"
        )));
    }
    if mu < 0.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "mu must be non-negative; got {mu}"
        )));
    }
    if alpha <= 0.0 || beta <= 0.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "alpha and beta must be positive; got alpha={alpha}, beta={beta}"
        )));
    }
    if alpha >= 1.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "alpha (= L1 norm) must be < 1 for the branching process to be subcritical; got {alpha}"
        )));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut events: Vec<f64> = Vec::with_capacity(64);

    // Immigrants
    let n_imm = Poisson::new(mu * t_horizon)
        .map_err(|e| IntensifyError::InvalidParam(format!("Poisson(μT): {e}")))?
        .sample(&mut rng) as usize;
    let mut queue: Vec<f64> = (0..n_imm).map(|_| rng.r#gen::<f64>() * t_horizon).collect();

    // Process the queue of all events; each spawns offspring lazily.
    let pois_alpha = Poisson::new(alpha)
        .map_err(|e| IntensifyError::InvalidParam(format!("Poisson(α): {e}")))?;
    let exp_beta = Exp::new(beta)
        .map_err(|e| IntensifyError::InvalidParam(format!("Exp(β): {e}")))?;

    while let Some(t_parent) = queue.pop() {
        events.push(t_parent);
        let n_off = pois_alpha.sample(&mut rng) as usize;
        for _ in 0..n_off {
            let delay: f64 = exp_beta.sample(&mut rng);
            let t_child = t_parent + delay;
            if t_child <= t_horizon {
                queue.push(t_child);
            }
        }
    }

    events.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(events)
}

/// Multivariate branching simulation for ExponentialKernel matrix.
///
/// `mu` length M; `alpha` flat row-major M×M (α_{i,j} = influence of source j
/// on target i); `beta` shared across all cells.
pub fn simulate_mv_exp_branching(
    t_horizon: f64,
    mu: &[f64],
    alpha: &[f64],
    beta: f64,
    seed: u64,
) -> Result<Vec<Vec<f64>>> {
    if t_horizon <= 0.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "t_horizon must be positive; got {t_horizon}"
        )));
    }
    let m = mu.len();
    if m == 0 {
        return Err(IntensifyError::InvalidParam("mu is empty".into()));
    }
    if alpha.len() != m * m {
        return Err(IntensifyError::InvalidParam(format!(
            "alpha length ({}) must be M·M ({})",
            alpha.len(),
            m * m
        )));
    }
    if beta <= 0.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "beta must be positive; got {beta}"
        )));
    }
    // Spectral radius of |α| should be < 1 for subcriticality, but we don't
    // enforce — caller is responsible. Return a large event count gracefully
    // if the process explodes.

    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_events: Vec<Vec<f64>> = (0..m).map(|_| Vec::with_capacity(64)).collect();
    let mut queue: Vec<(f64, usize)> = Vec::with_capacity(64); // (time, dim)

    // Immigrants per dim
    for (dim, &mu_d) in mu.iter().enumerate() {
        if mu_d <= 0.0 {
            continue;
        }
        let n_imm = Poisson::new(mu_d * t_horizon)
            .map_err(|e| IntensifyError::InvalidParam(format!("Poisson(μ_{dim}·T): {e}")))?
            .sample(&mut rng) as usize;
        for _ in 0..n_imm {
            let t = rng.r#gen::<f64>() * t_horizon;
            queue.push((t, dim));
        }
    }

    let exp_beta = Exp::new(beta)
        .map_err(|e| IntensifyError::InvalidParam(format!("Exp(β): {e}")))?;

    // Process queue
    while let Some((t_parent, dim_parent)) = queue.pop() {
        all_events[dim_parent].push(t_parent);
        // Offspring in each dim k come from kernel α_{k, dim_parent}
        for dim_child in 0..m {
            let a = alpha[dim_child * m + dim_parent];
            if a <= 0.0 {
                continue;
            }
            // Sample offspring count Poisson(α_{k, dim_parent}). For small α
            // (typical < 1) this is usually 0 or 1.
            let pois = Poisson::new(a).map_err(|e| {
                IntensifyError::InvalidParam(format!("Poisson(α_{dim_child}{dim_parent}): {e}"))
            })?;
            let n_off = pois.sample(&mut rng) as usize;
            for _ in 0..n_off {
                let delay: f64 = exp_beta.sample(&mut rng);
                let t_child = t_parent + delay;
                if t_child <= t_horizon {
                    queue.push((t_child, dim_child));
                }
            }
        }
    }

    for v in all_events.iter_mut() {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    Ok(all_events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uni_branching_seedable() {
        let ev1 = simulate_uni_exp_branching(20.0, 0.3, 0.4, 1.5, 42).unwrap();
        let ev2 = simulate_uni_exp_branching(20.0, 0.3, 0.4, 1.5, 42).unwrap();
        assert_eq!(ev1, ev2);
        let ev3 = simulate_uni_exp_branching(20.0, 0.3, 0.4, 1.5, 43).unwrap();
        assert_ne!(ev1, ev3);
        for w in ev1.windows(2) {
            assert!(w[0] <= w[1]);
        }
        for &t in &ev1 {
            assert!((0.0..=20.0).contains(&t));
        }
    }

    #[test]
    fn uni_branching_mean_count_matches_theory() {
        // Mean total event count = μ·T / (1 - α)
        let mu = 0.5; let alpha = 0.4; let beta = 1.0; let T = 100.0;
        let mut counts = Vec::new();
        for seed in 0..50_u64 {
            counts.push(
                simulate_uni_exp_branching(T, mu, alpha, beta, seed).unwrap().len() as f64,
            );
        }
        let mean: f64 = counts.iter().sum::<f64>() / counts.len() as f64;
        let theory = mu * T / (1.0 - alpha);
        assert!(
            (mean - theory).abs() < 0.3 * theory,
            "mean count {mean} too far from theoretical {theory}"
        );
    }

    #[test]
    fn rejects_supercritical_alpha() {
        // alpha >= 1 would cause an infinite expected event count
        assert!(simulate_uni_exp_branching(10.0, 0.5, 1.0, 1.0, 0).is_err());
        assert!(simulate_uni_exp_branching(10.0, 0.5, 1.5, 1.0, 0).is_err());
    }

    #[test]
    fn mv_branching_basic() {
        let mu = vec![0.3, 0.2];
        let alpha = vec![0.2, 0.05, 0.05, 0.25]; // row-major
        let ev = simulate_mv_exp_branching(20.0, &mu, &alpha, 1.5, 42).unwrap();
        assert_eq!(ev.len(), 2);
        for d in &ev {
            for w in d.windows(2) {
                assert!(w[0] <= w[1]);
            }
            for &t in d {
                assert!((0.0..=20.0).contains(&t));
            }
        }
    }
}
