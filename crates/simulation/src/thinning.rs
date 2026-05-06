//! Ogata's thinning algorithm for Hawkes processes with exponential kernels.
//!
//! Univariate:
//!   λ(t) = μ + Σ_{j: t_j < t} α·β·exp(-β·(t - t_j))
//!
//! Multivariate (shared β):
//!   λ_m(t) = μ_m + Σ_k Σ_{j: src_j=k, t_j<t} α_{m,k}·β·exp(-β·(t - t_j))
//!
//! Algorithm (per Ogata 1981):
//!   1. Maintain an upper bound `lambda_max` on the intensity.
//!   2. Propose t_new = t + Exp(1/lambda_max).
//!   3. Compute the current intensity λ(t_new).
//!   4. If λ(t_new) > lambda_max, bump the bound and continue.
//!   5. Otherwise accept with probability λ(t_new) / lambda_max; on
//!      acceptance, append t_new and update recursive state.

use intensify_core::{IntensifyError, Result};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Exp};

/// Univariate ExponentialKernel Hawkes via Ogata thinning.
///
/// Returns sorted event timestamps in `[0, t_horizon]`. Maintains the
/// recursive state R after each accepted event so per-proposal work is
/// O(1) (a decay + scalar arithmetic).
pub fn simulate_uni_exp_hawkes(
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

    let mut rng = StdRng::seed_from_u64(seed);
    let mut events: Vec<f64> = Vec::with_capacity(64);

    // R is the post-absorb running sum at time `t_last_state`. Decay to time
    // t via exp(-β·(t - t_last_state)) when reading.
    let mut r_post = 0.0_f64;
    let mut t_last_state = 0.0_f64;

    let mut lambda_max = mu * 1.5 + 1e-6;
    let mut t = 0.0_f64;

    loop {
        // Sample inter-arrival from Exp(lambda_max).
        let exp_dist = Exp::new(lambda_max)
            .map_err(|e| IntensifyError::InvalidParam(format!("exponential distribution: {e}")))?;
        let dt: f64 = exp_dist.sample(&mut rng);
        t += dt;
        if t >= t_horizon {
            break;
        }

        // Current intensity = μ + α·β·R_at_t where R_at_t = exp(-β·(t - t_last_state))·r_post.
        let r_at_t = (-beta * (t - t_last_state)).exp() * r_post;
        let current_lambda = mu + alpha * beta * r_at_t;

        if current_lambda > lambda_max {
            // Bump and try again from the same t (homogeneous-Poisson upper-bound trick;
            // matches the existing Python `lambda_max = current_lambda * 1.5`).
            lambda_max = current_lambda * 1.5;
            continue;
        }

        let u: f64 = rng.random();
        if u <= current_lambda / lambda_max {
            events.push(t);
            // Absorb: R(t)+1; advance state anchor to `t`.
            r_post = r_at_t + 1.0;
            t_last_state = t;
        }
    }

    Ok(events)
}

/// Multivariate ExponentialKernel Hawkes (shared β) via Ogata thinning.
///
/// `mu` has length M (baselines per dim). `alpha` is a flat row-major
/// (M·M) matrix where `alpha[i·M + j]` = influence of source j on
/// target i. `beta` is the shared decay across all cells.
///
/// Returns `M` per-dimension event vectors, each sorted ascending.
pub fn simulate_mv_exp_hawkes(
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
    for (i, &b) in mu.iter().enumerate() {
        if b < 0.0 {
            return Err(IntensifyError::InvalidParam(format!(
                "mu[{i}] must be non-negative; got {b}"
            )));
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut histories: Vec<Vec<f64>> = (0..m).map(|_| Vec::with_capacity(64)).collect();

    // Per-source running state: r_post[k] = post-absorb sum of decayed
    // contributions from past events of source k, at time `t_last_state`.
    // λ_m(t) = μ_m + β·Σ_k α_{m,k}·exp(-β·(t - t_last_state))·r_post[k]
    let mut r_post = vec![0.0_f64; m];
    let mut t_last_state = 0.0_f64;

    let mu_sum: f64 = mu.iter().sum();
    let mut lambda_max = mu_sum * 1.5 + 1e-6;
    let mut t = 0.0_f64;

    let mut lam_vec = vec![0.0_f64; m];

    loop {
        let exp_dist = Exp::new(lambda_max)
            .map_err(|e| IntensifyError::InvalidParam(format!("exponential distribution: {e}")))?;
        let dt: f64 = exp_dist.sample(&mut rng);
        t += dt;
        if t >= t_horizon {
            break;
        }

        // Decay factor and per-source decayed states.
        let decay = (-beta * (t - t_last_state)).exp();
        // Per-target intensity: λ_m = μ_m + Σ_k α_{m,k}·β·decay·r_post[k]
        let mut total_lambda = 0.0_f64;
        for tgt in 0..m {
            let mut excit = 0.0_f64;
            let row = tgt * m;
            for src in 0..m {
                excit += alpha[row + src] * decay * r_post[src];
            }
            let lam_m = mu[tgt] + beta * excit;
            lam_vec[tgt] = lam_m;
            total_lambda += lam_m;
        }

        if total_lambda > lambda_max {
            lambda_max = total_lambda * 1.5;
            continue;
        }

        let u: f64 = rng.random();
        if u > total_lambda / lambda_max {
            continue;
        }

        // Sample which dim by inverse-CDF on lam_vec normalized.
        let u2: f64 = rng.random();
        let mut cum = 0.0_f64;
        let mut chosen = m - 1;
        for tgt in 0..m {
            cum += lam_vec[tgt] / total_lambda;
            if u2 <= cum {
                chosen = tgt;
                break;
            }
        }

        histories[chosen].push(t);
        // Absorb: per-source state at time `t` is decayed-then-incremented.
        // We carry forward the "post-absorb" representation by first
        // decaying every component to `t` (new anchor), then bumping the
        // chosen dim by 1.
        for k in 0..m {
            r_post[k] *= decay;
        }
        r_post[chosen] += 1.0;
        t_last_state = t;
    }

    Ok(histories)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uni_exp_simulator_runs_and_is_seedable() {
        let ev1 = simulate_uni_exp_hawkes(10.0, 0.5, 0.3, 1.5, 42).unwrap();
        let ev2 = simulate_uni_exp_hawkes(10.0, 0.5, 0.3, 1.5, 42).unwrap();
        assert_eq!(ev1, ev2, "same seed must give identical events");
        let ev3 = simulate_uni_exp_hawkes(10.0, 0.5, 0.3, 1.5, 43).unwrap();
        assert_ne!(ev1, ev3, "different seeds must give different events");

        // All times in [0, T] and sorted
        for w in ev1.windows(2) {
            assert!(w[0] <= w[1]);
        }
        for &t in &ev1 {
            assert!(t >= 0.0 && t < 10.0);
        }
    }

    #[test]
    fn uni_exp_baseline_only_matches_poisson_mean() {
        // With α=ε, the process is essentially a homogeneous Poisson at rate μ.
        // Mean event count over 100 seeds, T=200, μ=0.5 should be ≈ 100.
        let mut counts = Vec::new();
        for seed in 0..100u64 {
            let ev = simulate_uni_exp_hawkes(200.0, 0.5, 1e-6, 1.5, seed).unwrap();
            counts.push(ev.len() as f64);
        }
        let mean: f64 = counts.iter().sum::<f64>() / counts.len() as f64;
        // Expected = μ·T = 100; allow 3σ tolerance (σ = √100 = 10, so 3σ ≈ 30).
        assert!(
            (mean - 100.0).abs() < 30.0,
            "Poisson-limit mean {mean} too far from 100"
        );
    }

    #[test]
    fn uni_exp_self_excitation_increases_count() {
        // Compare event counts for self-exciting vs Poisson-like; self-exciting should be higher.
        let mut self_count = 0_usize;
        let mut poisson_count = 0_usize;
        for seed in 0..30u64 {
            let ev_se = simulate_uni_exp_hawkes(50.0, 0.3, 0.5, 1.0, seed).unwrap();
            let ev_p = simulate_uni_exp_hawkes(50.0, 0.3, 1e-6, 1.0, seed).unwrap();
            self_count += ev_se.len();
            poisson_count += ev_p.len();
        }
        // Self-exciting should produce ~2× as many events on average for these params
        // (branching ratio 0.5 → expected ~1/(1-0.5) = 2× the baseline rate).
        assert!(
            self_count > poisson_count,
            "self-exciting count {} should exceed Poisson count {}",
            self_count,
            poisson_count,
        );
    }

    #[test]
    fn mv_exp_simulator_basic() {
        // 2D, identity-coupling-ish.
        let mu = vec![0.4, 0.3];
        let alpha = vec![0.2, 0.05, 0.05, 0.25]; // row-major
        let beta = 1.5;

        let ev = simulate_mv_exp_hawkes(20.0, &mu, &alpha, beta, 42).unwrap();
        assert_eq!(ev.len(), 2);
        for dim_events in &ev {
            for w in dim_events.windows(2) {
                assert!(w[0] <= w[1], "events must be sorted within each dim");
            }
            for &t in dim_events {
                assert!(t >= 0.0 && t < 20.0);
            }
        }
    }

    #[test]
    fn mv_exp_seedable() {
        let mu = vec![0.3, 0.3];
        let alpha = vec![0.1, 0.05, 0.05, 0.1];
        let ev1 = simulate_mv_exp_hawkes(10.0, &mu, &alpha, 1.5, 42).unwrap();
        let ev2 = simulate_mv_exp_hawkes(10.0, &mu, &alpha, 1.5, 42).unwrap();
        assert_eq!(ev1, ev2);
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(simulate_uni_exp_hawkes(-1.0, 0.5, 0.3, 1.5, 0).is_err());
        assert!(simulate_uni_exp_hawkes(10.0, -0.1, 0.3, 1.5, 0).is_err());
        assert!(simulate_uni_exp_hawkes(10.0, 0.5, 0.0, 1.5, 0).is_err());
        assert!(simulate_uni_exp_hawkes(10.0, 0.5, 0.3, 0.0, 1.5 as u64).is_err());

        // mv: mu length wrong
        let bad_alpha = vec![0.1; 4];
        assert!(simulate_mv_exp_hawkes(10.0, &[0.3, 0.3, 0.3], &bad_alpha, 1.5, 0).is_err());
    }
}
