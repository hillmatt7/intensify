//! Univariate exponential Hawkes negative log-likelihood with closed-form
//! gradient (Ozaki 1979).
//!
//! Model: λ(t) = μ + Σ_{j: t_j < t} α·β·exp(-β·(t - t_j))
//!
//! Recursive sufficient statistic (pre-event):
//!   R_0 = 0
//!   R_{i+1} = exp(-β·dt_{i+1})·(R_i + 1),  where dt_i = t_i - t_{i-1}
//!   so λ(t_i) = μ + α·β·R_i.
//!
//! Closed-form gradient propagates ∂R/∂β through the same recursion.
//! Compensator gradients are explicit sums over events.

/// Negative log-likelihood and gradient of the univariate exp Hawkes
/// process at parameters `(mu, alpha, beta)` over events `times` on
/// [0, t_horizon]. Times must be sorted and lie in [0, t_horizon].
///
/// Returns `(neg_loglik, [d/dmu, d/dalpha, d/dbeta])`.
///
/// O(N) time, O(1) auxiliary state. Hot path; do not allocate.
#[inline]
pub fn uni_exp_neg_ll_with_grad(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
) -> (f64, [f64; 3]) {
    let n = times.len();
    if n == 0 {
        // L = -μ·T; ∇L = [-T, 0, 0]; neg = [T, 0, 0]
        return (mu * t_horizon, [t_horizon, 0.0, 0.0]);
    }

    // Post-absorb state R̃ and its derivative w.r.t. β.
    let mut r_post = 0.0_f64;
    let mut dr_post_db = 0.0_f64;
    let mut t_prev = 0.0_f64;

    // Sums for the log-intensity term and its gradients.
    let mut log_lam_sum = 0.0_f64;
    let mut grad_mu_log = 0.0_f64; // Σ 1/λ_i
    let mut grad_alpha_log = 0.0_f64; // Σ β·R_i / λ_i
    let mut grad_beta_log = 0.0_f64; // Σ α·(R_i + β·∂R_i/∂β) / λ_i

    for &t in times {
        let dt = t - t_prev;
        let e = (-beta * dt).exp();
        let r_pre = e * r_post;
        // ∂r_pre/∂β = -dt·e·r_post + e·∂r_post/∂β = -dt·r_pre + e·dr_post_db
        let dr_pre_db = -dt * r_pre + e * dr_post_db;

        let lam = mu + alpha * beta * r_pre;
        // Floor matches the JAX path's `jnp.maximum(lam, 1e-30)` to
        // keep logs finite for catastrophically bad init parameters.
        let lam_safe = lam.max(1e-30);
        log_lam_sum += lam_safe.ln();
        let inv_lam = 1.0 / lam_safe;
        grad_mu_log += inv_lam;
        grad_alpha_log += beta * r_pre * inv_lam;
        grad_beta_log += alpha * (r_pre + beta * dr_pre_db) * inv_lam;

        // Absorb the event into the running state (and its derivative).
        r_post = r_pre + 1.0;
        dr_post_db = dr_pre_db;
        t_prev = t;
    }

    // Compensator: μ·T + α·Σ_i (1 - exp(-β·(T - t_i)))
    // ∂C/∂α = Σ_i (1 - exp(-β·(T - t_i)))
    // ∂C/∂β = α·Σ_i (T - t_i)·exp(-β·(T - t_i))
    let mut comp_alpha_term = 0.0_f64;
    let mut comp_beta_grad_term = 0.0_f64;
    for &t in times {
        let tail = t_horizon - t;
        let e = (-beta * tail).exp();
        comp_alpha_term += 1.0 - e;
        comp_beta_grad_term += tail * e;
    }

    let comp = mu * t_horizon + alpha * comp_alpha_term;
    let neg_ll = comp - log_lam_sum;
    let grad_mu = t_horizon - grad_mu_log;
    let grad_alpha = comp_alpha_term - grad_alpha_log;
    let grad_beta = alpha * comp_beta_grad_term - grad_beta_log;

    (neg_ll, [grad_mu, grad_alpha, grad_beta])
}

/// Value-only entry point for std-error computation. Same math, no
/// gradient bookkeeping (slightly cheaper inner loop).
#[inline]
pub fn uni_exp_neg_ll(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
) -> f64 {
    let n = times.len();
    if n == 0 {
        return mu * t_horizon;
    }

    let mut r_post = 0.0_f64;
    let mut t_prev = 0.0_f64;
    let mut log_lam_sum = 0.0_f64;

    for &t in times {
        let dt = t - t_prev;
        let r_pre = (-beta * dt).exp() * r_post;
        let lam = (mu + alpha * beta * r_pre).max(1e-30);
        log_lam_sum += lam.ln();
        r_post = r_pre + 1.0;
        t_prev = t;
    }

    let mut comp_alpha_term = 0.0_f64;
    for &t in times {
        comp_alpha_term += 1.0 - (-beta * (t_horizon - t)).exp();
    }
    let comp = mu * t_horizon + alpha * comp_alpha_term;
    comp - log_lam_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Numerical-gradient sanity check: Rust's analytic gradient should
    /// match a 5-point stencil to ~1e-6.
    fn numerical_grad(
        times: &[f64],
        t: f64,
        params: [f64; 3],
        h: f64,
    ) -> [f64; 3] {
        let mut g = [0.0; 3];
        for i in 0..3 {
            let mut p_plus = params;
            let mut p_minus = params;
            let mut p_plus2 = params;
            let mut p_minus2 = params;
            p_plus[i] += h;
            p_minus[i] -= h;
            p_plus2[i] += 2.0 * h;
            p_minus2[i] -= 2.0 * h;
            let f_p = uni_exp_neg_ll(times, t, p_plus[0], p_plus[1], p_plus[2]);
            let f_m = uni_exp_neg_ll(times, t, p_minus[0], p_minus[1], p_minus[2]);
            let f_p2 = uni_exp_neg_ll(times, t, p_plus2[0], p_plus2[1], p_plus2[2]);
            let f_m2 = uni_exp_neg_ll(times, t, p_minus2[0], p_minus2[1], p_minus2[2]);
            g[i] = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
        }
        g
    }

    #[test]
    fn empty_events_returns_compensator_only() {
        let (val, grad) = uni_exp_neg_ll_with_grad(&[], 10.0, 0.5, 0.3, 1.5);
        assert_relative_eq!(val, 5.0, max_relative = 1e-15); // μ·T
        assert_relative_eq!(grad[0], 10.0, max_relative = 1e-15); // ∂/∂μ = T
        assert_eq!(grad[1], 0.0);
        assert_eq!(grad[2], 0.0);
    }

    #[test]
    fn analytic_grad_matches_numerical_simple() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let t_horizon = 6.0;
        let params = [0.4, 0.3, 1.2];
        let (_v, g_analytic) = uni_exp_neg_ll_with_grad(
            &times, t_horizon, params[0], params[1], params[2],
        );
        let g_numerical = numerical_grad(&times, t_horizon, params, 1e-5);
        for i in 0..3 {
            assert_relative_eq!(g_analytic[i], g_numerical[i], max_relative = 1e-6);
        }
    }

    #[test]
    fn analytic_grad_matches_numerical_near_stationarity() {
        // alpha → 1 (high branching ratio); gradient surface is steep.
        let times = [0.1, 0.3, 0.7, 1.4, 2.1, 3.0, 4.0];
        let t_horizon = 5.0;
        let params = [0.2, 0.95, 1.5];
        let (_v, g_analytic) = uni_exp_neg_ll_with_grad(
            &times, t_horizon, params[0], params[1], params[2],
        );
        let g_numerical = numerical_grad(&times, t_horizon, params, 1e-6);
        for i in 0..3 {
            assert_relative_eq!(g_analytic[i], g_numerical[i], max_relative = 1e-5);
        }
    }

    #[test]
    fn analytic_grad_matches_numerical_low_decay() {
        // beta → 0 (slow decay, persistent excitation).
        let times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let t_horizon = 4.0;
        let params = [0.2, 0.3, 0.05];
        let (_v, g_analytic) = uni_exp_neg_ll_with_grad(
            &times, t_horizon, params[0], params[1], params[2],
        );
        let g_numerical = numerical_grad(&times, t_horizon, params, 1e-5);
        for i in 0..3 {
            assert_relative_eq!(g_analytic[i], g_numerical[i], max_relative = 1e-5);
        }
    }

    #[test]
    fn neg_ll_decomposes_correctly() {
        // neg_L = compensator - log_intensity_sum. Verify by reconstructing
        // both halves independently.
        let times = [0.5, 1.1, 2.0, 3.7];
        let t_horizon = 5.0;
        let mu = 0.4;
        let alpha = 0.3;
        let beta = 1.2;

        let (val, _) = uni_exp_neg_ll_with_grad(&times, t_horizon, mu, alpha, beta);

        // Reconstruct compensator by hand.
        let comp: f64 = mu * t_horizon
            + alpha
                * times
                    .iter()
                    .map(|&t| 1.0 - (-beta * (t_horizon - t)).exp())
                    .sum::<f64>();

        // Reconstruct log-intensity sum by brute force (O(N²)).
        let log_lam: f64 = (0..times.len())
            .map(|i| {
                let r: f64 = (0..i)
                    .map(|j| (-beta * (times[i] - times[j])).exp())
                    .sum();
                (mu + alpha * beta * r).ln()
            })
            .sum();

        assert_relative_eq!(val, comp - log_lam, max_relative = 1e-12);
    }

    #[test]
    fn value_and_value_grad_agree() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let t_horizon = 6.0;
        let mu = 0.4;
        let alpha = 0.3;
        let beta = 1.2;
        let (v1, _) = uni_exp_neg_ll_with_grad(&times, t_horizon, mu, alpha, beta);
        let v2 = uni_exp_neg_ll(&times, t_horizon, mu, alpha, beta);
        assert_relative_eq!(v1, v2, max_relative = 1e-15);
    }
}
