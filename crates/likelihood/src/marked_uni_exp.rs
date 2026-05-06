//! Marked univariate exponential Hawkes negative log-likelihood with
//! closed-form gradient.
//!
//! Model:
//!   λ(t) = μ + Σ_{j: t_j < t} g(m_j)·α·β·exp(-β·(t - t_j))
//!
//! `g` is the mark-influence function applied to each event's mark. Since
//! `g(m_j)` is **constant w.r.t. the optimization parameters** (μ, α, β),
//! the Python caller pre-computes the vector `g_values = [g(m_0), g(m_1),
//! ...]` once before the optimizer starts and passes it directly. This
//! way builtin influence kinds (linear/log/power) and user-supplied
//! callables are all handled uniformly by the same Rust hot loop — no
//! per-pair Python callback overhead, no enum dispatch.
//!
//! Recursive form (ExponentialKernel only):
//!   R_post = R_pre + g(m_i)
//!   R_pre  = exp(-β·dt)·R_post_prev   (decay then absorb-with-weight)
//!
//! Compensator: μ·T + α·Σ_j g(m_j)·(1 - exp(-β·(T - t_j)))

use intensify_core::{IntensifyError, Result};

/// Negative log-likelihood and gradient for marked univariate exp Hawkes
/// at parameters (μ, α, β). `times` and `g_values` (precomputed mark
/// weights) must have the same length; times sorted on `[0, t_horizon]`.
///
/// Returns `(neg_loglik, [d/dμ, d/dα, d/dβ])`.
pub fn marked_uni_exp_neg_ll_with_grad(
    times: &[f64],
    g_values: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
) -> Result<(f64, [f64; 3])> {
    let n = times.len();
    if g_values.len() != n {
        return Err(IntensifyError::InvalidParam(format!(
            "times.len() ({}) != g_values.len() ({})",
            n,
            g_values.len()
        )));
    }
    if n == 0 {
        return Ok((mu * t_horizon, [t_horizon, 0.0, 0.0]));
    }

    let mut r_post = 0.0_f64;
    let mut dr_post_db = 0.0_f64;
    let mut t_prev = 0.0_f64;

    let mut log_lam_sum = 0.0_f64;
    let mut grad_mu_log = 0.0_f64;
    let mut grad_alpha_log = 0.0_f64;
    let mut grad_beta_log = 0.0_f64;

    for i in 0..n {
        let t_i = times[i];
        let dt = t_i - t_prev;
        let e = (-beta * dt).exp();
        let r_pre = e * r_post;
        let dr_pre_db = -dt * r_pre + e * dr_post_db;

        let lam = mu + alpha * beta * r_pre;
        let lam_safe = lam.max(1e-30);
        log_lam_sum += lam_safe.ln();
        let inv_lam = 1.0 / lam_safe;
        grad_mu_log += inv_lam;
        grad_alpha_log += beta * r_pre * inv_lam;
        grad_beta_log += alpha * (r_pre + beta * dr_pre_db) * inv_lam;

        // Absorb with precomputed mark weight (g_values[i] is constant w.r.t. params).
        r_post = r_pre + g_values[i];
        dr_post_db = dr_pre_db;
        t_prev = t_i;
    }

    // Compensator: μ·T + α·Σ_j g_j·(1 - exp(-β·(T - t_j)))
    let mut comp_alpha_term = 0.0_f64;
    let mut comp_beta_grad_term = 0.0_f64;
    for j in 0..n {
        let g_j = g_values[j];
        let tail = t_horizon - times[j];
        let e = (-beta * tail).exp();
        comp_alpha_term += g_j * (1.0 - e);
        comp_beta_grad_term += g_j * tail * e;
    }

    let comp = mu * t_horizon + alpha * comp_alpha_term;
    let neg_ll = comp - log_lam_sum;

    let grad_mu = t_horizon - grad_mu_log;
    let grad_alpha = comp_alpha_term - grad_alpha_log;
    let grad_beta = alpha * comp_beta_grad_term - grad_beta_log;

    Ok((neg_ll, [grad_mu, grad_alpha, grad_beta]))
}

/// Value-only entry. ~30% cheaper than value+grad.
pub fn marked_uni_exp_neg_ll(
    times: &[f64],
    g_values: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
) -> Result<f64> {
    let n = times.len();
    if g_values.len() != n {
        return Err(IntensifyError::InvalidParam(format!(
            "times.len() ({}) != g_values.len() ({})",
            n,
            g_values.len()
        )));
    }
    if n == 0 {
        return Ok(mu * t_horizon);
    }

    let mut r_post = 0.0_f64;
    let mut t_prev = 0.0_f64;
    let mut log_lam_sum = 0.0_f64;

    for i in 0..n {
        let t_i = times[i];
        let r_pre = (-beta * (t_i - t_prev)).exp() * r_post;
        let lam = (mu + alpha * beta * r_pre).max(1e-30);
        log_lam_sum += lam.ln();
        r_post = r_pre + g_values[i];
        t_prev = t_i;
    }

    let mut comp_alpha_term = 0.0_f64;
    for j in 0..n {
        comp_alpha_term += g_values[j] * (1.0 - (-beta * (t_horizon - times[j])).exp());
    }
    Ok(mu * t_horizon + alpha * comp_alpha_term - log_lam_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn brute_force(
        times: &[f64],
        g_values: &[f64],
        t_horizon: f64,
        mu: f64,
        alpha: f64,
        beta: f64,
    ) -> f64 {
        let mut log_lam = 0.0_f64;
        for i in 0..times.len() {
            let mut excit = 0.0_f64;
            for j in 0..i {
                excit += g_values[j] * alpha * beta * (-beta * (times[i] - times[j])).exp();
            }
            log_lam += (mu + excit).max(1e-30).ln();
        }
        let mut comp = mu * t_horizon;
        for j in 0..times.len() {
            comp += g_values[j] * alpha * (1.0 - (-beta * (t_horizon - times[j])).exp());
        }
        comp - log_lam
    }

    #[test]
    fn empty_events() {
        let (val, grad) = marked_uni_exp_neg_ll_with_grad(&[], &[], 10.0, 0.5, 0.3, 1.5).unwrap();
        assert_relative_eq!(val, 5.0, max_relative = 1e-15);
        assert_eq!(grad, [10.0, 0.0, 0.0]);
    }

    #[test]
    fn matches_brute_force_linear_marks() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let g_values = [1.0, 2.0, 0.5, 1.5, 0.8]; // linear: g(m)=m
        let (val, _) =
            marked_uni_exp_neg_ll_with_grad(&times, &g_values, 6.0, 0.4, 0.3, 1.2).unwrap();
        let bf = brute_force(&times, &g_values, 6.0, 0.4, 0.3, 1.2);
        assert_relative_eq!(val, bf, max_relative = 1e-12);
    }

    #[test]
    fn matches_brute_force_arbitrary_g() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        // Caller may have applied any function (log, power, custom callable, etc.)
        let g_log: Vec<f64> = [1.0_f64, 2.0, 0.5, 1.5, 0.8]
            .iter()
            .map(|&m| (1.0 + m).ln())
            .collect();
        let g_pow: Vec<f64> = [1.0_f64, 2.0, 0.5, 1.5, 0.8]
            .iter()
            .map(|&m| m.powf(0.5))
            .collect();
        let g_custom = vec![0.7, 1.4, 0.2, 0.9, 1.1]; // arbitrary user-supplied
        for g in [g_log, g_pow, g_custom] {
            let (val, _) = marked_uni_exp_neg_ll_with_grad(&times, &g, 6.0, 0.4, 0.3, 1.2).unwrap();
            let bf = brute_force(&times, &g, 6.0, 0.4, 0.3, 1.2);
            assert_relative_eq!(val, bf, max_relative = 1e-12, epsilon = 1e-15);
        }
    }

    #[test]
    fn analytic_grad_matches_finite_difference() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let g_values = [1.0, 2.0, 0.5, 1.5, 0.8];
        let params = [0.4, 0.3, 1.2];

        let (_, g_a) = marked_uni_exp_neg_ll_with_grad(
            &times, &g_values, 6.0, params[0], params[1], params[2],
        )
        .unwrap();

        let h = 1e-6;
        for i in 0..3 {
            let bump = |delta: f64| -> f64 {
                let mut p = params;
                p[i] += delta;
                marked_uni_exp_neg_ll(&times, &g_values, 6.0, p[0], p[1], p[2]).unwrap()
            };
            let f_p = bump(h);
            let f_m = bump(-h);
            let f_p2 = bump(2.0 * h);
            let f_m2 = bump(-2.0 * h);
            let g_n = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
            assert_relative_eq!(g_a[i], g_n, max_relative = 1e-6);
        }
    }
}
