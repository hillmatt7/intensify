//! Marked univariate exponential Hawkes negative log-likelihood with
//! closed-form gradient.
//!
//! Model:
//!   λ(t) = μ + Σ_{j: t_j < t} g(m_j)·α·β·exp(-β·(t - t_j))
//!
//! where g: ℝ → ℝ is the mark-influence function. Three built-in kinds:
//!   * linear:  g(m) = m
//!   * log:     g(m) = log(1 + m)
//!   * power:   g(m) = max(0, m)^p   for fixed exponent `mark_power`
//!
//! Free parameters during MLE are (μ, α, β); mark_power is fixed.
//! Recursive form (ExponentialKernel only):
//!   R_{i+1}^pre = exp(-β·(t_{i+1} - t_i))·(R_i^pre·_implicit_ + g(m_i))
//! Concretely: decay then absorb-with-weight, just like uni_exp but the
//! +1 absorb becomes +g(m).
//!
//! Compensator: μ·T + α·Σ_j g(m_j)·(1 - exp(-β·(T - t_j)))

use intensify_core::{IntensifyError, Result};

/// Mark influence kind. Matches the Python MarkedHawkes `mark_influence`
/// argument.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarkInfluence {
    Linear,
    Log,
    Power(f64),
}

impl MarkInfluence {
    #[inline]
    fn apply(&self, m: f64) -> f64 {
        match self {
            Self::Linear => m,
            Self::Log => (1.0 + m).ln(),
            Self::Power(p) => m.max(0.0).powf(*p),
        }
    }
}

/// Negative log-likelihood and gradient for marked univariate exp Hawkes
/// at parameters (μ, α, β). `times` and `marks` must have the same length;
/// times sorted on `[0, t_horizon]`.
///
/// Returns `(neg_loglik, [d/dμ, d/dα, d/dβ])`.
pub fn marked_uni_exp_neg_ll_with_grad(
    times: &[f64],
    marks: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
    influence: MarkInfluence,
) -> Result<(f64, [f64; 3])> {
    let n = times.len();
    if marks.len() != n {
        return Err(IntensifyError::InvalidParam(format!(
            "times.len() ({}) != marks.len() ({})",
            n,
            marks.len()
        )));
    }
    if n == 0 {
        return Ok((mu * t_horizon, [t_horizon, 0.0, 0.0]));
    }

    // R_post = (mark-weighted) running sum of decayed past events.
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

        // Absorb with mark weight.
        let g_m = influence.apply(marks[i]);
        r_post = r_pre + g_m;
        dr_post_db = dr_pre_db; // d/dβ of (r_pre + g_m) = d/dβ of r_pre  (g_m has no β dependence)
        t_prev = t_i;
    }

    // Compensator: μ·T + α·Σ_j g(m_j)·(1 - exp(-β·(T - t_j)))
    let mut comp_alpha_term = 0.0_f64;       // Σ g_j·(1 - e_j)
    let mut comp_beta_grad_term = 0.0_f64;   // Σ g_j·(T-t_j)·e_j
    for j in 0..n {
        let g_j = influence.apply(marks[j]);
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
    marks: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
    influence: MarkInfluence,
) -> Result<f64> {
    let n = times.len();
    if marks.len() != n {
        return Err(IntensifyError::InvalidParam(format!(
            "times.len() ({}) != marks.len() ({})",
            n,
            marks.len()
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
        r_post = r_pre + influence.apply(marks[i]);
        t_prev = t_i;
    }

    let mut comp_alpha_term = 0.0_f64;
    for j in 0..n {
        comp_alpha_term += influence.apply(marks[j])
            * (1.0 - (-beta * (t_horizon - times[j])).exp());
    }
    Ok(mu * t_horizon + alpha * comp_alpha_term - log_lam_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn brute_force(
        times: &[f64],
        marks: &[f64],
        t_horizon: f64,
        mu: f64,
        alpha: f64,
        beta: f64,
        influence: MarkInfluence,
    ) -> f64 {
        let mut log_lam = 0.0_f64;
        for i in 0..times.len() {
            let mut excit = 0.0_f64;
            for j in 0..i {
                excit += influence.apply(marks[j])
                    * alpha * beta * (-beta * (times[i] - times[j])).exp();
            }
            log_lam += (mu + excit).max(1e-30).ln();
        }
        let mut comp = mu * t_horizon;
        for j in 0..times.len() {
            comp += influence.apply(marks[j])
                * alpha * (1.0 - (-beta * (t_horizon - times[j])).exp());
        }
        comp - log_lam
    }

    #[test]
    fn empty_events() {
        let (val, grad) = marked_uni_exp_neg_ll_with_grad(
            &[], &[], 10.0, 0.5, 0.3, 1.5, MarkInfluence::Linear,
        ).unwrap();
        assert_relative_eq!(val, 5.0, max_relative = 1e-15);
        assert_eq!(grad, [10.0, 0.0, 0.0]);
    }

    #[test]
    fn matches_brute_force_linear() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let marks = [1.0, 2.0, 0.5, 1.5, 0.8];
        let (val, _) = marked_uni_exp_neg_ll_with_grad(
            &times, &marks, 6.0, 0.4, 0.3, 1.2, MarkInfluence::Linear,
        ).unwrap();
        let bf = brute_force(&times, &marks, 6.0, 0.4, 0.3, 1.2, MarkInfluence::Linear);
        assert_relative_eq!(val, bf, max_relative = 1e-12);
    }

    #[test]
    fn matches_brute_force_log_and_power() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let marks = [1.0, 2.0, 0.5, 1.5, 0.8];
        for influence in [
            MarkInfluence::Log,
            MarkInfluence::Power(0.5),
            MarkInfluence::Power(2.0),
        ] {
            let (val, _) = marked_uni_exp_neg_ll_with_grad(
                &times, &marks, 6.0, 0.4, 0.3, 1.2, influence,
            ).unwrap();
            let bf = brute_force(&times, &marks, 6.0, 0.4, 0.3, 1.2, influence);
            assert_relative_eq!(val, bf, max_relative = 1e-12, epsilon = 1e-15);
        }
    }

    #[test]
    fn analytic_grad_matches_finite_difference() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let marks = [1.0, 2.0, 0.5, 1.5, 0.8];
        let params = [0.4, 0.3, 1.2];

        let (_, g_a) = marked_uni_exp_neg_ll_with_grad(
            &times, &marks, 6.0, params[0], params[1], params[2], MarkInfluence::Linear,
        ).unwrap();

        let h = 1e-6;
        for i in 0..3 {
            let bump = |delta: f64| -> f64 {
                let mut p = params;
                p[i] += delta;
                marked_uni_exp_neg_ll(&times, &marks, 6.0, p[0], p[1], p[2], MarkInfluence::Linear).unwrap()
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
