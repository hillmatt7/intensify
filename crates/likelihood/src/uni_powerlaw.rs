//! Univariate power-law Hawkes negative log-likelihood with closed-form
//! gradient.
//!
//! Model: λ(t) = μ + Σ_{j: t_j < t} α·(t - t_j + c)^{-(1+β)}.
//!
//! No recursive form — pairs (i, j) with j < i contribute independently
//! to λ(t_i). Computation is O(N²).
//!
//! ### Closed-form gradients
//!
//! Per-event log-intensity term log λ_i = log(μ + Σ_{j<i} α·s_{ij}^{-(1+β)})
//! where s_{ij} = t_i - t_j + c. Derivatives:
//!   ∂log λ_i/∂μ = 1/λ_i
//!   ∂log λ_i/∂α = (Σ_j s_{ij}^{-(1+β)}) / λ_i
//!   ∂log λ_i/∂β = -α·(Σ_j s_{ij}^{-(1+β)}·log s_{ij}) / λ_i
//!   ∂log λ_i/∂c = -(1+β)·α·(Σ_j s_{ij}^{-(2+β)}) / λ_i
//!
//! Compensator C = μ·T + (α/β)·Σ_j [c^{-β} − (T - t_j + c)^{-β}]. Let
//! T_j = T - t_j + c. Then:
//!   ∂C/∂μ = T
//!   ∂C/∂α = (1/β)·Σ_j (c^{-β} − T_j^{-β})
//!   ∂C/∂β = Σ_j ∂/∂β [(α/β)·(c^{-β} − T_j^{-β})]
//!         = -(α/β²)·(c^{-β} − T_j^{-β}) +
//!           (α/β)·(-c^{-β}·log c + T_j^{-β}·log T_j)   summed over j
//!   ∂C/∂c = α·Σ_j (T_j^{-β-1} − c^{-β-1})

use intensify_core::Result;

/// Negative log-likelihood and gradient at parameters `(mu, alpha, beta, c)`
/// for univariate power-law Hawkes over events `times` on `[0, t_horizon]`.
///
/// Returns `(neg_loglik, [d/dμ, d/dα, d/dβ, d/dc])`.
///
/// O(N²) time, O(1) auxiliary state. Hot path; do not allocate.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn uni_powerlaw_neg_ll_with_grad(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
    c: f64,
) -> Result<(f64, [f64; 4])> {
    let n = times.len();
    if n == 0 {
        // L = -μ·T; ∇L = [-T, 0, 0, 0]; neg = [T, 0, 0, 0]
        return Ok((mu * t_horizon, [t_horizon, 0.0, 0.0, 0.0]));
    }

    let exponent_lam = -(1.0 + beta); // for λ contribution: s^{exponent_lam}
                                      // Note: exponent_dc = -(2+β) = exponent_lam - 1, so
                                      //   s^{exponent_dc} = s^{exponent_lam} / s = p_lam / s.
                                      // Saves one f64::powf call per (i, j) pair — N²/2 transcendentals at large N.

    let mut log_lam_sum = 0.0_f64;
    let mut acc_inv_lam = 0.0_f64;
    let mut acc_alpha_log = 0.0_f64;
    let mut acc_beta_log = 0.0_f64;
    let mut acc_c_log = 0.0_f64;

    for i in 0..n {
        let t_i = times[i];
        let mut sum_pow_lam = 0.0_f64;
        let mut sum_pow_lam_log = 0.0_f64;
        let mut sum_pow_dc = 0.0_f64;
        for j in 0..i {
            let s = t_i - times[j] + c;
            let p_lam = s.powf(exponent_lam);
            sum_pow_lam += p_lam;
            sum_pow_lam_log += p_lam * s.ln();
            sum_pow_dc += p_lam / s;
        }
        let lam = mu + alpha * sum_pow_lam;
        let lam_safe = lam.max(1e-30);
        log_lam_sum += lam_safe.ln();
        let inv_lam = 1.0 / lam_safe;

        acc_inv_lam += inv_lam;
        acc_alpha_log += sum_pow_lam * inv_lam;
        acc_beta_log += sum_pow_lam_log * inv_lam;
        acc_c_log += sum_pow_dc * inv_lam;
    }

    // Compensator C = μ·T + (α/β)·Σ_j (c^{-β} - T_j^{-β}),  T_j = T - t_j + c.
    let c_neg_beta = c.powf(-beta);
    let c_neg_beta_minus_one = c.powf(-beta - 1.0);
    let log_c = c.ln();

    let mut comp_alpha_term = 0.0_f64; // Σ_j (c^{-β} - T_j^{-β})
    let mut comp_beta_log_diff = 0.0_f64; // Σ_j (-c^{-β}·log c + T_j^{-β}·log T_j)
    let mut comp_c_pow_diff = 0.0_f64; // Σ_j (T_j^{-β-1} - c^{-β-1})
    for &t in times {
        let tj = t_horizon - t + c;
        let tj_neg_beta = tj.powf(-beta);
        let tj_neg_beta_minus_one = tj.powf(-beta - 1.0);
        let log_tj = tj.ln();

        comp_alpha_term += c_neg_beta - tj_neg_beta;
        comp_beta_log_diff += -c_neg_beta * log_c + tj_neg_beta * log_tj;
        comp_c_pow_diff += tj_neg_beta_minus_one - c_neg_beta_minus_one;
    }

    let comp = mu * t_horizon + (alpha / beta) * comp_alpha_term;
    let neg_ll = comp - log_lam_sum;

    let grad_mu = t_horizon - acc_inv_lam;
    let grad_alpha = (1.0 / beta) * comp_alpha_term - acc_alpha_log;
    let grad_beta = -(alpha / (beta * beta)) * comp_alpha_term
        + (alpha / beta) * comp_beta_log_diff
        + alpha * acc_beta_log;
    let grad_c = alpha * comp_c_pow_diff + (1.0 + beta) * alpha * acc_c_log;

    Ok((neg_ll, [grad_mu, grad_alpha, grad_beta, grad_c]))
}

/// Value-only entry. Same math; ~30% cheaper.
#[inline]
pub fn uni_powerlaw_neg_ll(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
    c: f64,
) -> Result<f64> {
    let n = times.len();
    if n == 0 {
        return Ok(mu * t_horizon);
    }
    let exponent_lam = -(1.0 + beta);
    let mut log_lam_sum = 0.0_f64;
    for i in 0..n {
        let t_i = times[i];
        let mut sum_pow = 0.0_f64;
        for j in 0..i {
            sum_pow += (t_i - times[j] + c).powf(exponent_lam);
        }
        log_lam_sum += (mu + alpha * sum_pow).max(1e-30).ln();
    }
    let c_neg_beta = c.powf(-beta);
    let mut comp_alpha_term = 0.0_f64;
    for &t in times {
        comp_alpha_term += c_neg_beta - (t_horizon - t + c).powf(-beta);
    }
    let comp = mu * t_horizon + (alpha / beta) * comp_alpha_term;
    Ok(comp - log_lam_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn numerical_grad(times: &[f64], t: f64, params: [f64; 4], h: f64) -> [f64; 4] {
        let mut g = [0.0; 4];
        for i in 0..4 {
            let bump = |delta: f64| {
                let mut p = params;
                p[i] += delta;
                uni_powerlaw_neg_ll(times, t, p[0], p[1], p[2], p[3]).unwrap()
            };
            let f_p = bump(h);
            let f_m = bump(-h);
            let f_p2 = bump(2.0 * h);
            let f_m2 = bump(-2.0 * h);
            g[i] = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
        }
        g
    }

    fn brute_force(times: &[f64], t_horizon: f64, mu: f64, alpha: f64, beta: f64, c: f64) -> f64 {
        let mut log_lam = 0.0_f64;
        for i in 0..times.len() {
            let mut excit = 0.0_f64;
            for j in 0..i {
                excit += alpha * (times[i] - times[j] + c).powf(-(1.0 + beta));
            }
            log_lam += (mu + excit).max(1e-30).ln();
        }
        let mut comp = mu * t_horizon;
        for &t in times {
            // ∫_0^{T-t} α(s+c)^{-(1+β)} ds = (α/β)(c^{-β} - (T-t+c)^{-β})
            comp += (alpha / beta) * (c.powf(-beta) - (t_horizon - t + c).powf(-beta));
        }
        comp - log_lam
    }

    #[test]
    fn empty_events() {
        let (val, grad) = uni_powerlaw_neg_ll_with_grad(&[], 10.0, 0.5, 0.4, 0.8, 0.3).unwrap();
        assert_relative_eq!(val, 5.0, max_relative = 1e-15);
        assert_relative_eq!(grad[0], 10.0, max_relative = 1e-15);
        assert_eq!(grad[1], 0.0);
        assert_eq!(grad[2], 0.0);
        assert_eq!(grad[3], 0.0);
    }

    #[test]
    fn matches_brute_force_value() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2, 6.4];
        let T = 7.0;
        let (mu, alpha, beta, c) = (0.4, 0.3, 0.7, 0.5);
        let (val, _) = uni_powerlaw_neg_ll_with_grad(&times, T, mu, alpha, beta, c).unwrap();
        let bf = brute_force(&times, T, mu, alpha, beta, c);
        assert_relative_eq!(val, bf, max_relative = 1e-12);
    }

    #[test]
    fn analytic_grad_matches_numerical_simple() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let T = 6.0;
        let params = [0.4, 0.3, 0.8, 0.5];
        let (_, g_a) =
            uni_powerlaw_neg_ll_with_grad(&times, T, params[0], params[1], params[2], params[3])
                .unwrap();
        let g_n = numerical_grad(&times, T, params, 1e-5);
        for i in 0..4 {
            assert_relative_eq!(g_a[i], g_n[i], max_relative = 5e-6);
        }
    }

    #[test]
    fn analytic_grad_matches_numerical_heavy_tail() {
        // β small → heavy tail
        let times = [0.1, 0.5, 1.0, 1.7, 2.5, 3.4];
        let T = 4.5;
        let params = [0.2, 0.4, 0.3, 0.2];
        let (_, g_a) =
            uni_powerlaw_neg_ll_with_grad(&times, T, params[0], params[1], params[2], params[3])
                .unwrap();
        let g_n = numerical_grad(&times, T, params, 1e-6);
        for i in 0..4 {
            assert_relative_eq!(g_a[i], g_n[i], max_relative = 1e-5);
        }
    }

    #[test]
    fn value_and_value_grad_consistent() {
        let times = [0.5, 1.1, 2.0, 3.7];
        let (v1, _) = uni_powerlaw_neg_ll_with_grad(&times, 5.0, 0.4, 0.3, 0.8, 0.5).unwrap();
        let v2 = uni_powerlaw_neg_ll(&times, 5.0, 0.4, 0.3, 0.8, 0.5).unwrap();
        assert_relative_eq!(v1, v2, max_relative = 1e-15);
    }
}
