//! Univariate ApproxPowerLawKernel Hawkes negative log-likelihood with
//! closed-form gradient.
//!
//! Model (Bacry-Muzy approximation):
//!   φ(t) = α · Σ_k w_k · β_k · exp(-β_k · t)
//! with:
//!   β_k = β_min · r^k       for k = 0..K-1   (geometric)
//!   u_k = β_k^{β_pow - 1}
//!   w_k = u_k / Σ_j u_j     (normalized)
//!
//! L1 norm of φ over [0, ∞) = α (because Σ_k w_k = 1).
//!
//! ### Free parameters and gradient chain rule
//!
//! Optimized: (μ, α, β_pow, β_min). Fixed structurally: (r, K).
//!
//! Define effective per-component amplitudes α_eff_k = α · w_k. Then
//! the model is identical to SumExponentialKernel with parameters
//! (μ, α_eff, β_k). Reuse `uni_sumexp_neg_ll_with_grad` for the
//! per-component gradient and chain through:
//!
//!   ∂L/∂α     = Σ_k ∂L/∂α_eff_k · w_k
//!   ∂L/∂β_pow = α · Σ_k ∂L/∂α_eff_k · w_k · (ln β_k − <ln β>_w)
//!     where <ln β>_w = Σ_j w_j · ln β_j
//!   ∂L/∂β_min = (1/β_min) · Σ_k β_k · ∂L/∂β_k
//!     (because ∂β_k/∂β_min = r^k = β_k/β_min and ∂w_k/∂β_min = 0
//!      — the β_min factor cancels in u_k/U)

use intensify_core::{IntensifyError, Result};

use crate::uni_sumexp::{uni_sumexp_neg_ll, uni_sumexp_neg_ll_with_grad};

fn build_betas_and_weights(
    beta_pow: f64,
    beta_min: f64,
    r: f64,
    n_components: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if beta_pow <= 0.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "beta_pow must be positive; got {beta_pow}"
        )));
    }
    if beta_min <= 0.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "beta_min must be positive; got {beta_min}"
        )));
    }
    if r <= 1.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "r must be > 1 for geometric spacing; got {r}"
        )));
    }
    if n_components == 0 {
        return Err(IntensifyError::InvalidParam(
            "n_components must be positive".into(),
        ));
    }

    let mut betas = Vec::with_capacity(n_components);
    let mut u = Vec::with_capacity(n_components);
    for k in 0..n_components {
        let b_k = beta_min * r.powi(k as i32);
        betas.push(b_k);
        u.push(b_k.powf(beta_pow - 1.0));
    }
    let u_sum: f64 = u.iter().sum();
    let weights: Vec<f64> = u.iter().map(|&u_k| u_k / u_sum).collect();
    Ok((betas, weights))
}

/// Returns `(neg_loglik, [d/dμ, d/dα, d/dβ_pow, d/dβ_min])`.
#[allow(clippy::too_many_arguments)]
pub fn uni_approx_powerlaw_neg_ll_with_grad(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta_pow: f64,
    beta_min: f64,
    r: f64,
    n_components: usize,
) -> Result<(f64, [f64; 4])> {
    let (betas, weights) = build_betas_and_weights(beta_pow, beta_min, r, n_components)?;
    if alpha <= 0.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "alpha must be positive; got {alpha}"
        )));
    }

    // Effective per-component amplitudes for the SumExp formulation.
    let alphas_eff: Vec<f64> = weights.iter().map(|&w| alpha * w).collect();

    let (val, gmu, ga_eff, gb) =
        uni_sumexp_neg_ll_with_grad(times, t_horizon, mu, &alphas_eff, &betas)?;

    // ∂L/∂α: Σ_k ∂L/∂α_eff_k · w_k
    let grad_alpha: f64 = (0..n_components).map(|k| ga_eff[k] * weights[k]).sum();

    // ∂L/∂β_pow: α · Σ_k ∂L/∂α_eff_k · w_k · (ln β_k − <ln β>_w)
    let mean_log_beta_weighted: f64 = (0..n_components).map(|k| weights[k] * betas[k].ln()).sum();
    let grad_beta_pow: f64 = alpha
        * (0..n_components)
            .map(|k| ga_eff[k] * weights[k] * (betas[k].ln() - mean_log_beta_weighted))
            .sum::<f64>();

    // ∂L/∂β_min: (1/β_min) · Σ_k β_k · ∂L/∂β_k
    let grad_beta_min: f64 = (0..n_components).map(|k| betas[k] * gb[k]).sum::<f64>() / beta_min;

    Ok((val, [gmu, grad_alpha, grad_beta_pow, grad_beta_min]))
}

#[allow(clippy::too_many_arguments)]
pub fn uni_approx_powerlaw_neg_ll(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta_pow: f64,
    beta_min: f64,
    r: f64,
    n_components: usize,
) -> Result<f64> {
    let (betas, weights) = build_betas_and_weights(beta_pow, beta_min, r, n_components)?;
    let alphas_eff: Vec<f64> = weights.iter().map(|&w| alpha * w).collect();
    uni_sumexp_neg_ll(times, t_horizon, mu, &alphas_eff, &betas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn brute_force(
        times: &[f64],
        t_horizon: f64,
        mu: f64,
        alpha: f64,
        beta_pow: f64,
        beta_min: f64,
        r: f64,
        n_components: usize,
    ) -> f64 {
        let (betas, weights) =
            build_betas_and_weights(beta_pow, beta_min, r, n_components).unwrap();
        let mut log_lam = 0.0_f64;
        for i in 0..times.len() {
            let mut excit = 0.0_f64;
            for j in 0..i {
                for k in 0..n_components {
                    excit +=
                        alpha * weights[k] * betas[k] * (-betas[k] * (times[i] - times[j])).exp();
                }
            }
            log_lam += (mu + excit).max(1e-30).ln();
        }
        let mut comp = mu * t_horizon;
        for &t in times {
            for k in 0..n_components {
                comp += alpha * weights[k] * (1.0 - (-betas[k] * (t_horizon - t)).exp());
            }
        }
        comp - log_lam
    }

    #[test]
    fn matches_brute_force() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let T = 6.0;
        let (mu, alpha, beta_pow, beta_min) = (0.4, 0.3, 1.5, 0.5);
        let (r, K) = (1.5, 5_usize);
        let (val, _) =
            uni_approx_powerlaw_neg_ll_with_grad(&times, T, mu, alpha, beta_pow, beta_min, r, K)
                .unwrap();
        let bf = brute_force(&times, T, mu, alpha, beta_pow, beta_min, r, K);
        assert_relative_eq!(val, bf, max_relative = 1e-12);
    }

    #[test]
    fn analytic_grad_matches_finite_difference() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2, 6.4];
        let T = 7.5;
        let params = [0.3, 0.4, 1.2, 0.4]; // (μ, α, β_pow, β_min)
        let r = 1.5;
        let K = 6_usize;

        let (_, g_a) = uni_approx_powerlaw_neg_ll_with_grad(
            &times, T, params[0], params[1], params[2], params[3], r, K,
        )
        .unwrap();

        let h = 1e-6;
        let bump = |i: usize, delta: f64| -> f64 {
            let mut p = params;
            p[i] += delta;
            uni_approx_powerlaw_neg_ll(&times, T, p[0], p[1], p[2], p[3], r, K).unwrap()
        };
        for i in 0..4 {
            let f_p = bump(i, h);
            let f_m = bump(i, -h);
            let f_p2 = bump(i, 2.0 * h);
            let f_m2 = bump(i, -2.0 * h);
            let g_n = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
            assert_relative_eq!(g_a[i], g_n, max_relative = 1e-5);
        }
    }

    #[test]
    fn rejects_invalid_params() {
        let t = [0.5, 1.0];
        // beta_pow ≤ 0
        assert!(uni_approx_powerlaw_neg_ll_with_grad(&t, 2.0, 0.3, 0.2, 0.0, 0.5, 1.5, 4).is_err());
        // beta_min ≤ 0
        assert!(uni_approx_powerlaw_neg_ll_with_grad(&t, 2.0, 0.3, 0.2, 1.0, 0.0, 1.5, 4).is_err());
        // r ≤ 1
        assert!(uni_approx_powerlaw_neg_ll_with_grad(&t, 2.0, 0.3, 0.2, 1.0, 0.5, 1.0, 4).is_err());
        // n_components = 0
        assert!(uni_approx_powerlaw_neg_ll_with_grad(&t, 2.0, 0.3, 0.2, 1.0, 0.5, 1.5, 0).is_err());
        // alpha ≤ 0
        assert!(uni_approx_powerlaw_neg_ll_with_grad(&t, 2.0, 0.3, 0.0, 1.0, 0.5, 1.5, 4).is_err());
    }
}
