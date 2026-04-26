//! Univariate sum-of-exponentials Hawkes negative log-likelihood with
//! closed-form gradient.
//!
//! Model:
//!   φ(t) = Σ_k α_k·β_k·exp(-β_k·t)
//!   λ(t) = μ + Σ_{j: t_j < t} φ(t - t_j)
//!
//! K independent recursive states, one per component:
//!   R_{i,k}^post = R_{i,k}^pre + 1   where  R_{i,k}^pre = exp(-β_k·dt)·R_{i-1,k}^post
//!
//! Free parameters: μ, K alphas, K betas. Total 1 + 2K.
//!
//! ### Closed-form gradients
//!
//! For each component k we track the running ∂R_k/∂β_k alongside R_k via the
//! same recursion (chain rule on exp(-β·dt)·R). Per-step cost: O(K).
//!
//! Log-intensity gradients (per event i):
//!   ∂log λ_i/∂μ      = 1/λ_i
//!   ∂log λ_i/∂α_k    = β_k·R_{i,k}^pre / λ_i
//!   ∂log λ_i/∂β_k    = α_k·(R_{i,k}^pre + β_k·∂R_{i,k}^pre/∂β_k) / λ_i
//!
//! Compensator gradients (over events j):
//!   ∂C/∂μ   = T
//!   ∂C/∂α_k = Σ_j (1 - exp(-β_k·(T - t_j)))
//!   ∂C/∂β_k = α_k·Σ_j (T - t_j)·exp(-β_k·(T - t_j))

use intensify_core::{IntensifyError, Result};

/// Returns `(neg_loglik, grad_mu, grad_alphas, grad_betas)`. `alphas` and
/// `betas` must have the same length K.
#[allow(clippy::too_many_arguments)]
pub fn uni_sumexp_neg_ll_with_grad(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    alphas: &[f64],
    betas: &[f64],
) -> Result<(f64, f64, Vec<f64>, Vec<f64>)> {
    let k_components = alphas.len();
    if betas.len() != k_components {
        return Err(IntensifyError::InvalidParam(format!(
            "alphas length ({}) != betas length ({})",
            alphas.len(),
            betas.len()
        )));
    }
    if k_components == 0 {
        return Err(IntensifyError::InvalidParam(
            "must have at least one (alpha, beta) component".into(),
        ));
    }
    for (i, &b) in betas.iter().enumerate() {
        if b <= 0.0 {
            return Err(IntensifyError::InvalidParam(format!(
                "betas[{}] must be positive; got {}",
                i, b
            )));
        }
    }

    let n = times.len();
    if n == 0 {
        return Ok((mu * t_horizon, t_horizon, vec![0.0; k_components], vec![0.0; k_components]));
    }

    // Per-component state: R_k^post and ∂R_k^post/∂β_k.
    let mut r_post = vec![0.0_f64; k_components];
    let mut dr_post_db = vec![0.0_f64; k_components];

    let mut log_lam_sum = 0.0_f64;
    let mut grad_mu_log = 0.0_f64;
    let mut grad_alpha_log = vec![0.0_f64; k_components];
    let mut grad_beta_log = vec![0.0_f64; k_components];

    let mut t_prev = 0.0_f64;
    for &t_i in times {
        let dt = t_i - t_prev;

        // λ_i = μ + Σ_k α_k·β_k·R_k^pre, where R_k^pre = exp(-β_k·dt)·r_post[k].
        // ∂R_k^pre/∂β_k = -dt·R_k^pre + exp(-β_k·dt)·∂r_post[k]/∂β_k.
        // Compute pre-event states and gradients in a single pass.
        let mut lam = mu;
        // We need r_pre and dr_pre values for the gradient block; stash them
        // in stack-friendly small vectors.
        let mut r_pre_vec = vec![0.0_f64; k_components];
        let mut dr_pre_db_vec = vec![0.0_f64; k_components];
        for k in 0..k_components {
            let b_k = betas[k];
            let e = (-b_k * dt).exp();
            let r_pre = e * r_post[k];
            let dr_pre_db = -dt * r_pre + e * dr_post_db[k];
            r_pre_vec[k] = r_pre;
            dr_pre_db_vec[k] = dr_pre_db;
            lam += alphas[k] * b_k * r_pre;
        }

        let lam_safe = lam.max(1e-30);
        log_lam_sum += lam_safe.ln();
        let inv_lam = 1.0 / lam_safe;
        grad_mu_log += inv_lam;
        for k in 0..k_components {
            let b_k = betas[k];
            let r_pre = r_pre_vec[k];
            let dr_pre = dr_pre_db_vec[k];
            grad_alpha_log[k] += b_k * r_pre * inv_lam;
            grad_beta_log[k] += alphas[k] * (r_pre + b_k * dr_pre) * inv_lam;
        }

        // Absorb event into all components.
        for k in 0..k_components {
            r_post[k] = r_pre_vec[k] + 1.0;
            dr_post_db[k] = dr_pre_db_vec[k];
        }
        t_prev = t_i;
    }

    // Compensator: μ·T + Σ_k α_k·Σ_j (1 - exp(-β_k·(T - t_j)))
    // ∂C/∂α_k = Σ_j (1 - exp(-β_k·(T - t_j)))
    // ∂C/∂β_k = α_k·Σ_j (T - t_j)·exp(-β_k·(T - t_j))
    let mut comp_alpha_term = vec![0.0_f64; k_components];
    let mut comp_beta_grad_term = vec![0.0_f64; k_components];
    for &t in times {
        let tail = t_horizon - t;
        for k in 0..k_components {
            let e = (-betas[k] * tail).exp();
            comp_alpha_term[k] += 1.0 - e;
            comp_beta_grad_term[k] += tail * e;
        }
    }

    let mut comp_kernel = 0.0_f64;
    for k in 0..k_components {
        comp_kernel += alphas[k] * comp_alpha_term[k];
    }
    let comp = mu * t_horizon + comp_kernel;
    let neg_ll = comp - log_lam_sum;

    let grad_mu = t_horizon - grad_mu_log;
    let grad_alphas: Vec<f64> = (0..k_components)
        .map(|k| comp_alpha_term[k] - grad_alpha_log[k])
        .collect();
    let grad_betas: Vec<f64> = (0..k_components)
        .map(|k| alphas[k] * comp_beta_grad_term[k] - grad_beta_log[k])
        .collect();

    Ok((neg_ll, grad_mu, grad_alphas, grad_betas))
}

/// Value-only entry. ~30% cheaper than the value+grad call.
pub fn uni_sumexp_neg_ll(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    alphas: &[f64],
    betas: &[f64],
) -> Result<f64> {
    let k_components = alphas.len();
    if betas.len() != k_components || k_components == 0 {
        return Err(IntensifyError::InvalidParam(
            "alphas/betas length mismatch or empty".into(),
        ));
    }
    let n = times.len();
    if n == 0 {
        return Ok(mu * t_horizon);
    }

    let mut r_post = vec![0.0_f64; k_components];
    let mut t_prev = 0.0_f64;
    let mut log_lam_sum = 0.0_f64;

    for &t_i in times {
        let dt = t_i - t_prev;
        let mut lam = mu;
        for k in 0..k_components {
            let r_pre = (-betas[k] * dt).exp() * r_post[k];
            lam += alphas[k] * betas[k] * r_pre;
            r_post[k] = r_pre + 1.0;
        }
        log_lam_sum += lam.max(1e-30).ln();
        t_prev = t_i;
    }

    let mut comp_kernel = 0.0_f64;
    for &t in times {
        let tail = t_horizon - t;
        for k in 0..k_components {
            comp_kernel += alphas[k] * (1.0 - (-betas[k] * tail).exp());
        }
    }
    Ok(mu * t_horizon + comp_kernel - log_lam_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn brute_force(
        times: &[f64],
        t_horizon: f64,
        mu: f64,
        alphas: &[f64],
        betas: &[f64],
    ) -> f64 {
        let k = alphas.len();
        let mut log_lam = 0.0_f64;
        for i in 0..times.len() {
            let mut excit = 0.0_f64;
            for j in 0..i {
                for kk in 0..k {
                    excit += alphas[kk] * betas[kk] * (-betas[kk] * (times[i] - times[j])).exp();
                }
            }
            log_lam += (mu + excit).max(1e-30).ln();
        }
        let mut comp = mu * t_horizon;
        for &t in times {
            let tail = t_horizon - t;
            for kk in 0..k {
                comp += alphas[kk] * (1.0 - (-betas[kk] * tail).exp());
            }
        }
        comp - log_lam
    }

    #[test]
    fn empty_events() {
        let (val, gmu, ga, gb) = uni_sumexp_neg_ll_with_grad(
            &[], 10.0, 0.5, &[0.2, 0.1], &[1.0, 5.0],
        ).unwrap();
        assert_relative_eq!(val, 5.0, max_relative = 1e-15);
        assert_relative_eq!(gmu, 10.0, max_relative = 1e-15);
        assert_eq!(ga, vec![0.0, 0.0]);
        assert_eq!(gb, vec![0.0, 0.0]);
    }

    #[test]
    fn matches_brute_force_2_components() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let T = 6.0;
        let (mu, alphas, betas) = (0.4, vec![0.2, 0.1], vec![1.0, 3.5]);
        let (val, ..) = uni_sumexp_neg_ll_with_grad(&times, T, mu, &alphas, &betas).unwrap();
        let bf = brute_force(&times, T, mu, &alphas, &betas);
        assert_relative_eq!(val, bf, max_relative = 1e-12);
    }

    #[test]
    fn matches_brute_force_4_components() {
        let times = [0.1, 0.5, 1.0, 1.7, 2.5, 3.4, 4.0, 4.8];
        let T = 5.5;
        let mu = 0.3;
        let alphas = vec![0.10, 0.08, 0.05, 0.03];
        let betas = vec![0.5, 1.5, 4.0, 10.0];
        let (val, ..) = uni_sumexp_neg_ll_with_grad(&times, T, mu, &alphas, &betas).unwrap();
        let bf = brute_force(&times, T, mu, &alphas, &betas);
        assert_relative_eq!(val, bf, max_relative = 1e-12);
    }

    #[test]
    fn analytic_grad_matches_numerical() {
        let times = [0.5, 1.1, 2.0, 3.7];
        let T = 5.0;
        let mu = 0.4;
        let alphas = vec![0.2, 0.1];
        let betas = vec![1.0, 3.0];

        let (_, gmu_a, ga_a, gb_a) = uni_sumexp_neg_ll_with_grad(
            &times, T, mu, &alphas, &betas,
        ).unwrap();

        let h = 1e-6;
        let bump = |dmu: f64, da: &[f64], db: &[f64]| -> f64 {
            let mu_p = mu + dmu;
            let a_p: Vec<f64> = alphas.iter().zip(da).map(|(a, d)| a + d).collect();
            let b_p: Vec<f64> = betas.iter().zip(db).map(|(b, d)| b + d).collect();
            uni_sumexp_neg_ll(&times, T, mu_p, &a_p, &b_p).unwrap()
        };
        let zero = vec![0.0; 2];

        // ∂μ
        let f_p = bump(h, &zero, &zero);
        let f_m = bump(-h, &zero, &zero);
        let f_p2 = bump(2.0 * h, &zero, &zero);
        let f_m2 = bump(-2.0 * h, &zero, &zero);
        let gmu_n = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
        assert_relative_eq!(gmu_a, gmu_n, max_relative = 1e-6);

        // ∂α_k
        for k in 0..2 {
            let mut da = zero.clone(); da[k] = h;
            let mut da_m = zero.clone(); da_m[k] = -h;
            let mut da_p2 = zero.clone(); da_p2[k] = 2.0 * h;
            let mut da_m2 = zero.clone(); da_m2[k] = -2.0 * h;
            let f_p = bump(0.0, &da, &zero);
            let f_m = bump(0.0, &da_m, &zero);
            let f_p2 = bump(0.0, &da_p2, &zero);
            let f_m2 = bump(0.0, &da_m2, &zero);
            let g_n = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
            assert_relative_eq!(ga_a[k], g_n, max_relative = 1e-6);
        }

        // ∂β_k
        for k in 0..2 {
            let mut db = zero.clone(); db[k] = h;
            let mut db_m = zero.clone(); db_m[k] = -h;
            let mut db_p2 = zero.clone(); db_p2[k] = 2.0 * h;
            let mut db_m2 = zero.clone(); db_m2[k] = -2.0 * h;
            let f_p = bump(0.0, &zero, &db);
            let f_m = bump(0.0, &zero, &db_m);
            let f_p2 = bump(0.0, &zero, &db_p2);
            let f_m2 = bump(0.0, &zero, &db_m2);
            let g_n = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
            assert_relative_eq!(gb_a[k], g_n, max_relative = 1e-6);
        }
    }

    #[test]
    fn value_and_grad_consistent() {
        let times = [0.3, 0.9, 1.6, 2.2];
        let (v1, ..) = uni_sumexp_neg_ll_with_grad(&times, 3.0, 0.4, &[0.2, 0.1], &[1.0, 5.0]).unwrap();
        let v2 = uni_sumexp_neg_ll(&times, 3.0, 0.4, &[0.2, 0.1], &[1.0, 5.0]).unwrap();
        assert_relative_eq!(v1, v2, max_relative = 1e-15);
    }
}
