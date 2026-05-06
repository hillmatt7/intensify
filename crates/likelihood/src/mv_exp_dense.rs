//! Multivariate exponential Hawkes negative log-likelihood with **per-cell β**
//! (joint-decay mode). Closed-form analytic gradient.
//!
//! Differs from `mv_exp_recursive` in that every (m, k) cell has its own
//! β_{m,k} that the optimizer fits jointly with α_{m,k} and μ_m. Tick
//! does not support this case (its decay is shared and fixed).
//!
//! Model:
//!   λ_m(t) = μ_m + Σ_{j: t_j < t} α_{m, src_j}·β_{m, src_j}·exp(-β_{m, src_j}·(t - t_j))
//!
//! ### Algorithm
//!
//! Per-cell recursive state: R_{m,k}(t) = Σ_{j: src_j = k, t_j ≤ t} exp(-β_{m,k}·(t - t_j))
//!
//! At each event (t_i, src=s):
//!   1. Decay: R_{m,k} *= exp(-β_{m,k}·dt) for all (m, k); also decay ∂R/∂β
//!   2. Intensity:  λ_s = μ_s + Σ_k α_{s,k}·β_{s,k}·R_{s,k}  (only row m = s)
//!   3. Update gradients of log λ_s w.r.t. (μ_s, α_{s,:}, β_{s,:})
//!   4. Absorb: R_{m, s} += 1 for all m  (column s, all rows; ∂R/∂β unchanged
//!      since +1 is independent of β)
//!
//! Compensator is computed in a separate sweep over events at the end:
//!   C_m = μ_m·T + Σ_j α_{m, src_j}·(1 - exp(-β_{m, src_j}·(T - t_j)))
//!
//! Total work O(N·M²) for the main pass + O(N·M) for the compensator.
//! State: 2·M² scalars (R and ∂R/∂β).
//!
//! ### Coefficient layout (matches intensify Python)
//!
//! `coeffs = [μ_0..μ_{M-1}, (α_{0,0}, β_{0,0}), (α_{0,1}, β_{0,1}), ...,
//!            (α_{M-1, M-1}, β_{M-1, M-1})]`  — total length M + 2·M².
//! α_{m,k} at index `M + 2·(m·M + k)`, β_{m,k} at `M + 2·(m·M + k) + 1`.

use intensify_core::{IntensifyError, Result};

/// Compute the dense (joint-decay) multivariate exp Hawkes
/// neg-log-likelihood and its closed-form gradient.
///
/// Inputs:
/// - `times` (length N): event times sorted globally on `[0, end_time]`
/// - `sources` (length N): source dim of each event, in `0..M`
/// - `mu` (length M): baseline intensities
/// - `alpha` (length M·M, row-major): α_{m,k} at index `m·M + k`
/// - `beta`  (length M·M, row-major): β_{m,k} at index `m·M + k`
///
/// Returns `(neg_loglik, grad_mu, grad_alpha, grad_beta)`. Gradients are
/// row-major flat vectors matching the input layout.
#[allow(clippy::too_many_arguments)]
pub fn mv_exp_dense_neg_ll_with_grad(
    times: &[f64],
    sources: &[i64],
    end_time: f64,
    m: usize,
    mu: &[f64],
    alpha: &[f64],
    beta: &[f64],
) -> Result<(f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = times.len();
    if sources.len() != n {
        return Err(IntensifyError::InvalidParam(format!(
            "times.len() ({}) != sources.len() ({})",
            n,
            sources.len()
        )));
    }
    if mu.len() != m {
        return Err(IntensifyError::InvalidParam(format!(
            "mu.len() ({}) != M ({})",
            mu.len(),
            m
        )));
    }
    if alpha.len() != m * m || beta.len() != m * m {
        return Err(IntensifyError::InvalidParam(format!(
            "alpha/beta length ({}/{}) != M² ({})",
            alpha.len(),
            beta.len(),
            m * m
        )));
    }

    // Per-cell recursive state R_{m,k} and its β-derivative, flat row-major.
    let mut r = vec![0.0_f64; m * m];
    let mut dr_db = vec![0.0_f64; m * m];

    // Output accumulators.
    let mut log_lam_sum = 0.0_f64;
    let mut grad_mu = vec![0.0_f64; m];
    let mut grad_alpha = vec![0.0_f64; m * m];
    let mut grad_beta = vec![0.0_f64; m * m];

    let mut t_prev = 0.0_f64;
    for i in 0..n {
        let t_i = times[i];
        let src = sources[i];
        if !(0..m as i64).contains(&src) {
            return Err(IntensifyError::InvalidParam(format!(
                "sources[{}] = {} not in [0, {})",
                i, src, m
            )));
        }
        let s = src as usize;
        let dt = t_i - t_prev;

        // Step 1: decay all M² states.
        // ∂(R·e^{-β·dt})/∂β = e^{-β·dt}·∂R/∂β - dt·R·e^{-β·dt}
        for mm in 0..m {
            for kk in 0..m {
                let idx = mm * m + kk;
                let b_mk = beta[idx];
                let e = (-b_mk * dt).exp();
                let r_old = r[idx];
                r[idx] = r_old * e;
                dr_db[idx] = e * dr_db[idx] - dt * r_old * e;
            }
        }

        // Step 2: intensity λ_s at this event = μ_s + Σ_k α_{s,k}·β_{s,k}·R_{s,k}.
        let mu_s = mu[s];
        let mut lam = mu_s;
        let row_s = s * m;
        for kk in 0..m {
            let idx = row_s + kk;
            lam += alpha[idx] * beta[idx] * r[idx];
        }
        let lam_safe = lam.max(1e-30);
        log_lam_sum += lam_safe.ln();
        let inv_lam = 1.0 / lam_safe;

        // Step 3: log-intensity gradients (only the s-row contributes).
        // ∂log λ_s / ∂μ_s = 1/λ_s
        // ∂log λ_s / ∂α_{s,k} = β_{s,k}·R_{s,k} / λ_s
        // ∂log λ_s / ∂β_{s,k} = α_{s,k}·(R_{s,k} + β_{s,k}·∂R_{s,k}/∂β) / λ_s
        // Negate (we accumulate into the *negative* log-likelihood gradient).
        grad_mu[s] -= inv_lam;
        for kk in 0..m {
            let idx = row_s + kk;
            let a_sk = alpha[idx];
            let b_sk = beta[idx];
            let r_sk = r[idx];
            let dr_sk = dr_db[idx];
            grad_alpha[idx] -= b_sk * r_sk * inv_lam;
            grad_beta[idx] -= a_sk * (r_sk + b_sk * dr_sk) * inv_lam;
        }

        // Step 4: absorb the event in column s for all rows m.
        for mm in 0..m {
            r[mm * m + s] += 1.0;
            // dr_db unchanged: ∂(R+1)/∂β = ∂R/∂β.
        }

        t_prev = t_i;
    }

    // Compensator and its gradients.
    // C = Σ_m [μ_m·T + Σ_j α_{m, src_j}·(1 - exp(-β_{m, src_j}·(T - t_j)))]
    let mut total_comp = 0.0_f64;
    for mm in 0..m {
        let mu_m = mu[mm];
        total_comp += mu_m * end_time;
        grad_mu[mm] += end_time;

        let row_m = mm * m;
        for j in 0..n {
            let t_j = times[j];
            let s_j = sources[j] as usize;
            let idx = row_m + s_j;
            let a_mk = alpha[idx];
            let b_mk = beta[idx];
            let tail = end_time - t_j;
            let e = (-b_mk * tail).exp();
            let one_minus_e = 1.0 - e;
            total_comp += a_mk * one_minus_e;
            grad_alpha[idx] += one_minus_e;
            grad_beta[idx] += a_mk * tail * e;
        }
    }

    let neg_ll = total_comp - log_lam_sum;
    Ok((neg_ll, grad_mu, grad_alpha, grad_beta))
}

/// Value-only entry point. Identical math without the gradient
/// bookkeeping; ~30% faster.
#[allow(clippy::too_many_arguments)]
pub fn mv_exp_dense_neg_ll(
    times: &[f64],
    sources: &[i64],
    end_time: f64,
    m: usize,
    mu: &[f64],
    alpha: &[f64],
    beta: &[f64],
) -> Result<f64> {
    let n = times.len();
    if sources.len() != n {
        return Err(IntensifyError::InvalidParam(format!(
            "times.len() ({}) != sources.len() ({})",
            n,
            sources.len()
        )));
    }
    if mu.len() != m || alpha.len() != m * m || beta.len() != m * m {
        return Err(IntensifyError::InvalidParam(format!(
            "shape mismatch: mu/alpha/beta = {}/{}/{}, expected M={}, M²={}",
            mu.len(),
            alpha.len(),
            beta.len(),
            m,
            m * m,
        )));
    }

    let mut r = vec![0.0_f64; m * m];
    let mut log_lam_sum = 0.0_f64;
    let mut t_prev = 0.0_f64;

    for i in 0..n {
        let t_i = times[i];
        let s = sources[i] as usize;
        if s >= m {
            return Err(IntensifyError::InvalidParam(format!(
                "sources[{}] = {} not in [0, {})",
                i, s, m
            )));
        }
        let dt = t_i - t_prev;
        for mm in 0..m {
            for kk in 0..m {
                let idx = mm * m + kk;
                r[idx] *= (-beta[idx] * dt).exp();
            }
        }

        let row_s = s * m;
        let mut lam = mu[s];
        for kk in 0..m {
            let idx = row_s + kk;
            lam += alpha[idx] * beta[idx] * r[idx];
        }
        log_lam_sum += lam.max(1e-30).ln();

        for mm in 0..m {
            r[mm * m + s] += 1.0;
        }
        t_prev = t_i;
    }

    let mut total_comp = 0.0_f64;
    for mm in 0..m {
        total_comp += mu[mm] * end_time;
        let row_m = mm * m;
        for j in 0..n {
            let s_j = sources[j] as usize;
            let idx = row_m + s_j;
            let tail = end_time - times[j];
            total_comp += alpha[idx] * (1.0 - (-beta[idx] * tail).exp());
        }
    }

    Ok(total_comp - log_lam_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Brute-force O(N²·M²) reference. Unrolls every kernel evaluation.
    fn brute_force(
        times: &[f64],
        sources: &[i64],
        end_time: f64,
        m: usize,
        mu: &[f64],
        alpha: &[f64],
        beta: &[f64],
    ) -> f64 {
        let n = times.len();
        let mut log_lam_sum = 0.0_f64;
        for i in 0..n {
            let t_i = times[i];
            let s_i = sources[i] as usize;
            let mut lam = mu[s_i];
            for j in 0..i {
                let t_j = times[j];
                let s_j = sources[j] as usize;
                let idx = s_i * m + s_j;
                lam += alpha[idx] * beta[idx] * (-beta[idx] * (t_i - t_j)).exp();
            }
            log_lam_sum += lam.max(1e-30).ln();
        }
        let mut comp = 0.0_f64;
        for mm in 0..m {
            comp += mu[mm] * end_time;
            for j in 0..n {
                let s_j = sources[j] as usize;
                let idx = mm * m + s_j;
                comp += alpha[idx] * (1.0 - (-beta[idx] * (end_time - times[j])).exp());
            }
        }
        comp - log_lam_sum
    }

    fn make_2d_test() -> (Vec<f64>, Vec<i64>, f64, usize, Vec<f64>, Vec<f64>, Vec<f64>) {
        let times = vec![0.5, 0.7, 1.3, 1.5, 2.1, 2.4, 3.0, 3.5, 3.8];
        let sources = vec![0, 1, 0, 1, 0, 1, 0, 1, 0];
        let end_time = 4.5;
        let m = 2;
        let mu = vec![0.4, 0.3];
        let alpha = vec![0.20, 0.10, 0.15, 0.25]; // row-major
        let beta = vec![1.50, 1.20, 0.90, 1.80];
        (times, sources, end_time, m, mu, alpha, beta)
    }

    #[test]
    fn matches_brute_force_2d_distinct_betas() {
        let (times, sources, T, m, mu, a, b) = make_2d_test();
        let (rust, ..) =
            mv_exp_dense_neg_ll_with_grad(&times, &sources, T, m, &mu, &a, &b).unwrap();
        let brute = brute_force(&times, &sources, T, m, &mu, &a, &b);
        assert_relative_eq!(rust, brute, max_relative = 1e-12);
    }

    #[test]
    fn matches_brute_force_3d_dense() {
        let times: Vec<f64> = (0..30).map(|i| 0.1 + 0.13 * (i as f64)).collect();
        let sources: Vec<i64> = (0..30).map(|i| (i as i64) % 3).collect();
        let end_time = 4.5;
        let m = 3;
        let mu = vec![0.20, 0.25, 0.15];
        let alpha = vec![0.10, 0.05, 0.08, 0.07, 0.12, 0.04, 0.06, 0.09, 0.11];
        let beta = vec![1.5, 0.8, 1.2, 0.9, 1.6, 1.1, 1.3, 0.7, 1.4];

        let (rust, ..) =
            mv_exp_dense_neg_ll_with_grad(&times, &sources, end_time, m, &mu, &alpha, &beta)
                .unwrap();
        let brute = brute_force(&times, &sources, end_time, m, &mu, &alpha, &beta);
        assert_relative_eq!(rust, brute, max_relative = 1e-12);
    }

    #[test]
    fn analytic_grad_matches_finite_difference() {
        let (times, sources, T, m, mu, alpha, beta) = make_2d_test();
        let n_params = m + 2 * m * m;

        let (_v, gm, ga, gb) =
            mv_exp_dense_neg_ll_with_grad(&times, &sources, T, m, &mu, &alpha, &beta).unwrap();
        let mut analytic = vec![0.0_f64; n_params];
        analytic[..m].copy_from_slice(&gm);
        for cell in 0..m * m {
            analytic[m + 2 * cell] = ga[cell];
            analytic[m + 2 * cell + 1] = gb[cell];
        }

        let pack = |mu: &[f64], a: &[f64], b: &[f64]| -> Vec<f64> {
            let mut v = mu.to_vec();
            for cell in 0..m * m {
                v.push(a[cell]);
                v.push(b[cell]);
            }
            v
        };
        let unpack = |x: &[f64]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let mu = x[..m].to_vec();
            let mut a = vec![0.0; m * m];
            let mut b = vec![0.0; m * m];
            for cell in 0..m * m {
                a[cell] = x[m + 2 * cell];
                b[cell] = x[m + 2 * cell + 1];
            }
            (mu, a, b)
        };

        let x0 = pack(&mu, &alpha, &beta);

        // 5-point stencil per parameter
        let h = 1e-6;
        let mut numeric = vec![0.0_f64; n_params];
        for idx in 0..n_params {
            let bump = |delta: f64| {
                let mut x = x0.clone();
                x[idx] += delta;
                let (mu_p, a_p, b_p) = unpack(&x);
                mv_exp_dense_neg_ll(&times, &sources, T, m, &mu_p, &a_p, &b_p).unwrap()
            };
            let f_p = bump(h);
            let f_m = bump(-h);
            let f_p2 = bump(2.0 * h);
            let f_m2 = bump(-2.0 * h);
            numeric[idx] = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
        }
        for i in 0..n_params {
            assert_relative_eq!(analytic[i], numeric[i], max_relative = 1e-6);
        }
    }

    #[test]
    fn value_and_grad_consistent() {
        let (times, sources, T, m, mu, a, b) = make_2d_test();
        let (v1, ..) = mv_exp_dense_neg_ll_with_grad(&times, &sources, T, m, &mu, &a, &b).unwrap();
        let v2 = mv_exp_dense_neg_ll(&times, &sources, T, m, &mu, &a, &b).unwrap();
        assert_relative_eq!(v1, v2, max_relative = 1e-15);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let times = vec![0.5, 1.0];
        let sources_bad_dim = vec![0_i64, 5_i64];
        assert!(mv_exp_dense_neg_ll_with_grad(
            &times,
            &sources_bad_dim,
            2.0,
            2,
            &[0.1, 0.1],
            &[0.1, 0.1, 0.1, 0.1],
            &[1.0, 1.0, 1.0, 1.0],
        )
        .is_err());

        let bad_alpha = vec![0.1, 0.1]; // wrong size for M=2
        assert!(mv_exp_dense_neg_ll_with_grad(
            &times,
            &[0_i64, 1_i64],
            2.0,
            2,
            &[0.1, 0.1],
            &bad_alpha,
            &[1.0, 1.0, 1.0, 1.0],
        )
        .is_err());
    }
}
