//! Univariate nonparametric (piecewise-constant) Hawkes negative log-likelihood
//! with closed-form gradient.
//!
//! Model: λ(t) = μ + Σ_{j: t_j < t} φ(t - t_j), where φ is piecewise-constant
//! on bins defined by `edges` (length K+1, edges[0]=0) with heights `values`
//! (length K, non-negative).
//!
//! No recursive form — pairs (i, j) contribute via a binary-search bin lookup.
//! Computation is O(N²·log K), with the constant factor MUCH lower than the
//! existing JAX path (which currently performs `kernel.evaluate(jnp.array([lag]))[0]`
//! per pair, allocating a JAX array each time — see ISSUES.md item #8 — and is
//! effectively unusable above N=300).
//!
//! ### Free parameters
//!
//! μ + K bin heights = K+1 free parameters. Edges are *fixed* during a fit;
//! they're chosen up-front by the user (e.g. `select_bin_count_aic`).
//!
//! ### Closed-form gradients
//!
//! For each event i and each pair j < i, let k_{ij} = bin_index(t_i - t_j).
//! If t_i - t_j ≥ edges[K] then the pair contributes 0 (out of support).
//!
//!   λ_i = μ + Σ_{j<i: k_{ij} < K} values[k_{ij}]
//!   ∂log λ_i/∂μ = 1/λ_i
//!   ∂log λ_i/∂values[k] = (1/λ_i)·#{j<i : k_{ij} = k}
//!
//! Compensator: C = μ·T + Σ_j ∫₀^{T-t_j} φ(s)ds. Each term is a piecewise sum.
//! The total compensator over j contributes to bin k an "overlap weight":
//!
//!   ∂C/∂μ = T
//!   ∂C/∂values[k] = Σ_j max(0, min(T-t_j, edges[k+1]) - edges[k])
//!
//! i.e., for each event j, the length of the intersection of bin k's interval
//! [edges[k], edges[k+1]] with [0, T-t_j].

use intensify_core::{IntensifyError, Result};

#[inline]
fn bin_index(edges: &[f64], t: f64) -> Option<usize> {
    let last_edge = *edges.last().unwrap();
    if t < 0.0 || t >= last_edge {
        return None;
    }
    let kp1 = edges.partition_point(|&e| e <= t);
    Some(kp1 - 1)
}

/// Negative log-likelihood and gradient for univariate piecewise-constant
/// Hawkes at parameters `(mu, values)`. `edges` is fixed; not differentiated.
///
/// Returns `(neg_loglik, grad_mu, grad_values)`. `grad_values` has the same
/// length as `values`.
pub fn uni_nonparametric_neg_ll_with_grad(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    edges: &[f64],
    values: &[f64],
) -> Result<(f64, f64, Vec<f64>)> {
    if edges.len() != values.len() + 1 {
        return Err(IntensifyError::InvalidParam(format!(
            "edges length ({}) must be values length ({}) + 1",
            edges.len(),
            values.len(),
        )));
    }
    let k_bins = values.len();

    let n = times.len();
    if n == 0 {
        return Ok((mu * t_horizon, t_horizon, vec![0.0; k_bins]));
    }

    // Per-event log-intensity sum + gradient accumulators.
    let mut log_lam_sum = 0.0_f64;
    let mut grad_mu_log = 0.0_f64; // Σ_i 1/λ_i
                                   // For ∂log λ_i / ∂values[k], we need Σ_{j<i} 1[k_{ij}=k] for each i, then divide by λ_i.
                                   // Accumulate Σ_i count(k) / λ_i directly.
    let mut grad_values_log = vec![0.0_f64; k_bins];
    // Per-event temporary count of (j<i) hits per bin; reused across i.
    let mut bin_counts = vec![0_usize; k_bins];

    for i in 0..n {
        let t_i = times[i];
        // Reset per-i counts.
        for c in bin_counts.iter_mut() {
            *c = 0;
        }
        let mut excit = 0.0_f64;
        for j in 0..i {
            if let Some(k) = bin_index(edges, t_i - times[j]) {
                bin_counts[k] += 1;
                excit += values[k];
            }
        }
        let lam = mu + excit;
        let lam_safe = lam.max(1e-30);
        log_lam_sum += lam_safe.ln();
        let inv_lam = 1.0 / lam_safe;
        grad_mu_log += inv_lam;
        for k in 0..k_bins {
            grad_values_log[k] += (bin_counts[k] as f64) * inv_lam;
        }
    }

    // Compensator and its gradients.
    // C = μ·T + Σ_j ∫₀^{T-t_j} φ(s)ds
    // For each event j, the ∫_{0}^{T-t_j} contribution to bin k is:
    //   max(0, min(T-t_j, edges[k+1]) - edges[k])  if T-t_j > edges[k]; else 0
    let mut comp_kernel = 0.0_f64;
    let mut grad_values_comp = vec![0.0_f64; k_bins];
    for &t in times {
        let tail = t_horizon - t;
        if tail <= 0.0 {
            continue;
        }
        for k in 0..k_bins {
            let lo = edges[k];
            if tail <= lo {
                break;
            }
            let hi = edges[k + 1];
            let upper = if hi <= tail { hi } else { tail };
            let weight = upper - lo;
            comp_kernel += values[k] * weight;
            grad_values_comp[k] += weight;
        }
    }

    let comp = mu * t_horizon + comp_kernel;
    let neg_ll = comp - log_lam_sum;

    let grad_mu = t_horizon - grad_mu_log;
    let grad_values: Vec<f64> = (0..k_bins)
        .map(|k| grad_values_comp[k] - grad_values_log[k])
        .collect();

    Ok((neg_ll, grad_mu, grad_values))
}

/// Value-only entry. Same math; ~30% cheaper.
pub fn uni_nonparametric_neg_ll(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    edges: &[f64],
    values: &[f64],
) -> Result<f64> {
    if edges.len() != values.len() + 1 {
        return Err(IntensifyError::InvalidParam(format!(
            "edges length ({}) must be values length ({}) + 1",
            edges.len(),
            values.len(),
        )));
    }
    let n = times.len();
    if n == 0 {
        return Ok(mu * t_horizon);
    }

    let mut log_lam_sum = 0.0_f64;
    for i in 0..n {
        let t_i = times[i];
        let mut excit = 0.0_f64;
        for j in 0..i {
            if let Some(k) = bin_index(edges, t_i - times[j]) {
                excit += values[k];
            }
        }
        log_lam_sum += (mu + excit).max(1e-30).ln();
    }

    let mut comp_kernel = 0.0_f64;
    for &t in times {
        let tail = t_horizon - t;
        if tail <= 0.0 {
            continue;
        }
        for k in 0..values.len() {
            let lo = edges[k];
            if tail <= lo {
                break;
            }
            let hi = edges[k + 1];
            let upper = if hi <= tail { hi } else { tail };
            comp_kernel += values[k] * (upper - lo);
        }
    }

    Ok(mu * t_horizon + comp_kernel - log_lam_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn brute_force(times: &[f64], t_horizon: f64, mu: f64, edges: &[f64], values: &[f64]) -> f64 {
        let mut log_lam = 0.0_f64;
        for i in 0..times.len() {
            let mut excit = 0.0_f64;
            for j in 0..i {
                let lag = times[i] - times[j];
                // linear scan
                for k in 0..values.len() {
                    if lag >= edges[k] && lag < edges[k + 1] {
                        excit += values[k];
                        break;
                    }
                }
            }
            log_lam += (mu + excit).max(1e-30).ln();
        }
        let mut comp = mu * t_horizon;
        for &t in times {
            let tail = t_horizon - t;
            if tail <= 0.0 {
                continue;
            }
            for k in 0..values.len() {
                if edges[k] >= tail {
                    break;
                }
                let upper = if edges[k + 1] <= tail {
                    edges[k + 1]
                } else {
                    tail
                };
                comp += values[k] * (upper - edges[k]);
            }
        }
        comp - log_lam
    }

    #[test]
    fn empty_events() {
        let edges = vec![0.0, 1.0, 2.0];
        let values = vec![0.5, 0.3];
        let (val, gmu, gv) =
            uni_nonparametric_neg_ll_with_grad(&[], 10.0, 0.5, &edges, &values).unwrap();
        assert_relative_eq!(val, 5.0, max_relative = 1e-15);
        assert_relative_eq!(gmu, 10.0, max_relative = 1e-15);
        assert_eq!(gv, vec![0.0, 0.0]);
    }

    #[test]
    fn matches_brute_force_value() {
        let times = [0.5, 1.1, 1.9, 2.4, 3.0, 3.7, 4.5, 5.2];
        let T = 6.0;
        let mu = 0.4;
        let edges = vec![0.0, 0.3, 1.0, 2.5];
        let values = vec![0.5, 0.3, 0.1];

        let (val, ..) = uni_nonparametric_neg_ll_with_grad(&times, T, mu, &edges, &values).unwrap();
        let bf = brute_force(&times, T, mu, &edges, &values);
        assert_relative_eq!(val, bf, max_relative = 1e-12);
    }

    #[test]
    fn analytic_grad_matches_numerical() {
        let times = [0.5, 1.1, 1.9, 2.4, 3.0, 3.7, 4.5];
        let T = 5.0;
        let mu = 0.3;
        let edges = vec![0.0, 0.5, 1.5, 3.0];
        let values = vec![0.4, 0.2, 0.1];

        let (_, gmu_a, gv_a) =
            uni_nonparametric_neg_ll_with_grad(&times, T, mu, &edges, &values).unwrap();

        let h = 1e-6;
        let bump = |dmu: f64, dv: &[f64]| -> f64 {
            let mu_p = mu + dmu;
            let v_p: Vec<f64> = values.iter().zip(dv).map(|(v, d)| v + d).collect();
            uni_nonparametric_neg_ll(&times, T, mu_p, &edges, &v_p).unwrap()
        };
        let zero_dv = vec![0.0; values.len()];

        // ∂/∂μ
        let f_p = bump(h, &zero_dv);
        let f_m = bump(-h, &zero_dv);
        let f_p2 = bump(2.0 * h, &zero_dv);
        let f_m2 = bump(-2.0 * h, &zero_dv);
        let gmu_n = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
        assert_relative_eq!(gmu_a, gmu_n, max_relative = 1e-6);

        // ∂/∂values[k]
        for k in 0..values.len() {
            let mut dv_p = zero_dv.clone();
            dv_p[k] = h;
            let mut dv_m = zero_dv.clone();
            dv_m[k] = -h;
            let mut dv_p2 = zero_dv.clone();
            dv_p2[k] = 2.0 * h;
            let mut dv_m2 = zero_dv.clone();
            dv_m2[k] = -2.0 * h;
            let f_p = bump(0.0, &dv_p);
            let f_m = bump(0.0, &dv_m);
            let f_p2 = bump(0.0, &dv_p2);
            let f_m2 = bump(0.0, &dv_m2);
            let gv_n = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
            assert_relative_eq!(gv_a[k], gv_n, max_relative = 1e-6);
        }
    }

    #[test]
    fn out_of_support_lags_excluded() {
        // Events are far apart; lag t_1 - t_0 = 5 > last edge = 2. Excitation = 0.
        let times = [0.0, 5.0];
        let T = 10.0;
        let mu = 0.4;
        let edges = vec![0.0, 1.0, 2.0];
        let values = vec![0.5, 0.3];
        let (val, ..) = uni_nonparametric_neg_ll_with_grad(&times, T, mu, &edges, &values).unwrap();
        // λ at event 0 = μ; λ at event 1 = μ (lag 5 is out of support)
        // log_lam_sum = 2·log(μ) = 2·log(0.4)
        // C = μ·T + integrate(T - 0) + integrate(T - 5)
        //   = 0.4·10 + L1 + L1   (both tails > last edge so full L1)
        //   = 4 + 0.5·1 + 0.3·1 + (0.5·1 + 0.3·1) = 4 + 0.8 + 0.8 = 5.6
        let expected = 4.0 + 0.8 + 0.8 - 2.0 * 0.4_f64.ln();
        assert_relative_eq!(val, expected, max_relative = 1e-12);
    }
}
