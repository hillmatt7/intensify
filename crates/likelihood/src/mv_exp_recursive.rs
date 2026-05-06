//! Multivariate exponential Hawkes negative log-likelihood with closed-form
//! gradient. Modeled directly on tick's
//! `ModelHawkesExpKernLogLikSingle` / `ModelHawkesLogLikSingle`
//! (BSD-3-Clause, X-DataInitiative/tick).
//!
//! Model:
//!   λ_i(t) = μ_i + Σ_{j} Σ_{t_m^j < t} α_{i,j}·β·exp(-β·(t - t_m^j))
//!
//! β is **fixed** in this recursive variant (the case where
//! `fit_decay = False` in the Python wrapper). For the joint-decay
//! variant where β is fit, see `mv_exp_dense.rs` (Phase 2).
//!
//! ### Strategy (tick's)
//!
//! Pre-compute per-target weights that depend only on β + timestamps:
//!
//! * `g[i][k][j]` = Σ_{m: t_m^j < t_i^k} β·exp(-β·(t_i^k - t_m^j))
//! * `G[i][k][j]` = compensator increment over (t_i^{k-1}, t_i^k] from source j
//!   (with t_i^{-1} := 0 and a virtual final endpoint t_i^{N_i} := T)
//! * `sum_G[i][j]` = Σ_k G[i][k][j] — total compensator from j to i over [0, T]
//!
//! Then per-dimension loss/grad reduces to dot products — independent per
//! row, so each dim is computed in parallel via rayon.
//!
//! ### Coefficient layout (matches tick)
//!
//! `coeffs = [μ_0, μ_1, ..., μ_{M-1}, α_{0,:}, α_{1,:}, ..., α_{M-1,:}]`
//!
//! Total length `M + M·M`. `α[i][j]` (influence of source j on target i)
//! lives at index `M + i·M + j`. The gradient is written into `out` with
//! the same layout.
//!
//! ### Conventions
//!
//! Returns the **unnormalized negative log-likelihood** (compensator −
//! log-intensity-sum). Tick normalizes by `n_total_jumps`; we don't,
//! to match `python/intensify/core/inference/mle.py` and the univariate
//! Rust path.

use intensify_core::{IntensifyError, Result};
use ndarray::{Array1, Array2};

/// Per-fit precomputed weights and the data needed to evaluate
/// loss + gradient for the multivariate exp Hawkes process when β is fixed.
///
/// Construct once with `new(timestamps, end_time, decay)`; call
/// `loss_and_grad(coeffs, out)` repeatedly inside the L-BFGS loop. The
/// weight precomputation runs once; subsequent calls only do O(N·M) of
/// dot-products and divisions.
#[derive(Debug)]
pub struct MvExpRecursiveLogLik {
    n_dims: usize,
    end_time: f64,
    n_total_jumps: usize,
    decay: f64,

    /// Per-dimension event arrays, length `n_dims`. Each is sorted
    /// non-decreasing on `[0, end_time]`.
    timestamps: Vec<Vec<f64>>,

    /// `g[i]` has shape `(n_jumps_i, M)`. `g[i][k, j]` = excitation weight
    /// at the k-th event of dim i from source j.
    g: Vec<Array2<f64>>,

    /// `sum_G[i]` shape `(M,)`. Sum over k of `G[i][k, j]`.
    sum_g_compensator: Vec<Array1<f64>>,
}

impl MvExpRecursiveLogLik {
    /// Construct + precompute weights. Validates that timestamps are
    /// sorted, non-negative, and lie inside [0, end_time].
    pub fn new(timestamps: Vec<Vec<f64>>, end_time: f64, decay: f64) -> Result<Self> {
        if decay <= 0.0 {
            return Err(IntensifyError::InvalidParam(format!(
                "decay must be positive; got {decay}"
            )));
        }
        if end_time <= 0.0 {
            return Err(IntensifyError::InvalidParam(format!(
                "end_time must be positive; got {end_time}"
            )));
        }
        let n_dims = timestamps.len();
        if n_dims == 0 {
            return Err(IntensifyError::InvalidParam(
                "timestamps must not be empty (need at least 1 dimension)".into(),
            ));
        }

        // Validate event arrays.
        for (i, ev) in timestamps.iter().enumerate() {
            for k in 1..ev.len() {
                if ev[k] < ev[k - 1] {
                    return Err(IntensifyError::NonMonotoneEvents {
                        index: k,
                        prev: ev[k - 1],
                        curr: ev[k],
                    });
                }
            }
            if let Some(&first) = ev.first() {
                if first < 0.0 {
                    return Err(IntensifyError::EventOutOfHorizon {
                        value: first,
                        horizon: end_time,
                    });
                }
            }
            if let Some(&last) = ev.last() {
                if last > end_time {
                    return Err(IntensifyError::EventOutOfHorizon {
                        value: last,
                        horizon: end_time,
                    });
                }
            }
            let _ = i;
        }

        let n_total_jumps: usize = timestamps.iter().map(|t| t.len()).sum();

        // Precompute per-dim weights.
        let mut g = Vec::with_capacity(n_dims);
        let mut sum_g_compensator = Vec::with_capacity(n_dims);
        for i in 0..n_dims {
            let n_i = timestamps[i].len();
            let (g_i, _gc_i, sgc_i) =
                compute_weights_dim_i(i, &timestamps, end_time, decay, n_dims, n_i);
            g.push(g_i);
            sum_g_compensator.push(sgc_i);
        }

        Ok(Self {
            n_dims,
            end_time,
            n_total_jumps,
            decay,
            timestamps,
            g,
            sum_g_compensator,
        })
    }

    pub fn n_dims(&self) -> usize {
        self.n_dims
    }
    pub fn end_time(&self) -> f64 {
        self.end_time
    }
    pub fn decay(&self) -> f64 {
        self.decay
    }
    pub fn n_total_jumps(&self) -> usize {
        self.n_total_jumps
    }

    /// Number of free coefficients (`M + M·M`).
    pub fn n_coeffs(&self) -> usize {
        self.n_dims + self.n_dims * self.n_dims
    }

    /// Compute total negative log-likelihood and its gradient at `coeffs`.
    /// `out` must have length `n_coeffs()`. Returns `loss`; writes
    /// gradient in-place to `out`.
    pub fn loss_and_grad(&self, coeffs: &[f64], out: &mut [f64]) -> Result<f64> {
        let n_coeffs = self.n_coeffs();
        if coeffs.len() != n_coeffs {
            return Err(IntensifyError::InvalidParam(format!(
                "coeffs length mismatch: expected {}, got {}",
                n_coeffs,
                coeffs.len()
            )));
        }
        if out.len() != n_coeffs {
            return Err(IntensifyError::InvalidParam(format!(
                "out length mismatch: expected {}, got {}",
                n_coeffs,
                out.len()
            )));
        }

        // Zero the gradient.
        for slot in out.iter_mut() {
            *slot = 0.0;
        }

        let mut total_loss = 0.0;
        for i in 0..self.n_dims {
            total_loss += self.loss_and_grad_dim_i(i, coeffs, out)?;
        }
        Ok(total_loss)
    }

    /// Value-only entry; cheaper than value+grad by a constant factor.
    /// Used by `_finite_difference_std_errors`.
    pub fn loss(&self, coeffs: &[f64]) -> Result<f64> {
        let n_coeffs = self.n_coeffs();
        if coeffs.len() != n_coeffs {
            return Err(IntensifyError::InvalidParam(format!(
                "coeffs length mismatch: expected {}, got {}",
                n_coeffs,
                coeffs.len()
            )));
        }
        let mut total_loss = 0.0;
        for i in 0..self.n_dims {
            total_loss += self.loss_dim_i(i, coeffs);
        }
        Ok(total_loss)
    }

    fn alpha_i_range(&self, i: usize) -> std::ops::Range<usize> {
        let m = self.n_dims;
        let start = m + i * m;
        let end = m + (i + 1) * m;
        start..end
    }

    fn loss_dim_i(&self, i: usize, coeffs: &[f64]) -> f64 {
        let m = self.n_dims;
        let mu_i = coeffs[i];
        let alpha_i = &coeffs[self.alpha_i_range(i)];

        // Compensator part: μ_i·T + α_i · sum_G[i]
        let mut loss = mu_i * self.end_time;
        for j in 0..m {
            loss += alpha_i[j] * self.sum_g_compensator[i][j];
        }

        // − Σ_k log(λ_i_k)
        let g_i = &self.g[i];
        let n_i = self.timestamps[i].len();
        for k in 0..n_i {
            let mut s = mu_i;
            for j in 0..m {
                s += alpha_i[j] * g_i[(k, j)];
            }
            // Floor matches the Python `jnp.maximum(lam, 1e-30)` and the
            // univariate Rust path. Tick raises here; we soften because
            // L-BFGS-B occasionally probes mildly infeasible points.
            loss -= s.max(1e-30).ln();
        }
        loss
    }

    fn loss_and_grad_dim_i(&self, i: usize, coeffs: &[f64], out: &mut [f64]) -> Result<f64> {
        let m = self.n_dims;
        let mu_i = coeffs[i];
        let alpha_i_range = self.alpha_i_range(i);
        let alpha_i = &coeffs[alpha_i_range.clone()];

        let g_i = &self.g[i];
        let n_i = self.timestamps[i].len();

        // Compensator + its gradients (μ_i·T + α_i·sum_G[i]).
        let mut loss = mu_i * self.end_time;
        for j in 0..m {
            loss += alpha_i[j] * self.sum_g_compensator[i][j];
        }
        out[i] += self.end_time;
        for j in 0..m {
            out[alpha_i_range.start + j] += self.sum_g_compensator[i][j];
        }

        // −Σ_k log λ + its gradients (−Σ_k 1/λ for μ; −Σ_k g_k_j/λ for α_j).
        for k in 0..n_i {
            let mut s = mu_i;
            for j in 0..m {
                s += alpha_i[j] * g_i[(k, j)];
            }
            let s_safe = s.max(1e-30);
            loss -= s_safe.ln();
            let inv_s = 1.0 / s_safe;
            out[i] -= inv_s;
            for j in 0..m {
                out[alpha_i_range.start + j] -= g_i[(k, j)] * inv_s;
            }
        }

        Ok(loss)
    }
}

/// Tick-style weight recursion for one target dim i.
///
/// For each source dim j, walks the k-axis (target events of dim i, plus
/// the final endpoint at end_time) maintaining a single forward pointer
/// `ij` into the j-event array. Each `ij` is consumed once total across
/// the whole k loop, so the inner walk is amortized O(N_i + N_j).
fn compute_weights_dim_i(
    i: usize,
    timestamps: &[Vec<f64>],
    end_time: f64,
    decay: f64,
    m: usize,
    n_i: usize,
) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
    let t_i = &timestamps[i];

    // g shape (n_i, M); g_comp shape (n_i + 1, M); sum_g_comp shape (M,).
    let mut g = Array2::<f64>::zeros((n_i, m));
    let mut g_comp = Array2::<f64>::zeros((n_i + 1, m));
    let mut sum_g_comp = Array1::<f64>::zeros(m);

    for j in 0..m {
        let t_j = &timestamps[j];
        let n_j = t_j.len();
        let mut ij = 0_usize;

        for k in 0..=n_i {
            let t_i_k = if k < n_i { t_i[k] } else { end_time };

            if k > 0 {
                let dt = t_i_k - t_i[k - 1];
                let ebt = (-decay * dt).exp();

                if k < n_i {
                    g[(k, j)] = g[(k - 1, j)] * ebt;
                }
                // Compensator from old events: Σ_{old} (1 - exp(-β(t_i_k - t_m^j)))
                //   = (Σ_{old} β·exp(-β(t_i_{k-1} - t_m^j))) · (1 - ebt)/β
                //   = g[k-1, j] · (1 - ebt) / β
                g_comp[(k, j)] = g[(k - 1, j)] * (1.0 - ebt) / decay;
            }

            // Walk new j-events in (t_i_{k-1}, t_i_k]: each contributes
            // β·exp(-β(t_i_k - t_m^j)) to g, and (1 - exp(...)) to g_comp.
            while ij < n_j && t_j[ij] < t_i_k {
                let ebt = (-decay * (t_i_k - t_j[ij])).exp();
                if k < n_i {
                    g[(k, j)] += decay * ebt;
                }
                g_comp[(k, j)] += 1.0 - ebt;
                ij += 1;
            }

            sum_g_comp[j] += g_comp[(k, j)];
        }
    }

    (g, g_comp, sum_g_comp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Brute-force (O(N²·M²)) reference for the multivariate exp Hawkes
    /// neg-log-likelihood. Independent of the recursive precomputation,
    /// so a useful internal cross-check.
    fn brute_force_neg_ll(
        timestamps: &[Vec<f64>],
        end_time: f64,
        mu: &[f64],
        alpha: &Array2<f64>, // (M, M); alpha[i, j]
        beta: f64,
    ) -> f64 {
        let m = timestamps.len();
        let mut log_lam_sum = 0.0;
        for i in 0..m {
            for &t_i_k in &timestamps[i] {
                let mut excit = 0.0;
                for j in 0..m {
                    for &t_m_j in &timestamps[j] {
                        if t_m_j < t_i_k {
                            excit += alpha[(i, j)] * beta * (-beta * (t_i_k - t_m_j)).exp();
                        }
                    }
                }
                let lam = mu[i] + excit;
                log_lam_sum += lam.max(1e-30).ln();
            }
        }
        let mut comp = 0.0;
        for i in 0..m {
            comp += mu[i] * end_time;
            for j in 0..m {
                let mut int_ij = 0.0;
                for &t_m_j in &timestamps[j] {
                    if t_m_j <= end_time {
                        int_ij += 1.0 - (-beta * (end_time - t_m_j)).exp();
                    }
                }
                comp += alpha[(i, j)] * int_ij;
            }
        }
        comp - log_lam_sum
    }

    fn pack_coeffs(mu: &[f64], alpha: &Array2<f64>) -> Vec<f64> {
        let m = mu.len();
        let mut v = Vec::with_capacity(m + m * m);
        v.extend_from_slice(mu);
        for i in 0..m {
            for j in 0..m {
                v.push(alpha[(i, j)]);
            }
        }
        v
    }

    #[test]
    fn matches_brute_force_simple_2d() {
        let timestamps = vec![vec![0.5, 1.3, 2.1, 3.0, 3.8], vec![0.7, 1.5, 2.4, 3.5]];
        let end_time = 4.5;
        let beta = 1.5;
        let mu = vec![0.4, 0.3];
        let alpha = ndarray::array![[0.2, 0.1], [0.15, 0.25]];

        let model = MvExpRecursiveLogLik::new(timestamps.clone(), end_time, beta).unwrap();
        let coeffs = pack_coeffs(&mu, &alpha);
        let mut grad = vec![0.0; coeffs.len()];
        let loss = model.loss_and_grad(&coeffs, &mut grad).unwrap();

        let brute = brute_force_neg_ll(&timestamps, end_time, &mu, &alpha, beta);
        assert_relative_eq!(loss, brute, max_relative = 1e-12);

        let loss_only = model.loss(&coeffs).unwrap();
        assert_relative_eq!(loss_only, loss, max_relative = 1e-15);
    }

    #[test]
    fn matches_brute_force_3d_dense() {
        let timestamps = vec![
            vec![0.1, 0.4, 0.9, 1.5, 2.0, 2.7, 3.4, 4.0, 4.5, 4.9],
            vec![0.2, 0.7, 1.2, 1.8, 2.5, 3.0, 3.6, 4.2],
            vec![0.3, 0.8, 1.4, 2.1, 2.8, 3.5, 4.1, 4.7, 4.95],
        ];
        let end_time = 5.0;
        let beta = 2.0;
        let mu = vec![0.2, 0.25, 0.15];
        let alpha = ndarray::array![[0.10, 0.05, 0.08], [0.07, 0.12, 0.04], [0.06, 0.09, 0.11]];

        let model = MvExpRecursiveLogLik::new(timestamps.clone(), end_time, beta).unwrap();
        let coeffs = pack_coeffs(&mu, &alpha);
        let mut grad = vec![0.0; coeffs.len()];
        let loss = model.loss_and_grad(&coeffs, &mut grad).unwrap();
        let brute = brute_force_neg_ll(&timestamps, end_time, &mu, &alpha, beta);
        assert_relative_eq!(loss, brute, max_relative = 1e-12);
    }

    #[test]
    fn analytic_grad_matches_finite_difference() {
        let timestamps = vec![
            vec![0.5, 1.3, 2.1, 3.0],
            vec![0.7, 1.5, 2.4],
            vec![0.3, 1.0, 2.0, 2.9],
        ];
        let end_time = 3.5;
        let beta = 1.4;
        let mu = vec![0.3, 0.25, 0.4];
        let alpha = ndarray::array![[0.10, 0.05, 0.08], [0.07, 0.12, 0.04], [0.06, 0.09, 0.11]];
        let coeffs = pack_coeffs(&mu, &alpha);
        let n_coeffs = coeffs.len();

        let model = MvExpRecursiveLogLik::new(timestamps, end_time, beta).unwrap();
        let mut analytic = vec![0.0; n_coeffs];
        let _ = model.loss_and_grad(&coeffs, &mut analytic).unwrap();

        // 5-point stencil
        let h = 1e-6;
        for idx in 0..n_coeffs {
            let mut p_p = coeffs.clone();
            let mut p_m = coeffs.clone();
            let mut p_p2 = coeffs.clone();
            let mut p_m2 = coeffs.clone();
            p_p[idx] += h;
            p_m[idx] -= h;
            p_p2[idx] += 2.0 * h;
            p_m2[idx] -= 2.0 * h;
            let f_p = model.loss(&p_p).unwrap();
            let f_m = model.loss(&p_m).unwrap();
            let f_p2 = model.loss(&p_p2).unwrap();
            let f_m2 = model.loss(&p_m2).unwrap();
            let numeric = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
            assert_relative_eq!(analytic[idx], numeric, max_relative = 1e-6);
        }
    }

    #[test]
    fn rejects_invalid_inputs() {
        // empty timestamps
        assert!(MvExpRecursiveLogLik::new(vec![], 1.0, 1.0).is_err());
        // negative decay
        assert!(MvExpRecursiveLogLik::new(vec![vec![0.5]], 1.0, 0.0).is_err());
        // event past end_time
        assert!(MvExpRecursiveLogLik::new(vec![vec![0.5, 2.0]], 1.0, 1.0).is_err());
        // non-monotone events
        assert!(MvExpRecursiveLogLik::new(vec![vec![0.5, 0.3]], 1.0, 1.0).is_err());
    }

    #[test]
    fn empty_dim_handled() {
        // Some dims may have zero events.
        let timestamps = vec![vec![0.5, 1.3, 2.1], vec![]];
        let end_time = 3.0;
        let beta = 1.5;
        let mu = vec![0.3, 0.2];
        let alpha = ndarray::array![[0.1, 0.05], [0.08, 0.12]];

        let model = MvExpRecursiveLogLik::new(timestamps.clone(), end_time, beta).unwrap();
        let coeffs = pack_coeffs(&mu, &alpha);
        let mut grad = vec![0.0; coeffs.len()];
        let loss = model.loss_and_grad(&coeffs, &mut grad).unwrap();
        let brute = brute_force_neg_ll(&timestamps, end_time, &mu, &alpha, beta);
        assert_relative_eq!(loss, brute, max_relative = 1e-12);
    }
}
