//! Univariate nonlinear Hawkes negative log-likelihood with closed-form
//! gradient (under ExponentialKernel pre-intensity + builtin link function).
//!
//! Model:
//!   z(t)  = μ + Σ_{j: t_j < t} α·β·exp(-β·(t - t_j))   [linear pre-intensity]
//!   λ(t)  = link(z(t))                                  [nonneg intensity]
//!
//! Free parameters: μ, α, β. Link function is fixed at construction.
//! The compensator C = ∫₀^T link(z(s))ds has no closed form for any
//! non-trivial link, so we approximate it with **trapezoidal rule on a
//! uniform grid of `n_quad` points** — matching the existing Python
//! reference for cross-validation.
//!
//! ### Recursive pre-intensity
//!
//! Same R recursion as uni_exp:
//!   R_{i+1}^pre = exp(-β·dt_{i+1}) · (R_i^pre + 1)
//!   z_i = μ + α·β·R_i^pre   at each event
//!
//! Between events, R(s) decays continuously: for s ∈ (t_i, t_{i+1}),
//!   R(s) = exp(-β·(s - t_i)) · R_i^post   where R_i^post = R_i^pre + 1.
//! For grid points on `[0, t_0]`, R(s) = 0.
//!
//! ### Link functions
//!
//! - **softplus**:    link(z) = log(1 + e^z),  link'(z) = sigmoid(z)
//! - **relu**:        link(z) = max(0, z),     link'(z) = 1[z>0]
//! - **sigmoid_5**:   link(z) = 5 / (1 + e^{-z}),  link'(z) = link(z)·(1 - link(z)/5)
//! - **identity_pos**: link(z) = max(z, 1e-12), link'(z) = 1[z>1e-12]
//!
//! ### Closed-form gradient
//!
//! For the log-intensity sum: ∂log λ_i/∂param = link'(z_i) / λ_i · ∂z_i/∂param
//! For the compensator: ∂C/∂param = ∫ link'(z(s)) · ∂z(s)/∂param ds
//!
//! ∂z/∂μ = 1; ∂z/∂α = β·R; ∂z/∂β = α·(R + β·∂R/∂β)
//! Track ∂R/∂β through the same recursion as in uni_exp.

use intensify_core::{IntensifyError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkKind {
    Softplus,
    Relu,
    Sigmoid { scale: f64 },
    IdentityPos,
}

impl LinkKind {
    /// link(z)
    #[inline]
    fn apply(self, z: f64) -> f64 {
        match self {
            Self::Softplus => {
                if z > 35.0 {
                    z
                } else {
                    (1.0 + z.exp()).ln()
                }
            }
            Self::Relu => z.max(0.0),
            Self::Sigmoid { scale } => {
                let z_clip = z.clamp(-40.0, 40.0);
                scale / (1.0 + (-z_clip).exp())
            }
            Self::IdentityPos => z.max(1e-12),
        }
    }

    /// link'(z)
    #[inline]
    fn derivative(self, z: f64) -> f64 {
        match self {
            Self::Softplus => 1.0 / (1.0 + (-z).exp()),
            Self::Relu => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Sigmoid { scale } => {
                let s = self.apply(z);
                s * (1.0 - s / scale)
            }
            Self::IdentityPos => {
                if z > 1e-12 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Compute (z_at_grid, dR_dbeta_at_grid, R_at_grid) for the uniform grid of
/// `n_quad` points on `[0, T]`. Each grid point's pre-intensity is derived
/// from the recursive R state evolved across the events it has seen so far.
fn compute_grid_states(
    times: &[f64],
    t_horizon: f64,
    beta: f64,
    n_quad: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_q = n_quad.max(8);
    let grid: Vec<f64> = (0..n_q)
        .map(|i| t_horizon * (i as f64) / ((n_q - 1) as f64))
        .collect();

    // R_post and ∂R_post/∂β maintained through events; for each grid point
    // we decay from the most recent absorbed R_post by (s - t_last_event).
    let mut r_post = 0.0_f64;
    let mut dr_post_db = 0.0_f64;
    let mut t_last_event = 0.0_f64; // time of most recent absorb (or 0 before any events)

    let mut event_idx = 0_usize;
    let n = times.len();

    let mut r_at_grid = vec![0.0_f64; n_q];
    let mut dr_db_at_grid = vec![0.0_f64; n_q];

    for (g, &s) in grid.iter().enumerate() {
        // Process any events whose time has passed s (advance r_post + dr through them).
        while event_idx < n && times[event_idx] <= s {
            let t_e = times[event_idx];
            let dt = t_e - t_last_event;
            let e = (-beta * dt).exp();
            // Pre-event state at t_e:
            let r_pre = e * r_post;
            let dr_pre_db = -dt * r_pre + e * dr_post_db;
            // Absorb the event (no β dependence in +1):
            r_post = r_pre + 1.0;
            dr_post_db = dr_pre_db;
            t_last_event = t_e;
            event_idx += 1;
        }
        // R(s) = decay r_post by (s - t_last_event)
        let dt_g = s - t_last_event;
        let e_g = (-beta * dt_g).exp();
        r_at_grid[g] = e_g * r_post;
        dr_db_at_grid[g] = -dt_g * r_at_grid[g] + e_g * dr_post_db;
    }

    (grid, r_at_grid, dr_db_at_grid, /* placeholder */ vec![])
}

/// Returns `(neg_loglik, [d/dμ, d/dα, d/dβ])`.
#[allow(clippy::too_many_arguments)]
pub fn nonlinear_uni_exp_neg_ll_with_grad(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
    link: LinkKind,
    n_quad: usize,
) -> Result<(f64, [f64; 3])> {
    if t_horizon <= 0.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "t_horizon must be positive; got {t_horizon}"
        )));
    }
    if beta <= 0.0 {
        return Err(IntensifyError::InvalidParam(format!(
            "beta must be positive; got {beta}"
        )));
    }
    let n = times.len();

    // Compensator + its gradient via numerical quadrature on uniform grid.
    let (grid, r_at_grid, dr_db_at_grid, _) =
        compute_grid_states(times, t_horizon, beta, n_quad);
    let n_q = grid.len();

    // λ_grid[g] = link(z_g),  z_g = μ + α·β·R(s_g)
    // Gradient terms accumulate via trapezoid weights.
    let mut comp_val = 0.0_f64;
    let mut grad_mu_comp = 0.0_f64;
    let mut grad_alpha_comp = 0.0_f64;
    let mut grad_beta_comp = 0.0_f64;

    let mut prev_z = 0.0_f64;
    let mut prev_lam = 0.0_f64;
    let mut prev_lprime = 0.0_f64;
    let mut prev_dr_db = 0.0_f64;
    let mut prev_r = 0.0_f64;
    for g in 0..n_q {
        let r = r_at_grid[g];
        let dr_db = dr_db_at_grid[g];
        let z = mu + alpha * beta * r;
        let lam = link.apply(z);
        let lprime = link.derivative(z);

        if g > 0 {
            let h = grid[g] - grid[g - 1];
            // Trapezoid contribution:
            comp_val += 0.5 * h * (prev_lam + lam);
            // ∂C/∂μ contribution = ∫ link'(z) · 1 ds
            grad_mu_comp += 0.5 * h * (prev_lprime + lprime);
            // ∂C/∂α contribution = ∫ link'(z) · β·R ds
            grad_alpha_comp += 0.5 * h * (prev_lprime * beta * prev_r + lprime * beta * r);
            // ∂C/∂β contribution = ∫ link'(z) · α·(R + β·∂R/∂β) ds
            grad_beta_comp += 0.5 * h
                * (prev_lprime * alpha * (prev_r + beta * prev_dr_db)
                    + lprime * alpha * (r + beta * dr_db));
        }
        prev_z = z;
        prev_lam = lam;
        prev_lprime = lprime;
        prev_dr_db = dr_db;
        prev_r = r;
    }
    let _ = prev_z;

    // Log-intensity sum at events: same recursive walk as uni_exp.
    let mut log_lam_sum = 0.0_f64;
    let mut grad_mu_log = 0.0_f64;
    let mut grad_alpha_log = 0.0_f64;
    let mut grad_beta_log = 0.0_f64;

    let mut r_post = 0.0_f64;
    let mut dr_post_db = 0.0_f64;
    let mut t_prev = 0.0_f64;
    for &t_i in times {
        let dt = t_i - t_prev;
        let e = (-beta * dt).exp();
        let r_pre = e * r_post;
        let dr_pre_db = -dt * r_pre + e * dr_post_db;

        let z_i = mu + alpha * beta * r_pre;
        let lam_i = link.apply(z_i);
        let lprime_i = link.derivative(z_i);
        let lam_safe = lam_i.max(1e-30);
        log_lam_sum += lam_safe.ln();
        let inv_lam = 1.0 / lam_safe;
        grad_mu_log += lprime_i * inv_lam;
        grad_alpha_log += lprime_i * beta * r_pre * inv_lam;
        grad_beta_log += lprime_i * alpha * (r_pre + beta * dr_pre_db) * inv_lam;

        r_post = r_pre + 1.0;
        dr_post_db = dr_pre_db;
        t_prev = t_i;
    }
    let _ = n;

    let neg_ll = comp_val - log_lam_sum;
    let grad_mu = grad_mu_comp - grad_mu_log;
    let grad_alpha = grad_alpha_comp - grad_alpha_log;
    let grad_beta = grad_beta_comp - grad_beta_log;
    Ok((neg_ll, [grad_mu, grad_alpha, grad_beta]))
}

/// Value-only entry point.
pub fn nonlinear_uni_exp_neg_ll(
    times: &[f64],
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
    link: LinkKind,
    n_quad: usize,
) -> Result<f64> {
    if t_horizon <= 0.0 || beta <= 0.0 {
        return Err(IntensifyError::InvalidParam(
            "t_horizon and beta must be positive".into(),
        ));
    }
    let (grid, r_at_grid, _, _) =
        compute_grid_states(times, t_horizon, beta, n_quad);
    let n_q = grid.len();

    // Compensator via trapezoid
    let mut comp = 0.0_f64;
    for g in 1..n_q {
        let z_p = mu + alpha * beta * r_at_grid[g - 1];
        let z_c = mu + alpha * beta * r_at_grid[g];
        comp += 0.5 * (grid[g] - grid[g - 1]) * (link.apply(z_p) + link.apply(z_c));
    }

    // Log-intensity sum
    let mut r_post = 0.0_f64;
    let mut t_prev = 0.0_f64;
    let mut log_lam_sum = 0.0_f64;
    for &t_i in times {
        let r_pre = (-beta * (t_i - t_prev)).exp() * r_post;
        let z_i = mu + alpha * beta * r_pre;
        log_lam_sum += link.apply(z_i).max(1e-30).ln();
        r_post = r_pre + 1.0;
        t_prev = t_i;
    }

    Ok(comp - log_lam_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn empty_events_with_softplus() {
        // λ(t) = softplus(μ); compensator = T·softplus(μ).
        let mu = 0.5_f64;
        let val = nonlinear_uni_exp_neg_ll(
            &[], 10.0, mu, 0.0001, 1.0, LinkKind::Softplus, 256,
        ).unwrap();
        let expected_comp = 10.0 * (1.0 + mu.exp()).ln();
        assert_relative_eq!(val, expected_comp, max_relative = 1e-6);
    }

    #[test]
    fn analytic_grad_matches_finite_difference_softplus() {
        let times = [0.5, 1.1, 2.0, 3.7, 5.2];
        let T = 6.0;
        let n_quad = 1024;
        let params = [0.4, 0.3, 1.2];

        let (_, g_a) = nonlinear_uni_exp_neg_ll_with_grad(
            &times, T, params[0], params[1], params[2], LinkKind::Softplus, n_quad,
        ).unwrap();

        let h = 1e-5;
        for i in 0..3 {
            let bump = |delta: f64| -> f64 {
                let mut p = params;
                p[i] += delta;
                nonlinear_uni_exp_neg_ll(
                    &times, T, p[0], p[1], p[2], LinkKind::Softplus, n_quad,
                ).unwrap()
            };
            let f_p = bump(h);
            let f_m = bump(-h);
            let f_p2 = bump(2.0 * h);
            let f_m2 = bump(-2.0 * h);
            let g_n = (-f_p2 + 8.0 * f_p - 8.0 * f_m + f_m2) / (12.0 * h);
            // Looser tol — quadrature truncation error contributes
            assert_relative_eq!(g_a[i], g_n, max_relative = 1e-3);
        }
    }
}
