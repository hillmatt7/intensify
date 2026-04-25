//! Exponential kernel: φ(t) = α·β·exp(-β·t) for t ≥ 0.
//!
//! - L1 norm = |α| (branching ratio for unsigned kernels)
//! - Closed-form integral: ∫₀^t φ(τ)dτ = α·(1 - exp(-β·t))
//! - Recursive form: R(t) = exp(-β·dt)·(1 + R(t⁻)) where dt = t - t_{prev}

use intensify_core::{IntensifyError, Result};

/// Exponential Hawkes excitation kernel.
///
/// Mirrors `python/intensify/core/kernels/exponential.py`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "intensify._libintensify.kernels", name = "ExponentialKernel")
)]
pub struct ExponentialKernel {
    pub alpha: f64,
    pub beta: f64,
    pub allow_signed: bool,
}

impl ExponentialKernel {
    /// Validate parameters and construct.
    ///
    /// Errors if β ≤ 0, or if α ≤ 0 when `allow_signed = false`, or
    /// if α = 0 when `allow_signed = true`.
    pub fn new(alpha: f64, beta: f64, allow_signed: bool) -> Result<Self> {
        if beta <= 0.0 {
            return Err(IntensifyError::InvalidParam(format!(
                "beta must be positive; got {beta}"
            )));
        }
        if !allow_signed && alpha <= 0.0 {
            return Err(IntensifyError::InvalidParam(format!(
                "alpha must be positive (use allow_signed=True for signed); got {alpha}"
            )));
        }
        if allow_signed && alpha == 0.0 {
            return Err(IntensifyError::InvalidParam(
                "alpha must be non-zero when allow_signed=True".into(),
            ));
        }
        Ok(Self {
            alpha,
            beta,
            allow_signed,
        })
    }

    /// φ(t) = α·β·exp(-β·t).
    #[inline]
    pub fn evaluate(&self, t: f64) -> f64 {
        self.alpha * self.beta * (-self.beta * t).exp()
    }

    /// ∫₀^t φ(τ)dτ = α·(1 - exp(-β·t)).
    #[inline]
    pub fn integrate(&self, t: f64) -> f64 {
        self.alpha * (1.0 - (-self.beta * t).exp())
    }

    /// L1 norm = |α|. The branching ratio for unsigned kernels.
    #[inline]
    pub fn l1_norm(&self) -> f64 {
        self.alpha.abs()
    }

    /// Multiply α by `factor`. Used by `MultivariateHawkes.project_params()`
    /// to project a non-stationary fit into the stationary regime.
    #[inline]
    pub fn scale(&mut self, factor: f64) {
        self.alpha *= factor;
    }

    /// Whether the recursive O(N) likelihood path applies. Signed kernels
    /// need the general O(N²) path under nonlinear links.
    #[inline]
    pub fn has_recursive_form(&self) -> bool {
        !self.allow_signed
    }

    /// Recursive sufficient statistic update: R_i = exp(-β·dt)·(1 + R_{i-1}).
    #[inline]
    pub fn recursive_state_update(&self, state: f64, dt: f64) -> f64 {
        (-self.beta * dt).exp() * (1.0 + state)
    }

    /// Decay-only update: R_dec = exp(-β·dt)·R.
    #[inline]
    pub fn recursive_decay(&self, state: f64, dt: f64) -> f64 {
        (-self.beta * dt).exp() * state
    }

    /// Post-event absorption: R⁺ = R + 1 (the new event increments the
    /// running sum of decaying contributions).
    #[inline]
    pub fn recursive_absorb(&self, state: f64) -> f64 {
        state + 1.0
    }

    /// Excitation contribution to the intensity: α·β·R.
    #[inline]
    pub fn recursive_intensity_excitation(&self, state: f64) -> f64 {
        self.alpha * self.beta * state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rejects_invalid_params() {
        assert!(ExponentialKernel::new(0.5, 0.0, false).is_err());
        assert!(ExponentialKernel::new(0.5, -1.0, false).is_err());
        assert!(ExponentialKernel::new(0.0, 1.0, false).is_err());
        assert!(ExponentialKernel::new(-0.5, 1.0, false).is_err());
        assert!(ExponentialKernel::new(0.0, 1.0, true).is_err());
    }

    #[test]
    fn accepts_signed_when_allowed() {
        let k = ExponentialKernel::new(-0.5, 1.0, true).unwrap();
        assert_eq!(k.alpha, -0.5);
        assert_eq!(k.beta, 1.0);
        assert!(k.allow_signed);
        assert!(!k.has_recursive_form());
    }

    #[test]
    fn evaluate_at_zero_is_alpha_beta() {
        let k = ExponentialKernel::new(0.3, 1.5, false).unwrap();
        assert_relative_eq!(k.evaluate(0.0), 0.3 * 1.5, max_relative = 1e-15);
    }

    #[test]
    fn integral_zero_to_inf_equals_alpha() {
        let k = ExponentialKernel::new(0.3, 1.5, false).unwrap();
        // For large t, integral → α
        assert_relative_eq!(k.integrate(1e6), 0.3, max_relative = 1e-12);
    }

    #[test]
    fn l1_norm_is_abs_alpha() {
        let k = ExponentialKernel::new(0.3, 1.5, false).unwrap();
        assert_relative_eq!(k.l1_norm(), 0.3, max_relative = 1e-15);

        let k_signed = ExponentialKernel::new(-0.3, 1.5, true).unwrap();
        assert_relative_eq!(k_signed.l1_norm(), 0.3, max_relative = 1e-15);
    }

    #[test]
    fn integral_matches_numerical_quadrature() {
        // Trapezoidal vs closed form should agree to ~6 decimals at N=10000
        let k = ExponentialKernel::new(0.4, 2.0, false).unwrap();
        let t_max = 5.0;
        let n: usize = 10_000;
        let dt = t_max / (n as f64);
        let mut quad = 0.0;
        for i in 0..n {
            let a = (i as f64) * dt;
            let b = ((i + 1) as f64) * dt;
            quad += 0.5 * (k.evaluate(a) + k.evaluate(b)) * dt;
        }
        assert_relative_eq!(k.integrate(t_max), quad, max_relative = 1e-6);
    }

    #[test]
    fn scale_multiplies_alpha() {
        let mut k = ExponentialKernel::new(0.8, 1.0, false).unwrap();
        k.scale(0.5);
        assert_relative_eq!(k.alpha, 0.4, max_relative = 1e-15);
        assert_eq!(k.beta, 1.0);
    }

    #[test]
    fn decay_then_absorb_matches_brute_force() {
        // The likelihood path uses `decay → compute λ from R_dec → absorb`.
        // The post-absorb state R(t_i) should equal the brute-force sum
        //   Σ_{j ≤ i} exp(-β·(t_i - t_j))
        // i.e. all past events including the just-absorbed one.
        let k = ExponentialKernel::new(0.4, 1.2, false).unwrap();
        let times = [0.5, 1.1, 2.0, 3.7];

        let mut r = 0.0;
        let mut last_t = 0.0;
        let mut r_after_absorb = Vec::new();
        for &t in &times {
            let dt = t - last_t;
            r = k.recursive_decay(r, dt);
            r = k.recursive_absorb(r);
            r_after_absorb.push(r);
            last_t = t;
        }

        for i in 0..times.len() {
            let brute: f64 = (0..=i).map(|j| (-k.beta * (times[i] - times[j])).exp()).sum();
            assert_relative_eq!(r_after_absorb[i], brute, max_relative = 1e-12);
        }
    }

    #[test]
    fn intensity_excitation_at_event_uses_pre_event_state() {
        // Excitation at event i should be α·β · Σ_{j < i} exp(-β·(t_i - t_j)).
        // (Strict inequality: the event itself contributes to R *after*
        // the intensity is read; pre-event R sees only past events.)
        let k = ExponentialKernel::new(0.5, 1.5, false).unwrap();
        let times = [0.3, 0.9, 1.7];

        let mut r = 0.0;
        let mut last_t = 0.0;
        let mut excitations = Vec::new();
        for &t in &times {
            let dt = t - last_t;
            r = k.recursive_decay(r, dt);
            excitations.push(k.recursive_intensity_excitation(r));
            r = k.recursive_absorb(r);
            last_t = t;
        }

        for i in 0..times.len() {
            let brute: f64 = (0..i)
                .map(|j| k.alpha * k.beta * (-k.beta * (times[i] - times[j])).exp())
                .sum();
            assert_relative_eq!(excitations[i], brute, max_relative = 1e-12);
        }
    }
}
