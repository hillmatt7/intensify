//! Power-law Hawkes kernel: φ(t) = α·(t + c)^{-(1+β)} for t > 0.
//!
//! - Indefinite integral: ∫(τ+c)^{-(1+β)} dτ = -(1/β)·(τ+c)^{-β}
//! - Definite integral 0..t: (α/β)·[c^{-β} − (t+c)^{-β}]
//! - L1 norm: (α/β)·c^{-β}  (finite for any β > 0, c > 0; can exceed 1 for heavy tails)
//! - **No recursive form** — likelihood path is O(N²).

use intensify_core::{IntensifyError, Result};

/// Power-law excitation kernel.
///
/// Mirrors `python/intensify/core/kernels/power_law.py`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "intensify._libintensify.kernels", name = "PowerLawKernel")
)]
pub struct PowerLawKernel {
    pub alpha: f64,
    pub beta: f64,
    pub c: f64,
}

impl PowerLawKernel {
    pub fn new(alpha: f64, beta: f64, c: f64) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(IntensifyError::InvalidParam(format!(
                "alpha must be positive; got {alpha}"
            )));
        }
        if beta <= 0.0 {
            return Err(IntensifyError::InvalidParam(format!(
                "beta must be positive; got {beta}"
            )));
        }
        if c <= 0.0 {
            return Err(IntensifyError::InvalidParam(format!(
                "c must be positive (singularity at t=0 otherwise); got {c}"
            )));
        }
        Ok(Self { alpha, beta, c })
    }

    /// φ(t) = α·(t + c)^{-(1+β)}
    #[inline]
    pub fn evaluate(&self, t: f64) -> f64 {
        self.alpha * (t + self.c).powf(-(1.0 + self.beta))
    }

    /// ∫₀^t φ(τ)dτ = (α/β)·[c^{-β} − (t+c)^{-β}]
    #[inline]
    pub fn integrate(&self, t: f64) -> f64 {
        let c_pow = self.c.powf(-self.beta);
        let tc_pow = (t + self.c).powf(-self.beta);
        (self.alpha / self.beta) * (c_pow - tc_pow)
    }

    /// L1 norm = (α/β)·c^{-β}.
    #[inline]
    pub fn l1_norm(&self) -> f64 {
        (self.alpha / self.beta) * self.c.powf(-self.beta)
    }

    /// Multiply α by `factor` (used for stationarity projection).
    #[inline]
    pub fn scale(&mut self, factor: f64) {
        self.alpha *= factor;
    }

    /// PowerLaw has no recursive sufficient statistic.
    #[inline]
    pub fn has_recursive_form(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rejects_invalid_params() {
        assert!(PowerLawKernel::new(0.0, 1.0, 0.5).is_err());
        assert!(PowerLawKernel::new(-0.1, 1.0, 0.5).is_err());
        assert!(PowerLawKernel::new(0.5, 0.0, 0.5).is_err());
        assert!(PowerLawKernel::new(0.5, 1.0, 0.0).is_err());
        assert!(PowerLawKernel::new(0.5, 1.0, -0.1).is_err());
    }

    #[test]
    fn evaluate_at_zero() {
        let k = PowerLawKernel::new(0.5, 0.8, 0.5).unwrap();
        // φ(0) = α·c^{-(1+β)}
        assert_relative_eq!(
            k.evaluate(0.0),
            0.5 * 0.5_f64.powf(-1.8),
            max_relative = 1e-15
        );
    }

    #[test]
    fn integral_to_inf_is_l1_norm() {
        let k = PowerLawKernel::new(0.4, 1.2, 0.3).unwrap();
        // For large t, integrate(t) → l1_norm
        assert_relative_eq!(k.integrate(1e9), k.l1_norm(), max_relative = 1e-9);
    }

    #[test]
    fn integral_matches_quadrature() {
        let k = PowerLawKernel::new(0.4, 0.8, 0.5).unwrap();
        let t_max = 5.0;
        let n = 100_000usize;
        let dt = t_max / n as f64;
        let mut quad = 0.0_f64;
        for i in 0..n {
            let a = i as f64 * dt;
            let b = (i + 1) as f64 * dt;
            quad += 0.5 * (k.evaluate(a) + k.evaluate(b)) * dt;
        }
        assert_relative_eq!(k.integrate(t_max), quad, max_relative = 1e-6);
    }
}
