//! Nonparametric piecewise-constant Hawkes kernel.
//!
//! φ(t) = values[k]   for  t ∈ [edges[k], edges[k+1])
//! φ(t) = 0           for  t ≥ edges[K]    (kernel has finite support)
//!
//! `edges` has length K+1 with edges[0] = 0, strictly increasing.
//! `values` has length K, non-negative bin heights.

use intensify_core::{IntensifyError, Result};

/// Nonparametric (piecewise-constant) excitation kernel.
///
/// Mirrors `python/intensify/core/kernels/nonparametric.py`.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(
        module = "intensify._libintensify.kernels",
        name = "NonparametricKernel"
    )
)]
pub struct NonparametricKernel {
    /// Bin edges: edges[0]=0 < edges[1] < ... < edges[K]. Length K+1.
    pub edges: Vec<f64>,
    /// Bin heights ≥ 0. Length K.
    pub values: Vec<f64>,
}

impl NonparametricKernel {
    pub fn new(edges: Vec<f64>, values: Vec<f64>) -> Result<Self> {
        if edges.is_empty() {
            return Err(IntensifyError::InvalidParam(
                "edges must be non-empty".into(),
            ));
        }
        if edges[0] != 0.0 {
            return Err(IntensifyError::InvalidParam(format!(
                "edges[0] must be 0.0; got {}",
                edges[0]
            )));
        }
        for k in 0..edges.len() - 1 {
            if edges[k].partial_cmp(&edges[k + 1]) != Some(std::cmp::Ordering::Less) {
                return Err(IntensifyError::InvalidParam(format!(
                    "edges must be strictly increasing; got edges[{}]={}, edges[{}]={}",
                    k,
                    edges[k],
                    k + 1,
                    edges[k + 1]
                )));
            }
        }
        if edges.len() != values.len() + 1 {
            return Err(IntensifyError::InvalidParam(format!(
                "len(edges) ({}) must be len(values)+1 ({})",
                edges.len(),
                values.len() + 1
            )));
        }
        for (i, &v) in values.iter().enumerate() {
            if v < 0.0 || !v.is_finite() {
                return Err(IntensifyError::InvalidParam(format!(
                    "values[{i}] must be a non-negative finite number; got {v}"
                )));
            }
        }
        Ok(Self { edges, values })
    }

    pub fn n_bins(&self) -> usize {
        self.values.len()
    }

    /// φ(t) = values[bin] where bin contains t; 0 if t ≥ last edge.
    #[inline]
    pub fn evaluate(&self, t: f64) -> f64 {
        match self.bin_index(t) {
            Some(k) => self.values[k],
            None => 0.0,
        }
    }

    /// ∫₀^t φ(τ)dτ — sum of partial bins up to t.
    pub fn integrate(&self, t: f64) -> f64 {
        let mut total = 0.0_f64;
        for k in 0..self.n_bins() {
            let lo = self.edges[k];
            let hi = self.edges[k + 1];
            if lo >= t {
                break;
            }
            let upper = if hi <= t { hi } else { t };
            total += self.values[k] * (upper - lo);
        }
        total
    }

    /// L1 norm = Σ values[k]·(edges[k+1] - edges[k]).
    pub fn l1_norm(&self) -> f64 {
        let mut total = 0.0_f64;
        for k in 0..self.n_bins() {
            total += self.values[k] * (self.edges[k + 1] - self.edges[k]);
        }
        total
    }

    /// Multiply all bin heights by `factor`. Used for stationarity projection.
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.values {
            *v *= factor;
        }
    }

    /// Returns Some(bin_index) if `t ∈ [edges[k], edges[k+1])` for some k,
    /// or None if `t < 0` or `t ≥ edges[K]`.
    #[inline]
    pub fn bin_index(&self, t: f64) -> Option<usize> {
        if t < 0.0 || t >= self.edges[self.n_bins()] {
            return None;
        }
        // Binary search: find largest k with edges[k] ≤ t.
        // partition_point returns the first index where the predicate fails.
        // edges is sorted ascending so partition_point(|&e| e <= t) gives k+1
        // where edges[k] is the last one ≤ t.
        let kp1 = self.edges.partition_point(|&e| e <= t);
        // kp1 is in [1, n_bins+1] because edges[0]=0 ≤ t (we checked t ≥ 0)
        // and t < edges[n_bins] (we checked above).
        Some(kp1 - 1)
    }

    /// Always false — nonparametric kernel has no recursive sufficient statistic.
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
    fn rejects_invalid_edges() {
        // edges[0] != 0
        assert!(NonparametricKernel::new(vec![0.5, 1.0], vec![0.1]).is_err());
        // not strictly increasing
        assert!(NonparametricKernel::new(vec![0.0, 1.0, 1.0], vec![0.1, 0.1]).is_err());
        assert!(NonparametricKernel::new(vec![0.0, 2.0, 1.0], vec![0.1, 0.1]).is_err());
        // mismatched lengths
        assert!(NonparametricKernel::new(vec![0.0, 1.0, 2.0], vec![0.1]).is_err());
        assert!(NonparametricKernel::new(vec![0.0, 1.0], vec![]).is_err());
        // negative values
        assert!(NonparametricKernel::new(vec![0.0, 1.0, 2.0], vec![-0.1, 0.2]).is_err());
    }

    #[test]
    fn bin_index_lookup() {
        let k = NonparametricKernel::new(vec![0.0, 0.5, 1.0, 2.5, 4.0], vec![0.4, 0.3, 0.2, 0.1])
            .unwrap();
        assert_eq!(k.bin_index(0.0), Some(0));
        assert_eq!(k.bin_index(0.49), Some(0));
        assert_eq!(k.bin_index(0.5), Some(1));
        assert_eq!(k.bin_index(0.99), Some(1));
        assert_eq!(k.bin_index(1.0), Some(2));
        assert_eq!(k.bin_index(2.4), Some(2));
        assert_eq!(k.bin_index(2.5), Some(3));
        assert_eq!(k.bin_index(3.99), Some(3));
        assert_eq!(k.bin_index(4.0), None);
        assert_eq!(k.bin_index(5.0), None);
        assert_eq!(k.bin_index(-0.1), None);
    }

    #[test]
    fn evaluate_piecewise() {
        let k = NonparametricKernel::new(vec![0.0, 1.0, 2.0], vec![0.5, 0.2]).unwrap();
        assert_eq!(k.evaluate(0.0), 0.5);
        assert_eq!(k.evaluate(0.5), 0.5);
        assert_eq!(k.evaluate(1.0), 0.2);
        assert_eq!(k.evaluate(1.99), 0.2);
        assert_eq!(k.evaluate(2.0), 0.0);
        assert_eq!(k.evaluate(5.0), 0.0);
    }

    #[test]
    fn integrate_full_and_partial_bins() {
        let k = NonparametricKernel::new(vec![0.0, 1.0, 3.0, 5.0], vec![0.4, 0.3, 0.2]).unwrap();
        // Whole first bin
        assert_relative_eq!(k.integrate(1.0), 0.4 * 1.0, max_relative = 1e-15);
        // Whole first + half of second
        assert_relative_eq!(
            k.integrate(2.0),
            0.4 * 1.0 + 0.3 * 1.0,
            max_relative = 1e-15
        );
        // All of first two + half of third
        assert_relative_eq!(
            k.integrate(4.0),
            0.4 * 1.0 + 0.3 * 2.0 + 0.2 * 1.0,
            max_relative = 1e-15,
        );
        // Beyond last edge: just the L1 norm
        assert_relative_eq!(k.integrate(100.0), k.l1_norm(), max_relative = 1e-15);
    }

    #[test]
    fn l1_norm_sums_bin_areas() {
        let k = NonparametricKernel::new(vec![0.0, 0.5, 1.5, 2.0], vec![0.3, 0.2, 0.1]).unwrap();
        // 0.3 * 0.5 + 0.2 * 1.0 + 0.1 * 0.5 = 0.15 + 0.2 + 0.05 = 0.4
        assert_relative_eq!(k.l1_norm(), 0.4, max_relative = 1e-15);
    }

    #[test]
    fn scale_multiplies_all_values() {
        let mut k = NonparametricKernel::new(vec![0.0, 1.0, 2.0], vec![0.4, 0.2]).unwrap();
        k.scale(0.5);
        assert_eq!(k.values, vec![0.2, 0.1]);
    }
}
