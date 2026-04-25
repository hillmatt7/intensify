# Frozen reference implementations (dev-only, JAX-based)

This directory holds frozen copies of the JAX-based likelihood, kernel, and
simulation implementations that pre-date the Rust port. They serve as
**cross-validation oracles** for `tests/test_rust_*.py` — assertions of the
form `rust_loglik(args) ≈ jax_loglik(args)` to 1e-10 across many seeds.

## Rules

1. **Nothing under `python/intensify/` may import from this directory.** It
   is dev-only and not packaged in any wheel. JAX is not a runtime fallback.
2. **Modules here are frozen.** Once a Phase populates a reference module,
   it does not change unless the cross-validation reveals a bug in the
   reference itself.
3. **JAX is a `[dev]` dependency only.** A user running `pip install
   intensify` will not see this directory or pull JAX.

## Contents

Populated incrementally as each Phase needs an oracle for its Rust port.

- (Phase 1) `kernel_exponential_jax.py` — frozen ExponentialKernel JAX evaluator
- (Phase 1) `mle_uni_exp_jax.py` — frozen `_recursive_likelihood` JAX impl
- (Phase 1) `mle_mv_exp_recursive_jax.py` — frozen `_neg_ll_mv_exp_recursive`
- (Phase 2) `mle_mv_exp_dense_jax.py` — frozen `_neg_ll_mv_exp` (joint-decay)
- (Phase 2) `mle_general_jax.py` — frozen `_general_likelihood`
- (Phase 3) per-kernel JAX evaluators + compensators

## Usage in tests

```python
# tests/test_rust_uni_exp.py
import numpy as np
from intensify._libintensify.likelihood import uni_exp_neg_ll_with_grad as rust_fn
from tests._reference.mle_uni_exp_jax import recursive_likelihood as jax_fn

def test_uni_exp_matches_jax_reference(seed_idx):
    rng = np.random.default_rng(seed_idx)
    times, T, mu, alpha, beta = _gen_seed(rng)
    rust_val, _ = rust_fn(times, T, mu, alpha, beta)
    jax_val = jax_fn(times, T, mu, alpha, beta)
    assert abs(rust_val - jax_val) < 1e-10
```
