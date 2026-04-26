# intensify

[![PyPI](https://img.shields.io/pypi/v/intensify.svg)](https://pypi.org/project/intensify/)
[![Python](https://img.shields.io/pypi/pyversions/intensify.svg)](https://pypi.org/project/intensify/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/hillmatt7/intensify/actions/workflows/ci.yml/badge.svg)](https://github.com/hillmatt7/intensify/actions/workflows/ci.yml)

A modern Python point process library with deep Hawkes specialization —
built for quantitative finance and computational neuroscience, tested on
real spike-train recordings, and Rust-accelerated end-to-end.

```bash
pip install intensify
```

Binary wheels are published for Linux (x86_64, aarch64), macOS (Intel and
Apple Silicon), and Windows (x86_64) on Python 3.10 – 3.12. Source builds
require a Rust toolchain — install via `pip install 'intensify[fast]'`.

## Quickstart

```python
import numpy as np
import intensify as its

# Simulate some event times
events = np.array([0.1, 0.5, 1.2, 1.8, 2.3, 3.1, 3.7, 4.4])
T = 5.0

# Fit a univariate Hawkes model with exponential kernel
model = its.Hawkes(mu=0.5, kernel=its.ExponentialKernel(alpha=0.3, beta=1.5))
result = model.fit(events, T=T)

print(f"Branching ratio: {result.branching_ratio_:.3f}")
print(f"Log-likelihood:  {result.log_likelihood:.3f}")
print(f"Fitted params:   {result.flat_params()}")

# Visualize fitted intensity and diagnostics
its.plot_intensity(result)
```

## Features

- **Core processes**: homogeneous / inhomogeneous Poisson, Cox (Log-Gaussian,
  Shot-Noise), Hawkes (univariate, multivariate, marked, nonlinear /
  inhibitory).
- **Kernel family**: Exponential, Sum-of-Exponentials, Power-Law, Approximate
  Power-Law (Bacry–Muzy), Nonparametric (piecewise-constant). Every kernel
  is supported in every MLE path.
- **Inference**: MLE with hand-derived closed-form gradients (no autodiff in
  the hot path); recursive `O(N)` likelihood for exponential-family kernels;
  EM and online (streaming) updates route through the same Rust core;
  Bayesian MCMC via numpyro (optional `[bayesian]` extra).
- **Diagnostics**: time-rescaling theorem (KS + QQ on inter-compensator
  increments — the mathematically correct form), AIC/BIC, residual intensity.
- **Simulation**: Ogata thinning (general) and cluster/branching
  (Galton–Watson) — both Rust-backed.
- **Stationarity enforcement**: projected gradient for multivariate Hawkes;
  spectral radius of the kernel-norm matrix reported on every multivariate
  `FitResult`.
- **Architecture**: Rust core (`intensify._libintensify`) for every kernel,
  likelihood, gradient, simulator, and compensator. Pure-Python user API.
  Loud `ImportError` if the compiled extension is missing.

## Why intensify?

intensify and [tick][] solve partly overlapping problems. Pick the right
tool:

| Capability | intensify | [tick][] |
|---|---|---|
| Joint MLE of (μ, α, β) — fits the decay for you | ✓ | — (decay must be supplied) |
| MLE for power-law, approx-power-law, nonparametric kernels | ✓ | — |
| Marked Hawkes fit with any kernel | ✓ | — |
| Nonlinear (softplus / sigmoid / relu) Hawkes, signed kernels | ✓ | — |
| Multivariate stationarity enforcement (projected gradient) | ✓ | — |
| Time-rescaling test on inter-compensator increments | ✓ | ✓ |
| Python 3.10 – 3.12 support, prebuilt wheels | ✓ | 3.8 only, C++ build |
| Sub-millisecond fits for exp kernels with known decay | ✓ | ✓ |
| Faster than tick at every benchmarked N (decay-given mv_exp) | ✓ | — |

Concrete numbers, head-to-head: [docs/benchmarks.md](docs/benchmarks.md).
Short version, on the apples-to-apples problem (decay-given mv_exp_5d):

| N | tick (ms) | **intensify 0.3.0** (ms) | speedup |
|---:|---:|---:|---:|
| 501 | 1.0 | **0.5** | 2.0× |
| 9,271 | 6.0 | **2.4** | 2.5× |
| 91,249 | 48.0 | **22.2** | 2.2× |

Joint-decay (β fit per cell) — tick can't do this at all — went from
1100 ms to **14 ms** at N=1099 (~80× vs the 0.2.0 JAX baseline). For
kernels tick doesn't ship (power-law, nonparametric, signed, marked,
nonlinear) intensify is the only option. ISSUES.md #8 is fixed: the
nonparametric path now finishes in <1 s at N=500 (previously killed
after 7 minutes). [pyhawkes][] is no longer usable — its transitive
deps depend on APIs removed from SciPy 1.0 in 2017.

[tick]: https://github.com/X-DataInitiative/tick
[pyhawkes]: https://github.com/slinderman/pyhawkes

## Documentation

Full docs: <https://hillmatt7.github.io/intensify>

- [Getting started](docs/getting_started.md)
- User guide: [inference](docs/user_guide/inference.md),
  [kernels](docs/user_guide/kernels.md),
  [simulation](docs/user_guide/simulation.md),
  [diagnostics](docs/user_guide/diagnostics.md)
- Domain guides: quantitative finance, computational neuroscience
- [API reference](https://hillmatt7.github.io/intensify/api_reference.html)
- [Tutorials](tutorials/) (Jupyter notebooks)

## Citation

If you use intensify in academic work, please cite it:

```bibtex
@software{intensify,
  author = {Hill, Matthew},
  title  = {intensify: A modern Python point process library with deep Hawkes specialization},
  year   = {2026},
  url    = {https://github.com/hillmatt7/intensify}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable form.

## Contributing & changes

- [Contributing guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Open issues](https://github.com/hillmatt7/intensify/issues)

## License

MIT — see [LICENSE](LICENSE).
