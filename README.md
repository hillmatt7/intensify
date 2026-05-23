# intensify

[![PyPI](https://img.shields.io/pypi/v/intensify.svg)](https://pypi.org/project/intensify/)
[![Python](https://img.shields.io/pypi/pyversions/intensify.svg)](https://pypi.org/project/intensify/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/hillmatt7/intensify/actions/workflows/ci.yml/badge.svg)](https://github.com/hillmatt7/intensify/actions/workflows/ci.yml)

A Python library for point process modeling — Poisson, Cox, and Hawkes — with
a Rust-backed implementation of the likelihood, gradient, and simulation hot
paths. Provides closed-form gradients, Ogata thinning, cluster simulation,
and time-rescaling goodness-of-fit diagnostics.

```bash
pip install intensify
```

Binary wheels are published for Linux (x86_64, aarch64), macOS (Intel and
Apple Silicon), and Windows (x86_64) on Python 3.10–3.12. Source builds
require a Rust toolchain; install via `pip install 'intensify[fast]'`.

## Quickstart

```python
import numpy as np
import intensify as its

# Simulate event times from a self-exciting process.
model = its.Hawkes(mu=0.6, kernel=its.ExponentialKernel(alpha=0.55, beta=1.4))
events = model.simulate(T=80.0, seed=1)

# Fit mu, alpha, and beta jointly from the observed events.
result = model.fit(events, T=80.0)

print(f"Branching ratio: {result.branching_ratio_:.3f}")
print(f"Log-likelihood:  {result.log_likelihood:.3f}")
print(f"Fitted params:   {result.flat_params()}")

# Visualize the fitted intensity.
fig = its.plot_intensity(result)
fig.savefig("quickstart_intensity.png", dpi=160)
```

Representative output:

```text
Branching ratio: 0.547
Log-likelihood:  -66.671
Fitted params:   {'mu': 0.6271952244498643, 'alpha': 0.5470266035451343, 'beta': 1.0562059205150198}
```

![Fitted Hawkes conditional intensity](docs/_static/quickstart_intensity.png)

Additional workflows:

```python
# Multivariate connectivity: estimate directed excitation strengths.
kernels = [
    [its.ExponentialKernel(0.20, 1.0), its.ExponentialKernel(0.05, 1.0)],
    [its.ExponentialKernel(0.10, 1.0), its.ExponentialKernel(0.25, 1.0)],
]
mh = its.MultivariateHawkes(n_dims=2, mu=[0.5, 0.6], kernel=kernels)
mv_events = mh.simulate(T=100.0, seed=4)
mv_result = mh.fit(mv_events, T=100.0, fit_decay=False)
print(mv_result.connectivity_matrix())

# Goodness of fit: time-rescaling theorem residuals.
from intensify.core.diagnostics.goodness_of_fit import time_rescaling_test

ks_stat, p_value = time_rescaling_test(result)
print(f"KS stat={ks_stat:.3f}, p={p_value:.3f}")

# Cox process: latent, time-varying intensity for event-stream data.
lgcp = its.LogGaussianCoxProcess(n_bins=80, mu_prior=-0.2, sigma_prior=0.6)
events = lgcp.simulate(T=10.0, seed=11)
print(len(events), "events from an LGCP prior sample")
```

## Features

- **Process families**:
  - Poisson: `HomogeneousPoisson`, `InhomogeneousPoisson` (callable
    intensity or piecewise-constant rates).
  - Cox: `LogGaussianCoxProcess` (LGCP), `ShotNoiseCoxProcess`.
  - Hawkes: univariate, multivariate, marked, nonlinear (softplus, sigmoid,
    relu, identity links), multivariate-nonlinear, and signed (inhibitory).
- **Kernels**: exponential, sum-of-exponentials, power-law, approximate
  power-law (Bacry–Muzy), and nonparametric (piecewise-constant). Each
  kernel is supported across all MLE code paths.
- **Inference**: MLE with hand-derived closed-form gradients (no autodiff
  in the hot path); recursive O(N) likelihood for exponential-family
  kernels; EM and online (streaming) updates routed through the same Rust
  core; optional Bayesian MCMC via NumPyro (`[bayesian]` extra).
- **Diagnostics**: time-rescaling theorem (KS and QQ on inter-compensator
  increments), AIC/BIC, and residual intensity.
- **Simulation**: Ogata thinning (general) and cluster/branching
  (Galton–Watson), both implemented in Rust.
- **Stationarity enforcement**: projected gradient for multivariate
  Hawkes; the spectral radius of the kernel-norm matrix is stored as
  `branching_ratio_` on every multivariate `FitResult`.
- **Architecture**: Rust core (`intensify._libintensify`) for kernel,
  likelihood, gradient, and simulation hot paths. Pure-Python user-facing
  API. The package raises `ImportError` at import time if the compiled
  extension is unavailable.

## Comparison with `tick`

[tick][] is the established Python library for Hawkes-process inference.
intensify covers a broader set of process families and kernels, and its
exponential-Hawkes path is competitive with `tick` on wall-clock time
while requiring less manual configuration. The summary below is intended
as a quick orientation for users choosing between the two; both libraries
remain valuable, and `tick` is a sensible choice when its feature set is
sufficient.

### Feature coverage

| Capability | intensify | [tick][] |
|---|---|---|
| Inhomogeneous Poisson (arbitrary rate function or piecewise-constant) | yes | simulation only |
| Log-Gaussian Cox Process (LGCP) | yes | — |
| Shot-Noise Cox Process | yes | — |
| Joint MLE of (μ, α, β); Hawkes decay fit from data | yes | decay must be supplied |
| MLE for power-law, approximate-power-law, and nonparametric kernels | yes | — |
| Marked Hawkes fit with any kernel | yes | — |
| Nonlinear (softplus/sigmoid/relu) Hawkes; signed kernels | yes | — |
| Multivariate stationarity enforcement (projected gradient) | yes | — |
| Time-rescaling test on inter-compensator increments | yes | yes |
| Python 3.10–3.12 support, prebuilt wheels | yes | 3.8 only, C++ build |

### Performance

The benchmark suite builds seeded synthetic Hawkes datasets, fits the same
model repeatedly, and reports median wall time over three runs. The
comparison with `tick` locks the exponential decay `β`, because `tick`
requires the user to supply it; intensify also runs in joint-decay mode,
where `β` is estimated from data. Full methodology and reproduction
commands are in [docs/benchmarks.md](docs/benchmarks.md).

Multivariate exponential, decay-given (`mv_exp_5d`), median wall time:

| N | tick (ms) | intensify 0.3.1 (ms) |
|---:|---:|---:|
| 501 | 1.0 | 0.5 |
| 2,249 | 2.0 | 0.8 |
| 9,271 | 6.0 | 2.4 |
| 27,519 | 15.0 | 6.9 |
| 91,249 | 48.0 | 22.2 |

Parameter-recovery RMSE on the same problems is within 0.01 of `tick` at
every N, and slightly lower at the larger sizes (full table in
[docs/benchmarks.md](docs/benchmarks.md)).

In joint-decay mode — where the kernel decay is fit alongside `μ` and
`α` rather than supplied by the user — `mv_exp_5d` at N=1099 runs in
about 14 ms. This is the path most lab users want, since avoiding a
separate cross-validation loop over `β` is one of the practical
motivations for the library. `tick` does not provide a comparable mode.

For kernels outside the exponential family — power-law,
approximate-power-law, nonparametric, signed, marked, and nonlinear —
intensify provides MLE paths that `tick` does not. The nonparametric
path is viable at modest N: a 500-event fit completes in under one
second after the binary-search bin lookup introduced in 0.3.0
(see [ISSUES.md][issues] #8).

[tick]: https://github.com/X-DataInitiative/tick
[pyhawkes]: https://github.com/slinderman/pyhawkes
[issues]: ISSUES.md

## Documentation

Full documentation: <https://hillmatt7.github.io/intensify>

- [Getting started](docs/getting_started.md)
- User guide: [inference](docs/user_guide/inference.md),
  [kernels](docs/user_guide/kernels.md),
  [simulation](docs/user_guide/simulation.md),
  [diagnostics](docs/user_guide/diagnostics.md)
- [API reference](https://hillmatt7.github.io/intensify/api_reference.html)
- [Tutorials](tutorials/) (Jupyter notebooks)

## Citation

If you use intensify in academic work, please cite it:

```bibtex
@software{intensify,
  author = {Hill, Matthew},
  title  = {intensify: A Python point process library with Rust-backed inference},
  year   = {2026},
  url    = {https://github.com/hillmatt7/intensify}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable form.

## Contributing and changes

- [Contributing guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Open issues](https://github.com/hillmatt7/intensify/issues)

## License

MIT — see [LICENSE](LICENSE).
