# Getting started

## Installation

Install from a checkout (with all optional dependencies):

```bash
pip install -e .[dev,docs,bayesian]
```

Or install only the core library:

```bash
pip install intensify
```

## Minimal Hawkes Example

```python
import intensify as its

model = its.Hawkes(mu=0.6, kernel=its.ExponentialKernel(alpha=0.55, beta=1.4))
events = model.simulate(T=80.0, seed=1)
result = model.fit(events, T=80.0)

print(f"Branching ratio: {result.branching_ratio_:.3f}")
print(f"Log-likelihood:  {result.log_likelihood:.3f}")
print(f"Fitted params:   {result.flat_params()}")
```

Representative output:

```text
Branching ratio: 0.547
Log-likelihood:  -66.671
Fitted params:   {'mu': 0.6271952244498643, 'alpha': 0.5470266035451343, 'beta': 1.0562059205150198}
```

The branching ratio is the expected number of triggered events per event. A
value below 1 indicates a stationary self-exciting process.

## Plot The Fitted Intensity

```python
fig = its.plot_intensity(result)
fig.savefig("quickstart_intensity.png", dpi=160)
```

![Fitted Hawkes conditional intensity](_static/quickstart_intensity.png)

## Accessing fitted parameters

After fitting, all scalar parameter values are available via `flat_params()`:

```python
result.flat_params()
# {'mu': 0.627..., 'alpha': 0.547..., 'beta': 1.056...}
```

The `params` dict on `FitResult` stores the raw objects (kernel objects, arrays)
which is useful for advanced introspection but less convenient for quick access.

## Multivariate Hawkes

```python
import intensify as its
from intensify import ExponentialKernel, MultivariateHawkes

kernels = [
    [ExponentialKernel(alpha=0.2, beta=1.0), ExponentialKernel(alpha=0.1, beta=1.0)],
    [ExponentialKernel(alpha=0.05, beta=1.0), ExponentialKernel(alpha=0.3, beta=1.0)],
]
model = MultivariateHawkes(n_dims=2, mu=[0.5, 0.8], kernel=kernels)

# events_by_dim is a list of arrays, one per dimension
result = model.fit(events_by_dim, T=100.0, method="mle")

print(f"Spectral radius (branching ratio): {result.branching_ratio_:.3f}")
W = result.connectivity_matrix()
print("Connectivity matrix:\n", W)
```

The connectivity matrix contains kernel L1 norms. Entry `(i, j)` is the
estimated excitation from source dimension `j` into target dimension `i`.

## Marked Hawkes

```python
import intensify as its

model = its.MarkedHawkes(
    mu=1.0, kernel=its.ExponentialKernel(alpha=0.3, beta=1.0), mark_power=0.5
)

# Two equivalent calling conventions:
result = model.fit(events, marks, T=100.0)
result = model.fit((events, marks), T=100.0)  # tuple form
```

## Diagnostics

```python
from intensify.core.diagnostics.goodness_of_fit import time_rescaling_test, qq_plot

ks_stat, p_value = time_rescaling_test(result)
print(f"KS test: stat={ks_stat:.4f}, p={p_value:.4f}")

# Or use the all-in-one diagnostic plot:
result.plot_diagnostics()
```

## Cox/LGCP Example

Use a Cox process when the event rate is driven by an unobserved,
time-varying intensity rather than direct self-excitation:

```python
lgcp = its.LogGaussianCoxProcess(n_bins=80, mu_prior=-0.2, sigma_prior=0.6)
spikes = lgcp.simulate(T=10.0, seed=11)
lgcp.set_last_window(10.0)

print(f"simulated spikes: {len(spikes)}")
print(f"intensity at t=5: {lgcp.intensity(5.0, spikes):.3f}")
```

## Visualization

```python
import intensify as its

its.plot_intensity(result)
its.plot_kernel(result.process.kernel)
its.plot_connectivity(result)        # for multivariate
its.plot_inter_event_intervals(events)
```

## Choosing kernels

| Use case | Kernel | Complexity |
|----------|--------|-----------|
| Single timescale decay | `ExponentialKernel` | O(N) recursive |
| Multiple timescales | `SumExponentialKernel` | O(N) recursive |
| Long memory (large N) | `ApproxPowerLawKernel` | O(N) recursive |
| Long memory (small N) | `PowerLawKernel` | O(N^2) general |
| Data-driven shape | `NonparametricKernel` | O(N^2) general |

## Regularization

Regularizers are available at the top level for multivariate fits:

```python
import intensify as its

# Explicit instance
result = model.fit(events_by_dim, T=T, regularization=its.L1(strength=0.01))

# String shorthand with default strength
result = model.fit(events_by_dim, T=T, regularization="l1")
result = model.fit(events_by_dim, T=T, regularization="elasticnet")
```

## Switching backends

JAX is the default for speed. Drop to the pure-NumPy backend any time — all
modules that cached `bt = get_backend()` at import time pick up the switch:

```python
import intensify as its
its.set_backend("numpy")
print(its.get_backend_name())  # -> 'numpy'
```

## Input validation

Every `fit()` entry point validates event arrays before running the
optimizer. An array that is unsorted, contains negative timestamps, or has
elements outside `[0, T]` raises `ValueError` immediately rather than
silently producing a wrong likelihood. If you preprocess your data
somewhere, sort and clip before calling `fit`.

## What `FitResult` contains

| Attribute | Description |
|-----------|-------------|
| `params` | Raw parameter dict (may contain kernel objects) |
| `flat_params()` | Scalar name->value dict |
| `log_likelihood` | Maximized log-likelihood |
| `aic`, `bic` | Information criteria |
| `std_errors` | Parameter standard errors (when available) |
| `branching_ratio_` | L1 norm (univariate) or spectral radius (multivariate) |
| `endogeneity_index_` | Fraction of events from self-excitation |
| `convergence_info` | Optimizer metadata |
| `process` | Fitted process object |
| `events`, `T` | Data used for fitting |

See the tutorials in the `tutorials/` directory and `user_guide/` for more.
