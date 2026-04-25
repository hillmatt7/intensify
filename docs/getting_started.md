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

## Minimal Hawkes example

```python
import numpy as np
import intensify as its

model = its.UnivariateHawkes(mu=0.5, kernel=its.ExponentialKernel(alpha=0.35, beta=1.2))
events = np.asarray(model.simulate(T=40.0, seed=0))
result = model.fit(events, T=40.0, method="mle")
print(result.summary())
```

## Accessing fitted parameters

After fitting, all scalar parameter values are available via `flat_params()`:

```python
result.flat_params()
# {'mu': 0.48, 'alpha': 0.33, 'beta': 1.18}
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
