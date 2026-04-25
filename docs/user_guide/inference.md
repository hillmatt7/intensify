# Inference

## Available methods

| Method | Description | Kernels | Backend |
|--------|-------------|---------|---------|
| `method="mle"` | Maximum likelihood via L-BFGS-B | All | JAX or NumPy |
| `method="em"` | EM algorithm (branching structure) | Exponential | JAX recommended |
| `method="online"` | Streaming SGD | Recursive only | NumPy |
| `method="bayesian"` | NUTS via NumPyro | Exponential | JAX (requires `numpyro`) |

## MLE inference

### Computation paths

The MLE engine automatically selects the likelihood computation path based on
`kernel.has_recursive_form()`:

- **Recursive (O(N))**: `ExponentialKernel`, `SumExponentialKernel`, `ApproxPowerLawKernel`
- **General (O(N^2))**: `PowerLawKernel`, `NonparametricKernel`, any custom kernel

A `PerformanceWarning` is emitted when the general path is used with more than
50,000 events (configurable via `its.config_set("recursive_warning_threshold", n)`).

### Stationarity enforcement

After optimization, `project_params()` is called to ensure the branching ratio
stays below 1.0. For univariate models, this scales `alpha` down if needed. For
multivariate, it scales each row of the kernel matrix.

### Basic usage

```python
import intensify as its

model = its.Hawkes(mu=0.5, kernel=its.ExponentialKernel(alpha=0.3, beta=1.5))
result = model.fit(events, T=100.0, method="mle")
```

### Regularized multivariate MLE

For multivariate models with many dimensions, L1 regularization on the
connectivity matrix promotes sparsity:

```python
import intensify as its

result = model.fit(
    events_by_dim, T=100.0, method="mle",
    regularization=its.L1(strength=0.01, off_diagonal_only=True)
)
```

`ElasticNet` combines L1 and L2 penalties:

```python
result = model.fit(
    events_by_dim, T=100.0, method="mle",
    regularization=its.ElasticNet(strength=0.01, l1_ratio=0.5)
)
```

Strings are accepted as shorthand and resolve to defaults:

```python
result = model.fit(events_by_dim, T=100.0, regularization="l1")
result = model.fit(events_by_dim, T=100.0, regularization="elasticnet")
```

## Interpreting `FitResult`

### Parameters

```python
result.flat_params()        # {'mu': 0.48, 'alpha': 0.33, 'beta': 1.18}
result.params               # raw dict with kernel objects
result.process.kernel.alpha # direct attribute access
```

### Diagnostics on the result

```python
result.branching_ratio_     # L1 norm or spectral radius
result.endogeneity_index_   # n/(1+n), fraction from self-excitation
result.log_likelihood
result.aic, result.bic
result.std_errors           # {'mu': 0.05, 'alpha': 0.03, 'beta': 0.12}
```

### Convergence information

```python
result.convergence_info
# {'iterations': 42, 'success': True, 'message': 'CONVERGENCE...', 'backend': 'numpy'}
```

If `convergence_info["success"]` is `False`, the optimizer did not converge and
the parameters may be unreliable. A warning is emitted automatically for
multivariate fits that fail to converge.

### Standard errors

Standard errors are computed from the Hessian at the MLE (finite-difference for
NumPy backend, exact `jax.hessian` for JAX). They are only computed when:

- The optimizer converged (`success=True`)
- The number of parameters is small enough (up to 12 for univariate, 24 for multivariate)

A condition number check warns when the Hessian is ill-conditioned.

## When to use each method

- **MLE**: Default choice. Fast, consistent estimates. Use for most problems.
- **EM**: More stable for nonparametric kernels. Slower convergence in high dimensions.
- **Online**: Real-time parameter tracking as events arrive. Approximate.
- **Bayesian**: Full posterior with uncertainty quantification. Requires NumPyro.

## Multivariate workflow

```python
model = its.MultivariateHawkes(n_dims=M, mu=mu_list, kernel=kernel_matrix)
result = model.fit(events_by_dim, T=T, method="mle")

W = result.connectivity_matrix()        # M x M L1 norm matrix
sig = result.significant_connections()   # M x M boolean mask
its.plot_connectivity(result)
```

## Marked Hawkes workflow

```python
model = its.MarkedHawkes(mu=1.0, kernel=its.ExponentialKernel(alpha=0.3, beta=1.0))
result = model.fit(events, marks, T=T)
# or: result = model.fit((events, marks), T=T)
```

## Kernel coverage by process

Since 0.1.1 every MLE path accepts every univariate-kernel type:

| Process | Exponential | SumExp | PowerLaw | ApproxPL | Nonparametric |
|---|---|---|---|---|---|
| `UnivariateHawkes` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `MultivariateHawkes` | ✓ | — | — | — | — |
| `MarkedHawkes` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `NonlinearHawkes` | ✓ (signed or not) | ✓ | ✓ | ✓ | ✓ |

Multivariate Hawkes still uses a homogeneous `ExponentialKernel` grid — see
the roadmap for mixed-kernel multivariate support.
