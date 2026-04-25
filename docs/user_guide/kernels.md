# Kernels

## Overview

Every Hawkes process uses a kernel (excitation function) that determines how
past events influence the current intensity. The kernel choice is the most
important modeling decision.

## Available kernels

### ExponentialKernel

$$\phi(t) = \alpha \cdot \beta \cdot e^{-\beta t}$$

| Parameter | Meaning | Typical range |
|-----------|---------|---------------|
| `alpha` | Jump size / branching ratio contribution | 0.01 -- 0.99 |
| `beta` | Decay rate (1/beta is the timescale) | 0.1 -- 100 |

- **L1 norm**: `alpha` (must be < 1 for stationarity)
- **Recursive**: Yes -- O(N) likelihood
- **Use when**: Single timescale, real-time applications, neural membrane decay

Set `allow_signed=True` for inhibitory processes (negative alpha). This is
required for `NonlinearHawkes` with link functions like softplus or sigmoid.

### SumExponentialKernel

$$\phi(t) = \sum_k \alpha_k \cdot \beta_k \cdot e^{-\beta_k t}$$

- Multiple decay timescales in a single kernel
- **Recursive**: Yes -- O(N) per component
- **Use when**: Multi-timescale dynamics (fast synaptic + slow modulatory)

### ApproxPowerLawKernel (Bacry-Muzy)

$$\phi(t) = \alpha \cdot \sum_k w_k \cdot e^{-\beta_k t}$$

Approximates power-law decay using geometrically spaced exponential components.

| Parameter | Meaning |
|-----------|---------|
| `alpha` | Overall amplitude |
| `r` | Ratio between successive decay rates |
| `n_components` | Number of exponential components |

- **Recursive**: Yes -- O(N), the key advantage over `PowerLawKernel`
- **Use when**: Long memory behavior at scale (HFT order flow, large datasets)

### PowerLawKernel

$$\phi(t) = \alpha \cdot (t + c)^{-(1+\beta)}$$

| Parameter | Meaning |
|-----------|---------|
| `alpha` | Amplitude |
| `c` | Offset (prevents singularity at t=0; must be > 0) |
| `beta` | Tail exponent |

- **Recursive**: No -- O(N^2)
- **Use when**: Heavy-tailed influence, small datasets where O(N^2) is acceptable
- For large N, prefer `ApproxPowerLawKernel`

### NonparametricKernel

$$\phi(t) = \sum_k a_k \cdot \mathbf{1}[t \in [\tau_k, \tau_{k+1})]$$

Piecewise-constant kernel with no parametric assumption.

| Parameter | Meaning |
|-----------|---------|
| `n_bins` | Number of bins |
| `max_lag` | Maximum time lag |

- **Recursive**: No -- O(N^2)
- **Performance**: Very slow for N > 100 due to O(N^2) per iteration.
  Consider using EM inference instead of MLE.
- **Use when**: Exploratory analysis, unknown kernel shape

## Computation path dispatch

The MLE engine selects the computation path automatically:

| `has_recursive_form()` | Path | Complexity |
|------------------------|------|-----------|
| `True` | Recursive likelihood via `recursive_state_update()` | O(N) |
| `False` | General likelihood via pairwise lag matrix | O(N^2) |

You never need to manage this manually -- the kernel declares its capability
and the inference engine dispatches accordingly.

## Choosing a kernel

| Scenario | Recommended | Reason |
|----------|------------|--------|
| HFT order flow, live trading | `ExponentialKernel` | Single timescale, O(N), Markov |
| Multi-timescale order flow | `SumExponentialKernel` | Multiple decay rates, O(N) |
| Market impact (large N) | `ApproxPowerLawKernel` | Long memory, O(N) |
| Market impact (small N) | `PowerLawKernel` | Exact long memory, O(N^2) ok |
| Neural EPSP / within-region | `ExponentialKernel` | Matches biophysical membrane decay |
| Cross-region neural | `PowerLawKernel` | Heavy-tailed inter-region influence |
| Unknown / exploratory | `NonparametricKernel` | Data-driven, use with EM |

## Stationarity

For a Hawkes process to be stationary, the L1 norm of the kernel must be < 1
(univariate) or the spectral radius of the kernel norm matrix must be < 1
(multivariate). The library enforces this via `project_params()` after
optimization. If the optimizer finds a non-stationary solution, the parameters
are projected back to the stationary manifold with a warning.
