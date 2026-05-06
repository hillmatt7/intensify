# Quantitative Finance Guide

Point processes are a natural fit for order-flow data: trades, quote updates,
cancellations, and market orders are timestamped events whose future rate often
depends on recent activity.

## Order-Flow Workflow

Start by mapping each event type to one dimension. A compact two-dimensional
setup might use buy-initiated and sell-initiated market orders:

```python
import numpy as np
import intensify as its

buy_times = np.asarray([0.3, 0.8, 1.4, 2.9, 3.1])
sell_times = np.asarray([0.5, 1.7, 2.1, 3.8])
events_by_side = [buy_times, sell_times]
T = 4.0

kernels = [
    [its.ExponentialKernel(0.20, 1.5), its.ExponentialKernel(0.05, 1.5)],
    [its.ExponentialKernel(0.08, 1.5), its.ExponentialKernel(0.18, 1.5)],
]
model = its.MultivariateHawkes(n_dims=2, mu=[0.6, 0.5], kernel=kernels)
result = model.fit(events_by_side, T=T, fit_decay=False)

print(result.connectivity_matrix())
print(result.branching_ratio_)
```

Interpret the fitted connectivity matrix as directed excitation. Large diagonal
entries indicate same-side clustering; large off-diagonal entries indicate
cross-excitation between buy and sell pressure.

## Practical Notes

- Use `fit_decay=False` when you already have a decay horizon from market
  microstructure assumptions and want an apples-to-apples tick-style fit.
- Use joint decay fitting when the decay horizon is part of the research
  question or varies by instrument.
- Check `result.branching_ratio_ < 1.0` before using the process for simulation
  or stress testing.
- Compare models with AIC/BIC and residual diagnostics on held-out windows.

## Regularized Connectivity

For larger event vocabularies, sparsity helps distinguish real excitation from
noise:

```python
result = model.fit(events_by_side, T=T, regularization=its.L1(strength=0.01))
mask = result.significant_connections(significance_level=0.05)
```

The mask is a first-pass screening tool. For production research, combine it
with stability checks across dates, venues, and liquidity regimes.
