# Computational Neuroscience Guide

Spike trains are event sequences, so intensify can model firing rates,
self-excitation, cross-neuron coupling, and latent intensity fluctuations with
the same API used for other point-process domains.

## Hawkes Coupling For Spike Trains

Use a multivariate Hawkes process when the scientific question is directed
interaction: does activity in neuron `j` increase or suppress the future rate
of neuron `i`?

```python
import numpy as np
import intensify as its

spikes_neuron_0 = np.asarray([0.02, 0.17, 0.41, 0.88])
spikes_neuron_1 = np.asarray([0.05, 0.21, 0.63])
events = [spikes_neuron_0, spikes_neuron_1]
T = 1.0

kernels = [
    [its.ExponentialKernel(0.10, 20.0), its.ExponentialKernel(0.04, 20.0)],
    [its.ExponentialKernel(0.06, 20.0), its.ExponentialKernel(0.12, 20.0)],
]
model = its.MultivariateHawkes(n_dims=2, mu=[2.0, 1.8], kernel=kernels)
result = model.fit(events, T=T, fit_decay=False)
print(result.connectivity_matrix())
```

The decay `beta` should match the time units of your spike timestamps. If
timestamps are in seconds, `beta=20` corresponds to a roughly 50 ms decay.

## LGCP For Latent Firing Rate

Use a Log-Gaussian Cox Process when spikes are driven by an unobserved,
smoothly varying rate rather than direct spike-to-spike excitation.

```python
lgcp = its.LogGaussianCoxProcess(n_bins=100, mu_prior=-0.5, sigma_prior=0.7)
spikes = lgcp.simulate(T=2.0, seed=7)
lgcp.set_last_window(2.0)

rate_at_one_second = lgcp.intensity(1.0, spikes)
print(f"{len(spikes)} spikes, rate at 1 s = {rate_at_one_second:.3f}")
```

LGCPs are useful for stimulus-driven or state-dependent firing where the
conditional intensity changes over time but individual spikes are not assumed
to trigger future spikes.

## Diagnostics

After fitting, run time-rescaling diagnostics:

```python
from intensify.core.diagnostics.goodness_of_fit import time_rescaling_test

ks_stat, p_value = time_rescaling_test(result)
print(ks_stat, p_value)
```

Interpret the p-value as a model comparison signal when parameters were fit on
the same data. For formal goodness-of-fit, reserve held-out trials or apply a
parametric bootstrap.
