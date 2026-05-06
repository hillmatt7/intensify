# Tutorials

The `tutorials/` directory contains runnable notebooks that move from basic
Hawkes simulation to multivariate connectivity and nonlinear models.

- `tutorials/01_basic_hawkes.ipynb`: simulate, fit, and diagnose a univariate
  Hawkes process.
- `tutorials/02_multivariate.ipynb`: estimate multivariate excitation and
  inspect the connectivity matrix.
- `tutorials/03_marked_and_online.ipynb`: model marks and streaming updates.
- `tutorials/04_connectivity_and_nonlinear.ipynb`: compare directed
  connectivity and nonlinear links.
- `tutorials/05_lgcp_neuroscience.ipynb`: simulate latent firing-rate data
  with a Log-Gaussian Cox Process.

Run a notebook from a checkout with:

```bash
pip install -e ".[dev,docs,bayesian]"
jupyter notebook tutorials/01_basic_hawkes.ipynb
```
