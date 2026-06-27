# Contributing to intensify

Thank you for considering a contribution. This project aims to be a reliable,
well-tested building block for point-process research in labs and industry.

## Ground rules

- **Correctness first.** Every numerical change should ship with a test that
  asserts the quantity it affects, not just that the code runs.
- **No silent fallbacks.** If a code path can produce a wrong answer under
  certain conditions (non-stationary fit, degenerate data, optimizer stall),
  surface it as a `warning` or `raise`, never as a silent sentinel.
- **Keep the Rust boundary clean.** Performance-critical paths route through
  the compiled `intensify._libintensify` extension. Validate inputs on the
  Python side before crossing into Rust, and keep any pure-Python fallback in
  sync with the compiled path.

## Development setup

The core is a Rust extension built with [maturin](https://www.maturin.rs/),
so a stable Rust toolchain is required (`rust-toolchain.toml` pins the
channel). An editable install compiles the extension on the fly.

```bash
git clone https://github.com/hillmatt7/intensify
cd intensify

# Create a virtualenv and install with dev + docs extras
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"

pre-commit install
```

## Running checks

These mirror the CI jobs in `.github/workflows/ci.yml`:

```bash
# Python lint + format
ruff check python/ tests/
ruff format --check python/ tests/

# Rust lint + format
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

# Tests
pytest                                                  # Python (with coverage)
cargo nextest run --workspace --exclude intensify-pyo3  # Rust

# Docs
sphinx-build -W -b html docs docs/_build/html
```

## Pull request checklist

- [ ] Tests cover the new or changed behavior.
- [ ] `pytest` passes locally (no new warnings unless intentional).
- [ ] `ruff check python/ tests/` is clean; Rust changes pass `cargo clippy`.
- [ ] Public API changes are documented in `CHANGELOG.md` under `## [Unreleased]`.
- [ ] If you changed math, a citation (paper or textbook chapter) appears in
      the docstring or in-line comment.

## Reporting bugs

Open an issue at
<https://github.com/hillmatt7/intensify/issues/new/choose>.
Please include a minimal reproducer, the observed vs expected behavior,
your Python version, and `intensify.__version__`.

## Security

See [`SECURITY.md`](SECURITY.md) for reporting security issues privately.
