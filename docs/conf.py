"""Sphinx configuration for intensify."""

from pathlib import Path
import tomllib

project = "intensify"
copyright = "2026, Matthew Hill"
author = "Matthew Hill"

_ROOT = Path(__file__).resolve().parents[1]
with (_ROOT / "pyproject.toml").open("rb") as f:
    release = tomllib.load(f)["project"]["version"]
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

autodoc_member_order = "bysource"
napoleon_google_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
