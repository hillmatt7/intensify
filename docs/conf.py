"""Sphinx configuration for intensify."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "intensify"
copyright = "2025, Matthew Hill"
author = "Matthew Hill"
release = "0.1.0-alpha"

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
