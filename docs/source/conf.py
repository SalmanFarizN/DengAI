# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add both project root and src directory to Python path
sys.path.insert(0, os.path.abspath("../../"))  # Point to your project root
sys.path.insert(0, os.path.abspath("../../src"))  # Point to src directory

project = "DengAI"
copyright = "2025, Your Name"
author = "Your Name"

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate from docstrings
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Support for Google/NumPy docstring styles
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
