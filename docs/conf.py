"""Sphinx configuration for the PyMC-Ray documentation."""

import shutil
from pathlib import Path

import pymc_ray

DOCS = Path(__file__).parent
ROOT = DOCS.parent

project = "PyMC-Ray"
author = "Jonathan Dekermanjian"
copyright = "2026, Jonathan Dekermanjian"
release = version = pymc_ray.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
templates_path = ["_templates"]

# The tutorial notebooks live in examples/, where they are run. Sphinx only reads files inside docs/,
# so they are copied in at build time. The copies are ignored by git.
for notebook in sorted((ROOT / "examples").glob("*.ipynb")):
    shutil.copy2(notebook, DOCS / "tutorials" / notebook.name)

# The notebooks are shown with their saved outputs. Running them would need a Ray cluster and real sampling time.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

autosummary_generate = True
# Each class page documents its methods through the class template in _templates/autosummary.
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable", None),
    "arviz": ("https://python.arviz.org/en/stable", None),
    "ray": ("https://docs.ray.io/en/latest", None),
    "narwhals": ("https://narwhals-dev.github.io/narwhals", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "PyMC-Ray"
html_theme_options = {
    "github_url": "https://github.com/Dekermanjian/pymc-ray",
    "show_toc_level": 2,
}
