# Installation

Install PyMC-Ray from the Python Package Index (PyPI) with pip:

```bash
pip install pymc-ray
```

PyMC-Ray requires Python 3.12 or later. Installing it also installs PyMC 6.3 or later and Ray 2.40 or later, if they are not already present.

## Optional packages

The [nutpie](https://github.com/pymc-devs/nutpie) sampler works with {class}`~pymc_ray.ShardedModel`, and PyMC uses nutpie by default whenever it is installed. Install it with `pip install nutpie`.

To start and manage Ray clusters, and to use the Ray dashboard, install Ray's default extras with `pip install "ray[default]"`.

## Development installation

To work on PyMC-Ray itself, clone the repository and install it in editable mode with the development extras:

```bash
git clone https://github.com/Dekermanjian/pymc-ray.git
cd pymc-ray
conda env create -f environment.yml # Python 3.13 + editable install with dev extras
conda activate pymc-ray
# or, in an existing environment:
pip install -e ".[dev]"
```

The [development page](development.md) explains how to run the tests and the code checks.
