# PyMC-Ray

PyMC-Ray provides utilities for scaling [PyMC](https://www.pymc.io) models to large datasets with [Ray](https://www.ray.io).

| Situation | Tool | Posterior |
|---|---|---|
| Data split by a key, and you want one independent model per group | {func}`~pymc_ray.fit_grouped` | Exact, per group |
| One model whose likelihood needs *all* the data, and the data are too large for one process or machine | {class}`~pymc_ray.ShardedModel` | Exact |
| Same, but you can accept an approximation in exchange for zero communication, and the model has no group-level (multilevel) structure | {func}`~pymc_ray.consensus_sample` | Approximate |

PyMC-Ray requires PyMC 6.3 or later, Python 3.12 or later, and Ray 2.40 or later. Posteriors are returned as `xarray.DataTree` objects, the same type that `pm.sample` returns.

```bash
pip install pymc-ray
```

The [user guide](user_guide/index.md) explains each tool and how to tune it. The [tutorials](tutorials/index.md) work through each tool from start to finish in a notebook. The [API reference](api.md) documents every function and class.

```{toctree}
:maxdepth: 2
:hidden:

installation
user_guide/index
tutorials/index
api
development
```
