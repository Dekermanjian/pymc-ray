# Development

The [installation page](installation.md#development-installation) explains how to set up a development environment. These commands run the tests, the code checks, and a benchmark:

```bash
pytest # starts a local 4-CPU Ray instance
pytest -m "not slow" # skip the multiprocess sampling tests
ruff check . && ruff format --check .
pre-commit install # once per clone: run ruff and the file checks on every commit
pre-commit run --all-files
python benchmarks/gradient_eval.py --model multilevel --rows 100000 1000000 --chains 4
```

Tests run against pandas, polars and pyarrow inputs.

To build this documentation locally, install the documentation extras and run Sphinx. The second command treats every warning as an error, as the continuous integration (CI) build does:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.

## Roadmap

- Send each shard only its own parameters. With `shard_by=`, a shard's rows use only their own groups' parameters, yet every step still sends each shard the full parameter vector, so per-step traffic does not shrink as shards are added. Declaring the group dimension (e.g. `local_dims=["store"]`) would let each shard receive only its groups' entries plus the global parameters, with the sliced and full log-likelihood and gradient compared at construction to catch a wrong declaration.

- Hierarchical consensus: shard by group, temper only the global priors, combine the global parameters, then resample each group's parameters given the combined draws.

- Pick `n_shards` × `n_replicas` automatically from a quick concurrent benchmark.

- Lower per-step latency with Ray Compiled Graphs.

- Multi-threaded shards (one actor per node evaluating its rows on several cores), to cut the number of actor calls per step.

- JAX backends for the shard functions; GPU actors.

- More subposterior combination rules (semiparametric, Weierstrass) and stochastic-gradient MCMC.

## License

Copyright 2026 Jonathan Dekermanjian

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/Dekermanjian/pymc-ray/blob/main/LICENSE) for the full text.
