# Posterior predictive checks and leave-one-out cross-validation

The observed data live on the shards, so in-sample checks run there too. Each shard handles its own rows, and the results come back in the order of your data.

```python
with pmr.ShardedModel(model_fn, data, n_shards=32, shard_by="group") as sharded:
    idata = sharded.sample(chains=4)
    ppc = sharded.sample_posterior_predictive(idata, num_draws=200, random_seed=1)
    elpd = sharded.loo(idata) # equals arviz.loo on the full data
    idata = sharded.compute_log_likelihood(idata) # the full pointwise arrays, if they fit

az.compare({"this model": elpd, "another model": other_elpd})
```

- **`sample_posterior_predictive`** returns `posterior_predictive` and `observed_data` groups, like `pm.sample_posterior_predictive`. The result holds chains × draws × rows values per variable. Thin it with `num_draws` (evenly spaced draws per chain). If even that will not fit in memory, pass `postprocess_fn(shard, model, result)` where it runs where each shard lives, can write that shard's draws to storage, and its return values (the paths) come back instead.

  ```python
  def save(shard, model, result):
      path = f"{output_dir}/ppc-shard-{shard}.nc" # storage that every node can reach
      result.to_netcdf(path)
      return path


  paths = sharded.sample_posterior_predictive(idata, postprocess_fn=save)
  ```

- **`loo`** computes the pointwise log-likelihood and PSIS-LOO on each shard and sends back only one value per row. It therefore works even when the full log-likelihood (chains × draws × rows values) would not fit anywhere. The result is ArviZ's own LOO object, and it prints as `arviz.loo` does, flags high Pareto k values, and works with `arviz.compare`.

- **`compute_log_likelihood`** returns the full pointwise arrays, for use with ArviZ directly when they fit.

- **Predictions on new data** that fit in memory don't need sharding: use `pm.sample_posterior_predictive(idata, model=model_fn(new_data))`. Sharded posteriors omit deterministics that depend on the data, because the driver computes them on zero rows. To recompute one on the data, pass it in `var_names`.

- **Posteriors from anywhere else**, such as `consensus_sample` or a saved posterior, `pmr.sample_posterior_predictive(idata, model_fn, data, n_shards=...)`, `pmr.compute_log_likelihood(...)` and `pmr.loo(...)` run the same computations on short-lived Ray tasks. Their API reference pages are {func}`~pymc_ray.sample_posterior_predictive`, {func}`~pymc_ray.compute_log_likelihood` and {func}`~pymc_ray.loo`.

- **Row order** is that of the input data, with or without `shard_by`. The exception is a Ray Dataset split with `shard_by`, whose original order is not known. Give the observations a dim whose coords come from an id column, so every value carries its label, or use `postprocess_fn`.
