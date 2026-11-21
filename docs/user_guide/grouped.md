# Many groups: `fit_grouped`

Write a function that builds the model for **one** group's data. {func}`~pymc_ray.fit_grouped` fits all groups in parallel, one Ray task per group. Its API reference page lists every argument.

```python
import pymc as pm
import pymc_ray as pmr


def model_fn(df): # receives one group's rows
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 5, 2)
        elasticity = pm.Normal("elasticity", -1, 1)
        sigma = pm.HalfNormal("sigma", 1)
        mu = alpha + elasticity * df["log_price"].to_numpy()
        pm.Normal("log_units", mu, sigma, observed=df["log_units"].to_numpy())
    return model


results = pmr.fit_grouped(
    sales_df,
    "store",
    model_fn,
    fit_kwargs={"draws": 1000, "chains": 4},
    random_seed=42,
)
results["store_007"] # DataTree for one store
results.summary(var_names=["elasticity"]) # one table, indexed by (store, variable)
results.failures # groups that raised, with worker tracebacks
```

- **Input.** Any eager dataframe, or a `ray.data.Dataset`. A Dataset is grouped and fitted on the cluster with `map_groups`, so the data never pass through the driver. Null group keys are reported as `None` whichever library produced them.

- **Parallelism.** By default each group runs its chains sequentially (`cores=1`) on one CPU, and parallelism comes from fitting many groups at once. Pass `fit_kwargs={"cores": 4}` to give each group 4 CPUs instead.

- **Memory.** `postprocess_fn(key, model, result, random_seed)` runs on the worker before results are returned. Return a summary, or write the posterior to storage and return its path, so that 100k groups do not have to fit in driver memory (see [examples/grouped_models.py](https://github.com/Dekermanjian/pymc-ray/blob/main/examples/grouped_models.py)).

- **Reproducibility.** Each group's seeds are derived from `random_seed` and the group key, so a group gets the same draws however the work is scheduled, and whichever other groups are in the run. `postprocess_fn` receives its own seed, separate from the one used for fitting, for any random work it does.

- **Errors.** `on_error="raise"` (default; cancels pending groups), `"warn"` or `"ignore"`.

- **Other fitters.** `fit_fn` is called as `fit_fn(model=model, random_seed=seed, **fit_kwargs)`, so `pm.fit` or a wrapper around nutpie or `pm.find_MAP` also work.

- **Posterior predictive checks.** Each worker holds its group's model, so add them in `postprocess_fn`:

  ```python
  def with_ppc(key, model, idata, random_seed):
      return pm.sample_posterior_predictive(
          idata, model=model, extend_inferencedata=True, random_seed=random_seed
      )


  results = pmr.fit_grouped(sales_df, "store", model_fn, postprocess_fn=with_ppc, random_seed=42)
  ```
