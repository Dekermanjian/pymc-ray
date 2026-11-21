# API reference

Every name on this page is importable from the top-level package, for example `pymc_ray.fit_grouped`.

```{eval-rst}
.. currentmodule:: pymc_ray
```

## Many groups

```{eval-rst}
.. autosummary::
   :toctree: generated

   fit_grouped
   GroupedResults
   GroupFailure
   GroupFitError
```

## One model on sharded data

```{eval-rst}
.. autosummary::
   :toctree: generated

   ShardedModel
```

## Consensus Monte Carlo

```{eval-rst}
.. autosummary::
   :toctree: generated

   consensus_sample
   combine_subposteriors
   temper_prior
```

## Posterior predictive checks and cross-validation

```{eval-rst}
.. autosummary::
   :toctree: generated

   sample_posterior_predictive
   compute_log_likelihood
   loo
```

## Timing

```{eval-rst}
.. autosummary::
   :toctree: generated

   time_logp_dlogp
```
