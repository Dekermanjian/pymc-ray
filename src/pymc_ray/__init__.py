"""Utilities for scaling PyMC models to large datasets with Ray.

- :func:`fit_grouped`: fit one independent model per group of a dataset, in parallel.
- :class:`ShardedModel`: exact inference for one model whose data are spread across
  Ray actors; the likelihood and its gradient are evaluated shard-by-shard and summed.
- :func:`consensus_sample`: approximate inference by sampling shards independently and
  combining the subposterior draws (consensus Monte Carlo).
- :func:`sample_posterior_predictive`, :func:`compute_log_likelihood` and :func:`loo`:
  posterior checks and cross-validation over sharded data, for any posterior.
"""

from pymc_ray._timing import time_logp_dlogp
from pymc_ray.consensus import combine_subposteriors, consensus_sample, temper_prior
from pymc_ray.grouped import GroupedResults, GroupFailure, GroupFitError, fit_grouped
from pymc_ray.predictive import compute_log_likelihood, loo, sample_posterior_predictive
from pymc_ray.sharded import ShardedModel

__version__ = "0.1.0"

__all__ = [
    "GroupFailure",
    "GroupFitError",
    "GroupedResults",
    "ShardedModel",
    "combine_subposteriors",
    "compute_log_likelihood",
    "consensus_sample",
    "fit_grouped",
    "loo",
    "sample_posterior_predictive",
    "temper_prior",
    "time_logp_dlogp",
]
