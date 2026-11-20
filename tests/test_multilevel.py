"""Multilevel models: the case where sampling gets slow and approximations get fragile."""

import numpy as np
import pymc as pm
import pytest
import xarray as xr

import pymc_ray as pmr

from helpers import assert_same_density
from models import multilevel_model, simulate_multilevel

TRUE_GROUP_SDS = [1.0, 0.5]
SAMPLE_KWARGS = {"draws": 400, "tune": 400, "chains": 2, "compute_convergence_checks": False}


@pytest.fixture(scope="module")
def multilevel_data():
    return simulate_multilevel(mean_group_size=60, seed=0)


def posterior_with_deterministics(idata, model) -> xr.Dataset:
    posterior = idata["posterior"].to_dataset()
    missing = [d.name for d in model.deterministics if d.name not in posterior]
    if missing:
        computed = pm.compute_deterministics(posterior, var_names=missing, model=model, progressbar=False)
        posterior = xr.merge([posterior, computed])
    return posterior


def test_sharded_density_is_exact(multilevel_data):
    # LKJCholeskyCov has its own packed-Cholesky transform; groups span every shard.
    with pmr.ShardedModel(multilevel_model, multilevel_data, n_shards=4) as sharded:
        assert_same_density(sharded.model, multilevel_model(multilevel_data))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Consensus Monte Carlo is biased for multilevel models: tempering the group-level "
        "prior lets every shard absorb group variation into the local parameters, so each "
        "shard (and the combination) underestimates the group-level sds by ~1/sqrt(n_shards)."
    ),
)
def test_consensus_recovers_group_level_sds(multilevel_data):
    combined = pmr.consensus_sample(multilevel_model, multilevel_data, n_shards=4, sample_kwargs=SAMPLE_KWARGS, random_seed=3)
    posterior = posterior_with_deterministics(combined, multilevel_model(multilevel_data))
    group_sds = posterior["chol_stds"].mean(("chain", "draw")).to_numpy()
    np.testing.assert_allclose(group_sds, TRUE_GROUP_SDS, rtol=0.25)
