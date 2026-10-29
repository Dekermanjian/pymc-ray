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


@pytest.fixture(scope="module")
def reference(multilevel_data):
    """Full-data NUTS on one process: the posterior every method is judged against."""
    model = multilevel_model(multilevel_data)
    return pm.sample(model=model, cores=1, random_seed=1, progressbar=False, **SAMPLE_KWARGS)


def posterior_with_deterministics(idata, model) -> xr.Dataset:
    posterior = idata["posterior"].to_dataset()
    missing = [d.name for d in model.deterministics if d.name not in posterior]
    if missing:
        computed = pm.compute_deterministics(posterior, var_names=missing, model=model, progressbar=False)
        posterior = xr.merge([posterior, computed])
    return posterior


def assert_matches_reference(got: xr.Dataset, ref: xr.Dataset, var: str, mean_tol, sd_rtol):
    """Posterior means agree within ``mean_tol`` reference sds; sds agree within ``sd_rtol``."""
    dims = ("chain", "draw")
    ref_sd = ref[var].std(dims)
    z = np.abs(got[var].mean(dims) - ref[var].mean(dims)) / ref_sd
    assert float(z.max()) < mean_tol, f"{var}: posterior means differ by {float(z.max()):.2f} sd"
    np.testing.assert_allclose(got[var].std(dims), ref_sd, rtol=sd_rtol, err_msg=var)


def test_sharded_density_is_exact(multilevel_data):
    # LKJCholeskyCov has its own packed-Cholesky transform; groups span every shard.
    with pmr.ShardedModel(multilevel_model, multilevel_data, n_shards=4) as sharded:
        assert_same_density(sharded.model, multilevel_model(multilevel_data))


@pytest.mark.slow
def test_sharded_posterior_matches_full_data_nuts(multilevel_data, reference):
    model = multilevel_model(multilevel_data)
    with pmr.ShardedModel(multilevel_model, multilevel_data, n_shards=2) as sharded:
        idata = sharded.sample(cores=2, random_seed=2, progressbar=False, **SAMPLE_KWARGS)
    got = posterior_with_deterministics(idata, model)
    ref = posterior_with_deterministics(reference, model)
    for var in ["mu", "chol_stds", "chol_corr", "sigma", "effects"]:
        assert_matches_reference(got, ref, var, mean_tol=0.35, sd_rtol=0.25)


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
