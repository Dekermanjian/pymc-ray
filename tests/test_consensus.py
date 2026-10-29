import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import pymc_ray as pmr

from helpers import to_backend
from models import normal_mean_model, regression_model

SAMPLE_KWARGS = {"draws": 1000, "tune": 500, "chains": 2, "compute_convergence_checks": False}


@pytest.fixture(scope="module")
def normal_data() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({"y": rng.normal(2.5, 1.0, size=4_000)})


def exact_normal_mean_posterior(y, prior_sd=10.0, noise_sd=1.0):
    precision = 1 / prior_sd**2 + len(y) / noise_sd**2
    return (y.sum() / noise_sd**2) / precision, precision**-0.5


def test_temper_prior(regression_data):
    power = 0.25
    full = regression_model(regression_data)
    tempered = regression_model(regression_data)
    pmr.temper_prior(tempered, power)

    point = full.initial_point()
    point["sigma_log__"] = np.array(0.3)
    point["beta"] = np.array([0.5, -1.0])
    prior = full.compile_fn(full.logp(vars=full.free_RVs, jacobian=False), point_fn=True)(point)
    expected = full.compile_logp()(point) - (1 - power) * prior
    np.testing.assert_allclose(tempered.compile_logp()(point), expected)


def synthetic_subposteriors(means, covs, n_chains=2, n_draws=3000, seed=0):
    rng = np.random.default_rng(seed)
    subposteriors = []
    for mean, cov in zip(means, covs, strict=True):
        draws = rng.multivariate_normal(mean, cov, size=(n_chains, n_draws))
        posterior = xr.Dataset({"theta": (("chain", "draw", "theta_dim_0"), draws)})
        subposteriors.append(xr.DataTree.from_dict({"posterior": posterior}))
    return subposteriors


@pytest.mark.parametrize("method", ["consensus", "gaussian"])
def test_combination_of_gaussian_subposteriors_is_exact(method):
    means = [np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([2.0, 2.0])]
    covs = [
        np.array([[1.0, 0.3], [0.3, 0.5]]),
        np.eye(2) * 0.5,
        np.array([[2.0, -0.5], [-0.5, 1.0]]),
    ]
    precisions = [np.linalg.inv(c) for c in covs]
    expected_cov = np.linalg.inv(sum(precisions))
    expected_mean = expected_cov @ sum(p @ m for p, m in zip(precisions, means, strict=True))

    with pm.Model() as model:
        pm.Normal("theta", 0, 1, shape=2)
    combined = pmr.combine_subposteriors(synthetic_subposteriors(means, covs), model, method=method, random_seed=0)
    draws = combined["posterior"]["theta"].to_numpy().reshape(-1, 2)
    assert combined["posterior"]["theta"].dims == ("chain", "draw", "theta_dim_0")
    np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=0.03)
    np.testing.assert_allclose(np.cov(draws, rowvar=False), expected_cov, atol=0.03)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_matches_conjugate_posterior(normal_data, backend):
    combined = pmr.consensus_sample(
        normal_mean_model,
        to_backend(normal_data, backend),
        n_shards=4,
        sample_kwargs=SAMPLE_KWARGS,
        random_seed=2,
    )
    mean, sd = exact_normal_mean_posterior(normal_data["y"].to_numpy())
    mu = combined["posterior"]["mu"]
    assert mu.sizes == {"chain": 2, "draw": 1000}
    assert float(mu.mean()) == pytest.approx(mean, abs=0.2 * sd)
    assert float(mu.std()) == pytest.approx(sd, rel=0.1)


@pytest.mark.parametrize(("method", "weights"), [("consensus", "diag"), ("gaussian", "full")])
def test_constrained_parameters_against_full_data_nuts(regression_data, method, weights):
    combined, subposteriors = pmr.consensus_sample(
        regression_model,
        regression_data,
        n_shards=4,
        sample_kwargs=SAMPLE_KWARGS,
        method=method,
        weights=weights,
        random_seed=4,
        return_subposteriors=True,
    )
    assert len(subposteriors) == 4
    reference = pm.sample(model=regression_model(regression_data), random_seed=5, progressbar=False, **SAMPLE_KWARGS)
    for var in ["intercept", "beta", "sigma"]:
        got, ref = combined["posterior"][var], reference["posterior"][var]
        sample_dims = ("chain", "draw")
        np.testing.assert_allclose(got.mean(sample_dims), ref.mean(sample_dims), atol=0.02)
        np.testing.assert_allclose(got.std(sample_dims), ref.std(sample_dims), rtol=0.15)
    assert float(combined["posterior"]["sigma"].min()) > 0


def test_missing_unconstrained_draws_raise(regression_data):
    model = regression_model(regression_data)
    idata = pm.sample(model=model, draws=50, tune=50, chains=1, progressbar=False)
    with pytest.raises(ValueError, match="include_transformed"):
        pmr.combine_subposteriors([idata, idata], model)
