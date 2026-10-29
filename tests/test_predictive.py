"""Posterior predictive sampling, log-likelihood and LOO over sharded data."""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import ray
import xarray as xr

import pymc_ray as pmr

from models import (
    column,
    deterministic_regression_model,
    labeled_regression_model,
    multilevel_model,
    regression_model,
    simulate_multilevel,
)

SAMPLE_KWARGS = {"draws": 200, "tune": 200, "chains": 2, "random_seed": 1, "progressbar": False}


@pytest.fixture(scope="module")
def data(regression_data) -> pd.DataFrame:
    return regression_data.assign(row_id=[f"row{i}" for i in range(len(regression_data))])


@pytest.fixture(scope="module")
def posterior(data) -> xr.DataTree:
    """A full-data posterior, so sharded and in-process results can be compared exactly."""
    return pm.sample(model=regression_model(data), **SAMPLE_KWARGS)


@pytest.fixture(scope="module")
def multilevel_data() -> pd.DataFrame:
    return simulate_multilevel(mean_group_size=40, seed=1)


@pytest.fixture(scope="module")
def multilevel_posterior(multilevel_data) -> xr.DataTree:
    return pm.sample(model=multilevel_model(multilevel_data), **SAMPLE_KWARGS)


def in_process_log_lik(posterior, model) -> xr.Dataset:
    return pm.compute_log_likelihood(posterior, model=model, extend_inferencedata=False, progressbar=False)


def full_data_loo(posterior, model, **kwargs):
    log_lik = in_process_log_lik(posterior, model)
    tree = xr.DataTree.from_dict({"posterior": posterior["posterior"].to_dataset(), "log_likelihood": log_lik})
    return az.loo(tree, pointwise=True, **kwargs)


def assert_same_loo(got, expected):
    for field in ("elpd", "se", "p", "n_samples", "n_data_points", "good_k", "warning"):
        np.testing.assert_allclose(got[field], expected[field], rtol=1e-8, err_msg=field)
    np.testing.assert_allclose(got.elpd_i, expected.elpd_i, rtol=1e-8)
    np.testing.assert_allclose(got.pareto_k, expected.pareto_k, rtol=1e-8)


# Log-likelihood: deterministic, so it must match exactly and in the original row order.


def test_log_likelihood_matches_in_process(data, posterior, as_backend):
    native = as_backend(data)
    with pmr.ShardedModel(regression_model, native, n_shards=3) as sm:
        got = sm.compute_log_likelihood(posterior, extend_inferencedata=False)
    expected = in_process_log_lik(posterior, regression_model(native))
    np.testing.assert_allclose(got["log_likelihood"]["y"], expected["y"], rtol=1e-10)


def test_log_likelihood_with_shard_by_restores_row_order(multilevel_data, multilevel_posterior):
    with pmr.ShardedModel(multilevel_model, multilevel_data, n_shards=4, shard_by="group") as sm:
        got = sm.compute_log_likelihood(multilevel_posterior, extend_inferencedata=False)
    expected = in_process_log_lik(multilevel_posterior, multilevel_model(multilevel_data))
    np.testing.assert_allclose(got["log_likelihood"]["y"], expected["y"], rtol=1e-10)


def test_log_likelihood_extends_and_thins(data, posterior):
    tree = posterior.copy()
    with pmr.ShardedModel(regression_model, data, n_shards=2) as sm:
        extended = sm.compute_log_likelihood(tree, num_draws=50)
    assert extended is tree
    assert tree["log_likelihood"]["y"].shape == (2, 50, len(data))


# Posterior predictive: random, so check row order through the observed data.


@pytest.mark.parametrize("shard_by", [None, "group"])
def test_posterior_predictive_rows_in_order(data, posterior, shard_by):
    with pmr.ShardedModel(regression_model, data, n_shards=3, shard_by=shard_by) as sm:
        result = sm.sample_posterior_predictive(posterior, random_seed=2)
    np.testing.assert_array_equal(result["observed_data"]["y"], data["y"])
    y_rep = result["posterior_predictive"]["y"]
    assert y_rep.shape == (2, 200, len(data))
    # Each row's draws centre on that row's prediction, so they track the observed values
    # (about 0.9 here, since the model ignores the data's group offsets; ~0 if scrambled).
    assert np.corrcoef(y_rep.mean(("chain", "draw")), data["y"])[0, 1] > 0.8


def test_posterior_predictive_matches_in_process_in_distribution(data, posterior):
    with pmr.ShardedModel(regression_model, data, n_shards=3) as sm:
        sharded = sm.sample_posterior_predictive(posterior, random_seed=2)["posterior_predictive"]
    full = pm.sample_posterior_predictive(posterior, model=regression_model(data), random_seed=3, progressbar=False)["posterior_predictive"]
    dims = ("chain", "draw")
    # Per-row means differ only by Monte Carlo error: about sigma / sqrt(400 draws).
    diff = np.abs(sharded["y"].mean(dims) - full["y"].mean(dims))
    assert float(diff.mean()) < 0.1
    np.testing.assert_allclose(float(sharded["y"].std()), float(full["y"].std()), rtol=0.02)


def test_posterior_predictive_is_reproducible(data, posterior):
    with pmr.ShardedModel(regression_model, data, n_shards=2) as sm:
        a = sm.sample_posterior_predictive(posterior, random_seed=5, num_draws=20)
        b = sm.sample_posterior_predictive(posterior, random_seed=5, num_draws=20)
        c = sm.sample_posterior_predictive(posterior, random_seed=6, num_draws=20)
    y = "posterior_predictive"
    assert a[y]["y"].sizes["draw"] == 20
    np.testing.assert_array_equal(a[y]["y"], b[y]["y"])
    assert not np.array_equal(a[y]["y"], c[y]["y"])


def test_postprocess_fn_runs_on_each_shard(data, posterior):
    def describe(shard, model, result):
        return shard, result["posterior_predictive"]["y"].sizes["y_dim_0"]

    with pmr.ShardedModel(regression_model, data, n_shards=3) as sm:
        per_shard = sm.sample_posterior_predictive(posterior, postprocess_fn=describe)
    assert [shard for shard, _ in per_shard] == [0, 1, 2]
    assert sum(rows for _, rows in per_shard) == len(data)


def test_posterior_predictive_extends_idata(data, posterior):
    tree = posterior.copy()
    with pmr.ShardedModel(regression_model, data, n_shards=2) as sm:
        result = sm.sample_posterior_predictive(tree, num_draws=10, extend_inferencedata=True)
    assert result is tree
    assert {"posterior", "posterior_predictive", "observed_data"} <= set(tree.children)


# LOO: computed shard by shard, it must equal arviz.loo on the full data.


def test_loo_matches_arviz(data, posterior):
    with pmr.ShardedModel(regression_model, data, n_shards=3) as sm:
        got = sm.loo(posterior)
    assert_same_loo(got, full_data_loo(posterior, regression_model(data)))


def test_loo_with_shard_by_matches_arviz(multilevel_data, multilevel_posterior):
    with (
        pmr.ShardedModel(multilevel_model, multilevel_data, n_shards=4, shard_by="group") as sm,
        # A few observations have high Pareto k: arviz.loo warns about them, and so must we.
        pytest.warns(UserWarning, match="Pareto"),
    ):
        got = sm.loo(multilevel_posterior)
    with pytest.warns(UserWarning, match="Pareto"):
        expected = full_data_loo(multilevel_posterior, multilevel_model(multilevel_data))
    assert got.warning and expected.warning
    assert_same_loo(got, expected)


def test_loo_works_with_compare(data, posterior):
    with pmr.ShardedModel(regression_model, data, n_shards=2) as sm:
        sharded = sm.loo(posterior)
    full = full_data_loo(posterior, regression_model(data))
    comparison = az.compare({"sharded": sharded, "full": full})
    assert len(comparison) == 2
    np.testing.assert_allclose(comparison["elpd_diff"].to_numpy(), 0, atol=1e-8)


# Posteriors from anywhere: short-lived Ray tasks load the shards.


def test_standalone_functions_match_in_process(data, posterior):
    model = regression_model(data)
    log_lik = pmr.compute_log_likelihood(posterior, regression_model, data, n_shards=3, extend_inferencedata=False)
    np.testing.assert_allclose(log_lik["log_likelihood"]["y"], in_process_log_lik(posterior, model)["y"], rtol=1e-10)
    assert_same_loo(pmr.loo(posterior, regression_model, data, n_shards=3), full_data_loo(posterior, model))
    result = pmr.sample_posterior_predictive(posterior, regression_model, data, n_shards=3, shard_by="group", num_draws=10)
    np.testing.assert_array_equal(result["observed_data"]["y"], data["y"])


def test_ray_dataset_without_shard_by_keeps_order(data, posterior):
    dataset = ray.data.from_pandas(data).repartition(4)
    log_lik = pmr.compute_log_likelihood(posterior, regression_model, dataset, n_shards=3, extend_inferencedata=False)
    expected = in_process_log_lik(posterior, regression_model(data))
    np.testing.assert_allclose(log_lik["log_likelihood"]["y"], expected["y"], rtol=1e-10)


def test_ray_dataset_with_shard_by_needs_labeled_rows(data, posterior):
    dataset = ray.data.from_pandas(data).repartition(4)
    with pytest.raises(ValueError, match="original order"):
        pmr.compute_log_likelihood(posterior, regression_model, dataset, n_shards=3, shard_by="group")
    # With observations labeled by an id column, rows can be matched up by label.
    log_lik = pmr.compute_log_likelihood(
        posterior,
        labeled_regression_model,
        dataset,
        n_shards=3,
        shard_by="group",
        extend_inferencedata=False,
    )["log_likelihood"]["y"]
    expected = in_process_log_lik(posterior, labeled_regression_model(data))["y"]
    np.testing.assert_allclose(log_lik.sel(obs_id=expected["obs_id"]), expected, rtol=1e-10)


# Predictions on new data with the driver's posterior.


def test_per_row_deterministics_do_not_block_new_data_predictions(data):
    with pmr.ShardedModel(deterministic_regression_model, data, n_shards=2) as sm:
        idata = sm.sample(**SAMPLE_KWARGS)
    # Computed on the driver's zero-row model, "mu" would be empty: it is dropped.
    assert "mu" not in idata["posterior"]
    new = data.iloc[:7]
    predictions = pm.sample_posterior_predictive(idata, model=deterministic_regression_model(new), progressbar=False)
    assert predictions["posterior_predictive"]["y"].shape == (2, 200, 7)
    assert column(new, "y").shape == (7,)
