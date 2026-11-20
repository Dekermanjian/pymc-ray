import arviz as az
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytest
import ray

import pymc_ray as pmr

from helpers import to_backend
from models import failing_on_store_b, group_mean_model

FIT_KWARGS = {"draws": 300, "tune": 300, "chains": 2, "compute_convergence_checks": False}
TRUE_MEANS = {"a": -3.0, "b": 0.0, "c": 4.0}


@pytest.fixture(scope="module")
def sales() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame(
            {
                "store": store,
                "region": np.where(np.arange(200) % 2 == 0, "north", "south"),
                "y": rng.normal(mean, 1.0, size=200),
            }
        )
        for store, mean in TRUE_MEANS.items()
    ]
    return pd.concat(frames, ignore_index=True)


def posterior_mean(idata, var="mu"):
    return float(idata["posterior"][var].mean())


def expecting(native_type):
    """group_mean_model, but fail unless groups arrive as ``native_type``."""

    def model_fn(group):
        assert type(group) is native_type, f"got {type(group)}"
        return group_mean_model(group)

    return model_fn


def test_fits_every_group(sales, as_backend):
    data = as_backend(sales)
    results = pmr.fit_grouped(
        data,
        "store",
        expecting(type(data)),
        fit_kwargs=FIT_KWARGS,
        random_seed=1,
        progressbar=False,
    )
    assert list(results) == sorted(TRUE_MEANS)
    assert not results.failures
    for store, mean in TRUE_MEANS.items():
        assert posterior_mean(results[store]) == pytest.approx(mean, abs=0.3)


def test_seeds_do_not_depend_on_which_groups_run(sales):
    kwargs = {"fit_kwargs": FIT_KWARGS, "random_seed": 7, "progressbar": False}
    everything = pmr.fit_grouped(sales, "store", group_mean_model, **kwargs)
    only_b = pmr.fit_grouped(sales[sales["store"] == "b"], "store", group_mean_model, **kwargs)
    np.testing.assert_allclose(everything["b"]["posterior"]["mu"].to_numpy(), only_b["b"]["posterior"]["mu"].to_numpy(), rtol=1e-8, atol=1e-8)


def test_postprocess_gets_its_own_seed(sales):
    def seeds(random_seed):
        return pmr.fit_grouped(
            sales,
            "store",
            group_mean_model,
            fit_fn=lambda model, random_seed: random_seed,
            postprocess_fn=lambda key, model, fit_seed, seed: (fit_seed, seed),
            random_seed=random_seed,
            progressbar=False,
        )

    seeded = seeds(7)
    assert all(fit_seed != seed for fit_seed, seed in seeded.values())
    assert len({seed for _, seed in seeded.values()}) == len(TRUE_MEANS)
    assert all(seed is None for _, seed in seeds(None).values())


def test_postprocess_draws_do_not_depend_on_which_groups_run(sales):
    def with_ppc(key, model, idata, random_seed):
        ppc = pm.sample_posterior_predictive(idata, model=model, random_seed=random_seed, progressbar=False)
        return ppc["posterior_predictive"]["y"].to_numpy()

    kwargs = {
        "fit_kwargs": FIT_KWARGS,
        "postprocess_fn": with_ppc,
        "random_seed": 7,
        "progressbar": False,
    }
    everything = pmr.fit_grouped(sales, "store", group_mean_model, **kwargs)
    only_b = pmr.fit_grouped(sales[sales["store"] == "b"], "store", group_mean_model, **kwargs)
    np.testing.assert_allclose(everything["b"], only_b["b"], rtol=1e-8, atol=1e-8)


def test_multi_column_keys_are_tuples(sales, as_backend):
    results = pmr.fit_grouped(
        as_backend(sales),
        ["store", "region"],
        group_mean_model,
        fit_kwargs=FIT_KWARGS,
        postprocess_fn=lambda key, model, idata, seed: posterior_mean(idata),
        progressbar=False,
    )
    assert set(results) == {(s, r) for s in TRUE_MEANS for r in ("north", "south")}
    assert results[("c", "north")] == pytest.approx(4.0, abs=0.4)


def test_postprocess_runs_on_worker(sales):
    results = pmr.fit_grouped(
        sales,
        "store",
        group_mean_model,
        fit_kwargs=FIT_KWARGS,
        postprocess_fn=lambda key, model, idata, seed: (key, model.name, az.summary(idata)),
        progressbar=False,
    )
    key, _, summary = results["a"]
    assert key == "a"
    assert "mu" in summary.index


def test_summary_is_indexed_by_group_and_variable(sales):
    results = pmr.fit_grouped(sales, "store", group_mean_model, fit_kwargs=FIT_KWARGS, progressbar=False)
    summary = results.summary()
    assert summary.index.names == ["store", "variable"]
    assert summary.loc[("c", "mu"), "mean"] == pytest.approx(4.0, abs=0.3)


def test_on_error_raise(sales):
    with pytest.raises(pmr.GroupFitError, match="bad store"):
        pmr.fit_grouped(sales, "store", failing_on_store_b, fit_kwargs=FIT_KWARGS, progressbar=False)


def test_on_error_warn_keeps_successful_groups(sales):
    with pytest.warns(UserWarning, match="group 'b' failed"):
        results = pmr.fit_grouped(
            sales,
            "store",
            failing_on_store_b,
            fit_kwargs=FIT_KWARGS,
            on_error="warn",
            progressbar=False,
        )
    assert set(results) == {"a", "c"}
    assert set(results.failures) == {"b"}
    assert "ValueError" in results.failures["b"].traceback


def test_custom_fit_fn(sales):
    def find_map(model, random_seed):
        return pm.find_MAP(model=model, progressbar=False, seed=random_seed)

    results = pmr.fit_grouped(sales, "store", group_mean_model, fit_fn=find_map, progressbar=False)
    assert float(results["c"]["mu"]) == pytest.approx(4.0, abs=0.2)


def test_null_keys_are_none(sales, as_backend):
    data = as_backend(sales.assign(store=sales["store"].where(sales["store"] != "b", None)))
    results = pmr.fit_grouped(
        data,
        "store",
        group_mean_model,
        fit_kwargs=FIT_KWARGS,
        postprocess_fn=lambda key, model, idata, seed: posterior_mean(idata),
        progressbar=False,
    )
    assert set(results) == {"a", "c", None}
    assert results[None] == pytest.approx(0.0, abs=0.3)


def test_lazy_frames_are_rejected(sales):
    with pytest.raises(TypeError, match="Lazy frames are not supported"):
        pmr.fit_grouped(pl.from_pandas(sales).lazy(), "store", group_mean_model)


def test_ray_dataset_input(sales, backend):
    dataset = ray.data.from_pandas(sales)
    results = pmr.fit_grouped(
        dataset,
        "store",
        expecting(type(to_backend(sales, backend))),
        fit_kwargs=FIT_KWARGS,
        random_seed=1,
        backend=backend,
    )
    assert set(results) == set(TRUE_MEANS)
    for store, mean in TRUE_MEANS.items():
        assert posterior_mean(results[store]) == pytest.approx(mean, abs=0.3)


def test_ray_dataset_multi_column_keys(sales):
    results = pmr.fit_grouped(
        ray.data.from_pandas(sales),
        ["store", "region"],
        group_mean_model,
        fit_kwargs=FIT_KWARGS,
        postprocess_fn=lambda key, model, idata, seed: posterior_mean(idata),
    )
    assert set(results) == {(s, r) for s in TRUE_MEANS for r in ("north", "south")}
