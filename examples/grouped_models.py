"""Fit a separate price-elasticity model for every store, in parallel on Ray.

Run with: python examples/grouped_models.py
"""

import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import ray

import pymc_ray as pmr


def simulate_sales(n_stores: int = 40, weeks: int = 104, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    elasticity = rng.normal(-1.5, 0.4, size=n_stores)
    baseline = rng.normal(5.0, 0.5, size=n_stores)
    store = np.repeat(np.arange(n_stores), weeks)
    log_price = rng.normal(0.0, 0.2, size=store.size)
    log_units = baseline[store] + elasticity[store] * log_price + rng.normal(0, 0.3, store.size)
    return pd.DataFrame({"store": [f"store_{s:03d}" for s in store], "log_price": log_price, "log_units": log_units})


def elasticity_model(df: pd.DataFrame) -> pm.Model:
    """Called once per store with that store's rows."""
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 5, 2)
        elasticity = pm.Normal("elasticity", -1, 1)
        sigma = pm.HalfNormal("sigma", 1)
        mu = alpha + elasticity * df["log_price"].to_numpy()
        pm.Normal("log_units", mu, sigma, observed=df["log_units"].to_numpy())
    return model


def main():
    ray.init(log_to_driver=False)
    sales = simulate_sales()
    fit_kwargs = {"draws": 1000, "tune": 1000, "chains": 4}

    # 1. In-memory DataFrame: results come back as one DataTree per store.
    results = pmr.fit_grouped(sales, "store", elasticity_model, fit_kwargs=fit_kwargs, random_seed=42)
    print(results)
    print(results.summary(var_names=["elasticity"]).head())

    # 2. Data too large for the driver: read it as a Ray Dataset, and persist each
    #    posterior from the worker so only file paths come back to the driver.
    with tempfile.TemporaryDirectory() as tmp:
        sales.to_parquet(Path(tmp) / "sales.parquet")
        out_dir = Path(tmp) / "posteriors"
        out_dir.mkdir()

        def save_posterior(store, model, idata, random_seed):
            path = out_dir / f"{store}.nc"
            idata.to_netcdf(path)
            return str(path)

        paths = pmr.fit_grouped(
            ray.data.read_parquet(tmp),
            "store",
            elasticity_model,
            fit_kwargs=fit_kwargs,
            postprocess_fn=save_posterior,
            random_seed=42,
        )
        first = next(iter(paths))
        print(f"{len(paths)} posteriors written, e.g. {paths[first]}")
        print(az.summary(az.from_netcdf(paths[first]), var_names=["elasticity"]))


if __name__ == "__main__":
    main()
