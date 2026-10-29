"""Model functions shared by the tests (importable by Ray workers).

They read data through narwhals, so the same function accepts pandas, polars or
pyarrow; this is the pattern recommended for library-agnostic model functions.
"""

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pymc as pm

GROUPS = [0, 1, 2]
N_MULTILEVEL_GROUPS = 30


def column(data, name: str) -> np.ndarray:
    return nw.from_native(data, eager_only=True)[name].to_numpy()


def columns(data, *names: str) -> np.ndarray:
    return nw.from_native(data, eager_only=True).select(*names).to_numpy()


def regression_model(data) -> pm.Model:
    with pm.Model() as model:
        intercept = pm.Normal("intercept", 0, 5)
        beta = pm.Normal("beta", 0, 5, shape=2)
        sigma = pm.HalfNormal("sigma", 2)
        mu = intercept + columns(data, "x0", "x1") @ beta
        pm.Normal("y", mu, sigma, observed=column(data, "y"))
    return model


def deterministic_regression_model(data) -> pm.Model:
    """regression_model, plus a Deterministic with one value per row."""
    with pm.Model() as model:
        intercept = pm.Normal("intercept", 0, 5)
        beta = pm.Normal("beta", 0, 5, shape=2)
        sigma = pm.HalfNormal("sigma", 2)
        mu = pm.Deterministic("mu", intercept + columns(data, "x0", "x1") @ beta)
        pm.Normal("y", mu, sigma, observed=column(data, "y"))
    return model


def labeled_regression_model(data) -> pm.Model:
    """regression_model with observations labeled by the data's row_id column."""
    with pm.Model(coords={"obs_id": column(data, "row_id")}) as model:
        intercept = pm.Normal("intercept", 0, 5)
        beta = pm.Normal("beta", 0, 5, shape=2)
        sigma = pm.HalfNormal("sigma", 2)
        mu = intercept + columns(data, "x0", "x1") @ beta
        pm.Normal("y", mu, sigma, observed=column(data, "y"), dims="obs_id")
    return model


def regression_model_dict(data: dict) -> pm.Model:
    with pm.Model() as model:
        intercept = pm.Normal("intercept", 0, 5)
        beta = pm.Normal("beta", 0, 5, shape=2)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("y", intercept + data["X"] @ beta, sigma, observed=data["y"])
    return model


def hierarchical_model(data) -> pm.Model:
    # Coords come from the global GROUPS list, so every shard has the same parameters.
    with pm.Model(coords={"group": GROUPS}) as model:
        mu_a = pm.Normal("mu_a", 0, 5)
        sigma_a = pm.HalfNormal("sigma_a", 2)
        a = pm.Normal("a", mu_a, sigma_a, dims="group")
        beta = pm.Normal("beta", 0, 5, shape=2)
        sigma = pm.HalfNormal("sigma", 2)
        mu = a[column(data, "group")] + columns(data, "x0", "x1") @ beta
        pm.Normal("y", mu, sigma, observed=column(data, "y"))
    return model


def multilevel_model(data) -> pm.Model:
    """Correlated varying intercepts and slopes, non-centered, with an LKJ prior."""
    group = column(data, "group").astype("int64")
    coords = {"group": np.arange(N_MULTILEVEL_GROUPS), "effect": ["intercept", "slope"]}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", 0, 5, dims="effect")
        chol, _, _ = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0, shape=2), compute_corr=True)
        z = pm.Normal("z", 0, 1, dims=("group", "effect"))
        effects = pm.Deterministic("effects", mu + z @ chol.T, dims=("group", "effect"))
        sigma = pm.HalfNormal("sigma", 1)
        y_hat = effects[group, 0] + effects[group, 1] * column(data, "x")
        pm.Normal("y", y_hat, sigma, observed=column(data, "y"))
    return model


def simulate_multilevel(n_groups: int = N_MULTILEVEL_GROUPS, mean_group_size: int = 100, seed: int = 0) -> pd.DataFrame:
    """Imbalanced groups (sizes 5 to 2x the mean), shuffled so groups span shards."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(5, 2 * mean_group_size, size=n_groups)
    sds, corr = np.array([1.0, 0.5]), 0.4
    cov = np.diag(sds) @ np.array([[1, corr], [corr, 1]]) @ np.diag(sds)
    effects = rng.multivariate_normal([1.0, -0.5], cov, size=n_groups)
    group = np.repeat(np.arange(n_groups), sizes)
    x = rng.normal(size=group.size)
    y = effects[group, 0] + effects[group, 1] * x + rng.normal(scale=0.8, size=group.size)
    order = rng.permutation(group.size)
    return pd.DataFrame({"group": group[order], "x": x[order], "y": y[order]})


def data_dependent_coords_model(data) -> pm.Model:
    # Anti-pattern: the parameter shape depends on the rows the model receives.
    groups, index = np.unique(column(data, "group"), return_inverse=True)
    with pm.Model(coords={"group": groups}) as model:
        a = pm.Normal("a", 0, 5, dims="group")
        pm.Normal("y", a[index], 1, observed=column(data, "y"))
    return model


def discrete_parameter_model(data) -> pm.Model:
    with pm.Model() as model:
        k = pm.Poisson("k", 3)
        pm.Normal("y", k, 1, observed=column(data, "y"))
    return model


def potential_likelihood_model(data) -> pm.Model:
    with pm.Model() as model:
        intercept = pm.Normal("intercept", 0, 5)
        beta = pm.Normal("beta", 0, 5, shape=2)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Potential("ridge", -0.5 * (beta**2).sum())
        mu = intercept + columns(data, "x0", "x1") @ beta
        dist = pm.Normal.dist(mu, sigma)
        pm.Potential("loglik", pm.logp(dist, column(data, "y")).sum())
    return model


def normal_mean_model(data) -> pm.Model:
    """Conjugate model: known noise, Normal(0, 10) prior on the mean."""
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 10)
        pm.Normal("y", mu, 1.0, observed=column(data, "y"))
    return model


def group_mean_model(data) -> pm.Model:
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 10)
        sigma = pm.HalfNormal("sigma", 5)
        pm.Normal("y", mu, sigma, observed=column(data, "y"))
    return model


def failing_on_store_b(data) -> pm.Model:
    if "b" in column(data, "store"):
        raise ValueError("bad store")
    return group_mean_model(data)
