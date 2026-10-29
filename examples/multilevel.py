"""A multilevel model on a large dataset, sampled exactly with ShardedModel.

Correlated varying intercepts and slopes for many groups (non-centered, LKJ prior): the
kind of model where NUTS needs long trajectories and sampling gets slow. Uses polars
throughout; any dataframe library supported by narwhals works the same way.

Run with: python examples/multilevel.py --rows 1000000 --groups 1000 --compare
"""

import argparse
import time

import narwhals.stable.v2 as nw
import numpy as np
import polars as pl
import pymc as pm
import ray
import xarray as xr

import pymc_ray as pmr

HYPERPARAMETERS = ["mu", "chol_stds", "chol_corr", "sigma"]


def simulate(n_rows: int, n_groups: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    sds, corr = np.array([1.0, 0.5]), 0.4
    cov = np.diag(sds) @ np.array([[1, corr], [corr, 1]]) @ np.diag(sds)
    effects = rng.multivariate_normal([1.0, -0.5], cov, size=n_groups)
    # Imbalanced groups: a few large, many small.
    group = rng.choice(n_groups, size=n_rows, p=rng.dirichlet(np.full(n_groups, 0.5)))
    x = rng.normal(size=n_rows)
    y = effects[group, 0] + effects[group, 1] * x + rng.normal(scale=0.8, size=n_rows)
    return pl.DataFrame({"group": group, "x": x, "y": y})


def make_model_fn(n_groups: int):
    # The number of groups is global metadata, captured here, never read from the rows,
    # so every shard builds a model with the same parameters.
    def model_fn(data) -> pm.Model:
        frame = nw.from_native(data, eager_only=True)
        group = frame["group"].to_numpy()
        coords = {"group": np.arange(n_groups), "effect": ["intercept", "slope"]}
        with pm.Model(coords=coords) as model:
            mu = pm.Normal("mu", 0, 5, dims="effect")
            chol, _, _ = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0, shape=2), compute_corr=True)
            z = pm.Normal("z", 0, 1, dims=("group", "effect"))
            effects = pm.Deterministic("effects", mu + z @ chol.T, dims=("group", "effect"))
            sigma = pm.HalfNormal("sigma", 1)
            y_hat = effects[group, 0] + effects[group, 1] * frame["x"].to_numpy()
            pm.Normal("y", y_hat, sigma, observed=frame["y"].to_numpy())
        return model

    return model_fn


def summarize(idata: xr.DataTree) -> dict[str, str]:
    posterior = idata["posterior"]
    rows = {}
    for var in HYPERPARAMETERS:
        values = posterior[var].stack(sample=("chain", "draw"))
        if var == "chol_corr":
            values = values[0, 1]
        for i, (m, s) in enumerate(
            zip(
                np.atleast_1d(values.mean("sample")),
                np.atleast_1d(values.std("sample")),
                strict=True,
            )
        ):
            rows[f"{var}[{i}]"] = f"{m:7.3f} ± {s:.3f}"
    steps = idata["sample_stats"]["n_steps"]
    rows["leapfrog steps/draw"] = f"{float(steps.mean()):7.1f}"
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--groups", type=int, default=1_000)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--replicas", type=int, default=1, help="copies of every shard actor")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--compare", action="store_true", help="also run in-process NUTS")
    args = parser.parse_args()
    sample_kwargs = {"draws": args.draws, "tune": args.draws, "chains": 4, "progressbar": False}

    ray.init(log_to_driver=False)
    data = simulate(args.rows, args.groups)
    model_fn = make_model_fn(args.groups)
    results = {}

    start = time.perf_counter()
    with pmr.ShardedModel(model_fn, data, n_shards=args.shards, n_replicas=args.replicas) as sharded:
        print(f"{sharded}: {sharded.benchmark() * 1e3:.1f} ms per logp+gradient")
        idata = sharded.sample(random_seed=1, **sample_kwargs)
    layout = f"{args.shards} shards x {args.replicas} replicas"
    results[f"ShardedModel ({layout})"] = (time.perf_counter() - start, idata)

    if args.compare:
        model = model_fn(data)
        print(f"in-process: {pmr.time_logp_dlogp(model) * 1e3:.1f} ms per logp+gradient")
        start = time.perf_counter()
        idata = pm.sample(model=model, cores=4, mp_ctx="spawn", random_seed=1, **sample_kwargs)
        results["pm.sample (in-process)"] = (time.perf_counter() - start, idata)

    summaries = {name: summarize(idata) for name, (_, idata) in results.items()}
    names = list(summaries)
    print(f"\n{'':22}" + "".join(f"{name:>30}" for name in names))
    print(f"{'wall time':22}" + "".join(f"{results[n][0]:>28.0f} s" for n in names))
    for row in next(iter(summaries.values())):
        print(f"{row:22}" + "".join(f"{summaries[n][row]:>30}" for n in names))


if __name__ == "__main__":
    main()
