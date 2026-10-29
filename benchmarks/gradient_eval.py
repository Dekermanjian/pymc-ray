"""Time one joint logp + gradient evaluation: in-process PyMC vs. ShardedModel.

NUTS makes one such evaluation per leapfrog step, so this ratio is (roughly) the
sampling speedup. Layouts are SHARDSxREPLICAS; with --chains N, N processes evaluate
concurrently, as chains do while sampling. Usage:

    python benchmarks/gradient_eval.py --model logistic --rows 100000 1000000
    python benchmarks/gradient_eval.py --model multilevel --rows 1000000 \
        --chains 4 --layouts 8x1 4x2 2x4 3x4
"""

import argparse
from collections.abc import Callable

import narwhals.stable.v2 as nw
import numpy as np
import polars as pl
import pymc as pm
import ray

import pymc_ray as pmr

N_FEATURES = 20
ROWS_PER_GROUP = 1_000


def logistic_model_fn(n_rows: int) -> Callable:
    def model_fn(data) -> pm.Model:
        frame = nw.from_native(data, eager_only=True)
        X = frame.select([f"x{i}" for i in range(N_FEATURES)]).to_numpy()
        with pm.Model() as model:
            intercept = pm.Normal("intercept", 0, 2)
            beta = pm.Normal("beta", 0, 1, shape=N_FEATURES)
            pm.Bernoulli("y", logit_p=intercept + X @ beta, observed=frame["y"].to_numpy())
        return model

    return model_fn


def simulate_logistic(n_rows: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, N_FEATURES))
    beta = rng.normal(scale=0.5, size=N_FEATURES)
    y = rng.random(n_rows) < 1 / (1 + np.exp(-(X @ beta - 0.5)))
    return pl.DataFrame(X, schema=[f"x{i}" for i in range(N_FEATURES)]).with_columns(y=pl.Series(y.astype("int8")))


def multilevel_model_fn(n_rows: int, n_groups: int | None = None) -> Callable:
    """Correlated varying intercepts and slopes (non-centered, LKJ prior)."""
    n_groups = n_groups or max(1, n_rows // ROWS_PER_GROUP)

    def model_fn(data) -> pm.Model:
        frame = nw.from_native(data, eager_only=True)
        group = frame["group"].to_numpy()
        coords = {"group": np.arange(n_groups), "effect": ["intercept", "slope"]}
        with pm.Model(coords=coords) as model:
            mu = pm.Normal("mu", 0, 5, dims="effect")
            chol, _, _ = pm.LKJCholeskyCov("chol", n=2, eta=2.0, sd_dist=pm.Exponential.dist(1.0, shape=2))
            z = pm.Normal("z", 0, 1, dims=("group", "effect"))
            effects = mu + z @ chol.T
            sigma = pm.HalfNormal("sigma", 1)
            y_hat = effects[group, 0] + effects[group, 1] * frame["x"].to_numpy()
            pm.Normal("y", y_hat, sigma, observed=frame["y"].to_numpy())
        return model

    return model_fn


def simulate_multilevel(n_rows: int, seed: int = 0, n_groups: int | None = None) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n_groups = n_groups or max(1, n_rows // ROWS_PER_GROUP)
    effects = rng.multivariate_normal([1.0, -0.5], [[1.0, 0.2], [0.2, 0.25]], size=n_groups)
    group = rng.integers(0, n_groups, size=n_rows)
    x = rng.normal(size=n_rows)
    y = effects[group, 0] + effects[group, 1] * x + rng.normal(scale=0.8, size=n_rows)
    return pl.DataFrame({"group": group, "x": x, "y": y})


MODELS = {
    "logistic": (logistic_model_fn, simulate_logistic),
    "multilevel": (multilevel_model_fn, simulate_multilevel),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="logistic")
    parser.add_argument("--rows", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--layouts", nargs="+", default=["1x1", "2x1", "4x1", "8x1", "12x1"])
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--shard-by", help="keep this column's groups on one shard each")
    parser.add_argument(
        "--relays",
        choices=["auto", "on", "off"],
        default="auto",
        help="one relay per node (auto: on when shards span several nodes)",
    )
    parser.add_argument("--evals", type=int, default=20)
    args = parser.parse_args()
    relays = {"auto": None, "on": True, "off": False}[args.relays]
    make_model_fn, simulate = MODELS[args.model]
    layouts = [tuple(int(n) for n in layout.split("x")) for layout in args.layouts]

    ray.init(log_to_driver=False, include_dashboard=False)
    print(f"{args.chains} concurrent chain(s); layouts are shards x replicas")
    print(f"{'rows':>10} {'in-process':>12} " + " ".join(f"{s:>7}" for s in args.layouts))
    for n_rows in args.rows:
        data, model_fn = simulate(n_rows), make_model_fn(n_rows)
        baseline = pmr.time_logp_dlogp(model_fn(data), args.evals, args.chains)
        cells = []
        for n_shards, n_replicas in layouts:
            with pmr.ShardedModel(
                model_fn,
                data,
                n_shards=n_shards,
                n_replicas=n_replicas,
                shard_by=args.shard_by,
                relays=relays,
            ) as sharded:
                cells.append(sharded.benchmark(args.evals, args.chains))
        row = f"{n_rows:>10,} {baseline * 1e3:>9.2f} ms "
        row += " ".join(f"{t * 1e3:>4.1f} ms" for t in cells)
        print(row, flush=True)


if __name__ == "__main__":
    main()
