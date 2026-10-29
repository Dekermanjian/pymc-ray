"""One logistic regression on more data than you want in a single process.

Compares the two strategies in pymc-ray:

- ``ShardedModel``: exact. NUTS runs on the driver; every logp/gradient evaluation is
  summed over shards held by Ray actors.
- ``consensus_sample``: approximate. Each shard is sampled independently and the draws
  are combined (consensus Monte Carlo); no communication while sampling.

Run with: python examples/large_data.py --rows 2000000
"""

import argparse
import time

import numpy as np
import pandas as pd
import pymc as pm
import ray

import pymc_ray as pmr

N_FEATURES = 10


def simulate(n_rows: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, N_FEATURES))
    beta = rng.normal(scale=0.5, size=N_FEATURES)
    y = rng.random(n_rows) < 1 / (1 + np.exp(-(X @ beta - 0.5)))
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
    df["y"] = y.astype("int8")
    return df, beta


def logistic_model(df: pd.DataFrame) -> pm.Model:
    # Parameter shapes come from the fixed feature list, never from the rows received.
    X = df[[f"x{i}" for i in range(N_FEATURES)]].to_numpy()
    with pm.Model(coords={"feature": [f"x{i}" for i in range(N_FEATURES)]}) as model:
        intercept = pm.Normal("intercept", 0, 2)
        beta = pm.Normal("beta", 0, 1, dims="feature")
        pm.Bernoulli("y", logit_p=intercept + X @ beta, observed=df["y"].to_numpy())
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--draws", type=int, default=500)
    args = parser.parse_args()
    sample_kwargs = {"draws": args.draws, "tune": args.draws, "chains": 4, "random_seed": 1}

    ray.init(log_to_driver=False)
    df, true_beta = simulate(args.rows)
    # With real data, pass ray.data.read_parquet(...) instead: each actor then loads its
    # own shard and the full dataset never has to fit on the driver.

    start = time.perf_counter()
    with pmr.ShardedModel(logistic_model, df, n_shards=args.shards) as sharded:
        in_process = pmr.time_logp_dlogp(logistic_model(df))
        print(f"{sharded}: {sharded.benchmark() * 1e3:.1f} ms per logp+gradient")
        print(f"  (full data in one process: {in_process * 1e3:.1f} ms)")
        exact = sharded.sample(**sample_kwargs)
    exact_time = time.perf_counter() - start

    start = time.perf_counter()
    approx = pmr.consensus_sample(
        logistic_model,
        df,
        n_shards=args.shards,
        sample_kwargs={k: v for k, v in sample_kwargs.items() if k != "random_seed"},
        random_seed=1,
    )
    approx_time = time.perf_counter() - start

    def beta_summary(idata):
        beta = idata["posterior"]["beta"]
        return beta.mean(("chain", "draw")).to_numpy(), beta.std(("chain", "draw")).to_numpy()

    (exact_mean, exact_sd), (approx_mean, approx_sd) = beta_summary(exact), beta_summary(approx)
    print(f"\nShardedModel (exact):  {exact_time:6.1f} s")
    print(f"consensus (approx):    {approx_time:6.1f} s\n")
    print(
        pd.DataFrame(
            {
                "true": true_beta,
                "exact_mean": exact_mean,
                "consensus_mean": approx_mean,
                "exact_sd": exact_sd,
                "consensus_sd": approx_sd,
            },
            index=[f"x{i}" for i in range(N_FEATURES)],
        ).round(4)
    )


if __name__ == "__main__":
    main()
