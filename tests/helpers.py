"""Test utilities that are not model functions (Ray workers never import this)."""

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pyarrow as pa

BACKENDS = ["pandas", "polars", "pyarrow"]


def to_backend(df: pd.DataFrame, backend: str):
    """Convert a pandas DataFrame to the native frame of a narwhals backend."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    return nw.from_arrow(table, backend=backend).to_native()


def random_points(model, n=3, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    base = model.initial_point()
    return [{k: v + rng.normal(scale=scale, size=np.shape(v)) for k, v in base.items()} for _ in range(n)]


def assert_same_density(sharded_model, full_model):
    """The sharded model must have exactly the full-data logp and gradient."""
    logp_sharded, logp_full = sharded_model.compile_logp(), full_model.compile_logp()
    dlogp_sharded, dlogp_full = sharded_model.compile_dlogp(), full_model.compile_dlogp()
    for point in random_points(full_model):
        np.testing.assert_allclose(logp_sharded(point), logp_full(point), rtol=1e-10)
        np.testing.assert_allclose(dlogp_sharded(point), dlogp_full(point), rtol=1e-8, atol=1e-8)
