from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ray

from helpers import BACKENDS, to_backend


@pytest.fixture(scope="session", autouse=True)
def ray_cluster():
    # Workers must be able to import tests/models.py to unpickle the model functions.
    tests_dir = str(Path(__file__).parent)
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"env_vars": {"PYTHONPATH": tests_dir}},
    )
    yield
    ray.shutdown()


@pytest.fixture(params=BACKENDS)
def backend(request) -> str:
    return request.param


@pytest.fixture
def as_backend(backend):
    """Convert a pandas DataFrame to the dataframe library under test."""
    return lambda df: to_backend(df, backend)


@pytest.fixture(scope="session")
def regression_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 2_000
    x = rng.normal(size=(n, 2))
    group = rng.integers(0, 3, size=n)
    y = 0.5 + x @ np.array([1.0, -2.0]) + np.array([-1.0, 0.0, 1.0])[group]
    y += rng.normal(scale=0.7, size=n)
    return pd.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "group": group, "y": y})
