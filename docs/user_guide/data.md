# Data: bring your own dataframe library

Data handling goes through [narwhals](https://narwhals-dev.github.io/narwhals/), so you are not tied to pandas. Every function accepts:

- **An eager dataframe from any library narwhals supports**: pandas, polars, pyarrow, modin, cuDF and others. Your `model_fn` receives row subsets **of the same type**: pass polars in, get polars shards or groups back.

- **A `ray.data.Dataset`** for data larger than the driver's memory, e.g. `ray.data.read_parquet(...)`. Each worker loads only its own rows, as Arrow, and converts them to the library named by `backend=` (`"pandas"` by default; also `"polars"`, `"pyarrow"`, ...).

- **A dict of equal-length arrays** ({class}`~pymc_ray.ShardedModel` and {func}`~pymc_ray.consensus_sample` only).

Lazy frames (polars `LazyFrame`, DuckDB, ...) are rejected rather than silently collected onto the driver: collect them yourself, or read the data as a Ray Dataset.

To write one `model_fn` that works with any of these libraries, read columns through narwhals:

```python
import narwhals.stable.v2 as nw


def model_fn(data):
    frame = nw.from_native(data, eager_only=True)
    X = frame.select("x0", "x1").to_numpy()
    y = frame["y"].to_numpy()
    with pm.Model() as model:
        ...
    return model
```
