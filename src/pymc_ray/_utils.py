"""Helpers shared across pymc-ray: data containers, seeding and Ray setup.

pymc-ray accepts three kinds of data:

- an eager dataframe from any library narwhals supports (pandas, polars, pyarrow,
  modin, cuDF, ...); model functions receive row subsets of the same native type,
- a ``dict`` mapping names to arrays that share their leading (row) dimension,
- a ``ray.data.Dataset``, which is materialized on the worker that owns each shard or
  group and converted to the dataframe library named by ``backend``.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from contextlib import contextmanager
from itertools import pairwise
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pyarrow as pa
import pymc as pm
import ray

Data = Any
"""A native eager dataframe, a dict of arrays, or a ``ray.data.Dataset``."""

ModelFn = Callable[[Any], pm.Model]
"""``model_fn(data) -> pm.Model``.

Called with a dataframe or a dict of arrays, never a Ray Dataset, which is materialized first.
"""


def ensure_ray_initialized() -> None:
    """Connect to Ray (or start a local instance) if this process is not connected yet."""
    if not ray.is_initialized():
        ray.init()


def is_ray_dataset(data: Any) -> bool:
    """Whether ``data`` is a ``ray.data.Dataset``.

    Parameters
    ----------
    data : object
        Any object.

    Returns
    -------
    bool
        True for a Ray Dataset.
    """
    return isinstance(data, ray.data.Dataset)


def as_frame(data: Any) -> nw.DataFrame | None:
    """Wrap an eager dataframe with narwhals; return None for containers that are not one.

    Parameters
    ----------
    data : object
        A native dataframe, or any other container.

    Returns
    -------
    narwhals.DataFrame or None
        The wrapped dataframe, or None if ``data`` is not an eager dataframe.

    Raises
    ------
    TypeError
        If ``data`` is a lazy frame.
    """
    frame = nw.from_native(data, pass_through=True)
    if isinstance(frame, nw.LazyFrame):
        raise TypeError(
            "Lazy frames are not supported. Collect it first, or load the data as a ray.data.Dataset so that it never has to fit on the driver."
        )
    return frame if isinstance(frame, nw.DataFrame) else None


def num_rows(data: Data) -> int:
    """Count the rows of an eager dataframe or of a dict of arrays.

    Parameters
    ----------
    data : dataframe or dict of arrays
        The container (not a Ray Dataset).

    Returns
    -------
    int
        The number of rows; for a dict, the shared length of its arrays.

    Raises
    ------
    ValueError
        If the arrays of a dict differ in length.
    TypeError
        If ``data`` is neither a dataframe nor a mapping.
    """
    frame = as_frame(data)
    if frame is not None:
        return len(frame)
    if isinstance(data, Mapping):
        lengths = {len(value) for value in data.values()}
        if len(lengths) != 1:
            raise ValueError(f"All arrays in a data dict must share their first dimension, got lengths {lengths}")
        return lengths.pop()
    raise TypeError(
        "data must be a dataframe supported by narwhals (pandas, polars, pyarrow, ...), "
        f"a dict of arrays or a ray.data.Dataset, got {type(data).__name__}"
    )


def slice_rows(data: Data, start: int, stop: int) -> Data:
    """Rows ``start`` to ``stop`` (exclusive) of a dataframe or dict of arrays.

    Parameters
    ----------
    data : dataframe or dict of arrays
        The container to slice.
    start, stop : int
        Positions of the first row and of one past the last row.

    Returns
    -------
    dataframe or dict of numpy.ndarray
        The rows, as a native dataframe of the same library or a dict of arrays.
    """
    frame = as_frame(data)
    if frame is not None:
        # Python ints: narwhals reads a slice of numpy integers as labels, not positions.
        return frame[int(start) : int(stop)].to_native()
    return {name: np.asarray(value)[start:stop] for name, value in data.items()}


def empty_like(data: Data) -> Data:
    """Zero-row version of ``data`` with the same columns and dtypes.

    Parameters
    ----------
    data : dataframe or dict of arrays
        The container.

    Returns
    -------
    dataframe or dict of numpy.ndarray
        Zero rows of ``data``.
    """
    return slice_rows(data, 0, 0)


def take_rows(data: Data, rows: np.ndarray) -> Data:
    """Rows of a dataframe or dict of arrays at the given positions.

    Parameters
    ----------
    data : dataframe or dict of arrays
        The container to index.
    rows : numpy.ndarray
        Integer positions of the rows to take, in the order to return them.

    Returns
    -------
    dataframe or dict of numpy.ndarray
        The rows, as a native dataframe of the same library or a dict of arrays.
    """
    frame = as_frame(data)
    if frame is not None:
        return frame[rows].to_native()
    return {name: np.asarray(value)[rows] for name, value in data.items()}


def split_data(data: Any, n_shards: int, by: Sequence[str] | None = None) -> list[Any]:
    """Split ``data`` into ``n_shards`` shards of (nearly) equal size.

    Without ``by``, shards are contiguous row ranges. With ``by``, all rows sharing a value
    of the ``by`` column(s) go to the same shard, and groups are spread so that shards
    hold similar numbers of rows. Ray Datasets are split with ``split_at_indices`` so that
    no rows are dropped and no shard is empty, both of which would bias inference.

    Parameters
    ----------
    data : dataframe, dict of arrays or ray.data.Dataset
        The data to split.
    n_shards : int
        Number of shards.
    by : sequence of str, optional
        Column(s) whose rows must stay on the same shard.

    Returns
    -------
    list
        The shards, each the same kind of container as ``data``.

    Raises
    ------
    ValueError
        If ``n_shards`` is less than 1 or exceeds the number of rows (or of groups, with
        ``by``).
    """
    return split_with_positions(data, n_shards, by)[0]


def split_with_positions(data: Any, n_shards: int, by: Sequence[str] | None = None) -> tuple[list[Any], list[np.ndarray] | None]:
    """Like :func:`split_data`, also returning the positions in ``data`` of each shard's rows.

    Positions are None for Ray Datasets, whose rows the driver never sees. Without ``by``
    their shards are still contiguous and in order; with ``by`` the original order is lost.

    Parameters
    ----------
    data : dataframe, dict of arrays or ray.data.Dataset
        The data to split.
    n_shards : int
        Number of shards.
    by : sequence of str, optional
        Column(s) whose rows must stay on the same shard.

    Returns
    -------
    shards : list
        The shards, as returned by :func:`split_data`.
    positions : list of numpy.ndarray or None
        For each shard, the positions of its rows in ``data``; None for a Ray Dataset.

    Raises
    ------
    ValueError
        If ``n_shards`` is less than 1 or exceeds the number of rows (or of groups, with
        ``by``).
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be at least 1, got {n_shards}")
    if is_ray_dataset(data):
        if by is not None:
            return _split_dataset_by_key(data, n_shards, list(by)), None
        n = data.count()
        if n_shards > n:
            raise ValueError(f"Cannot split {n} rows into {n_shards} non-empty shards")
        bounds = np.linspace(0, n, n_shards + 1).astype(int)
        return data.split_at_indices(bounds[1:-1].tolist()), None
    if by is not None:
        positions = _positions_by_key(data, n_shards, list(by))
        return [take_rows(data, rows) for rows in positions], positions
    n = num_rows(data)
    if n_shards > n:
        raise ValueError(f"Cannot split {n} rows into {n_shards} non-empty shards")
    bounds = np.linspace(0, n, n_shards + 1).astype(int)
    shards = [slice_rows(data, start, stop) for start, stop in pairwise(bounds)]
    return shards, [np.arange(start, stop) for start, stop in pairwise(bounds)]


_SHARD_COLUMN = "__pymc_ray_shard"


def _assign_groups(counts: np.ndarray, n_shards: int) -> np.ndarray:
    """Shard of each group: largest groups first, each to the shard with fewest rows.

    Parameters
    ----------
    counts : numpy.ndarray
        Number of rows of each group.
    n_shards : int
        Number of shards.

    Returns
    -------
    numpy.ndarray
        The shard index of each group.

    Raises
    ------
    ValueError
        If there are fewer groups than shards.
    """
    if n_shards > len(counts):
        raise ValueError(f"Cannot split {len(counts)} groups into {n_shards} shards")
    shards = [(0, shard) for shard in range(n_shards)]
    assignment = np.empty(len(counts), dtype=np.int64)
    for group in np.argsort(-counts, kind="stable"):
        rows, shard = heapq.heappop(shards)
        assignment[group] = shard
        heapq.heappush(shards, (rows + int(counts[group]), shard))
    return assignment


def _positions_by_key(data: Data, n_shards: int, columns: list[str]) -> list[np.ndarray]:
    """Positions of each shard's rows when the groups of ``columns`` are kept together.

    Parameters
    ----------
    data : dataframe or dict of arrays
        The data to split.
    n_shards : int
        Number of shards.
    columns : list of str
        Columns (or dict keys) whose distinct combinations of values define the groups.

    Returns
    -------
    list of numpy.ndarray
        For each shard, the sorted positions of its rows in ``data``.

    Raises
    ------
    ValueError
        If a column is missing or, for a dataframe, contains nulls, or if there are fewer
        groups than shards.
    """
    frame = as_frame(data)
    if frame is not None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"shard_by columns {missing} are not in the data")
        if any(frame[column].is_null().any() for column in columns):
            raise ValueError(f"shard_by columns {columns} must not contain nulls")
        keys = [frame[column].to_numpy() for column in columns]
    else:
        num_rows(data)  # validates the container
        missing = [column for column in columns if column not in data]
        if missing:
            raise ValueError(f"shard_by keys {missing} are not in the data")
        keys = [np.asarray(data[column]) for column in columns]

    # One dense code per distinct key (re-densified per column so codes stay small).
    codes = np.zeros(len(keys[0]), dtype=np.int64)
    for values in keys:
        _, inverse = np.unique(values, return_inverse=True)
        _, codes = np.unique(codes * (inverse.max() + 1) + inverse, return_inverse=True)
    counts = np.bincount(codes)
    shard_of_row = _assign_groups(counts, n_shards)[codes]
    return [np.flatnonzero(shard_of_row == shard) for shard in range(n_shards)]


def _split_dataset_by_key(dataset: Any, n_shards: int, columns: list[str]) -> list[Any]:
    """Tag each row with its group's shard, sort by shard, and split at the boundaries.

    Parameters
    ----------
    dataset : ray.data.Dataset
        The data to split.
    n_shards : int
        Number of shards.
    columns : list of str
        Columns whose distinct combinations of values define the groups.

    Returns
    -------
    list of ray.data.Dataset
        The shards, with each group's rows on a single shard.

    Raises
    ------
    ValueError
        If a column contains nulls, or if there are fewer groups than shards.
    """
    sizes = dataset.groupby(columns).count().to_pandas()
    if sizes[columns].isna().to_numpy().any():
        raise ValueError(f"shard_by columns {columns} must not contain nulls")
    (count_column,) = [column for column in sizes.columns if column not in columns]
    sizes[_SHARD_COLUMN] = _assign_groups(sizes[count_column].to_numpy(), n_shards)
    shard_rows = sizes.groupby(_SHARD_COLUMN)[count_column].sum().sort_index().to_numpy()
    lookup = sizes[[*columns, _SHARD_COLUMN]]

    def tag(batch):
        """Add the shard column to a batch by joining it with the groups' shards.

        Parameters
        ----------
        batch : pandas.DataFrame
            A batch of rows of ``dataset``.

        Returns
        -------
        pandas.DataFrame
            ``batch`` with a column holding each row's shard.
        """
        return batch.merge(lookup, on=columns, how="left", validate="many_to_one")

    ordered = dataset.map_batches(tag, batch_format="pandas").sort(_SHARD_COLUMN)
    splits = ordered.split_at_indices(np.cumsum(shard_rows)[:-1].tolist())
    return [split.drop_columns([_SHARD_COLUMN]) for split in splits]


def from_arrow(table: pa.Table, backend: str) -> Data:
    """Convert an Arrow table to a native dataframe of the narwhals ``backend``.

    Parameters
    ----------
    table : pyarrow.Table
        The table to convert.
    backend : str
        The dataframe library, by narwhals name (``"pandas"``, ``"polars"``, ...).

    Returns
    -------
    dataframe
        The table as a native dataframe of ``backend``.
    """
    return nw.from_arrow(table, backend=backend).to_native()


def materialize(shard: Any, backend: str) -> Data:
    """Load a Ray Dataset shard as a ``backend`` dataframe; other containers pass through.

    Parameters
    ----------
    shard : object
        A Ray Dataset shard, or data already in memory.
    backend : str
        The dataframe library to load a Ray Dataset shard as, by narwhals name.

    Returns
    -------
    dataframe or dict of arrays
        The shard's rows, in memory.
    """
    if not is_ray_dataset(shard):
        return shard
    return from_arrow(pa.concat_tables(ray.get(shard.to_arrow_refs())), backend)


def normalize_key(key: Hashable) -> Hashable:
    """Convert a group key to plain Python scalars.

    Dataframe libraries return different scalar types for the same group (numpy
    integers, NaN or None for nulls); normalizing makes keys, and the seeds derived from
    them, identical whichever library and code path produced them.

    Parameters
    ----------
    key : hashable
        A group key: a scalar, or a tuple of scalars for several columns.

    Returns
    -------
    hashable
        The key, with numpy scalars converted to Python scalars and NaN to None.
    """
    if isinstance(key, tuple):
        return tuple(normalize_key(part) for part in key)
    if isinstance(key, np.generic):
        key = key.item()
    if isinstance(key, float) and math.isnan(key):
        return None
    return key


def seeds_for_key(base_seed: int, key: Hashable, n: int) -> list[int]:
    """Derive ``n`` reproducible seeds for ``key`` that do not depend on scheduling order.

    The first seeds do not depend on ``n``, so asking for more seeds leaves them unchanged.

    Parameters
    ----------
    base_seed : int
        The user's base seed.
    key : hashable
        The group key. It is hashed through its ``repr``, so normalize it first with
        :func:`normalize_key`.
    n : int
        Number of seeds.

    Returns
    -------
    list of int
        The seeds, each a 32-bit unsigned integer.
    """
    digest = hashlib.sha256(repr(key).encode()).digest()
    sequence = np.random.SeedSequence([base_seed, int.from_bytes(digest[:8], "little")])
    return [int(seed) for seed in sequence.generate_state(n)]


def spawn_seeds(base_seed: int | None, n: int) -> list[int | None]:
    """Derive ``n`` independent seeds from ``base_seed``, or ``n`` Nones without one.

    Parameters
    ----------
    base_seed : int or None
        The user's base seed. None leaves every stream unseeded.
    n : int
        Number of seeds.

    Returns
    -------
    list of int or None
        One seed per stream, from the children of ``numpy.random.SeedSequence(base_seed)``.
    """
    if base_seed is None:
        return [None] * n
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(base_seed).spawn(n)]


def free_rv_shapes(model: pm.Model) -> dict[str, tuple[int, ...]]:
    """Shape of every free random variable of ``model``, by name.

    Parameters
    ----------
    model : pm.Model
        The model.

    Returns
    -------
    dict of str to tuple of int
        Each free random variable's name, mapped to its shape.
    """
    shapes = model.eval_rv_shapes()
    return {rv.name: tuple(int(d) for d in shapes[rv.name]) for rv in model.free_RVs}


def validate_likelihood_potentials(model: pm.Model, names: tuple[str, ...]) -> None:
    """Check that every name in ``names`` is a Potential of ``model``.

    Parameters
    ----------
    model : pm.Model
        The model.
    names : tuple of str
        The names given as ``likelihood_potentials``.

    Raises
    ------
    ValueError
        If a name is not a Potential of the model.
    """
    unknown = set(names) - {potential.name for potential in model.potentials}
    if unknown:
        raise ValueError(f"likelihood_potentials {sorted(unknown)} are not Potentials of the model")


@contextmanager
def progress_bar(total: int | None, enabled: bool, desc: str):
    """Context manager for an optional tqdm progress bar.

    Parameters
    ----------
    total : int or None
        Expected number of updates; None if unknown.
    enabled : bool
        Whether to show the bar.
    desc : str
        Label printed before the bar.

    Yields
    ------
    tqdm.auto.tqdm or None
        The bar, or None when it is disabled.
    """
    if not enabled:
        yield None
        return
    from tqdm.auto import tqdm

    with tqdm(total=total, desc=desc) as bar:
        yield bar
