"""Fit one independent PyMC model per group of a dataset, in parallel on a Ray cluster."""

from __future__ import annotations

import traceback
import warnings
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from contextlib import closing, suppress
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Literal

import cloudpickle
import narwhals.stable.v2 as nw
import pandas as pd
import pyarrow as pa
import pymc as pm
import ray

from pymc_ray._utils import (
    ModelFn,
    as_frame,
    ensure_ray_initialized,
    from_arrow,
    is_ray_dataset,
    normalize_key,
    progress_bar,
    seeds_for_key,
)

FitFn = Callable[..., Any]
PostprocessFn = Callable[[Hashable, pm.Model, Any, int | None], Any]
Outcome = tuple[Hashable, bool, Any]

_DEFAULT_SAMPLE_KWARGS = {"progressbar": False, "cores": 1}


@dataclass(frozen=True)
class GroupFailure:
    """A group whose model could not be built or fit.

    Attributes
    ----------
    key : hashable
        The group's key.
    error : str
        ``repr`` of the exception.
    traceback : str
        The formatted traceback from the Ray worker.
    """

    key: Hashable
    error: str
    traceback: str


class GroupFitError(RuntimeError):
    """Raised by :func:`fit_grouped` when a group fails and ``on_error="raise"``.

    Parameters
    ----------
    failure : GroupFailure
        The failed group, also available as the ``failure`` attribute.
    """

    def __init__(self, failure: GroupFailure):
        super().__init__(f"Fitting group {failure.key!r} failed: {failure.error}\n\nTraceback from the Ray worker:\n{failure.traceback}")
        self.failure = failure


class GroupedResults(Mapping):
    """Mapping from group key to that group's fit result.

    With the default ``fit_fn`` each value is the ``xarray.DataTree`` returned by
    ``pm.sample``. Groups that failed are excluded and reported in :attr:`failures`.

    Parameters
    ----------
    results : dict
        The fit result of every group that succeeded, by group key.
    failures : dict of hashable to GroupFailure
        Every group that failed, by group key.
    group_by : list of str
        The grouping columns.

    Attributes
    ----------
    failures : dict of hashable to GroupFailure
        Every group that failed, by group key.
    group_by : list of str
        The grouping columns, which name the key levels of :meth:`summary`.
    """

    def __init__(
        self,
        results: dict[Hashable, Any],
        failures: dict[Hashable, GroupFailure],
        group_by: list[str],
    ):
        self._results = results
        self.failures = failures
        self.group_by = group_by

    def __getitem__(self, key: Hashable) -> Any:
        """Fit result of the group ``key``.

        Parameters
        ----------
        key : hashable
            A group key: a scalar for one grouping column, a tuple for several.

        Returns
        -------
        object
            The group's (post-processed) fit result.

        Raises
        ------
        KeyError
            If no group has this key, or the group failed.
        """
        return self._results[key]

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over the keys of the fitted groups.

        Returns
        -------
        iterator of hashable
            The group keys, sorted when they are comparable.
        """
        return iter(self._results)

    def __len__(self) -> int:
        """Count the fitted groups.

        Returns
        -------
        int
            The number of groups that were fit successfully.
        """
        return len(self._results)

    def __repr__(self) -> str:
        """Show the counts of fitted and failed groups.

        Returns
        -------
        str
            The representation, e.g. ``GroupedResults(10 fitted, 0 failed)``.
        """
        return f"GroupedResults({len(self)} fitted, {len(self.failures)} failed)"

    def summary(self, **summary_kwargs) -> pd.DataFrame:
        """Concatenate ``arviz.summary`` of every group, indexed by group key and variable.

        Parameters
        ----------
        **summary_kwargs
            Passed to ``arviz.summary`` for every group.

        Returns
        -------
        pandas.DataFrame
            One row per group and variable, indexed by the ``group_by`` columns and
            ``variable``. Empty when no group was fit.
        """
        import arviz as az

        names = [*self.group_by, "variable"]
        frames = {key: pd.DataFrame(az.summary(res, **summary_kwargs)) for key, res in self.items()}
        if not frames:
            return pd.DataFrame(index=pd.MultiIndex.from_arrays([[]] * len(names), names=names))
        return pd.concat(frames, names=names)


class _GroupTask:
    """Builds and fits the model of a single group. Runs inside a Ray worker.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(group) -> pm.Model``.
    fit_fn : callable
        Called as ``fit_fn(model=model, random_seed=seed, **fit_kwargs)``.
    fit_kwargs : dict
        Extra keyword arguments for ``fit_fn``.
    postprocess_fn : callable or None
        ``postprocess_fn(key, model, result, random_seed) -> Any``, applied to the result.
    random_seed : int or None
        Base seed from which each group's seeds are derived.
    """

    def __init__(
        self,
        model_fn: ModelFn,
        fit_fn: FitFn,
        fit_kwargs: dict[str, Any],
        postprocess_fn: PostprocessFn | None,
        random_seed: int | None,
    ):
        self.model_fn = model_fn
        self.fit_fn = fit_fn
        self.fit_kwargs = fit_kwargs
        self.postprocess_fn = postprocess_fn
        self.random_seed = random_seed

    def __call__(self, key: Hashable, group: Any) -> Outcome:
        """Build, fit and post-process the model of one group, catching any failure.

        Parameters
        ----------
        key : hashable
            The group's normalized key, from which its seeds are derived.
        group : dataframe
            The group's rows.

        Returns
        -------
        key : hashable
            The group's key.
        ok : bool
            Whether the group was fit successfully.
        result : object or GroupFailure
            The (post-processed) fit result, or the failure.
        """
        try:
            model = self.model_fn(group)
            fit_seed, postprocess_seed = (None, None) if self.random_seed is None else seeds_for_key(self.random_seed, key, 2)
            result = self.fit_fn(model=model, random_seed=fit_seed, **self.fit_kwargs)
            if self.postprocess_fn is not None:
                result = self.postprocess_fn(key, model, result, postprocess_seed)
        except Exception as exc:
            return key, False, GroupFailure(key, repr(exc), traceback.format_exc())
        return key, True, result


@ray.remote
def _fit_group(task: _GroupTask, key: Hashable, group: Any) -> Outcome:
    """Run ``task`` on one group in a Ray worker.

    Parameters
    ----------
    task : _GroupTask
        The fitting task shared by all groups.
    key : hashable
        The group's normalized key.
    group : dataframe
        The group's rows.

    Returns
    -------
    tuple of (hashable, bool, object)
        The group's outcome; see :meth:`_GroupTask.__call__`.
    """
    return task(key, group)


def fit_grouped(
    data: Any,
    group_by: str | Sequence[str],
    model_fn: ModelFn,
    *,
    fit_fn: FitFn | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    postprocess_fn: PostprocessFn | None = None,
    num_cpus: float | None = None,
    max_in_flight: int | None = None,
    random_seed: int | None = None,
    on_error: Literal["raise", "warn", "ignore"] = "raise",
    progressbar: bool = True,
    backend: str = "pandas",
) -> GroupedResults:
    """Fit an independent PyMC model to every group of ``data`` in parallel with Ray.

    Parameters
    ----------
    data : dataframe or ray.data.Dataset
        The full dataset: an eager dataframe from any library narwhals supports (pandas,
        polars, pyarrow, ...), or a Ray Dataset, which is grouped and fit on the cluster
        without ever being collected on the driver, so it can exceed driver memory.
    group_by : str or sequence of str
        Column(s) defining the groups. Keys are scalars for a single column and tuples
        for a list of columns. Null keys are reported as ``None``.
    model_fn : callable
        ``model_fn(group) -> pm.Model``. Called on a Ray worker with the rows of one
        group, as a dataframe of the same library as ``data`` (or ``backend`` for a Ray
        Dataset). Must be picklable (closures and lambdas are fine).
    fit_fn : callable, optional
        Called as ``fit_fn(model=model, random_seed=seed, **fit_kwargs)``. Defaults to
        ``pm.sample``; ``pm.fit`` or a custom wrapper (e.g. around nutpie) also work.
    fit_kwargs : dict, optional
        Extra keyword arguments for ``fit_fn``. With the default ``fit_fn`` these
        default to ``progressbar=False, cores=1``: chains run sequentially within a
        group, and parallelism comes from fitting many groups at once.
    postprocess_fn : callable, optional
        ``postprocess_fn(key, model, result, random_seed) -> Any``, applied on the worker
        before the result is sent back. Use it to add to results (e.g. posterior
        predictive samples, seeded with ``random_seed``), to shrink them (e.g. return
        ``az.summary``), or to write them to storage and return a path, so that thousands
        of groups do not have to fit in driver memory. ``random_seed`` is derived like the
        seed passed to ``fit_fn`` but differs from it, and is None without a base seed.
    num_cpus : float, optional
        CPUs reserved per group. Defaults to ``fit_kwargs["cores"]`` (1 by default).
    max_in_flight : int, optional
        Upper bound on groups submitted but not yet finished (dataframe input only),
        which bounds object-store memory. Defaults to twice the number of groups the
        cluster can fit concurrently.
    random_seed : int, optional
        Base seed. Each group's seeds (for ``fit_fn`` and ``postprocess_fn``) are derived
        from it and the group key, so results are reproducible regardless of scheduling
        order.
    on_error : {"raise", "warn", "ignore"}
        What to do when a group fails. Failures are always recorded in
        :attr:`GroupedResults.failures`; with ``"raise"``, pending groups are cancelled.
    progressbar : bool
        Show a progress bar over groups (dataframe input; Ray Data reports its own).
    backend : str
        Ray Dataset input only: the dataframe library ``model_fn`` receives each group
        as, by narwhals name (``"pandas"``, ``"polars"``, ``"pyarrow"``, ...).

    Returns
    -------
    GroupedResults
        Mapping from group key to the (post-processed) fit result.

    Raises
    ------
    GroupFitError
        If a group fails and ``on_error="raise"``.
    ValueError
        If ``on_error`` is not one of the allowed values.
    TypeError
        If ``data`` is neither a dataframe nor a Ray Dataset.
    """
    if on_error not in ("raise", "warn", "ignore"):
        raise ValueError(f"on_error must be 'raise', 'warn' or 'ignore', got {on_error!r}")
    single_column = isinstance(group_by, str)
    columns = [group_by] if single_column else list(group_by)

    fit_kwargs = dict(fit_kwargs or {})
    if fit_fn is None:
        fit_fn = pm.sample
        fit_kwargs = _DEFAULT_SAMPLE_KWARGS | fit_kwargs
    if num_cpus is None:
        num_cpus = fit_kwargs.get("cores") or 1
    task = _GroupTask(model_fn, fit_fn, fit_kwargs, postprocess_fn, random_seed)

    ensure_ray_initialized()
    if is_ray_dataset(data):
        outcomes = _dataset_outcomes(data, columns, single_column, task, num_cpus, backend)
        total = None
    elif (frame := as_frame(data)) is not None:
        total = len(frame.unique(subset=columns))
        if max_in_flight is None:
            slots = int(ray.cluster_resources().get("CPU", 1) // num_cpus)
            max_in_flight = 2 * max(1, slots)
        outcomes = _frame_outcomes(frame, columns, single_column, task, num_cpus, max_in_flight)
    else:
        raise TypeError(
            f"data must be a dataframe supported by narwhals (pandas, polars, pyarrow, ...) or a ray.data.Dataset, got {type(data).__name__}"
        )

    results: dict[Hashable, Any] = {}
    failures: dict[Hashable, GroupFailure] = {}
    with closing(outcomes), progress_bar(total, progressbar, desc="Fitting groups") as bar:
        for key, ok, payload in outcomes:
            if bar is not None:
                bar.update()
            if ok:
                results[key] = payload
                continue
            failures[key] = payload
            if on_error == "raise":
                raise GroupFitError(payload)
            if on_error == "warn":
                warnings.warn(f"Fitting group {key!r} failed: {payload.error}", stacklevel=2)
    # Results arrive in completion order; present them in key order when keys are comparable.
    with suppress(TypeError):
        results = dict(sorted(results.items(), key=itemgetter(0)))
    return GroupedResults(results, failures, columns)


def _frame_outcomes(
    frame: nw.DataFrame,
    columns: list[str],
    single_column: bool,
    task: _GroupTask,
    num_cpus: float,
    max_in_flight: int,
):
    """Submit one Ray task per group, keeping at most ``max_in_flight`` running.

    Parameters
    ----------
    frame : narwhals.DataFrame
        The full dataset.
    columns : list of str
        The grouping columns.
    single_column : bool
        Whether keys are scalars (one grouping column) rather than tuples.
    task : _GroupTask
        The fitting task shared by all groups.
    num_cpus : float
        CPUs reserved by each Ray task.
    max_in_flight : int
        Maximum number of tasks submitted but not yet finished.

    Yields
    ------
    tuple of (hashable, bool, object)
        Each group's outcome, in completion order. Tasks still pending when the generator
        is closed are cancelled.
    """
    task_ref = ray.put(task)
    fit_group = _fit_group.options(num_cpus=num_cpus)
    pending: list[ray.ObjectRef] = []
    try:
        for key, group in frame.group_by(*columns):
            if len(pending) >= max_in_flight:
                done, pending = ray.wait(pending, num_returns=1)
                yield from ray.get(done)
            key = normalize_key(key[0] if single_column else key)
            pending.append(fit_group.remote(task_ref, key, group.to_native()))
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            yield from ray.get(done)
    finally:
        for ref in pending:
            ray.cancel(ref)


def _dataset_outcomes(
    dataset: ray.data.Dataset,
    columns: list[str],
    single_column: bool,
    task: _GroupTask,
    num_cpus: float,
    backend: str,
):
    """Fit groups with ``Dataset.groupby().map_groups()`` so data never visits the driver.

    Parameters
    ----------
    dataset : ray.data.Dataset
        The full dataset.
    columns : list of str
        The grouping columns.
    single_column : bool
        Whether keys are scalars (one grouping column) rather than tuples.
    task : _GroupTask
        The fitting task shared by all groups.
    num_cpus : float
        CPUs reserved by each ``map_groups`` worker.
    backend : str
        The dataframe library ``model_fn`` receives each group as.

    Yields
    ------
    tuple of (hashable, bool, object)
        Each group's outcome.
    """

    def fit_group(group: pa.Table) -> pa.Table:
        """Fit one group and return its pickled outcome as a one-row table.

        Parameters
        ----------
        group : pyarrow.Table
            The group's rows.

        Returns
        -------
        pyarrow.Table
            A single ``outcome`` column holding the cloudpickled outcome.
        """
        key = tuple(group.column(column)[0].as_py() for column in columns)
        key = normalize_key(key[0] if single_column else key)
        outcome = task(key, from_arrow(group, backend))
        return pa.table({"outcome": pa.array([cloudpickle.dumps(outcome)], pa.large_binary())})

    fitted = dataset.groupby(columns).map_groups(fit_group, batch_format="pyarrow", num_cpus=num_cpus)
    for row in fitted.iter_rows():
        yield cloudpickle.loads(row["outcome"])
