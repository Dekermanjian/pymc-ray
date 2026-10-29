"""Posterior predictive sampling, pointwise log-likelihood and PSIS-LOO over sharded data.

The observed data stay where they are: on the shard actors of a :class:`ShardedModel`, or
on short-lived Ray tasks for a posterior from anywhere else (e.g. :func:`consensus_sample`).
Each shard handles its own rows. Posterior predictive draws and log-likelihoods are
stitched back into the original row order on the driver. For LOO, each shard reduces its
rows to one value per row first, so even data that never fit on the driver can be checked.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
import ray
import xarray as xr
from arviz_stats.utils import ELPDDataLOO

from pymc_ray._utils import (
    Data,
    ModelFn,
    ensure_ray_initialized,
    materialize,
    num_rows,
    spawn_seeds,
    split_with_positions,
)

PostprocessFn = Callable[[int, pm.Model, xr.DataTree], Any]
SAMPLE_DIMS = ("chain", "draw")


# Per-shard work: runs where the shard's data live, on its model.


def _free_variables(posterior: xr.Dataset, model: pm.Model) -> xr.Dataset:
    """Select the posterior's free variables only.

    The shard recomputes everything that depends on its rows, such as deterministics,
    from them.

    Parameters
    ----------
    posterior : xarray.Dataset
        A posterior.
    model : pm.Model
        The shard's model.

    Returns
    -------
    xarray.Dataset
        The draws of the model's free random variables.

    Raises
    ------
    ValueError
        If the posterior lacks draws of a free random variable of the model.
    """
    names = [rv.name for rv in model.free_RVs]
    missing = [name for name in names if name not in posterior]
    if missing:
        raise ValueError(f"The posterior lacks draws of the model's free variables {missing}")
    return posterior[names]


def shard_posterior_predictive(
    model: pm.Model,
    posterior: xr.Dataset,
    var_names: Sequence[str] | None,
    seed: int | None,
    shard: int,
    postprocess_fn: PostprocessFn | None,
) -> Any:
    """Posterior predictive draws for the rows of one shard.

    Parameters
    ----------
    model : pm.Model
        The shard's model, built on its rows.
    posterior : xarray.Dataset
        The posterior; only its free variables are used.
    var_names : sequence of str or None
        Variables to sample. None samples the observed variables.
    seed : int or None
        Seed for ``pm.sample_posterior_predictive``.
    shard : int
        The shard's index, passed to ``postprocess_fn``.
    postprocess_fn : callable or None
        ``postprocess_fn(shard, model, result) -> Any``, applied to the result.

    Returns
    -------
    dict of str to xarray.Dataset, or object
        The result's groups (``posterior_predictive`` and ``observed_data``) by name, or
        the value returned by ``postprocess_fn``.
    """
    trace = xr.DataTree.from_dict({"posterior": _free_variables(posterior, model)})
    result = pm.sample_posterior_predictive(trace, model=model, var_names=var_names, random_seed=seed, progressbar=False)
    if postprocess_fn is not None:
        return postprocess_fn(shard, model, result)
    return {name: result[name].to_dataset() for name in result.children}


def shard_log_likelihood(model: pm.Model, posterior: xr.Dataset, var_names: Sequence[str] | None) -> xr.Dataset:
    """Pointwise log-likelihood of the rows of one shard.

    Parameters
    ----------
    model : pm.Model
        The shard's model, built on its rows.
    posterior : xarray.Dataset
        The posterior; only its free variables are used.
    var_names : sequence of str or None
        Observed variables to compute. None computes all of them.

    Returns
    -------
    xarray.Dataset
        The log-likelihood of each observed variable, with ``chain``, ``draw`` and row
        dimensions.
    """
    trace = xr.DataTree.from_dict({"posterior": _free_variables(posterior, model)})
    return pm.compute_log_likelihood(trace, model=model, var_names=var_names, extend_inferencedata=False, progressbar=False)


def shard_loo(model: pm.Model, posterior: xr.Dataset, var_name: str | None, reff: float) -> tuple[xr.DataArray, xr.DataArray, float]:
    """Pointwise PSIS-LOO for this shard's rows: elpd_i, pareto_k and the shard's p_loo.

    Pareto smoothing works observation by observation, so this equals the full-data
    computation restricted to these rows.

    Parameters
    ----------
    model : pm.Model
        The shard's model, built on its rows.
    posterior : xarray.Dataset
        The posterior.
    var_name : str or None
        Observed variable to cross-validate. May be None only when the model has a
        single observed variable.
    reff : float
        Relative MCMC efficiency of the full posterior.

    Returns
    -------
    elpd_i : xarray.DataArray
        Pointwise expected log predictive density of the shard's rows.
    pareto_k : xarray.DataArray
        Pareto shape estimate of each row.
    p_loo : float
        The shard's contribution to the effective number of parameters.

    Raises
    ------
    ValueError
        If ``var_name`` is None and the model has several observed variables.
    """
    if var_name is None:
        if len(model.observed_RVs) != 1:
            names = [rv.name for rv in model.observed_RVs]
            raise ValueError(f"The model has several observed variables {names}; pass var_name")
        var_name = model.observed_RVs[0].name
    log_lik = shard_log_likelihood(model, posterior, [var_name])
    data = xr.DataTree.from_dict({"posterior": posterior, "log_likelihood": log_lik})
    with warnings.catch_warnings():
        # Pareto k warnings are raised once, for all rows, on the driver.
        warnings.simplefilter("ignore")
        result = az.loo(data, pointwise=True, var_name=var_name, reff=reff)
    return result.elpd_i, result.pareto_k, float(result.p)


# Where the shards live.


class ActorShards:
    """The shards of a ShardedModel: work runs on the actors that already hold them.

    Parameters
    ----------
    actors : list of ray.actor.ActorHandle
        One shard actor per shard, in shard order.
    positions : list of numpy.ndarray or None
        For each shard, the positions of its rows in the data; None for a Ray Dataset.
    rows : list of int
        Number of rows of each shard.
    ordered : bool
        Whether the shards' rows, concatenated, are in the order of the data. Used only
        when ``positions`` is None.
    """

    def __init__(self, actors, positions, rows, ordered):
        self.n_shards = len(actors)
        self.actors = actors
        self.positions = positions
        self.rows = rows
        self.ordered = ordered

    def run(self, work: Callable, per_shard_args: Sequence[tuple]) -> list[Any]:
        """Run ``work(model, *args)`` on the actor of every shard, with that shard's args.

        Parameters
        ----------
        work : callable
            ``work(model, *args)``, run on each shard's model.
        per_shard_args : sequence of tuple
            Extra positional arguments for each shard, in shard order.

        Returns
        -------
        list
            The result of every shard, in shard order.
        """
        calls = zip(self.actors, per_shard_args, strict=True)
        return ray.get([actor.run.remote(work, *args) for actor, args in calls])


@ray.remote
def _on_shard(model_fn: ModelFn, shard: Any, backend: str, work: Callable, *args) -> tuple:
    """Load a shard, build its model and run ``work`` on it, in a Ray task.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(data) -> pm.Model``.
    shard : object
        The shard's data, or a Ray Dataset shard to load as ``backend``.
    backend : str
        The dataframe library to load a Ray Dataset shard as.
    work : callable
        ``work(model, *args)``.
    *args
        Extra positional arguments for ``work``.

    Returns
    -------
    n_rows : int
        Number of rows of the shard.
    result : object
        The value returned by ``work``.
    """
    data = materialize(shard, backend)
    return num_rows(data), work(model_fn(data), *args)


class TaskShards:
    """Shards of ``data`` loaded by short-lived Ray tasks, for posteriors from anywhere.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(data) -> pm.Model``, called on the rows of each shard.
    data : dataframe, dict of arrays or ray.data.Dataset
        The observed data.
    n_shards : int
        Number of shards.
    shard_by : str, sequence of str or None
        Column(s) whose rows must stay on the same shard.
    backend : str
        Ray Dataset input only: the dataframe library ``model_fn`` receives.
    num_cpus_per_shard : float
        CPUs reserved by each task.
    """

    def __init__(self, model_fn, data, n_shards, shard_by, backend, num_cpus_per_shard):
        ensure_ray_initialized()
        by = [shard_by] if isinstance(shard_by, str) else shard_by
        self.shards, self.positions = split_with_positions(data, n_shards, by)
        self.n_shards = len(self.shards)
        self.ordered = self.positions is not None or by is None
        self.rows = None
        self._model_fn = ray.put(model_fn)
        self._backend = backend
        self._task = _on_shard.options(num_cpus=num_cpus_per_shard)

    def run(self, work: Callable, per_shard_args: Sequence[tuple]) -> list[Any]:
        """Run ``work(model, *args)`` on every shard, each in its own Ray task.

        Each task loads its shard and builds the model on it. The number of rows of every
        shard is recorded in :attr:`rows`.

        Parameters
        ----------
        work : callable
            ``work(model, *args)``, run on each shard's model.
        per_shard_args : sequence of tuple
            Extra positional arguments for each shard, in shard order.

        Returns
        -------
        list
            The result of every shard, in shard order.
        """
        calls = zip(self.shards, per_shard_args, strict=True)
        results = ray.get([self._task.remote(self._model_fn, s, self._backend, work, *a) for s, a in calls])
        self.rows = [rows for rows, _ in results]
        return [output for _, output in results]


# Putting the shards' rows back together.


def _row_dim(var: xr.DataArray) -> str | None:
    """Find the first dimension of ``var`` that is not ``chain`` or ``draw``.

    Parameters
    ----------
    var : xarray.DataArray
        A result of one shard.

    Returns
    -------
    str or None
        The dimension taken to index the shard's rows, or None if there is none.
    """
    dims = [dim for dim in var.dims if dim not in SAMPLE_DIMS]
    return dims[0] if dims else None


def _stitch_array(name: str, parts: list[xr.DataArray], shards) -> xr.DataArray:
    """Concatenate one variable's per-shard results along its row dimension.

    Rows are put back in the order of the data, and default row coords (``0`` to
    ``n - 1`` on every shard) are renumbered over all rows.

    Parameters
    ----------
    name : str
        The variable's name, for error messages.
    parts : list of xarray.DataArray
        The variable on every shard, in shard order.
    shards : ActorShards or TaskShards
        The shards that produced ``parts``.

    Returns
    -------
    xarray.DataArray
        The variable over all rows of the data.

    Raises
    ------
    ValueError
        If the variable has no dimension with one entry per row of each shard, or if its
        rows cannot be put back in order (a Ray Dataset split by ``shard_by``, with
        default row coords).
    """
    dim = _row_dim(parts[0])
    sizes = [p.sizes.get(dim) for p in parts]
    if dim is None or sizes != list(shards.rows):
        raise ValueError(
            f"{name!r} has no dimension with one entry per row of the data, so its draws "
            "cannot be split across shards. Sample it on the driver with "
            "pm.sample_posterior_predictive and the zero-row model."
        )
    default_coords = all(dim not in p.coords or np.array_equal(p[dim].values, np.arange(rows)) for p, rows in zip(parts, shards.rows, strict=True))
    stitched = xr.concat(parts, dim=dim)
    if shards.positions is not None:
        stitched = stitched.isel({dim: np.argsort(np.concatenate(shards.positions), kind="stable")})
    elif not shards.ordered and default_coords:
        raise ValueError(
            f"The rows of {name!r} cannot be put back in their original order: the data are "
            "a Ray Dataset split by shard_by. Give the observations a dim whose coords come "
            "from an id column, or use postprocess_fn to handle each shard's rows."
        )
    if default_coords:
        stitched = stitched.assign_coords({dim: np.arange(stitched.sizes[dim])})
    return stitched


def stitch(parts: list[xr.Dataset | xr.DataArray], shards) -> xr.Dataset | xr.DataArray:
    """Concatenate per-shard results along their row dimension, in the original row order.

    Parameters
    ----------
    parts : list of xarray.Dataset or list of xarray.DataArray
        The result of every shard, in shard order.
    shards : ActorShards or TaskShards
        The shards that produced ``parts``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The results over all rows of the data, of the same type as ``parts``.
    """
    if isinstance(parts[0], xr.DataArray):
        return _stitch_array(str(parts[0].name), parts, shards)
    stitched = {name: _stitch_array(name, [p[name] for p in parts], shards) for name in parts[0]}
    return xr.Dataset(stitched, attrs=parts[0].attrs)


# The computations, for either kind of shards.


def _posterior_of(idata: xr.DataTree | xr.Dataset) -> xr.Dataset:
    """Extract the posterior of ``idata``, without its empty variables.

    Parameters
    ----------
    idata : xarray.DataTree or xarray.Dataset
        A result with a ``posterior`` group, or the posterior itself.

    Returns
    -------
    xarray.Dataset
        The posterior, without zero-size variables.
    """
    posterior = idata["posterior"].to_dataset() if isinstance(idata, xr.DataTree) else idata
    return posterior.drop_vars([name for name in posterior.data_vars if posterior[name].size == 0])


def _thin(posterior: xr.Dataset, num_draws: int | None) -> xr.Dataset:
    """Keep ``num_draws`` evenly spaced draws of every chain.

    Parameters
    ----------
    posterior : xarray.Dataset
        The posterior.
    num_draws : int or None
        Number of draws to keep. None, or at least the number of draws, keeps them all.

    Returns
    -------
    xarray.Dataset
        The thinned posterior.
    """
    if num_draws is None or num_draws >= posterior.sizes["draw"]:
        return posterior
    keep = np.unique(np.linspace(0, posterior.sizes["draw"] - 1, num_draws).round().astype(int))
    return posterior.isel(draw=keep)


def relative_efficiency(posterior: xr.Dataset) -> float:
    """Relative MCMC efficiency for PSIS, as ``arviz.loo`` computes it by default.

    Parameters
    ----------
    posterior : xarray.Dataset
        The posterior.

    Returns
    -------
    float
        The mean effective sample size over all parameters, divided by the number of
        samples; 1 for a single chain.
    """
    n_chains, n_draws = posterior.sizes["chain"], posterior.sizes["draw"]
    if n_chains == 1:
        return 1.0
    ess = az.ess(posterior, method="mean")
    values = np.hstack([np.asarray(ess[name]).ravel() for name in ess.data_vars])
    return float(values.mean() / (n_chains * n_draws))


def _run_posterior_predictive(
    shards,
    idata,
    *,
    var_names=None,
    num_draws=None,
    random_seed=None,
    postprocess_fn=None,
    extend_inferencedata=False,
):
    """Posterior predictive sampling over either kind of shards.

    Shared by :func:`sample_posterior_predictive` and
    :meth:`ShardedModel.sample_posterior_predictive`.

    Parameters
    ----------
    shards : ActorShards or TaskShards
        Where each shard's draws are sampled.
    idata : xarray.DataTree or xarray.Dataset
        A posterior.
    var_names : sequence of str, optional
        Variables to sample. Defaults to the observed variables.
    num_draws : int, optional
        Use this many evenly spaced draws per chain.
    random_seed : int, optional
        Base seed; every shard gets an independent stream.
    postprocess_fn : callable, optional
        ``postprocess_fn(shard, model, result) -> Any``, applied to each shard's result
        where it is computed.
    extend_inferencedata : bool
        Add the groups to ``idata`` (a ``DataTree``) and return it.

    Returns
    -------
    xarray.DataTree or list
        The stitched groups (or ``idata`` extended with them), or the list of
        ``postprocess_fn`` results.
    """
    ensure_ray_initialized()
    posterior = ray.put(_thin(_posterior_of(idata), num_draws))
    seeds = spawn_seeds(random_seed, shards.n_shards)
    args = [(posterior, var_names, seed, i, postprocess_fn) for i, seed in enumerate(seeds)]
    results = shards.run(shard_posterior_predictive, args)
    if postprocess_fn is not None:
        return results
    groups = {name: stitch([result[name] for result in results], shards) for name in results[0]}
    if extend_inferencedata and isinstance(idata, xr.DataTree):
        for name, group in groups.items():
            idata[name] = group
        return idata
    return xr.DataTree.from_dict(groups)


def _run_log_likelihood(shards, idata, *, var_names=None, num_draws=None, extend_inferencedata=True):
    """Pointwise log-likelihood over either kind of shards.

    Shared by :func:`compute_log_likelihood` and :meth:`ShardedModel.compute_log_likelihood`.

    Parameters
    ----------
    shards : ActorShards or TaskShards
        Where each shard's log-likelihood is computed.
    idata : xarray.DataTree or xarray.Dataset
        A posterior.
    var_names : sequence of str, optional
        Observed variables to compute. Defaults to all of them.
    num_draws : int, optional
        Use this many evenly spaced draws per chain.
    extend_inferencedata : bool
        Add the ``log_likelihood`` group to ``idata`` (a ``DataTree``) and return it.

    Returns
    -------
    xarray.DataTree
        ``idata`` with a ``log_likelihood`` group, or a new tree holding only that group.
    """
    ensure_ray_initialized()
    posterior = ray.put(_thin(_posterior_of(idata), num_draws))
    args = [(posterior, var_names)] * shards.n_shards
    log_lik = stitch(shards.run(shard_log_likelihood, args), shards)
    if extend_inferencedata and isinstance(idata, xr.DataTree):
        idata["log_likelihood"] = log_lik
        return idata
    return xr.DataTree.from_dict({"log_likelihood": log_lik})


def _run_loo(shards, idata, *, var_name=None, reff=None) -> ELPDDataLOO:
    """PSIS-LOO over either kind of shards, combined from their pointwise values.

    Shared by :func:`loo` and :meth:`ShardedModel.loo`.

    Parameters
    ----------
    shards : ActorShards or TaskShards
        Where each shard's pointwise values are computed.
    idata : xarray.DataTree or xarray.Dataset
        A posterior.
    var_name : str, optional
        Observed variable to cross-validate. Required when there are several.
    reff : float, optional
        Relative MCMC efficiency. Defaults to :func:`relative_efficiency` of the posterior.

    Returns
    -------
    arviz_stats.utils.ELPDDataLOO
        The LOO estimate over all rows, as ``arviz.loo`` returns it.

    Warns
    -----
    UserWarning
        Once, for all rows, if a Pareto shape estimate exceeds the threshold for reliable
        importance sampling.
    """
    ensure_ray_initialized()
    posterior = _posterior_of(idata)
    if reff is None:
        reff = relative_efficiency(posterior)
    n_samples = posterior.sizes["chain"] * posterior.sizes["draw"]
    ref = ray.put(posterior)
    results = shards.run(shard_loo, [(ref, var_name, reff)] * shards.n_shards)
    elpd_i = stitch([elpd for elpd, _, _ in results], shards)
    pareto_k = stitch([k for _, k, _ in results], shards)

    values = np.asarray(elpd_i).ravel()
    n = values.size
    good_k = min(1 - 1 / np.log10(n_samples), 0.7) if n_samples > 1 else 0.7
    warning = bool(np.any(np.asarray(pareto_k) > good_k))
    if warning:
        warnings.warn(
            f"Estimated shape parameter of Pareto distribution is greater than {good_k:.2f} "
            "for one or more observations. Importance sampling is unreliable for them; "
            "consider a more robust model.",
            UserWarning,
            stacklevel=3,
        )
    return ELPDDataLOO(
        elpd=float(values.sum()),
        se=float(np.sqrt(n * np.var(values, ddof=1))) if n > 1 else 0.0,
        p=float(sum(p for _, _, p in results)),
        n_samples=n_samples,
        n_data_points=n,
        scale="log",
        warning=warning,
        good_k=good_k,
        elpd_i=elpd_i,
        pareto_k=pareto_k,
    )


# Public functions for posteriors from anywhere (consensus_sample, saved results, ...).


def sample_posterior_predictive(
    idata: xr.DataTree | xr.Dataset,
    model_fn: ModelFn,
    data: Data,
    n_shards: int,
    *,
    var_names: Sequence[str] | None = None,
    num_draws: int | None = None,
    random_seed: int | None = None,
    postprocess_fn: PostprocessFn | None = None,
    extend_inferencedata: bool = False,
    shard_by: str | Sequence[str] | None = None,
    backend: str = "pandas",
    num_cpus_per_shard: float = 1,
) -> xr.DataTree | list[Any]:
    """Posterior predictive draws for every row of ``data``, computed shard by shard.

    Like ``pm.sample_posterior_predictive``, but each Ray task samples the rows of one
    shard, so the observed data never have to fit in one process.

    Parameters
    ----------
    idata : xarray.DataTree or xarray.Dataset
        A posterior, e.g. from :func:`consensus_sample` or :meth:`ShardedModel.sample`.
        Only its free variables are used; everything that depends on the data is
        recomputed on the shards.
    model_fn : callable
        ``model_fn(data) -> pm.Model``, as for :class:`ShardedModel`.
    data : dataframe, dict of arrays or ray.data.Dataset
        The observed data. Each Ray task loads one shard.
    n_shards : int
        Number of shards (Ray tasks).
    var_names : sequence of str, optional
        Variables to sample. Defaults to the observed variables. Each must have one entry
        per row of the data along its first non-sample dimension.
    num_draws : int, optional
        Use this many evenly spaced draws per chain. The result holds
        ``chains x num_draws x rows`` values for every variable.
    random_seed : int, optional
        Seed for reproducible draws; every shard gets an independent stream.
    postprocess_fn : callable, optional
        ``postprocess_fn(shard, model, result) -> Any``, run on the worker with the shard's
        index, its model and its ``DataTree`` (``posterior_predictive`` and
        ``observed_data`` for that shard's rows). Use it to write each shard's draws to
        storage and return the paths, when the full result would not fit in memory. A list
        of the returned values, in shard order, is returned instead of a ``DataTree``.
    extend_inferencedata : bool
        Add the groups to ``idata`` (a ``DataTree``) and return it.
    shard_by : str or sequence of str, optional
        Keep each group's rows on one shard, as in :class:`ShardedModel`.
    backend : str
        Ray Dataset input only: the dataframe library ``model_fn`` receives.
    num_cpus_per_shard : float
        CPUs reserved by each task.

    Returns
    -------
    xarray.DataTree or list
        ``posterior_predictive`` and ``observed_data`` groups, with rows in the order of
        ``data``; or the list of ``postprocess_fn`` results.
    """
    shards = TaskShards(model_fn, data, n_shards, shard_by, backend, num_cpus_per_shard)
    return _run_posterior_predictive(
        shards,
        idata,
        var_names=var_names,
        num_draws=num_draws,
        random_seed=random_seed,
        postprocess_fn=postprocess_fn,
        extend_inferencedata=extend_inferencedata,
    )


def compute_log_likelihood(
    idata: xr.DataTree | xr.Dataset,
    model_fn: ModelFn,
    data: Data,
    n_shards: int,
    *,
    var_names: Sequence[str] | None = None,
    num_draws: int | None = None,
    extend_inferencedata: bool = True,
    shard_by: str | Sequence[str] | None = None,
    backend: str = "pandas",
    num_cpus_per_shard: float = 1,
) -> xr.DataTree:
    """Pointwise log-likelihood of every row of ``data``, computed shard by shard.

    Like ``pm.compute_log_likelihood``. The result holds ``chains x draws x rows`` values
    per observed variable; for LOO on data too large for that, use :func:`loo`.

    Parameters
    ----------
    idata : xarray.DataTree or xarray.Dataset
        A posterior, e.g. from :func:`consensus_sample` or :meth:`ShardedModel.sample`.
        Only its free variables are used; everything that depends on the data is
        recomputed on the shards.
    model_fn : callable
        ``model_fn(data) -> pm.Model``, as for :class:`ShardedModel`.
    data : dataframe, dict of arrays or ray.data.Dataset
        The observed data. Each Ray task loads one shard.
    n_shards : int
        Number of shards (Ray tasks).
    var_names : sequence of str, optional
        Observed variables to compute. Defaults to all of them.
    num_draws : int, optional
        Use this many evenly spaced draws per chain.
    extend_inferencedata : bool
        Add a ``log_likelihood`` group to ``idata`` (a ``DataTree``) and return it.
    shard_by : str or sequence of str, optional
        Keep each group's rows on one shard, as in :class:`ShardedModel`.
    backend : str
        Ray Dataset input only: the dataframe library ``model_fn`` receives.
    num_cpus_per_shard : float
        CPUs reserved by each task.

    Returns
    -------
    xarray.DataTree
        ``idata`` with a ``log_likelihood`` group, with rows in the order of ``data``; or,
        without ``extend_inferencedata`` (or for a ``Dataset``), a new tree holding only
        that group.
    """
    shards = TaskShards(model_fn, data, n_shards, shard_by, backend, num_cpus_per_shard)
    return _run_log_likelihood(
        shards,
        idata,
        var_names=var_names,
        num_draws=num_draws,
        extend_inferencedata=extend_inferencedata,
    )


def loo(
    idata: xr.DataTree | xr.Dataset,
    model_fn: ModelFn,
    data: Data,
    n_shards: int,
    *,
    var_name: str | None = None,
    reff: float | None = None,
    shard_by: str | Sequence[str] | None = None,
    backend: str = "pandas",
    num_cpus_per_shard: float = 1,
) -> ELPDDataLOO:
    """PSIS-LOO cross-validation, computed shard by shard.

    Each shard computes the pointwise log-likelihood of its rows and their PSIS-LOO
    values, and returns only one value per row. The result equals ``arviz.loo`` on the
    full data and works with ``arviz.compare``.

    Parameters
    ----------
    idata : xarray.DataTree or xarray.Dataset
        A posterior, e.g. from :func:`consensus_sample` or :meth:`ShardedModel.sample`.
        Only its free variables are used; everything that depends on the data is
        recomputed on the shards.
    model_fn : callable
        ``model_fn(data) -> pm.Model``, as for :class:`ShardedModel`.
    data : dataframe, dict of arrays or ray.data.Dataset
        The observed data. Each Ray task loads one shard.
    n_shards : int
        Number of shards (Ray tasks).
    var_name : str, optional
        Observed variable to cross-validate. Required when there are several.
    reff : float, optional
        Relative MCMC efficiency. Defaults to the value ``arviz.loo`` would compute from
        the posterior.
    shard_by : str or sequence of str, optional
        Keep each group's rows on one shard, as in :class:`ShardedModel`.
    backend : str
        Ray Dataset input only: the dataframe library ``model_fn`` receives.
    num_cpus_per_shard : float
        CPUs reserved by each task.

    Returns
    -------
    arviz_stats.utils.ELPDDataLOO
        The LOO estimate, with pointwise ``elpd_i`` and ``pareto_k`` in the order of
        ``data``.

    Warns
    -----
    UserWarning
        If a Pareto shape estimate exceeds the threshold for reliable importance
        sampling.
    """
    shards = TaskShards(model_fn, data, n_shards, shard_by, backend, num_cpus_per_shard)
    return _run_loo(shards, idata, var_name=var_name, reff=reff)
