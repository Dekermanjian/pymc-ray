"""Consensus Monte Carlo: embarrassingly parallel, approximate posterior inference.

The data are split into S shards and each shard's "subposterior", ``p(theta)^(1/S) * p(shard | theta)``,
is sampled independently on its own Ray worker. The product of the subposteriors is the full posterior.
The subposterior draws are combined to approximate it, which is exact when every
subposterior is Gaussian (Scott et al., 2016, "Bayes and big data: the consensus Monte
Carlo algorithm"). Draws are combined on the unconstrained scale and then transformed
back, so positivity, simplex and similar constraints are respected.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import ray
import xarray as xr

from pymc_ray._utils import (
    Data,
    ModelFn,
    empty_like,
    ensure_ray_initialized,
    materialize,
    spawn_seeds,
    split_data,
    validate_likelihood_potentials,
)

CombineMethod = Literal["consensus", "gaussian"]
Weights = Literal["full", "diag"]


def temper_prior(model: pm.Model, power: float, likelihood_potentials: Sequence[str] = ()) -> None:
    """Raise the prior of ``model`` to ``power``, in place, by adding a Potential.

    The prior is the log-density of every free random variable plus every Potential not
    listed in ``likelihood_potentials``.

    Parameters
    ----------
    model : pm.Model
        The model to modify. A Potential named ``pymc_ray_prior_tempering`` is added.
    power : float
        Exponent of the prior: ``1 / n_shards`` for a consensus subposterior.
    likelihood_potentials : sequence of str
        Potentials of ``model`` that depend on the data, which are not tempered.

    Raises
    ------
    ValueError
        If a name in ``likelihood_potentials`` is not a Potential of the model.
    """
    validate_likelihood_potentials(model, tuple(likelihood_potentials))
    terms = [pt.sum(pm.logp(rv, rv)) for rv in model.free_RVs]
    terms += [pt.sum(pot) for pot in model.potentials if pot.name not in likelihood_potentials]
    with model:
        pm.Potential("pymc_ray_prior_tempering", (power - 1) * pt.sum(pt.stack(terms)))


@ray.remote
def _sample_subposterior(
    model_fn: ModelFn,
    shard: Any,
    prior_power: float,
    likelihood_potentials: tuple[str, ...],
    sample_kwargs: dict[str, Any],
    seed: int | None,
    backend: str,
) -> tuple[xr.DataTree, Data]:
    """Sample the subposterior of one shard on a Ray worker.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(data) -> pm.Model``.
    shard : object
        The shard's data, or a Ray Dataset shard to load as ``backend``.
    prior_power : float
        Exponent of the prior, ``1 / n_shards``.
    likelihood_potentials : tuple of str
        Potentials that depend on the data, which are not tempered.
    sample_kwargs : dict
        Keyword arguments for ``pm.sample``.
    seed : int or None
        Seed for ``pm.sample``.
    backend : str
        The dataframe library to load a Ray Dataset shard as.

    Returns
    -------
    idata : xarray.DataTree
        The ``pm.sample`` result.
    empty : dataframe or dict of arrays
        Zero rows of the shard, to build the model on the driver.
    """
    data = materialize(shard, backend)
    model = model_fn(data)
    temper_prior(model, prior_power, likelihood_potentials)
    idata = pm.sample(model=model, random_seed=seed, **sample_kwargs)
    return idata, empty_like(data)


def consensus_sample(
    model_fn: ModelFn,
    data: Data,
    n_shards: int,
    *,
    sample_kwargs: dict[str, Any] | None = None,
    method: CombineMethod = "consensus",
    weights: Weights = "full",
    likelihood_potentials: Sequence[str] = (),
    num_cpus_per_shard: float | None = None,
    random_seed: int | None = None,
    return_subposteriors: bool = False,
    backend: str = "pandas",
) -> xr.DataTree | tuple[xr.DataTree, list[xr.DataTree]]:
    """Approximate the full-data posterior by combining independently sampled shards.

    .. warning::
        Do not use this for multilevel (hierarchical) models. Tempering the prior of the
        group-level parameters lets every shard attribute group variation to them, so
        each shard, and therefore the combination, underestimates the group-level
        standard deviations by roughly ``1/sqrt(n_shards)``. Use :class:`ShardedModel`,
        which is exact. Consensus suits models whose parameters are all informed by every
        shard, such as regressions without group-level effects.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(data) -> pm.Model``, called once per shard on a Ray worker. Its free
        random variables (names and shapes) must not depend on the rows it receives.
    data : dataframe, dict of arrays or ray.data.Dataset
        The full dataset: an eager dataframe from any library narwhals supports, a dict
        of equal-length arrays, or a Ray Dataset, which is split on the cluster and each
        shard is loaded by the worker that samples it.
    n_shards : int
        Number of shards. Each shard should hold enough rows that its subposterior is
        close to Gaussian; the approximation degrades as shards shrink.
    sample_kwargs : dict, optional
        Passed to ``pm.sample`` on every shard. Defaults to ``progressbar=False,
        cores=1, nuts_sampler="pymc"``. Every shard must produce the same number of
        chains and draws.
    method : {"consensus", "gaussian"}
        ``"consensus"`` averages the s-th draw of every shard, weighted by each shard's
        posterior precision (Scott et al., 2016). ``"gaussian"`` fits a Gaussian to each
        subposterior and samples from their product.
    weights : {"full", "diag"}
        Use full precision matrices or only their diagonal. ``"diag"`` scales to many
        parameters but ignores posterior correlations.
    likelihood_potentials : sequence of str
        Potentials in ``model_fn`` that depend on the data. All other Potentials are
        treated as part of the prior and tempered.
    num_cpus_per_shard : float, optional
        CPUs reserved per shard. Defaults to ``sample_kwargs["cores"]``.
    random_seed : int, optional
        Seed for sampling the shards and, for ``method="gaussian"``, for the combination.
    return_subposteriors : bool
        Also return the per-shard ``DataTree`` objects, e.g. to check convergence.
    backend : str
        Ray Dataset input only: the dataframe library ``model_fn`` receives each shard
        as, by narwhals name (``"pandas"``, ``"polars"``, ``"pyarrow"``, ...).

    Returns
    -------
    combined : xarray.DataTree
        A ``posterior`` group holding the combined draws of every free random variable.
        Deterministics are not included; recompute them with ``pm.compute_deterministics``.
    subposteriors : list of xarray.DataTree, optional
        The ``pm.sample`` result of every shard, in shard order. Only returned when
        ``return_subposteriors`` is True.
    """
    # PyMC's own NUTS: nutpie (PyMC's default when installed) does not return the
    # unconstrained draws that the combination needs.
    defaults = {"progressbar": False, "cores": 1, "nuts_sampler": "pymc"}
    sample_kwargs = {**defaults, **(sample_kwargs or {})}
    sample_kwargs["idata_kwargs"] = {
        **sample_kwargs.get("idata_kwargs", {}),
        "include_transformed": True,
    }
    if num_cpus_per_shard is None:
        num_cpus_per_shard = sample_kwargs.get("cores") or 1

    ensure_ray_initialized()
    seeds = spawn_seeds(random_seed, n_shards + 1)
    model_fn_ref = ray.put(model_fn)
    sample_shard = _sample_subposterior.options(num_cpus=num_cpus_per_shard)
    refs = [
        sample_shard.remote(
            model_fn_ref,
            shard,
            1 / n_shards,
            tuple(likelihood_potentials),
            sample_kwargs,
            seed,
            backend,
        )
        for shard, seed in zip(split_data(data, n_shards), seeds, strict=False)
    ]
    outcomes = ray.get(refs)
    subposteriors = [idata for idata, _ in outcomes]

    model = model_fn(outcomes[0][1])
    combined = combine_subposteriors(subposteriors, model, method=method, weights=weights, random_seed=seeds[-1])
    return (combined, subposteriors) if return_subposteriors else combined


def combine_subposteriors(
    subposteriors: Sequence[xr.DataTree],
    model: pm.Model,
    *,
    method: CombineMethod = "consensus",
    weights: Weights = "full",
    random_seed: int | None = None,
) -> xr.DataTree:
    """Combine subposterior draws into draws from (an approximation of) the full posterior.

    Parameters
    ----------
    subposteriors : sequence of xarray.DataTree
        One ``pm.sample`` result per shard, sampled with
        ``idata_kwargs={"include_transformed": True}`` and identical chains and draws.
    model : pm.Model
        Any instance of the model (e.g. built on zero rows), used for its parameter
        transforms and to map unconstrained draws back to the constrained scale.
    method : {"consensus", "gaussian"}
        How to combine the draws; see :func:`consensus_sample`.
    weights : {"full", "diag"}
        Use full precision matrices or only their diagonal; see :func:`consensus_sample`.
    random_seed : int, optional
        Seed for drawing from the product of Gaussians (``method="gaussian"``).

    Returns
    -------
    xarray.DataTree
        A ``posterior`` group holding the combined draws of every free random variable,
        on the constrained scale, with the chains and draws of the subposteriors.

    Raises
    ------
    ValueError
        If ``method`` or ``weights`` is invalid, if the subposteriors differ in chains or
        draws or lack their unconstrained draws, or if a subposterior covariance is
        singular (``weights="full"``).
    """
    if method not in ("consensus", "gaussian"):
        raise ValueError(f"method must be 'consensus' or 'gaussian', got {method!r}")
    if weights not in ("full", "diag"):
        raise ValueError(f"weights must be 'full' or 'diag', got {weights!r}")

    posteriors = [sub["posterior"].to_dataset() for sub in subposteriors]
    n_chains, n_draws = posteriors[0].sizes["chain"], posteriors[0].sizes["draw"]
    for posterior in posteriors:
        if (posterior.sizes["chain"], posterior.sizes["draw"]) != (n_chains, n_draws):
            raise ValueError("All subposteriors must have the same number of chains and draws")

    names = [value.name for value in model.value_vars]
    missing = [name for name in names if name not in posteriors[0]]
    if missing:
        raise ValueError(f"Subposteriors lack unconstrained draws for {missing}; sample them with idata_kwargs={{'include_transformed': True}}")
    shapes = [posteriors[0][name].shape[2:] for name in names]
    draws = [np.concatenate([p[name].to_numpy().reshape(n_chains * n_draws, -1) for name in names], axis=1) for p in posteriors]

    if method == "consensus":
        combined = _consensus_average(draws, weights)
    else:
        combined = _gaussian_product(draws, weights, np.random.default_rng(random_seed))

    to_constrained = pm.compile(model.value_vars, model.replace_rvs_by_values(model.free_RVs), on_unused_input="ignore")
    split_at = np.cumsum([int(np.prod(shape)) for shape in shapes])[:-1]
    constrained = [to_constrained(*(part.reshape(shape) for part, shape in zip(np.split(row, split_at), shapes, strict=True))) for row in combined]

    variables = {}
    for i, rv in enumerate(model.free_RVs):
        template = posteriors[0][rv.name]
        values = np.stack([sample[i] for sample in constrained]).reshape(template.shape)
        variables[rv.name] = template.copy(data=values)
    attrs = {
        **posteriors[0].attrs,
        "combined_by": "pymc-ray",
        "combination_method": method,
        "combination_weights": weights,
        "n_shards": len(subposteriors),
    }
    return xr.DataTree.from_dict({"posterior": xr.Dataset(variables, attrs=attrs)})


def _consensus_average(draws: list[np.ndarray], weights: Weights) -> np.ndarray:
    """Precision-weighted average of the s-th draw of every shard (Scott et al., 2016).

    Parameters
    ----------
    draws : list of numpy.ndarray
        One ``(n_samples, n_params)`` array of unconstrained draws per shard.
    weights : {"full", "diag"}
        Weight by full precision matrices or only by their diagonal.

    Returns
    -------
    numpy.ndarray
        The ``(n_samples, n_params)`` combined draws.
    """
    if weights == "diag":
        precisions = [1 / np.var(x, axis=0, ddof=1) for x in draws]
        return sum(w * x for w, x in zip(precisions, draws, strict=True)) / sum(precisions)
    precisions = [_precision(x) for x in draws]
    weighted_sum = sum(x @ w for w, x in zip(precisions, draws, strict=True))
    return np.linalg.solve(sum(precisions), weighted_sum.T).T


def _gaussian_product(draws: list[np.ndarray], weights: Weights, rng: np.random.Generator) -> np.ndarray:
    """Draws from the product of Gaussian approximations of every subposterior.

    Parameters
    ----------
    draws : list of numpy.ndarray
        One ``(n_samples, n_params)`` array of unconstrained draws per shard.
    weights : {"full", "diag"}
        Use full precision matrices or only their diagonal.
    rng : numpy.random.Generator
        Source of the new draws.

    Returns
    -------
    numpy.ndarray
        ``(n_samples, n_params)`` draws from the product.
    """
    if weights == "diag":
        precisions = [np.diag(1 / np.var(x, axis=0, ddof=1)) for x in draws]
    else:
        precisions = [_precision(x) for x in draws]
    covariance = np.linalg.inv(sum(precisions))
    mean = covariance @ sum(w @ x.mean(axis=0) for w, x in zip(precisions, draws, strict=True))
    return rng.multivariate_normal(mean, covariance, size=draws[0].shape[0])


def _precision(x: np.ndarray) -> np.ndarray:
    """Inverse of the sample covariance of draws.

    Parameters
    ----------
    x : numpy.ndarray
        ``(n_samples, n_params)`` draws.

    Returns
    -------
    numpy.ndarray
        The ``(n_params, n_params)`` precision matrix.

    Raises
    ------
    ValueError
        If the covariance is singular.
    """
    covariance = np.atleast_2d(np.cov(x, rowvar=False))
    try:
        return np.linalg.inv(covariance)
    except np.linalg.LinAlgError as err:
        raise ValueError(
            "A subposterior covariance is singular (more parameters than draws, or a "
            "parameter that never moved). Draw more samples or use weights='diag'."
        ) from err
