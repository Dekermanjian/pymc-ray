"""Timing of joint logp + gradient evaluations, alone or from concurrent chain processes."""

from __future__ import annotations

import multiprocessing
import time
import traceback

import cloudpickle
import numpy as np
import pymc as pm
import pytensor


def time_logp_dlogp(model: pm.Model, n_evals: int = 20, chains: int = 1) -> float:
    """Mean wall time, in seconds, of one joint logp and gradient evaluation of ``model``.

    NUTS performs one such evaluation per leapfrog step (up to ``2**max_treedepth`` per
    draw), so this bounds how fast the model can be sampled.

    With ``chains > 1``, evaluations run concurrently in that many spawned processes (as
    ``pm.sample`` runs chains) and the mean over processes is returned. Concurrent chains
    compete for memory bandwidth and, for a :class:`~pymc_ray.ShardedModel`, for the
    shard actors, so this is the number to compare when choosing shards and replicas.

    Parameters
    ----------
    model : pm.Model
        The model to time, evaluated at its initial point after one untimed warm-up
        evaluation. With ``chains > 1`` it must be picklable with cloudpickle.
    n_evals : int
        Number of timed evaluations per process.
    chains : int
        Number of processes that evaluate concurrently.

    Returns
    -------
    float
        Mean seconds per evaluation, averaged over processes when ``chains > 1``.

    Raises
    ------
    RuntimeError
        If a timing process fails; the message holds its traceback.
    """
    if chains == 1:
        logp_dlogp, args = _compile_logp_dlogp(model)
        logp_dlogp(*args)
        return _time_calls(logp_dlogp, args, n_evals)

    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(chains), ctx.Queue()
    payload = cloudpickle.dumps(model)
    processes = [ctx.Process(target=_timing_worker, args=(payload, n_evals, barrier, results), daemon=True) for _ in range(chains)]
    for process in processes:
        process.start()
    times = [results.get() for _ in processes]
    for process in processes:
        process.join()
    for outcome in times:
        if isinstance(outcome, str):
            raise RuntimeError(f"A timing process failed:\n{outcome}")
    return float(np.mean(times))


def _compile_logp_dlogp(model: pm.Model):
    """Compile the joint logp and its gradient, and the arguments to evaluate them at.

    Parameters
    ----------
    model : pm.Model
        The model to compile.

    Returns
    -------
    logp_dlogp : callable
        Function of the model's value variables that returns ``[logp, *grads]``.
    args : list of numpy.ndarray
        The model's initial point, in the order of its value variables.
    """
    logp = model.logp()
    values = model.value_vars
    logp_dlogp = pm.compile(values, [logp, *pytensor.grad(logp, values)])
    point = model.initial_point()
    return logp_dlogp, [point[value.name] for value in values]


def _time_calls(fn, args, n_evals: int) -> float:
    """Mean wall time, in seconds, of ``n_evals`` calls of ``fn(*args)``.

    Parameters
    ----------
    fn : callable
        The function to time.
    args : sequence
        Positional arguments passed to every call.
    n_evals : int
        Number of calls.

    Returns
    -------
    float
        Mean seconds per call.
    """
    start = time.perf_counter()
    for _ in range(n_evals):
        fn(*args)
    return (time.perf_counter() - start) / n_evals


def _timing_worker(payload: bytes, n_evals: int, barrier, results) -> None:
    """Time evaluations of a pickled model in a spawned process and report the result.

    Parameters
    ----------
    payload : bytes
        The model, pickled with cloudpickle.
    n_evals : int
        Number of timed evaluations.
    barrier : multiprocessing.Barrier
        Shared by all workers, which wait at it before timing. Aborted on failure, so
        that the other workers do not wait forever.
    results : multiprocessing.Queue
        Receives the mean seconds per evaluation, or the formatted traceback (a str) if
        this worker fails.
    """
    try:
        logp_dlogp, args = _compile_logp_dlogp(cloudpickle.loads(payload))
        # Warm up (compile, connect to Ray, lease a replica) before the clocks start, so
        # that every process times its evaluations while the others are evaluating too.
        logp_dlogp(*args)
        barrier.wait()
        results.put(_time_calls(logp_dlogp, args, n_evals))
    except BaseException:
        barrier.abort()
        results.put(traceback.format_exc())
